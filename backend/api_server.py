import sys, os, json, time, subprocess, threading, re, asyncio, secrets, contextvars
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model_manager import get_local_models
from driver_manager import get_local_drivers, get_active_drivers, find_driver_executable
from gpu_detector import detect_gpus
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import httpx

from pc_links import list_remote_models, resolve_remote_model
from served_profiles import get_override, list_overrides
from resource_safety import ResourcePressureError, resource_manager
from pairing import validate_peer_token

_server_process: Optional[subprocess.Popen] = None
_server_config: Dict[str, Any] = {}
_server_log: List[str] = []
_server_lock = threading.Lock()

app = FastAPI(title="Avalon Local Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PUBLIC_LLM_PATHS = {"/v1/models", "/v1/chat/completions", "/v1/messages"}


def _is_loopback(request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}


@app.middleware("http")
async def require_public_api_key(request, call_next):
    started = time.monotonic()
    client = request.client.host if request.client else "unknown"
    entry = None if request.url.path == "/v1/status" else _begin_request_log(request.method, request.url.path, client)
    if not _is_loopback(request) and request.url.path not in _PUBLIC_LLM_PATHS:
        from fastapi.responses import JSONResponse
        if entry:
            _finish_request_log(entry, 404, round((time.monotonic() - started) * 1000), "internal endpoint")
        return JSONResponse({"detail": "Only public LLM endpoints are available"}, status_code=404)
    if request.url.path in _PUBLIC_LLM_PATHS and not _is_loopback(request):
        expected = os.environ.get("AVALON_API_KEY", "")
        supplied = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = supplied or auth[7:].strip()
        valid_global_key = expected and supplied and secrets.compare_digest(supplied, expected)
        valid_peer_key = validate_peer_token(supplied)
        if not valid_global_key and not valid_peer_key:
            from fastapi.responses import JSONResponse
            if entry:
                _finish_request_log(entry, 401, round((time.monotonic() - started) * 1000), "invalid API key")
            return JSONResponse(
                {"error": {"message": "Valid Avalon API key required", "type": "authentication_error"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
    try:
        response = await call_next(request)
        if entry:
            _finish_request_log(entry, response.status_code, round((time.monotonic() - started) * 1000))
        return response
    except Exception as exc:
        if entry:
            _finish_request_log(entry, 500, round((time.monotonic() - started) * 1000), str(exc))
        raise


@app.exception_handler(ResourcePressureError)
async def resource_pressure_response(_request, exc: ResourcePressureError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        {
            "error": {
                "message": str(exc),
                "type": "resource_pressure",
                "resource": exc.snapshot,
            }
        },
        status_code=exc.status_code,
        headers={"Retry-After": str(exc.retry_after)},
    )

_stream_metrics_lock = threading.Lock()
_stream_metrics_seq = 0
_stream_metrics_by_model: Dict[str, Dict[str, Any]] = {}
STREAM_REQUEST_STALE_MS = 15000
_request_log_lock = threading.Lock()
_request_log: List[Dict[str, Any]] = []
REQUEST_LOG_LIMIT = 200
_request_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("avalon_request_context", default=None)


def _begin_request_log(method: str, path: str, client: str) -> Dict[str, Any]:
    entry = {
        "at_ms": int(time.time() * 1000),
        "method": method,
        "path": path,
        "status": 0,
        "duration_ms": 0,
        "client": client,
        "model_id": "",
        "model_source": "",
        "remote_model_id": "",
        "remote_pc": "",
        "streamed_token_estimate": 0,
    }
    with _request_log_lock:
        _request_log.append(entry)
        del _request_log[:-REQUEST_LOG_LIMIT]
    _request_context.set(entry)
    return entry


def _finish_request_log(entry: Dict[str, Any], status: int, duration_ms: int, detail: str = "") -> None:
    entry["status"] = status
    entry["duration_ms"] = duration_ms
    if detail:
        entry["detail"] = detail[:240]


def _annotate_request(**fields: Any) -> None:
    entry = _request_context.get()
    if entry is None:
        return
    for key, value in fields.items():
        if value is not None:
            entry[key] = value


def _annotate_latest_request(model_id: str, **fields: Any) -> None:
    with _request_log_lock:
        for entry in reversed(_request_log):
            if entry.get("status") == 0 and entry.get("model_id") == model_id:
                entry.update({key: value for key, value in fields.items() if value is not None})
                return


def _request_log_snapshot() -> List[Dict[str, Any]]:
    with _request_log_lock:
        return list(_request_log)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class OllamaChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    think: Optional[bool] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None
    format: Optional[Any] = None

class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str = ""
    messages: List[AnthropicMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# --- In-memory model cache for OpenVINO paths ---
_ov_pipes: Dict[str, Any] = {}

def _load_ov_model(model_path: str, model_id: str = ""):
    profile = _profile_for_request_model(model_id)
    ov_device = profile.get("driver") if profile and profile.get("runtime") == "openvino" else ""
    ov_device = ov_device or _server_config.get("openvino_device") or "NPU"
    cache_key = f"{ov_device}:{model_path}"
    if cache_key in _ov_pipes:
        return _ov_pipes[cache_key]
    try:
        from openvino_genai import LLMPipeline
        pipe = LLMPipeline(model_path, ov_device, {"MAX_PROMPT_LEN": 4096})
        _ov_pipes[cache_key] = pipe
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenVINO model on {ov_device}: {e}")

# --- GPU device listing for API server page ---

@app.get("/v1/available-devices")
async def available_devices():
    gpus = detect_gpus()
    backends = {}
    for g in gpus:
        b = g["backend"]
        if b not in backends:
            backends[b] = []
        backends[b].append({
            "index": g["index"],
            "name": g["name"],
            "vendor": g["vendor"],
            "memory_mb": g.get("memory_mb", 0),
        })
    return {"backends": backends, "gpus": gpus}


@app.get("/v1/status")
async def gateway_status():
    return {
        "status": "running",
        "config": {
            "model_id": _server_config.get("model_id", ""),
            "port": _server_config.get("port", 8787),
            "mode": _server_config.get("mode", "openai"),
            "device": _server_config.get("device", ""),
            "gguf_backend": _server_config.get("gguf_backend", _server_config.get("device", "")),
            "openvino_device": _server_config.get("openvino_device", "NPU"),
            "gpu_index": _server_config.get("gpu_index", ""),
        },
        "models": _served_model_catalog(),
        "resources": resource_manager.snapshot(),
        "request_log": _request_log_snapshot(),
    }

# --- OpenAI-compatible endpoints ---

@app.get("/v1/models")
async def list_models():
    models = []
    for m in _served_model_catalog():
        if m.get("is_draft") or m.get("serving_supported") is False:
            continue
        models.append({
            "id": m["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": m.get("source", "local"),
        })
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    remote = resolve_remote_model(req.model)
    if remote:
        _annotate_request(
            model_id=req.model,
            model_source="remote",
            remote_model_id=remote.get("remote_model_id", ""),
            remote_pc=remote.get("pc_id") or remote.get("name") or remote.get("local_model_id", ""),
        )
        return _proxy_openai_chat(remote, req)

    model_path = _find_model_path(req.model)
    _annotate_request(model_id=req.model, model_source="local", remote_model_id="", remote_pc="")
    if not model_path:
        raise HTTPException(404, f"Model '{req.model}' not found. Available: {[m['id'] for m in get_local_models()]}")

    is_ov = _is_openvino_model_path(model_path)
    reservation = await asyncio.to_thread(
        resource_manager.acquire,
        req.model,
        model_path,
        req.max_tokens or 512,
        _find_mtp_head(model_path) or "",
    )
    try:
        result = await asyncio.to_thread(
            _chat_ov if is_ov else _chat_llama,
            model_path,
            req,
            reservation,
        )
        if isinstance(result, StreamingResponse):
            return result
        if req.stream:
            return _stream_openai_result(result, reservation)
        reservation.release()
        return result
    except Exception:
        reservation.release()
        raise

# --- Anthropic-compatible endpoints ---

@app.post("/v1/messages")
async def messages(req: AnthropicRequest):
    remote = resolve_remote_model(req.model)
    if remote:
        _annotate_request(
            model_id=req.model,
            model_source="remote",
            remote_model_id=remote.get("remote_model_id", ""),
            remote_pc=remote.get("pc_id") or remote.get("name") or remote.get("local_model_id", ""),
        )
        return _proxy_anthropic_messages(remote, req)

    model_path = _find_model_path(req.model)
    _annotate_request(model_id=req.model, model_source="local", remote_model_id="", remote_pc="")
    if not model_path:
        raise HTTPException(404, f"Model '{req.model}' not found")

    is_ov = _is_openvino_model_path(model_path)
    reservation = await asyncio.to_thread(
        resource_manager.acquire,
        req.model,
        model_path,
        req.max_tokens or 512,
        _find_mtp_head(model_path) or "",
    )
    try:
        result = await asyncio.to_thread(
            _messages_ov if is_ov else _messages_llama,
            model_path,
            req,
            reservation,
        )
        reservation.release()
        return result
    except Exception:
        reservation.release()
        raise

# --- Ollama-compatible subset ---

@app.post("/api/chat")
async def ollama_chat(req: OllamaChatRequest):
    if req.stream:
        raise HTTPException(400, "Avalon Ollama compatibility currently supports stream=false only")

    remote = resolve_remote_model(req.model)
    if remote:
        return _proxy_ollama_chat(remote, req)

    options = req.options or {}
    messages = _ollama_messages_for_local_model(req.messages, req.think)
    chat_req = ChatRequest(
        model=req.model,
        messages=messages,
        max_tokens=int(options.get("num_predict", 512)),
        temperature=float(options.get("temperature", 0.7)),
        stream=False,
    )

    model_path = _find_model_path(chat_req.model)
    if not model_path:
        raise HTTPException(404, f"Model '{chat_req.model}' not found")

    reservation = await asyncio.to_thread(
        resource_manager.acquire,
        chat_req.model,
        model_path,
        chat_req.max_tokens or 512,
        _find_mtp_head(model_path) or "",
    )
    try:
        result = await asyncio.to_thread(
            _chat_ov if _is_openvino_model_path(model_path) else _chat_llama,
            model_path,
            chat_req,
            reservation,
        )
        reservation.release()
    except Exception:
        reservation.release()
        raise
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "model": chat_req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }

# --- Helpers ---

def _message_dicts(messages: List[Any]) -> List[Dict[str, str]]:
    result = []
    for message in messages:
        if hasattr(message, "model_dump"):
            result.append(message.model_dump())
        elif isinstance(message, dict):
            result.append(message)
        else:
            result.append({"role": getattr(message, "role", "user"), "content": getattr(message, "content", "")})
    return result


def _raise_remote_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = response.text
    try:
        parsed = response.json()
        detail = parsed.get("detail") or parsed.get("error") or detail
    except Exception:
        pass
    raise HTTPException(response.status_code, f"Remote PC error: {detail}")


def _proxy_openai_chat(remote: Dict[str, Any], req: ChatRequest) -> Dict:
    payload = {
        "model": remote["remote_model_id"],
        "messages": _message_dicts(req.messages),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{remote['base_url']}/v1/chat/completions", json=payload)
        _raise_remote_error(response)
        data = response.json()
        if isinstance(data, dict):
            data["model"] = remote["local_model_id"]
            _annotate_request(
                streamed_token_estimate=(data.get("usage") or {}).get("completion_tokens", 0),
            )
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Remote PC request failed: {e}")


def _proxy_anthropic_messages(remote: Dict[str, Any], req: AnthropicRequest) -> Dict:
    payload = {
        "model": remote["remote_model_id"],
        "messages": _message_dicts(req.messages),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{remote['base_url']}/v1/messages", json=payload)
            if response.status_code in (404, 405):
                chat_response = client.post(
                    f"{remote['base_url']}/v1/chat/completions",
                    json={
                        "model": remote["remote_model_id"],
                        "messages": payload["messages"],
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                        "stream": req.stream,
                    },
                )
                _raise_remote_error(chat_response)
                chat_data = chat_response.json()
                content = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                _annotate_request(
                    streamed_token_estimate=(chat_data.get("usage") or {}).get("completion_tokens", _estimate_text_tokens(content)),
                )
                return {
                    "id": chat_data.get("id", "msg_remote"),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                    "model": remote["local_model_id"],
                    "stop_reason": "end_turn",
                    "usage": chat_data.get("usage", {"input_tokens": 0, "output_tokens": 0}),
                }
        _raise_remote_error(response)
        data = response.json()
        if isinstance(data, dict):
            data["model"] = remote["local_model_id"]
            _annotate_request(
                streamed_token_estimate=(data.get("usage") or {}).get("output_tokens", 0),
            )
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Remote PC request failed: {e}")


def _proxy_ollama_chat(remote: Dict[str, Any], req: OllamaChatRequest) -> Dict:
    payload = {
        "model": remote["remote_model_id"],
        "messages": _message_dicts(req.messages),
        "stream": req.stream,
        "think": req.think,
        "options": req.options,
        "keep_alive": req.keep_alive,
        "format": req.format,
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{remote['base_url']}/api/chat", json=payload)
        _raise_remote_error(response)
        data = response.json()
        if isinstance(data, dict):
            data["model"] = remote["local_model_id"]
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Remote PC request failed: {e}")


def _is_openvino_model_path(model_path: str) -> bool:
    path = Path(model_path)
    return path.is_dir() and any(path.glob("openvino_*.xml"))

def _ollama_messages_for_local_model(messages: List[ChatMessage], think: Optional[bool]) -> List[ChatMessage]:
    if think is not False:
        return messages
    cloned = [ChatMessage(role=m.role, content=m.content) for m in messages]
    for message in reversed(cloned):
        if message.role == "user":
            message.content = f"{message.content}\n/no_think"
            break
    return cloned


def _base_profile_for_model(model: Dict[str, Any]) -> Dict[str, Any]:
    fmt = model.get("format", "gguf")
    runtime = "openvino" if fmt == "openvino" else "gguf"
    driver = (_server_config.get("openvino_device") or "NPU") if runtime == "openvino" else (
        _server_config.get("gguf_backend") or _server_config.get("device") or _default_gguf_backend()
    )
    return _served_model_profile(
        served_id=model["id"],
        source="local",
        runtime=runtime,
        driver=driver,
        model_file=_served_model_file(model),
        model=model,
    )


def _profile_for_request_model(model_id: str) -> Dict[str, Any]:
    if not model_id:
        return {}
    for model in get_local_models():
        profile = _base_profile_for_model(model)
        names = [model["id"], profile.get("served_id", ""), model.get("repo_id", "")]
        if model.get("files"):
            names.extend(model["files"])
        if any(name and (name == model_id or name in model_id or model_id in name) for name in names):
            return profile
    return {}


def _profile_model_path(model: Dict[str, Any], profile: Dict[str, Any]) -> str:
    model_file = profile.get("model_file", "")
    if not model_file:
        return ""
    if model.get("format") == "openvino":
        return model_file
    if os.path.isabs(model_file):
        return model_file
    return os.path.join(model.get("path", ""), model_file)

def _find_model_path(model_id: str) -> Optional[str]:
    if not model_id and _server_config.get("model_path"):
        return _server_config["model_path"]
    local_models = get_local_models()
    if not model_id and len(local_models) == 1:
        model_id = local_models[0]["id"]
    for m in local_models:
        if m.get("is_draft") or m.get("serving_supported") is False:
            continue
        profile = _base_profile_for_model(m)
        names = [m["id"], profile.get("served_id", ""), m.get("repo_id", "")]
        if m.get("files"):
            names.extend(m["files"])
        if any(name and (name == model_id or name in model_id or model_id in name) for name in names):
            override_path = _profile_model_path(m, profile)
            if override_path:
                return override_path
            if m.get("format") == "openvino" and m.get("openvino_path"):
                return m["path"]
            if m["files"]:
                return os.path.join(m["path"], m["files"][0])
    return None

def _chat_ov(model_path: str, req: ChatRequest, reservation=None) -> Dict:
    try:
        pipe = _load_ov_model(model_path, req.model)
        prompt = _format_chat_prompt_ov(req.messages)
        result = pipe.generate(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
        _annotate_request(model_id=req.model, model_source="local", streamed_token_estimate=_estimate_text_tokens(result))
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or os.path.basename(model_path),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        raise HTTPException(500, str(e))

def _messages_ov(model_path: str, req: AnthropicRequest, reservation=None) -> Dict:
    try:
        pipe = _load_ov_model(model_path, req.model)
        prompt = _format_chat_prompt_ov(req.messages)
        result = pipe.generate(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
        _annotate_request(model_id=req.model, model_source="local", streamed_token_estimate=_estimate_text_tokens(result))
        return {
            "id": "msg_local",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": result}],
            "model": req.model or os.path.basename(model_path),
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    except Exception as e:
        raise HTTPException(500, str(e))

def _format_chat_prompt_ov(messages: list) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user") if isinstance(m, dict) else m.role
        content = m.get("content", "") if isinstance(m, dict) else m.content
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

def _chat_llama(model_path: str, req: ChatRequest, reservation=None) -> Dict:
    return _run_llama_cli(model_path, req, "openai", reservation)

def _messages_llama(model_path: str, req: AnthropicRequest, reservation=None) -> Dict:
    return _run_llama_cli(model_path, req, "anthropic", reservation)

def _run_llama_cli(model_path: str, req: Any, api_format: str, reservation=None) -> Dict:
    profile = _profile_for_request_model(getattr(req, "model", ""))
    device = profile.get("driver") if profile and profile.get("runtime") == "gguf" else ""
    device = device or _server_config.get("gguf_backend") or _server_config.get("device") or _default_gguf_backend()
    driver = _find_active_gguf_driver(device)
    if not driver:
        raise HTTPException(500, f"No active driver with llama-cli found for GGUF models (device={device or 'any'})")

    cli_path = driver["cli_path"]
    prompt = _format_chat_prompt(req.messages) if hasattr(req, "messages") else str(req.messages)
    args = [
        cli_path,
        "-m",
        model_path,
        "--prompt",
        prompt,
        "-n",
        str(req.max_tokens),
        "--temp",
        str(req.temperature),
        "--single-turn",
        "--log-disable",
        "--no-display-prompt",
        "--simple-io",
    ]
    backend = driver.get("backend", "")
    if backend in ("cuda", "sycl", "vulkan", "openvino", "directml", "hip", "metal"):
        args += ["-ngl", "99"]
    gpu_index = _server_config.get("gpu_index", "")
    if gpu_index and backend in ("cuda", "sycl", "vulkan"):
        args += ["--main-gpu", gpu_index]
    mtp_head = _find_mtp_head(model_path)
    if mtp_head:
        args += [
            "--spec-draft-model",
            mtp_head,
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "3",
        ]
    if api_format == "openai" and getattr(req, "stream", False):
        return _stream_llama_cli_openai(args, req, model_path, reservation)
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
        output = r.stdout.strip()
        if r.returncode != 0:
            raise RuntimeError(f"llama-cli error: {r.stderr[:500]}")
        output = _strip_llama_output(output)
        output = _strip_model_thinking(output)
        _annotate_request(streamed_token_estimate=_estimate_text_tokens(output))
        if api_format == "openai":
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or os.path.basename(model_path),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        else:
            return {
                "id": "msg_local",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
                "model": req.model or os.path.basename(model_path),
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "llama-cli timed out")
    except Exception as e:
        raise HTTPException(500, str(e))


def _stream_llama_cli_openai(args: List[str], req: Any, model_path: str, reservation=None) -> StreamingResponse:
    created = int(time.time())
    model_id = req.model or os.path.basename(model_path)
    chunk_id = "chatcmpl-local"
    request_id = _begin_streaming_request(model_id)

    def encode_chunk(content: str = "", finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def iter_events():
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        _attach_streaming_process(model_id, request_id, proc.pid)
        stderr_parts: List[str] = []

        def drain_stderr() -> None:
            if not proc.stderr:
                return
            for line in proc.stderr:
                stderr_parts.append(line)

        threading.Thread(target=drain_stderr, daemon=True).start()

        try:
            visible_started = False
            prelude = ""
            emitted_any = False
            thinking_filter = _ThinkingBlockFilter()
            if proc.stdout:
                while True:
                    ch = proc.stdout.read(1)
                    if ch == "":
                        break
                    if not visible_started:
                        prelude += ch
                        stripped = _strip_llama_output(prelude)
                        if stripped and stripped != prelude.strip():
                            visible_started = True
                            _update_streaming_request(model_id, request_id, stripped)
                            visible = thinking_filter.feed(stripped)
                            if visible:
                                emitted_any = True
                                yield encode_chunk(visible)
                        elif len(prelude) > 65536:
                            # Avoid holding unbounded output if a driver does
                            # not echo prompts in a recognizable format.
                            visible_started = True
                            stripped = _strip_llama_output(prelude)
                            if stripped:
                                emitted_any = True
                                _update_streaming_request(model_id, request_id, stripped)
                                yield encode_chunk(stripped)
                        continue

                    _update_streaming_request(model_id, request_id, ch)
                    visible = thinking_filter.feed(ch)
                    if visible:
                        emitted_any = True
                        yield encode_chunk(visible)

            returncode = proc.wait(timeout=5)
            visible = thinking_filter.flush()
            if visible:
                emitted_any = True
                yield encode_chunk(visible)
            if returncode != 0:
                err = "".join(stderr_parts).strip()[:500]
                if not emitted_any:
                    _finish_streaming_request(model_id, request_id, "error")
                    yield encode_chunk(f"llama-cli error: {err or returncode}", "stop")
                    yield "data: [DONE]\n\n"
                    return
            _finish_streaming_request(model_id, request_id, "completed")
            yield encode_chunk("", "stop")
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            _finish_streaming_request(model_id, request_id, "cancelled")
            proc.terminate()
            raise
        except Exception as e:
            _finish_streaming_request(model_id, request_id, "error")
            proc.terminate()
            yield encode_chunk(f"stream error: {e}", "stop")
            yield "data: [DONE]\n\n"
        finally:
            if proc.poll() is None:
                proc.terminate()
            if reservation is not None:
                reservation.release()

    return StreamingResponse(
        iter_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_openai_result(result: Dict[str, Any], reservation=None) -> StreamingResponse:
    """Adapt runtimes without token callbacks to the OpenAI SSE contract."""
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    model_id = result.get("model", "")
    chunk_id = result.get("id", "chatcmpl-local")
    created = result.get("created", int(time.time()))

    def encode_chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def iter_events():
        try:
            if content:
                yield encode_chunk({"role": "assistant", "content": content})
            yield encode_chunk({}, "stop")
            yield "data: [DONE]\n\n"
        finally:
            if reservation is not None:
                reservation.release()

    return StreamingResponse(
        iter_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _begin_streaming_request(model_id: str) -> str:
    global _stream_metrics_seq
    now = int(time.time() * 1000)
    with _stream_metrics_lock:
        _stream_metrics_seq += 1
        request_id = f"stream-{now}-{_stream_metrics_seq}"
        current = {
            "id": request_id,
            "model": model_id,
            "status": "streaming",
            "started_at_ms": now,
            "updated_at_ms": now,
            "first_token_at_ms": None,
            "first_token_latency_ms": None,
            "streamed_token_estimate": 0,
            "streamed_chars": 0,
            "_content": "",
        }
        state = _stream_metrics_by_model.setdefault(model_id, {})
        state["current_request"] = current
        return request_id


def _update_streaming_request(model_id: str, request_id: str, content: str) -> None:
    if not content:
        return
    now = int(time.time() * 1000)
    with _stream_metrics_lock:
        current = _stream_metrics_by_model.get(model_id, {}).get("current_request")
        if not current or current.get("id") != request_id:
            return
        if not current["_content"]:
            current["first_token_at_ms"] = now
            current["first_token_latency_ms"] = now - current["started_at_ms"]
        current["_content"] += content
        current["updated_at_ms"] = now
        current["streamed_chars"] = len(current["_content"])
        current["streamed_token_estimate"] = _estimate_text_tokens(current["_content"])


def _attach_streaming_process(model_id: str, request_id: str, pid: int) -> None:
    with _stream_metrics_lock:
        current = _stream_metrics_by_model.get(model_id, {}).get("current_request")
        if not current or current.get("id") != request_id:
            return
        current["pid"] = pid


def _finish_streaming_request(model_id: str, request_id: str, status: str) -> None:
    now = int(time.time() * 1000)
    with _stream_metrics_lock:
        state = _stream_metrics_by_model.get(model_id)
        if not state:
            return
        current = state.get("current_request")
        if not current or current.get("id") != request_id:
            return
        current["status"] = status
        current["updated_at_ms"] = now
        current["finished_at_ms"] = now
        state["last_request"] = _public_streaming_request(current)
        state["current_request"] = None
        _annotate_latest_request(
            model_id,
            streamed_token_estimate=current.get("streamed_token_estimate", 0),
        )


def _streaming_request_metrics(model_id: str) -> Dict[str, Any]:
    now = int(time.time() * 1000)
    with _stream_metrics_lock:
        state = _stream_metrics_by_model.get(model_id, {})
        current = state.get("current_request")
        if current and _stream_request_is_stale(current, now):
            current["status"] = "stale"
            current["finished_at_ms"] = current.get("updated_at_ms", now)
            state["last_request"] = _public_streaming_request(current)
            state["current_request"] = None
            current = None
        last = state.get("last_request")
        return {
            "current_request": _public_streaming_request(current) if current else None,
            "last_request": _public_streaming_request(last) if last else None,
        }


def _public_streaming_request(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not request:
        return {}
    return {key: value for key, value in request.items() if not key.startswith("_")}


def _stream_request_is_stale(request: Dict[str, Any], now: int) -> bool:
    updated_at = int(request.get("updated_at_ms", request.get("started_at_ms", now)))
    if now - updated_at <= STREAM_REQUEST_STALE_MS:
        return False
    pid = request.get("pid")
    if isinstance(pid, int):
        return not _process_is_alive(pid)
    return True


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

def _format_chat_prompt(messages: list) -> str:
    parts = []
    for m in messages:
        role = getattr(m, "role", "user") if hasattr(m, "role") else m.get("role", "user")
        content = getattr(m, "content", "") if hasattr(m, "content") else m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

def _strip_llama_output(text: str) -> str:
    text = re.sub(r'^.*?assistant:\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\[ Prompt:.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\nExiting\.\.\.\s*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*<end>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'^assistant:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def _strip_model_thinking(text: str) -> str:
    text = re.sub(r'\[Start thinking\].*?\[End thinking\]\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class _ThinkingBlockFilter:
    start_marker = "[Start thinking]"
    end_marker = "[End thinking]"

    def __init__(self) -> None:
        self.pending = ""
        self.suppressing = False

    def feed(self, text: str) -> str:
        self.pending += text
        output: List[str] = []

        while self.pending:
            if self.suppressing:
                marker_index = self.pending.lower().find(self.end_marker.lower())
                if marker_index == -1:
                    return "".join(output)
                self.pending = self.pending[marker_index + len(self.end_marker):].lstrip()
                self.suppressing = False
                continue

            marker_index = self.pending.lower().find(self.start_marker.lower())
            if marker_index != -1:
                output.append(self.pending[:marker_index])
                self.pending = self.pending[marker_index + len(self.start_marker):]
                self.suppressing = True
                continue

            hold_len = self._partial_start_suffix_len(self.pending)
            if hold_len:
                output.append(self.pending[:-hold_len])
                self.pending = self.pending[-hold_len:]
                return "".join(output)

            output.append(self.pending)
            self.pending = ""

        return "".join(output)

    def flush(self) -> str:
        if self.suppressing:
            self.pending = ""
            return ""
        output = self.pending
        self.pending = ""
        return output

    def _partial_start_suffix_len(self, text: str) -> int:
        lower_text = text.lower()
        lower_marker = self.start_marker.lower()
        max_len = min(len(lower_text), len(lower_marker) - 1)
        for length in range(max_len, 0, -1):
            if lower_marker.startswith(lower_text[-length:]):
                return length
        return 0

def _default_gguf_backend() -> str:
    active = [d.get("backend", "") for d in get_active_drivers()]
    if "metal" in active:
        return "metal"
    for backend in active:
        if backend and backend != "npu":
            return backend
    return ""


def _find_active_gguf_driver(preferred_backend: str = "") -> Optional[Dict]:
    def with_cli(driver: Dict) -> Optional[Dict]:
        cli_path = find_driver_executable(driver["id"], ["llama-cli", "llama-cli.exe"])
        if cli_path:
            return {"cli_path": cli_path, "backend": driver.get("backend", "")}
        return None

    for d in get_active_drivers():
        if d["backend"] == "npu":
            continue
        if preferred_backend and d["backend"] != preferred_backend:
            continue
        found = with_cli(d)
        if found:
            return found
    for ld in get_local_drivers(None):
        if ld.get("active") and (not preferred_backend or ld["backend"] == preferred_backend):
            found = with_cli(ld)
            if found:
                return found
    if preferred_backend:
        return _find_active_gguf_driver("")
    return None

# --- Server lifecycle (called from main.py) ---

def _served_model_catalog() -> List[Dict[str, Any]]:
    models = []
    gguf_backend = _server_config.get("gguf_backend") or _server_config.get("device") or _default_gguf_backend()
    openvino_device = _server_config.get("openvino_device") or "NPU"
    for model in get_local_models():
        fmt = model.get("format", "gguf")
        runtime = "openvino" if fmt == "openvino" else "gguf"
        driver = openvino_device if fmt == "openvino" else gguf_backend
        model_file = _served_model_file(model)
        profile = _served_model_profile(
            served_id=model["id"],
            source="local",
            runtime=runtime,
            driver=driver,
            model_file=model_file,
            model=model,
        )
        stream_metrics = _streaming_request_metrics(profile["served_id"])
        models.append({
            "id": profile["served_id"],
            "base_id": model["id"],
            "served_id": profile["served_id"],
            "source": "local",
            "format": fmt,
            "runtime": runtime,
            "driver": profile["driver"],
            "model_file": profile["model_file"],
            "mtp": profile["mtp"],
            "mtp_head_path": profile["mtp_head_path"],
            "is_mtp_head": profile["is_mtp_head"],
            "dspark": profile["dspark"],
            "is_draft": bool(model.get("is_draft", False)),
            "serving_supported": model.get("serving_supported", True),
            "profile": profile,
            **stream_metrics,
        })
    for model in list_remote_models():
        served_id = model.get("id", "")
        profile = _served_model_profile(
            served_id=served_id,
            source="pc",
            runtime="remote",
            driver=model.get("owned_by", "linked PC"),
            model_file=model.get("remote_model_id", served_id),
            model=model,
        )
        stream_metrics = _streaming_request_metrics(profile["served_id"])
        models.append({
            "id": profile["served_id"],
            "base_id": served_id,
            "served_id": profile["served_id"],
            "source": "pc",
            "format": model.get("format", "remote"),
            "runtime": "remote",
            "driver": profile["driver"],
            "model_file": profile["model_file"],
            "mtp": profile["mtp"],
            "mtp_head_path": profile["mtp_head_path"],
            "is_mtp_head": profile["is_mtp_head"],
            "dspark": profile["dspark"],
            "is_draft": False,
            "serving_supported": True,
            "profile": profile,
            **stream_metrics,
        })
    return models


def _served_model_file(model: Dict[str, Any]) -> str:
    if model.get("format") == "openvino":
        return model.get("openvino_path") or model.get("path", "")
    files = model.get("files") or []
    if files:
        return os.path.join(model.get("path", ""), files[0])
    return model.get("path", "")


def _served_model_profile(
    served_id: str,
    source: str,
    runtime: str,
    driver: str,
    model_file: str,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    search_text = " ".join(
        str(value)
        for value in [
            served_id,
            model.get("id", ""),
            model.get("repo_id", ""),
            model.get("remote_model_id", ""),
            model_file,
            " ".join(model.get("files", []) or []),
        ]
        if value
    ).lower()
    override = get_override(served_id)
    served_id = override.get("served_id", served_id)
    driver = override.get("driver", driver)
    model_file = override.get("model_file", model_file)
    mtp_head_path = _find_mtp_head(model_file)
    is_mtp_head = _is_mtp_head_artifact(model_file, search_text)
    serving_config = {
        "api_mode": _server_config.get("mode", "openai"),
        "port": _server_config.get("port", 8787),
        "runtime": runtime,
        "driver": driver,
        "gguf_backend": _server_config.get("gguf_backend", _server_config.get("device", "")),
        "openvino_device": _server_config.get("openvino_device", "NPU"),
        "gpu_index": _server_config.get("gpu_index", ""),
        "max_tokens_default": 512,
        "temperature_default": 0.7,
        "mtp_head_path": mtp_head_path or "",
    }
    serving_config.update(override.get("serving_config", {}))
    if source == "pc":
        serving_config["remote"] = True
    return {
        "base_id": model.get("id", served_id),
        "served_id": served_id,
        "model_id": model.get("id", served_id),
        "repo_id": model.get("repo_id", ""),
        "source": source,
        "format": model.get("format", runtime),
        "runtime": runtime,
        "driver": driver,
        "model_file": model_file,
        "mtp": override.get("mtp", mtp_head_path is not None),
        "mtp_head_path": mtp_head_path,
        "is_mtp_head": is_mtp_head,
        "dspark": override.get("dspark", "dspark" in search_text or "d-spark" in search_text),
        "serving_config": serving_config,
    }


def _find_mtp_head(model_path: str) -> Optional[str]:
    if not model_path:
        return None
    model_dir = os.path.dirname(model_path)
    mtp_dir = os.path.join(model_dir, "MTP")
    if not os.path.isdir(mtp_dir):
        return None
    heads = [
        os.path.join(mtp_dir, f)
        for f in os.listdir(mtp_dir)
        if f.endswith(".gguf")
    ]
    if not heads:
        return None
    heads.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
    return heads[0]


def _is_mtp_head_artifact(model_file: str, search_text: str) -> bool:
    normalized = model_file.replace("\\", "/").lower()
    if "/mtp/" in normalized:
        return True
    return "mtp" in search_text and _find_mtp_head(model_file) is None

def _free_port(port: int):
    """Kill any process holding the given TCP port."""
    import socket
    def port_is_free() -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", port))
            s.close()
            return True
        except OSError:
            return False

    if port_is_free():
        return True

    try:
        import psutil
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status in ("LISTEN", "TIME_WAIT", "ESTABLISHED"):
                try:
                    proc = psutil.Process(conn.pid)
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    pass
    except Exception:
        pass

    if not port_is_free():
        try:
            out = subprocess.run(
                ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            for pid in out.stdout.split():
                if pid and int(pid) != os.getpid():
                    try:
                        os.kill(int(pid), 15)
                    except OSError:
                        pass
        except Exception:
            pass

    try:
        return port_is_free()
    except Exception:
        return False

def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Wait until the port is accepting connections."""
    import socket, time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("127.0.0.1", port))
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False

def _check_proc_alive(proc: subprocess.Popen) -> Optional[str]:
    """Check if process is alive; return error string if dead."""
    ret = proc.poll()
    if ret is not None:
        return f"Process exited with code {ret} before binding"
    return None

def _popen_creation_kwargs() -> Dict[str, Any]:
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}

def start_server(
    model_id: str = "",
    port: int = 8787,
    mode: str = "openai",
    device: str = "",
    gpu_index: str = "",
    gguf_backend: str = "",
    openvino_device: str = "",
):
    global _server_process, _server_config
    with _server_lock:
        if _server_process is not None:
            # Check if tracked process is still alive
            err = _check_proc_alive(_server_process)
            if err:
                _server_process = None
                _server_config = {}
            else:
                return {"status": "already_running", "port": _server_config.get("port")}

        if not gguf_backend:
            gguf_backend = device if device and device.upper() not in ("NPU", "CPU", "AUTO") else _default_gguf_backend()
        if not openvino_device:
            openvino_device = device if device.upper() in ("NPU", "CPU", "AUTO") else "NPU"
        device = gguf_backend
        if model_id and not _find_model_path(model_id):
            raise ValueError(f"Model '{model_id}' not found")

        # Free the port if occupied
        _free_port(port)

        _server_config = {
            "model_id": model_id,
            "port": port,
            "mode": mode,
            "device": device,
            "gguf_backend": gguf_backend,
            "openvino_device": openvino_device,
            "gpu_index": gpu_index,
        }

        _server_process = _start_gateway_server(port, mode, device, gpu_index, gguf_backend, openvino_device)

        if _server_process is None:
            _server_config = {}
            raise RuntimeError(f"Failed to start server - no driver or executable found for '{device}'")

        # Wait a moment and verify the process is alive
        import time as _time
        _time.sleep(3)
        err = _check_proc_alive(_server_process)
        if err:
            _server_process = None
            _server_config = {}
            raise RuntimeError(f"Server process died: {err}")

        # Wait for the port to be ready (up to 60s for model load)
        port_ok = _wait_for_port(port, timeout=60.0)
        if not port_ok:
            err = _check_proc_alive(_server_process)
            _server_process = None
            _server_config = {}
            raise RuntimeError(f"Server on port {port} failed to start within 60s" + (f" ({err})" if err else ""))

        return {"status": "started", "port": port}

def stop_server():
    global _server_process, _server_config
    with _server_lock:
        if _server_process is None:
            port = _server_config.get("port", 8787)
            if _gateway_is_running(port):
                _free_port(port)
                _server_config = {}
                return {"status": "stopped"}
            return {"status": "not_running"}
        port = _server_config.get("port", 8787)
        active_requests = _active_gateway_requests(port)
        if active_requests:
            return {
                "status": "busy",
                "message": "Gateway has active requests; stop was not performed.",
                "active_requests": active_requests,
            }
        if _server_process.poll() is None:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _server_process.kill()
                _server_process.wait(timeout=5)
        _server_process = None
        _server_config = {}
        if port:
            _free_port(port)
        return {"status": "stopped"}


def _active_gateway_requests(port: int) -> List[Dict[str, Any]]:
    snapshot = _gateway_snapshot(port)
    if not snapshot:
        return []
    active = []
    for model in snapshot.get("models", []):
        current = model.get("current_request")
        if current:
            active.append({
                "model": model.get("id", ""),
                **current,
            })
    return active

def server_status():
    global _server_process, _server_config
    with _server_lock:
        if _server_process is None:
            port = _server_config.get("port", 8787)
            snapshot = _gateway_snapshot(port)
            if snapshot:
                config = snapshot["config"]
                config["detected"] = True
                return {"status": "running", "config": config, "models": snapshot.get("models", []), "resources": snapshot.get("resources", {}), "request_log": snapshot.get("request_log", [])}
            return {"status": "stopped", "config": _server_config, "resources": resource_manager.snapshot()}
        ret = _server_process.poll()
        if ret is not None:
            _server_process = None
            port = _server_config.get("port", 8787)
            snapshot = _gateway_snapshot(port)
            if snapshot:
                config = snapshot["config"]
                return {"status": "running", "config": {**_server_config, **config, "detected": True}, "models": snapshot.get("models", []), "resources": snapshot.get("resources", {}), "request_log": snapshot.get("request_log", [])}
            return {"status": f"exited ({ret})", "config": _server_config, "resources": resource_manager.snapshot()}
        return {"status": "running", "config": _server_config, "models": _served_model_catalog(), "resources": resource_manager.snapshot(), "request_log": _request_log_snapshot()}


def _gateway_is_running(port: int) -> bool:
    return _gateway_snapshot(port) is not None


def _gateway_config(port: int) -> Optional[Dict[str, Any]]:
    snapshot = _gateway_snapshot(port)
    return snapshot["config"] if snapshot else None


def _gateway_snapshot(port: int) -> Optional[Dict[str, Any]]:
    try:
        with httpx.Client(timeout=2.0) as client:
            status_response = client.get(f"http://127.0.0.1:{port}/v1/status")
            if status_response.status_code == 200:
                data = status_response.json()
                config = data.get("config", {}) if isinstance(data, dict) else {}
                return {
                    "config": {
                    "model_id": config.get("model_id", ""),
                    "port": config.get("port", port),
                    "mode": config.get("mode", "openai"),
                    "device": config.get("device", ""),
                    "gguf_backend": config.get("gguf_backend", config.get("device", "")),
                    "openvino_device": config.get("openvino_device", "NPU"),
                    "gpu_index": config.get("gpu_index", ""),
                    },
                    "models": data.get("models", []) if isinstance(data, dict) else [],
                    "resources": data.get("resources", {}) if isinstance(data, dict) else {},
                    "request_log": data.get("request_log", []) if isinstance(data, dict) else [],
                }
            models_response = client.get(f"http://127.0.0.1:{port}/v1/models")
        if models_response.status_code == 200:
            return {
                "config": {"model_id": "", "port": port, "mode": "openai", "device": "", "gguf_backend": "", "openvino_device": "NPU", "gpu_index": ""},
                "models": [],
                "resources": {},
                "request_log": [],
            }
        return None
    except Exception:
        return None

def _start_gateway_server(
    port: int,
    mode: str = "openai",
    device: str = "",
    gpu_index: str = "",
    gguf_backend: str = "",
    openvino_device: str = "",
) -> subprocess.Popen:
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["LLAMA_DASH_API_PORT"] = str(port)
    env["LLAMA_DASH_API_MODE"] = mode
    env["LLAMA_DASH_API_DEVICE"] = device
    env["LLAMA_DASH_API_GGUF_BACKEND"] = gguf_backend
    env["LLAMA_DASH_API_OPENVINO_DEVICE"] = openvino_device
    env["LLAMA_DASH_API_GPU_INDEX"] = gpu_index
    env["LLAMA_DASH_API_MODEL_ID"] = _server_config.get("model_id", "")
    log_dir = Path(__file__).parent
    stdout_f = open(log_dir / "api_server_stdout.log", "a")
    stderr_f = open(log_dir / "api_server_stderr.log", "a")
    gateway_executable = os.environ.get("AVALON_GATEWAY_EXECUTABLE", "")
    if gateway_executable and Path(gateway_executable).exists():
        command = [gateway_executable, "--serve"]
    else:
        command = [sys.executable, str(script), "--serve"]
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=stdout_f, stderr=stderr_f,
        **_popen_creation_kwargs(),
    )
    return proc

def _start_llama_server(port: int, model_path: str, device: str = "cuda", gpu_index: str = "") -> Optional[subprocess.Popen]:
    for d in get_local_drivers(None):
        if d["backend"] != device:
            continue
        if not d.get("llama_bench_path"):
            continue
        driver_dir = Path(d["path"])
        server_exe_path = find_driver_executable(d["id"], ["llama-server", "llama-server.exe"])
        if not server_exe_path:
            server_exe = driver_dir / "extracted" / "llama-server.exe"
            nested = list(driver_dir.rglob("llama-server*"))
            if nested:
                server_exe_path = str(nested[0])
            else:
                continue
        args = [server_exe_path, "-m", model_path, "--port", str(port), "--host", "0.0.0.0"]
        if device in ("cuda", "sycl", "vulkan", "openvino", "directml", "hip", "metal"):
            args += ["-ngl", "99"]
            if gpu_index:
                if device == "sycl":
                    args += ["--main-gpu", gpu_index]
                elif device == "vulkan":
                    args += ["--main-gpu", gpu_index]
                elif device == "cuda":
                    args += ["--main-gpu", gpu_index]
        log_dir = Path(__file__).parent
        stdout_f = open(log_dir / "api_server_stdout.log", "a")
        stderr_f = open(log_dir / "api_server_stderr.log", "a")
        proc = subprocess.Popen(
            args,
            stdout=stdout_f, stderr=stderr_f,
            **_popen_creation_kwargs(),
        )
        return proc
    raise RuntimeError(f"No driver found for backend '{device}' with llama-server.exe")

# --- Standalone OpenVINO server ---

if __name__ == "__main__":
    if "--serve" in sys.argv or "--serve-ov" in sys.argv:
        port = int(os.environ.get("LLAMA_DASH_API_PORT", os.environ.get("OV_API_PORT", "8787")))
        mode = os.environ.get("LLAMA_DASH_API_MODE", "openai")
        device = os.environ.get("LLAMA_DASH_API_DEVICE", os.environ.get("OV_API_DEVICE", ""))
        gguf_backend = os.environ.get("LLAMA_DASH_API_GGUF_BACKEND", device)
        openvino_device = os.environ.get("LLAMA_DASH_API_OPENVINO_DEVICE", "NPU")
        gpu_index = os.environ.get("LLAMA_DASH_API_GPU_INDEX", "")
        model_id = os.environ.get("LLAMA_DASH_API_MODEL_ID", "")
        _server_config.update({
            "model_id": model_id,
            "port": port,
            "mode": mode,
            "device": device,
            "gguf_backend": gguf_backend,
            "openvino_device": openvino_device,
            "gpu_index": gpu_index,
        })
        uvicorn.run(
            app,
            host=os.environ.get("AVALON_GATEWAY_HOST", "127.0.0.1"),
            port=port,
            log_level="warning",
        )
    else:
        print("Usage: api_server.py [--serve]", file=sys.stderr)

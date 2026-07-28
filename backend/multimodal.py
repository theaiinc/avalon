"""Versioned multimodal test contracts, storage, and execution.

The multimodal runner is deliberately separate from ``benchmark_runner``:
llama.cpp benchmarks have a different result contract and must not be made to
understand audio, image, or video artifacts.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import httpx

from config import DATA_DIR

PROTOCOL_VERSION = "avalon.multimodal/v1"
MODALITIES = {"tts", "stt", "imagegen", "videogen"}
STATES = {"queued", "running", "cancelling", "cancelled", "failed", "succeeded"}
MAX_ARTIFACT_BYTES = int(os.environ.get("AVALON_MAX_ARTIFACT_BYTES", 100 * 1024 * 1024))
MAX_RUN_BYTES = int(os.environ.get("AVALON_MAX_RUN_BYTES", 250 * 1024 * 1024))
HTTP_TIMEOUT = float(os.environ.get("AVALON_MULTIMODAL_HTTP_TIMEOUT", "300"))

PROFILE_FILE = DATA_DIR / "multimodal_profiles.json"
CASE_FILE = DATA_DIR / "multimodal_cases.json"
RUN_FILE = DATA_DIR / "multimodal_runs.json"
ARTIFACT_DIR = DATA_DIR / "multimodal_artifacts"

_lock = threading.RLock()
_processes: Dict[str, subprocess.Popen] = {}


def _read(path: Path) -> list:
    try:
        with path.open() as fh:
            value = json.load(fh)
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _write(path: Path, value: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as fh:
        json.dump(value, fh, indent=2)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _find(items: Iterable[dict], item_id: str) -> Optional[dict]:
    return next((item for item in items if item.get("id") == item_id), None)


def _plugin_allowlist() -> dict:
    raw = os.environ.get("AVALON_PLUGIN_ALLOWLIST", "{}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AVALON_PLUGIN_ALLOWLIST must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("AVALON_PLUGIN_ALLOWLIST must be a JSON object")
    return value


def _is_private_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    return any(
        address[4][0].startswith(("10.", "192.168.", "127.", "::1", "169.254."))
        or address[4][0].startswith("172.") and 16 <= int(address[4][0].split(".")[1]) <= 31
        for address in addresses
    )


def validate_profile(profile: dict) -> dict:
    modality = profile.get("modality")
    mode = profile.get("mode")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {sorted(MODALITIES)}")
    if mode not in {"local", "http", "builtin"}:
        raise ValueError("mode must be 'local', 'http', or 'builtin'")
    clean = {
        "id": profile.get("id") or _id("profile"),
        "name": str(profile.get("name") or profile.get("model") or modality)[:120],
        "modality": modality,
        "mode": mode,
        "model": str(profile.get("model", ""))[:240],
        "protocol_version": PROTOCOL_VERSION,
        "timeout_sec": max(1, min(int(profile.get("timeout_sec", 300)), 3600)),
        "input_format": profile.get("input_format", "json"),
        "secret_ref": profile.get("secret_ref", ""),
        "model_path": str(profile.get("model_path", "")),
    }
    if clean["secret_ref"] and not re.fullmatch(r"env:[A-Za-z_][A-Za-z0-9_]*", clean["secret_ref"]):
        raise ValueError("secret_ref must use the env:NAME format")
    if clean["input_format"] not in {"json", "multipart"}:
        raise ValueError("input_format must be json or multipart")

    if mode == "builtin":
        if modality != "imagegen":
            raise ValueError("builtin adapter currently supports imagegen only")
    elif mode == "local":
        executable_id = profile.get("executable_id", "")
        allowlist = _plugin_allowlist()
        if not executable_id or executable_id not in allowlist:
            raise ValueError("local profile must reference an approved executable_id")
        clean["executable_id"] = executable_id
    else:
        url = str(profile.get("url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("HTTP profile requires an absolute http(s) URL")
        if _is_private_host(parsed.hostname) and not profile.get("allow_private_network"):
            raise ValueError("private or loopback HTTP targets require allow_private_network=true")
        clean.update({
            "url": url,
            "allow_private_network": bool(profile.get("allow_private_network", False)),
            "async": bool(profile.get("async", False)),
            "poll_url": str(profile.get("poll_url", "")),
        })
    return clean


def list_profiles() -> list:
    with _lock:
        return _read(PROFILE_FILE)


def capabilities() -> dict:
    """Return the protocol and locally approved adapter capabilities."""
    return {
        "protocol": PROTOCOL_VERSION,
        "modalities": sorted(MODALITIES),
        "modes": ["local", "http"],
        "approved_executables": sorted(_plugin_allowlist().keys()),
        "limits": {
            "max_artifact_bytes": MAX_ARTIFACT_BYTES,
            "max_run_bytes": MAX_RUN_BYTES,
        },
    }


def save_profile(profile: dict) -> dict:
    clean = validate_profile(profile)
    with _lock:
        items = _read(PROFILE_FILE)
        existing = _find(items, clean["id"])
        if not existing and not profile.get("id"):
            existing = next((
                item for item in items
                if all(item.get(key) == clean.get(key) for key in (
                    "name", "modality", "mode", "model", "model_path",
                    "url", "executable_id",
                ))
            ), None)
        if existing:
            if not profile.get("id"):
                clean["id"] = existing["id"]
            existing.update(clean)
            clean = existing
        else:
            items.append(clean)
        _write(PROFILE_FILE, items)
    return clean


def delete_profile(profile_id: str) -> bool:
    with _lock:
        items = _read(PROFILE_FILE)
        updated = [item for item in items if item.get("id") != profile_id]
        if len(updated) == len(items):
            return False
        _write(PROFILE_FILE, updated)
        return True


def list_cases() -> list:
    with _lock:
        return _read(CASE_FILE)


def save_case(case: dict) -> dict:
    if case.get("modality") not in MODALITIES:
        raise ValueError("case modality is unsupported")
    clean = {
        "id": case.get("id") or _id("case"),
        "name": str(case.get("name") or "Untitled test")[:120],
        "modality": case["modality"],
        "input": case.get("input", {}),
        "assertions": case.get("assertions", {}),
        "profile_id": case.get("profile_id", ""),
    }
    with _lock:
        items = _read(CASE_FILE)
        existing = _find(items, clean["id"])
        if not existing and not case.get("id"):
            existing = next((
                item for item in items
                if item.get("profile_id") == clean["profile_id"]
                and item.get("modality") == clean["modality"]
                and item.get("input") == clean["input"]
                and item.get("assertions") == clean["assertions"]
            ), None)
        if existing:
            if not case.get("id"):
                clean["id"] = existing["id"]
            existing.update(clean)
            clean = existing
        else:
            items.append(clean)
        _write(CASE_FILE, items)
    return clean


def delete_case(case_id: str) -> bool:
    with _lock:
        items = _read(CASE_FILE)
        updated = [item for item in items if item.get("id") != case_id]
        if len(updated) == len(items):
            return False
        _write(CASE_FILE, updated)
        return True


def _update_run(run_id: str, **changes: Any) -> Optional[dict]:
    with _lock:
        runs = _read(RUN_FILE)
        run = _find(runs, run_id)
        if not run:
            return None
        run.update(changes)
        _write(RUN_FILE, runs)
        return dict(run)


def list_runs() -> list:
    with _lock:
        return _read(RUN_FILE)


def get_run(run_id: str) -> Optional[dict]:
    with _lock:
        return _find(_read(RUN_FILE), run_id)


def _safe_suffix(mime: str, name: str = "") -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif", ".txt", ".json"}:
        return suffix
    return mimetypes.guess_extension(mime) or ".bin"


def _store_artifact(run_id: str, item: dict, total: int) -> dict:
    mime = str(item.get("mime") or "application/octet-stream").lower()
    if not re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+*-]+", mime):
        raise ValueError("artifact MIME type is invalid")
    if "data_base64" in item:
        try:
            payload = base64.b64decode(item["data_base64"], validate=True)
        except Exception as exc:
            raise ValueError("artifact data_base64 is invalid") from exc
    elif "text" in item:
        payload = str(item["text"]).encode()
    else:
        raise ValueError("artifact must contain data_base64 or text")
    if len(payload) > MAX_ARTIFACT_BYTES or total + len(payload) > MAX_RUN_BYTES:
        raise ValueError("artifact exceeds configured size limit")
    if mime.startswith("image/") and not payload.startswith((b"\x89PNG", b"\xff\xd8", b"RIFF")):
        raise ValueError("image artifact failed magic-byte validation")
    if mime == "application/json":
        json.loads(payload)
    directory = ARTIFACT_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    artifact_id = _id("artifact")
    filename = artifact_id + _safe_suffix(mime, item.get("name", ""))
    (directory / filename).write_bytes(payload)
    return {
        "id": artifact_id,
        "filename": filename,
        "mime": mime,
        "bytes": len(payload),
        "url": f"/api/multimodal/artifacts/{run_id}/{filename}",
    }


def _metric_payload(modality: str, payload: dict, elapsed: float, artifacts: list) -> dict:
    metrics = dict(payload.get("metrics") or {})
    metrics["latency_ms"] = round(elapsed * 1000, 2)
    metrics["output_bytes"] = sum(item["bytes"] for item in artifacts)
    if modality == "tts":
        chars = int(payload.get("characters", len(str(payload.get("input", "")))))
        metrics["characters_per_sec"] = round(chars / elapsed, 3) if elapsed else 0
        if payload.get("audio_duration_sec"):
            metrics["real_time_factor"] = round(elapsed / float(payload["audio_duration_sec"]), 4)
    elif modality == "stt" and payload.get("audio_duration_sec"):
        metrics["real_time_factor"] = round(elapsed / float(payload["audio_duration_sec"]), 4)
    elif modality == "imagegen":
        metrics["images_per_sec"] = round(int(payload.get("image_count", len(artifacts))) / elapsed, 3) if elapsed else 0
    elif modality == "videogen" and payload.get("video_duration_sec"):
        metrics["frames_per_sec"] = round(float(payload.get("frames", 0)) / elapsed, 3) if elapsed else 0
    return metrics


def _evaluate_assertions(payload: dict, assertions: dict) -> dict:
    checks = []
    if "transcript_contains" in assertions:
        actual = str(payload.get("transcript", "")).lower()
        expected = str(assertions["transcript_contains"]).lower()
        checks.append({"name": "transcript_contains", "passed": expected in actual})
    if "min_audio_duration_sec" in assertions:
        checks.append({
            "name": "min_audio_duration_sec",
            "passed": float(payload.get("audio_duration_sec", 0)) >= float(assertions["min_audio_duration_sec"]),
        })
    for key in ("min_width", "min_height", "min_frames"):
        if key in assertions:
            checks.append({"name": key, "passed": int(payload.get(key[4:], 0)) >= int(assertions[key])})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _headers(profile: dict) -> dict:
    ref = profile.get("secret_ref", "")
    if not ref:
        return {}
    name = ref[4:]
    secret = os.environ.get(name)
    if not secret:
        raise ValueError(f"secret reference {ref} is not configured")
    return {"Authorization": f"Bearer {secret}"}


def _run_local(run: dict, profile: dict, case: dict) -> dict:
    executable = _plugin_allowlist()[profile["executable_id"]]
    if not isinstance(executable, list) or not executable or not all(isinstance(x, str) for x in executable):
        raise ValueError("approved executable must be a non-empty argv list")
    request = {
        "protocol": PROTOCOL_VERSION,
        "event": "request",
        "run_id": run["id"],
        "modality": case["modality"],
        "model": profile.get("model", ""),
        "input": case.get("input", {}),
        "assertions": case.get("assertions", {}),
    }
    proc = subprocess.Popen(
        executable,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={k: v for k, v in os.environ.items() if k not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}},
    )
    _processes[run["id"]] = proc
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.close()
    events = []
    try:
        deadline = time.monotonic() + profile["timeout_sec"]
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line:
                event = json.loads(line)
                if event.get("protocol") != PROTOCOL_VERSION:
                    raise ValueError("plugin protocol version mismatch")
                events.append(event)
                if event.get("event") in {"result", "error"}:
                    break
            elif proc.poll() is not None:
                break
        else:
            raise TimeoutError("local adapter timed out")
        if proc.poll() is None:
            proc.wait(timeout=5)
        result = next((event for event in events if event.get("event") == "result"), None)
        error = next((event for event in events if event.get("event") == "error"), None)
        if error:
            raise ValueError(error.get("message", "local adapter failed"))
        if not result:
            raise ValueError("local adapter exited without a result event")
        return result
    finally:
        _processes.pop(run["id"], None)


def _run_http(profile: dict, case: dict) -> dict:
    request = {
        "protocol": PROTOCOL_VERSION,
        "modality": case["modality"],
        "model": profile.get("model", ""),
        "input": case.get("input", {}),
        "assertions": case.get("assertions", {}),
    }
    headers = _headers(profile)
    request_kwargs: dict = {"headers": headers}
    if profile.get("input_format") == "multipart":
        fields = {"protocol": PROTOCOL_VERSION, "modality": case["modality"],
                  "model": profile.get("model", ""),
                  "input": json.dumps(case.get("input", {})),
                  "assertions": json.dumps(case.get("assertions", {}))}
        files = {}
        for key, value in case.get("input", {}).items():
            if key.endswith("_base64") and isinstance(value, str):
                try:
                    files[key[:-7]] = (key[:-7], base64.b64decode(value, validate=True))
                except Exception as exc:
                    raise ValueError(f"multipart input {key} is invalid") from exc
        request_kwargs.update({"data": fields, "files": files})
    else:
        request_kwargs["json"] = request
    with httpx.Client(timeout=profile["timeout_sec"], follow_redirects=False) as client:
        response = client.post(profile["url"], **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type == "application/json":
            data = response.json()
            if profile.get("async") and data.get("status") in {"queued", "running"}:
                poll_url = data.get("poll_url") or profile.get("poll_url")
                if not poll_url or urlparse(poll_url).hostname != urlparse(profile["url"]).hostname:
                    raise ValueError("async poll_url must stay on the configured host")
                deadline = time.monotonic() + profile["timeout_sec"]
                while time.monotonic() < deadline:
                    poll = client.get(poll_url, headers=_headers(profile))
                    poll.raise_for_status()
                    data = poll.json()
                    if data.get("status") in {"succeeded", "failed"}:
                        break
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
            return data
        return {
            "artifacts": [{
                "mime": content_type or "application/octet-stream",
                "data_base64": base64.b64encode(response.content).decode(),
            }]
        }


def _run_builtin(run: dict, profile: dict, case: dict) -> dict:
    if profile["modality"] != "imagegen" or not profile.get("model_path"):
        raise ValueError("automatic image generation requires a local model path")
    from diffusion_runtime import generate_flux
    output = ARTIFACT_DIR / run["id"] / "generated.png"
    case_input = case.get("input", {})
    prompt = str(case_input.get("prompt") or case_input.get("text") or "")
    options = dict(case_input.get("options") or {})
    if case_input.get("image_base64"):
        options["image_base64"] = case_input["image_base64"]
    if not options.get("image_base64"):
        # Backward compatibility for agents that put an Avalon artifact path
        # in the prompt before the image-input field was exposed.
        reference = re.search(r"/multimodal_artifacts/(run_[^/\s\"']+)/([^/\s\"']+)", prompt)
        if reference:
            reference_path = ARTIFACT_DIR / reference.group(1) / reference.group(2)
            if reference_path.is_file() and reference_path.stat().st_size <= MAX_ARTIFACT_BYTES:
                options["image_base64"] = base64.b64encode(reference_path.read_bytes()).decode()
    generate_flux(
        profile["model_path"],
        prompt,
        output,
        options,
        on_process=lambda process: _processes.__setitem__(run["id"], process),
    )
    return {
        "artifacts": [{
            "mime": "image/png",
            "data_base64": base64.b64encode(output.read_bytes()).decode(),
            "name": output.name,
        }]
    }


def _execute(run: dict, profile: dict, case: dict) -> None:
    started = time.monotonic()
    _update_run(run["id"], state="running", started_at=_now(), progress={
        "stage": "starting", "detail": f"Starting {profile['mode']} {case['modality']} adapter",
    })
    try:
        if profile["mode"] == "local":
            _update_run(run["id"], progress={"stage": "generating", "detail": "Running the approved local plugin"})
            payload = _run_local(run, profile, case)
        elif profile["mode"] == "builtin":
            _update_run(run["id"], progress={"stage": "generating", "detail": "Provisioning runtime and generating the image"})
            payload = _run_builtin(run, profile, case)
        else:
            _update_run(run["id"], progress={"stage": "waiting", "detail": "Waiting for the HTTP adapter response"})
            payload = _run_http(profile, case)
        _update_run(run["id"], progress={"stage": "processing", "detail": "Saving artifacts and calculating metrics"})
        artifacts = []
        total = 0
        for item in payload.get("artifacts", []):
            artifact = _store_artifact(run["id"], item, total)
            total += artifact["bytes"]
            artifacts.append(artifact)
        metrics = _metric_payload(case["modality"], payload, time.monotonic() - started, artifacts)
        current = get_run(run["id"]) or {}
        if current.get("state") == "cancelling":
            _update_run(run["id"], state="cancelled", finished_at=_now())
        else:
            _update_run(run["id"], state="succeeded", finished_at=_now(), result={
                "protocol": PROTOCOL_VERSION, "modality": case["modality"],
                "metrics": metrics, "artifacts": artifacts,
                "transcript": payload.get("transcript"),
                "assertions": _evaluate_assertions(payload, case.get("assertions", {})),
            })
    except Exception as exc:
        state = "cancelled" if get_run(run["id"]).get("state") == "cancelling" else "failed"
        _update_run(run["id"], state=state, finished_at=_now(), error=str(exc))
    finally:
        _processes.pop(run["id"], None)


def create_run(profile_id: str, case_id: str) -> dict:
    profile = _find(list_profiles(), profile_id)
    case = _find(list_cases(), case_id)
    if not profile or not case:
        raise ValueError("profile_id and case_id must reference existing records")
    if profile["modality"] != case["modality"]:
        raise ValueError("profile and test case modalities must match")
    run = {
        "id": _id("run"), "protocol": PROTOCOL_VERSION,
        "profile_id": profile_id, "case_id": case_id,
        "modality": case["modality"], "state": "queued", "created_at": _now(),
        "timeout_sec": profile.get("timeout_sec", 120),
        "progress": {"stage": "queued", "detail": "Waiting for an available runner"},
    }
    with _lock:
        runs = _read(RUN_FILE)
        runs.append(run)
        _write(RUN_FILE, runs)
    threading.Thread(target=_execute, args=(run, profile, case), daemon=True).start()
    return run


def cancel_run(run_id: str) -> dict:
    run = get_run(run_id)
    if not run:
        raise ValueError("run not found")
    if run["state"] not in {"queued", "running"}:
        return run
    _update_run(run_id, state="cancelling")
    proc = _processes.get(run_id)
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            proc.terminate()
    return get_run(run_id) or run


def artifact_path(run_id: str, filename: str) -> Optional[Path]:
    candidate = (ARTIFACT_DIR / run_id / filename).resolve()
    root = (ARTIFACT_DIR / run_id).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def recover_runs() -> None:
    """Mark non-terminal runs from a previous process as failed."""
    with _lock:
        runs = _read(RUN_FILE)
        changed = False
        for run in runs:
            if run.get("state") in {"queued", "running", "cancelling"}:
                run["state"] = "failed"
                run["error"] = "dashboard restarted while the run was active"
                run["finished_at"] = _now()
                changed = True
        if changed:
            _write(RUN_FILE, runs)

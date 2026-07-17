import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import asyncio
import uuid
import os
import httpx
from progress import create_download, update, get as get_progress, list_active as list_active_downloads

from gpu_detector import detect_gpus, check_driver_updates
from driver_manager import (
    fetch_releases, get_local_drivers, download_driver,
    set_driver_active, remove_driver, get_active_drivers,
)
from model_manager import (
    search_models, list_files, get_local_models, download_model, download_openvino_model, remove_model,
)
from benchmark_runner import run_benchmark, list_results, get_result, delete_result, cancel_benchmark, list_active as list_active_benchmarks
from system_stats import collect as collect_system_stats
from known_issues import KNOWN_ISSUES
from api_server import start_server as api_start, stop_server as api_stop, server_status as api_status
from pc_links import list_links as list_pc_links, upsert_link as upsert_pc_link, remove_link as remove_pc_link, test_link as test_pc_link
from served_profiles import reset_override as reset_serving_profile, upsert_override as save_serving_profile
import pairing

app = FastAPI(title="Avalon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def start_pairing_discovery():
    pairing.start_discovery(int(os.environ.get("AVALON_PORT", "8771")))


@app.on_event("shutdown")
async def stop_pairing_discovery():
    pairing.stop_discovery()


@app.middleware("http")
async def protect_dashboard_from_lan(request: Request, call_next):
    """Expose only the one-time pairing accept route beyond loopback."""
    client = request.client.host if request.client else ""
    pairing_accept_path = request.url.path == "/api/pairing/accept"
    if client not in {"127.0.0.1", "::1", "localhost"} and not pairing_accept_path:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Dashboard controls are local-only"}, status_code=404)
    return await call_next(request)


class DownloadDriverRequest(BaseModel):
    tag: str
    backend: str


class SetDriverActiveRequest(BaseModel):
    driver_id: str
    active: bool


class RemoveDriverRequest(BaseModel):
    driver_id: str


class SearchModelsRequest(BaseModel):
    query: str
    limit: int = 20
    format: str = "gguf"


class DownloadModelRequest(BaseModel):
    repo_id: str
    filename: str


class DownloadOVModelRequest(BaseModel):
    repo_id: str


class RemoveModelRequest(BaseModel):
    model_id: str


class RunBenchmarkRequest(BaseModel):
    model_path: str
    backends: List[str]
    model_params: Optional[dict] = None
    bench_params: Optional[dict] = None


class RemoveResultRequest(BaseModel):
    result_id: str


class ApiServerStartRequest(BaseModel):
    model_id: str = ""
    port: int = 8787
    mode: str = "openai"  # openai, anthropic, both
    device: str = ""  # NPU, CPU, cuda, sycl, vulkan, openvino, etc.
    gguf_backend: str = ""
    openvino_device: str = "NPU"
    gpu_index: str = ""  # specific GPU index within the backend


class PcLinkRequest(BaseModel):
    name: str
    base_url: str
    id: str = ""


class PcLinkTestRequest(BaseModel):
    base_url: str


class PcLinkRemoveRequest(BaseModel):
    id: str


class PairingAcceptRequest(BaseModel):
    session_id: str = ""
    code: str
    peer_name: str = ""
    peer_public_key: str
    peer_signature: str


class PairingRemoveRequest(BaseModel):
    peer_id: str


class PairingConnectRequest(BaseModel):
    base_url: str
    session_id: str = ""
    code: str
    name: str = ""


class ServingProfileRequest(BaseModel):
    base_id: str
    profile: Dict[str, Any]


class ServingProfileResetRequest(BaseModel):
    base_id: str


class QuickTestRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 128
    temperature: float = 0.7
    format: str = "openai"


@app.get("/api/gpu/list")
async def list_gpus():
    gpus = detect_gpus()
    return {"gpus": gpus}


@app.get("/api/gpu/driver-updates")
async def gpu_driver_updates():
    gpus = detect_gpus()
    return {"updates": check_driver_updates(gpus)}


@app.get("/api/drivers")
async def list_drivers():
    releases = await fetch_releases()
    return {
        "local": get_local_drivers(releases),
        "active": get_active_drivers(),
    }


@app.get("/api/drivers/releases")
async def list_releases():
    releases = await fetch_releases()
    return {"releases": releases}


@app.post("/api/drivers/download")
async def download(req: DownloadDriverRequest):
    download_id = create_download()
    update(download_id, status="starting", percent=0, stage="Starting...")

    async def task():
        try:
            def prog(**kw):
                update(download_id, **kw)
            await download_driver(req.tag, req.backend, on_progress=prog)
        except Exception as e:
            update(download_id, status="error", percent=0, stage=str(e))

    asyncio.create_task(task())
    return {"download_id": download_id}


@app.post("/api/drivers/set-active")
async def set_active(req: SetDriverActiveRequest):
    try:
        return set_driver_active(req.driver_id, req.active)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/drivers/remove")
async def remove(req: RemoveDriverRequest):
    remove_driver(req.driver_id)
    return {"status": "ok"}


@app.get("/api/models/search")
async def search(q: str = "", limit: int = 20, format: str = "gguf"):
    return {"models": search_models(q, limit, format)}


@app.get("/api/models/files")
async def files(repo_id: str = "", format: str = "gguf"):
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id required")
    return {"files": list_files(repo_id, format)}


@app.get("/api/models")
async def list_models():
    return {"models": get_local_models()}


@app.post("/api/models/download")
async def download(req: DownloadModelRequest):
    download_id = create_download()
    update(download_id, status="starting", percent=0, stage="Starting...")

    async def task():
        try:
            def prog(**kw):
                update(download_id, **kw)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: download_model(req.repo_id, req.filename, on_progress=prog))
            update(download_id, status="done", percent=100, stage="Complete")
        except Exception as e:
            update(download_id, status="error", percent=0, stage=str(e))

    asyncio.create_task(task())
    return {"download_id": download_id}


@app.post("/api/models/download-openvino")
async def download_ov(req: DownloadOVModelRequest):
    download_id = create_download()
    update(download_id, status="starting", percent=0, stage="Starting...")

    async def task():
        try:
            def prog(**kw):
                update(download_id, **kw)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: download_openvino_model(req.repo_id, on_progress=prog))
            update(download_id, status="done", percent=100, stage="Complete")
        except Exception as e:
            update(download_id, status="error", percent=0, stage=str(e))

    asyncio.create_task(task())
    return {"download_id": download_id}


@app.post("/api/models/remove")
async def remove(req: RemoveModelRequest):
    remove_model(req.model_id)
    return {"status": "ok"}


@app.post("/api/benchmark/run")
async def benchmark(req: RunBenchmarkRequest):
    task_id = str(uuid.uuid4())[:8]
    loop = asyncio.get_event_loop()

    def task():
        return run_benchmark(
            model_path=req.model_path,
            backends=req.backends,
            task_id=task_id,
            model_params=req.model_params,
            bench_params=req.bench_params,
        )

    loop.run_in_executor(None, task)
    return {"task_id": task_id, "status": "started"}


@app.get("/api/benchmark/status/{task_id}")
async def benchmark_status(task_id: str):
    result = get_result(task_id)
    if result:
        return {"task_id": task_id, "status": "done", "result": result}
    active = list_active_benchmarks()
    if any(a["id"] == task_id for a in active):
        return {"task_id": task_id, "status": "running"}
    return {"task_id": task_id, "status": "not_found"}


@app.get("/api/benchmark/list-active")
async def benchmark_list_active():
    return {"active": list_active_benchmarks()}


@app.get("/api/benchmark/stats")
async def benchmark_stats():
    return collect_system_stats()


@app.post("/api/benchmark/cancel/{task_id}")
async def benchmark_cancel(task_id: str):
    ok = cancel_benchmark(task_id)
    return {"task_id": task_id, "cancelled": ok}


@app.get("/api/benchmark/results")
async def results():
    return {"results": list_results()}


@app.get("/api/benchmark/results/{result_id}")
async def result_detail(result_id: str):
    r = get_result(result_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not found")
    return r


@app.post("/api/benchmark/results/remove")
async def remove_result(req: RemoveResultRequest):
    delete_result(req.result_id)
    return {"status": "ok"}


@app.get("/api/downloads/progress/{download_id}")
async def download_progress(download_id: str):
    p = get_progress(download_id)
    if not p:
        raise HTTPException(status_code=404, detail="Download not found")
    return p


@app.get("/api/downloads/active")
async def active_downloads():
    return {"downloads": list_active_downloads()}


@app.get("/api/compatibility/issues")
async def compatibility_issues():
    return {"issues": KNOWN_ISSUES}


@app.get("/api/compatibility/check")
async def compatibility_check(backend: str = "", model_name: str = "", gpu_name: str = "", driver_version: str = ""):
    matches = []
    for issue in KNOWN_ISSUES:
        if backend and backend not in issue["backends"]:
            continue
        m = issue.get("match", {})
        name_patterns = m.get("model_name_contains", [])
        if name_patterns and model_name:
            if not any(p.lower() in model_name.lower() for p in name_patterns):
                continue
        gpu_patterns = m.get("gpu_name_contains", [])
        if gpu_patterns and gpu_name:
            if not any(p.lower() in gpu_name.lower() for p in gpu_patterns):
                continue
        if m.get("driver_version_below") and driver_version:
            if driver_version >= m["driver_version_below"]:
                continue
        if m.get("device_type") and gpu_name:
            continue
        matches.append(issue)
    return {"matches": matches}


@app.get("/api/api-server/status")
async def api_server_status():
    return api_status()


@app.post("/api/api-server/start")
async def api_server_start(req: ApiServerStartRequest):
    try:
        return api_start(
            req.model_id,
            req.port,
            req.mode,
            req.device,
            req.gpu_index,
            req.gguf_backend,
            req.openvino_device,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/api-server/stop")
async def api_server_stop():
    result = api_stop()
    if result.get("status") == "busy":
        raise HTTPException(status_code=409, detail=result)
    return result


@app.post("/api/api-server/quick-test")
async def api_server_quick_test(req: QuickTestRequest):
    port = api_status().get("config", {}).get("port", 8787)
    path = "/v1/messages" if req.format == "anthropic" else "/v1/chat/completions"
    payload = {
        "model": req.model,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=310.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{port}{path}",
                json=payload,
                headers={"X-API-Key": os.environ.get("AVALON_API_KEY", "")},
            )
        try:
            content = response.json()
        except ValueError:
            content = {"detail": response.text}
        from fastapi.responses import JSONResponse
        return JSONResponse(content=content, status_code=response.status_code)
    except Exception as exc:
        raise HTTPException(502, f"Gateway request failed: {exc}")


@app.post("/api/api-server/model-profile")
async def api_server_model_profile(req: ServingProfileRequest):
    try:
        return {"profile": save_serving_profile(req.base_id, req.profile), "status": api_status()}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@app.post("/api/api-server/model-profile/reset")
async def api_server_model_profile_reset(req: ServingProfileResetRequest):
    try:
        return {"removed": reset_serving_profile(req.base_id), "status": api_status()}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@app.get("/api/pc-links")
async def pc_links():
    return {"links": list_pc_links()}


@app.post("/api/pc-links")
async def save_pc_link(req: PcLinkRequest):
    try:
        link = upsert_pc_link(req.name, req.base_url, req.id)
        tested = test_pc_link(link["base_url"])
        return {"link": link, "test": tested}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pc-links/test")
async def test_pc_link_endpoint(req: PcLinkTestRequest):
    try:
        return test_pc_link(req.base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pc-links/remove")
async def remove_pc_link_endpoint(req: PcLinkRemoveRequest):
    return {"removed": remove_pc_link(req.id)}


@app.get("/api/pairing/info")
async def pairing_info():
    return pairing.device_info()


@app.post("/api/pairing/code")
async def pairing_code():
    return pairing.create_pairing_code()


@app.get("/api/pairing/discover")
async def pairing_discover():
    return {"devices": await asyncio.to_thread(pairing.discover_devices)}


@app.post("/api/pairing/accept")
async def pairing_accept(req: PairingAcceptRequest):
    try:
        return pairing.accept_pairing(
            req.session_id,
            req.code,
            req.peer_name,
            req.peer_public_key,
            req.peer_signature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/pairing/connect")
async def pairing_connect(req: PairingConnectRequest):
    local = pairing.device_info()
    try:
        remote_url = req.base_url.strip().rstrip("/")
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{remote_url}/api/pairing/accept",
                json={
                    "session_id": req.session_id,
                    "code": req.code,
                    "peer_name": local["device_name"],
                    "peer_public_key": local["public_key"],
                    "peer_signature": pairing.sign_pairing(req.session_id, req.code),
                },
            )
            response.raise_for_status()
            remote = response.json()
        remote_device = remote["device"]
        peer_id = remote_device["device_id"]
        pairing.store_peer_credential(peer_id, remote["api_key"])
        link = upsert_pc_link(
            req.name or remote_device.get("device_name", peer_id),
            remote_url,
            peer_id=peer_id,
        )
        return {"link": link, "device": remote_device}
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Pairing failed: {exc}")


@app.get("/api/pairing/peers")
async def pairing_peers():
    return {"peers": pairing.list_peers()}


@app.post("/api/pairing/peers/remove")
async def pairing_remove(req: PairingRemoveRequest):
    return {"removed": pairing.remove_peer(req.peer_id)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("AVALON_HOST", "127.0.0.1"),
        port=int(os.environ.get("AVALON_PORT", "8771")),
    )

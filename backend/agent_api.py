import json
import time
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8771"
DEFAULT_INFERENCE_URL = "http://127.0.0.1:8787"


class DashboardClient:
    def __init__(self, base_url: str = DEFAULT_DASHBOARD_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    def get(self, path: str, **params: Any) -> Any:
        clean_params = {key: value for key, value in params.items() if value is not None}
        return self._request("GET", path, params=clean_params)

    def post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("POST", path, json=payload or {})

    def list_gpus(self) -> Any:
        return self.get("/api/gpu/list")

    def list_models(self) -> Any:
        return self.get("/api/models")

    def search_models(self, query: str, limit: int = 20, model_format: str = "gguf") -> Any:
        return self.get("/api/models/search", q=query, limit=limit, format=model_format)

    def list_model_files(self, repo_id: str, model_format: str = "gguf") -> Any:
        return self.get("/api/models/files", repo_id=repo_id, format=model_format)

    def download_model(self, repo_id: str, filename: str) -> Any:
        return self.post("/api/models/download", {"repo_id": repo_id, "filename": filename})

    def download_openvino_model(self, repo_id: str) -> Any:
        return self.post("/api/models/download-openvino", {"repo_id": repo_id})

    def download_progress(self, download_id: str) -> Any:
        return self.get(f"/api/downloads/progress/{download_id}")

    def list_active_downloads(self) -> Any:
        return self.get("/api/downloads/active")

    def wait_for_download(self, download_id: str, interval: float = 2.0, timeout: float = 3600.0) -> Any:
        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            last_status = self.download_progress(download_id)
            status = last_status.get("status")
            if status in ("done", "error"):
                return last_status
            time.sleep(interval)
        raise TimeoutError(f"Download {download_id} did not finish within {timeout} seconds")

    def list_drivers(self) -> Any:
        return self.get("/api/drivers")

    def list_pc_links(self) -> Any:
        return self.get("/api/pc-links")

    def save_pc_link(self, name: str, base_url: str, pc_id: str = "") -> Any:
        return self.post("/api/pc-links", {"name": name, "base_url": base_url, "id": pc_id})

    def test_pc_link(self, base_url: str) -> Any:
        return self.post("/api/pc-links/test", {"base_url": base_url})

    def remove_pc_link(self, pc_id: str) -> Any:
        return self.post("/api/pc-links/remove", {"id": pc_id})

    def list_benchmark_results(self) -> Any:
        return self.get("/api/benchmark/results")

    def get_benchmark_result(self, result_id: str) -> Any:
        return self.get(f"/api/benchmark/results/{result_id}")

    def run_benchmark(
        self,
        model_path: str,
        backends: List[str],
        bench_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.post(
            "/api/benchmark/run",
            {
                "model_path": model_path,
                "backends": backends,
                "bench_params": bench_params,
                "model_params": model_params,
            },
        )

    def benchmark_status(self, task_id: str) -> Any:
        return self.get(f"/api/benchmark/status/{task_id}")

    def list_active_benchmarks(self) -> Any:
        return self.get("/api/benchmark/list-active")

    def cancel_benchmark(self, task_id: str) -> Any:
        return self.post(f"/api/benchmark/cancel/{task_id}")

    def inference_status(self) -> Any:
        return self.get("/api/api-server/status")

    def start_inference_api(
        self,
        model_id: str = "",
        port: int = 8787,
        mode: str = "openai",
        device: str = "",
        gpu_index: str = "",
    ) -> Any:
        return self.post(
            "/api/api-server/start",
            {
                "model_id": model_id,
                "port": port,
                "mode": mode,
                "device": device,
                "gpu_index": gpu_index,
            },
        )

    def stop_inference_api(self) -> Any:
        return self.post("/api/api-server/stop")


def parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def chat_completion(
    inference_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Any:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    with httpx.Client(timeout=300.0) as client:
        response = client.post(f"{inference_url.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

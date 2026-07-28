import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from agent_api import (
    DEFAULT_DASHBOARD_URL,
    DEFAULT_INFERENCE_URL,
    DashboardClient,
    chat_completion,
)


mcp = FastMCP("llama-dash")


def _client() -> DashboardClient:
    return DashboardClient(os.environ.get("LLAMA_DASH_URL", DEFAULT_DASHBOARD_URL))


@mcp.tool()
def list_gpus() -> Dict[str, Any]:
    """List GPUs detected by the dashboard backend."""
    return _client().list_gpus()


@mcp.tool()
def list_models() -> Dict[str, Any]:
    """List locally downloaded models."""
    return _client().list_models()


@mcp.tool()
def search_models(query: str, limit: int = 20, model_format: str = "gguf") -> Dict[str, Any]:
    """Search Hugging Face models that can be downloaded by the dashboard."""
    return _client().search_models(query, limit, model_format)


@mcp.tool()
def list_model_files(repo_id: str, model_format: str = "gguf") -> Dict[str, Any]:
    """List downloadable files for a Hugging Face model repository."""
    return _client().list_model_files(repo_id, model_format)


@mcp.tool()
def download_model(repo_id: str, filename: str) -> Dict[str, Any]:
    """Start downloading a GGUF model file."""
    return _client().download_model(repo_id, filename)


@mcp.tool()
def download_openvino_model(repo_id: str) -> Dict[str, Any]:
    """Start downloading an OpenVINO model snapshot."""
    return _client().download_openvino_model(repo_id)


@mcp.tool()
def download_progress(download_id: str) -> Dict[str, Any]:
    """Get progress for a model or driver download."""
    return _client().download_progress(download_id)


@mcp.tool()
def list_active_downloads() -> Dict[str, Any]:
    """List active downloads."""
    return _client().list_active_downloads()


@mcp.tool()
def list_drivers() -> Dict[str, Any]:
    """List local and active llama.cpp drivers."""
    return _client().list_drivers()


@mcp.tool()
def list_pc_links() -> Dict[str, Any]:
    """List linked PCs that provide remote models."""
    return _client().list_pc_links()


@mcp.tool()
def test_pc_link(base_url: str) -> Dict[str, Any]:
    """Test a remote PC URL by reading its /v1/models endpoint."""
    return _client().test_pc_link(base_url)


@mcp.tool()
def add_pc_link(name: str, base_url: str, pc_id: str = "") -> Dict[str, Any]:
    """Add or update a linked PC."""
    return _client().save_pc_link(name, base_url, pc_id)


@mcp.tool()
def remove_pc_link(pc_id: str) -> Dict[str, Any]:
    """Remove a linked PC."""
    return _client().remove_pc_link(pc_id)


@mcp.tool()
def run_benchmark(
    model_path: str,
    backends: List[str],
    bench_params: Optional[Dict[str, Any]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Start a benchmark task for a local model path and one or more backends."""
    return _client().run_benchmark(model_path, backends, bench_params, model_params)


@mcp.tool()
def benchmark_status(task_id: str) -> Dict[str, Any]:
    """Get benchmark task status."""
    return _client().benchmark_status(task_id)


@mcp.tool()
def list_active_benchmarks() -> Dict[str, Any]:
    """List currently active benchmark tasks."""
    return _client().list_active_benchmarks()


@mcp.tool()
def cancel_benchmark(task_id: str) -> Dict[str, Any]:
    """Cancel a running benchmark task."""
    return _client().cancel_benchmark(task_id)


@mcp.tool()
def multimodal_capabilities() -> Dict[str, Any]:
    """List multimodal testing protocol, modalities, modes, and limits."""
    return _client().multimodal_capabilities()


@mcp.tool()
def list_multimodal_profiles() -> Dict[str, Any]:
    """List saved TTS, STT, image-generation, and video-generation profiles."""
    return _client().list_multimodal_profiles()


@mcp.tool()
def save_multimodal_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update a multimodal adapter profile."""
    return _client().save_multimodal_profile(profile)


@mcp.tool()
def delete_multimodal_profile(profile_id: str) -> Dict[str, Any]:
    """Delete a saved multimodal adapter profile."""
    return _client().delete_multimodal_profile(profile_id)


@mcp.tool()
def list_multimodal_cases() -> Dict[str, Any]:
    """List saved multimodal test cases."""
    return _client().list_multimodal_cases()


@mcp.tool()
def save_multimodal_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update a reproducible multimodal test case."""
    return _client().save_multimodal_case(case)


@mcp.tool()
def delete_multimodal_case(case_id: str) -> Dict[str, Any]:
    """Delete a saved multimodal test case."""
    return _client().delete_multimodal_case(case_id)


@mcp.tool()
def list_multimodal_runs() -> Dict[str, Any]:
    """List multimodal test runs and their progress/results."""
    return _client().list_multimodal_runs()


@mcp.tool()
def create_multimodal_run(profile_id: str, case_id: str) -> Dict[str, Any]:
    """Queue a multimodal test run for a saved profile and case."""
    return _client().create_multimodal_run(profile_id, case_id)


@mcp.tool()
def run_multimodal_test(
    modality: str,
    model_id: str,
    prompt: str = "",
    reference_image_base64: str = "",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    guidance: float = 2.5,
    seed: int = -1,
    mode: str = "",
    model_path: str = "",
) -> Dict[str, Any]:
    """Run TTS, STT, image-generation, or video-generation.

    Use reference_image_base64 for Kontext-style image editing. For example,
    provide modality="imagegen", prompt="preserve identity, change pose",
    reference_image_base64="<PNG/JPEG base64>", width=1024, height=1024,
    steps=28, guidance=2.5, and seed=-1. Never use chat completions for
    multimodal models.
    """
    input_data: Dict[str, Any] = {
        "prompt": prompt,
        "text": prompt,
        "options": {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
        },
    }
    if reference_image_base64:
        input_data["image_base64"] = reference_image_base64
    return _client().run_multimodal_test(modality, model_id, input_data, mode or None, model_path)


@mcp.tool()
def get_multimodal_run(run_id: str) -> Dict[str, Any]:
    """Get one multimodal run, including progress, metrics, and artifacts."""
    return _client().get_multimodal_run(run_id)


@mcp.tool()
def cancel_multimodal_run(run_id: str) -> Dict[str, Any]:
    """Cancel a queued or running multimodal test."""
    return _client().cancel_multimodal_run(run_id)


@mcp.tool()
def list_benchmark_results() -> Dict[str, Any]:
    """List saved benchmark results."""
    return _client().list_benchmark_results()


@mcp.tool()
def get_benchmark_result(result_id: str) -> Dict[str, Any]:
    """Get one saved benchmark result by id."""
    return _client().get_benchmark_result(result_id)


@mcp.tool()
def inference_api_status() -> Dict[str, Any]:
    """Show local OpenAI/Anthropic-compatible inference API status."""
    return _client().inference_status()


@mcp.tool()
def start_inference_api(
    model_id: str = "",
    port: int = 8787,
    mode: str = "openai",
    device: str = "",
    gpu_index: str = "",
) -> Dict[str, Any]:
    """Start the local inference API for a downloaded model."""
    return _client().start_inference_api(model_id, port, mode, device, gpu_index)


@mcp.tool()
def stop_inference_api() -> Dict[str, Any]:
    """Stop the local inference API."""
    return _client().stop_inference_api()


@mcp.tool()
def chat_completion_openai(
    message: str,
    model: str = "",
    system: str = "",
    inference_url: str = DEFAULT_INFERENCE_URL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    reference_image_base64: str = "",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    guidance: float = 2.5,
    seed: int = -1,
) -> Dict[str, Any]:
    """Send text to an LLM, or edit/generate an image when a multimodal model is selected.

    For FLUX/Kontext image work, pass reference_image_base64 for the source image
    and use negative_prompt, width, height, steps, guidance, and seed for image
    preferences. This compatibility tool routes those requests to the image
    runner instead of /v1/chat/completions.
    """
    multimodal_markers = ("flux", "stable-diffusion", "imagegen", "text-to-image", "cogvideo", "tts", "whisper")
    if any(marker in model.lower() for marker in multimodal_markers):
        modality = "imagegen" if any(marker in model.lower() for marker in (
            "flux", "stable-diffusion", "imagegen", "text-to-image",
        )) else ("stt" if "whisper" in model.lower() else "tts")
        return _client().run_multimodal_test(
            modality,
            model,
            {
                "prompt": message,
                "text": message,
                **({"image_base64": reference_image_base64} if reference_image_base64 else {}),
                "options": {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed,
                },
            },
            "builtin" if modality == "imagegen" else None,
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    return chat_completion(inference_url, model, messages, max_tokens, temperature)


if __name__ == "__main__":
    mcp.run()

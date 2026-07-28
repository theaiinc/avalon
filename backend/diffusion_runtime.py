"""Small stable-diffusion.cpp runtime adapter for local image generation."""

import os
import platform
import subprocess
import zipfile
import base64
from pathlib import Path

import httpx

from config import DATA_DIR

RUNTIME_DIR = DATA_DIR / "runtimes" / "stable-diffusion-cpp"
ASSET_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
FLUX_FILES = {
    # The official FLUX repo is gated; this public mirror contains the same
    # standard 335 MB ae.safetensors VAE required by stable-diffusion.cpp.
    "vae": "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors",
    "clip_l": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    "t5xxl": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
}
QWEN_FILES = {
    "vae": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    "llm": "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(300, connect=30)) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
    partial.replace(destination)


def ensure_cli() -> Path:
    configured = os.environ.get("AVALON_SD_CLI")
    if configured and Path(configured).is_file():
        return Path(configured)
    for candidate in RUNTIME_DIR.rglob("sd-cli*"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30) as client:
        release = client.get(ASSET_API).json()
    system = platform.system()
    machine = platform.machine().lower()
    marker = "Darwin-macOS" if system == "Darwin" else ("Linux-Ubuntu" if system == "Linux" else "win")
    asset = next(
        (item for item in release.get("assets", [])
         if marker in item.get("name", "") and (machine in item.get("name", "").lower() or system == "Windows")),
        None,
    )
    if not asset:
        raise RuntimeError("No stable-diffusion.cpp binary is available for this platform")
    archive = RUNTIME_DIR / asset["name"]
    if not archive.exists():
        _download(asset["browser_download_url"], archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(RUNTIME_DIR)
    for candidate in RUNTIME_DIR.rglob("sd-cli*"):
        if candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | 0o111)
            return candidate
    raise RuntimeError("stable-diffusion.cpp archive did not contain sd-cli")


def ensure_flux_files() -> dict:
    paths = {}
    for key, url in FLUX_FILES.items():
        destination = RUNTIME_DIR / "models" / Path(url.split("?")[0]).name
        if not destination.exists():
            _download(url, destination)
        paths[key] = str(destination)
    return paths


def ensure_qwen_files() -> dict:
    paths = {}
    for key, url in QWEN_FILES.items():
        destination = RUNTIME_DIR / "models" / Path(url.split("?")[0]).name
        if not destination.exists():
            _download(url, destination)
        paths[key] = str(destination)
    return paths


def _resolve_diffusion_model(model_path: str) -> Path:
    """Accept Avalon model directories as well as direct GGUF paths."""
    candidate = Path(model_path).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"image model path does not exist: {candidate}")
    gguf_files = sorted(candidate.rglob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"no .gguf diffusion model found under: {candidate}")
    # Prefer the primary FLUX/diffusion file when a repository contains
    # auxiliary or quantized variants; otherwise use the only discovered file.
    preferred = [
        path for path in gguf_files
        if any(marker in path.name.lower() for marker in ("flux", "diffusion", "unet"))
    ]
    return preferred[0] if preferred else gguf_files[0]


def _cli_supports(cli: Path, argument: str) -> bool:
    try:
        help_result = subprocess.run(
            [str(cli), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return argument in f"{help_result.stdout}\n{help_result.stderr}"
    except (OSError, subprocess.SubprocessError):
        return False


def generate_flux(
    model_path: str,
    prompt: str,
    output_path: Path,
    options: dict | None = None,
    on_process=None,
) -> dict:
    options = options or {}
    diffusion_model = _resolve_diffusion_model(model_path)
    is_qwen = "qwen-image" in str(diffusion_model).lower() or "qwen_image" in str(diffusion_model).lower()
    steps = max(1, min(int(options.get("steps", 28 if is_qwen else 4)), 100))
    cli = ensure_cli()
    dependencies = ensure_qwen_files() if is_qwen else ensure_flux_files()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command_options = [
        "--cfg-scale", str(float(options.get("guidance", 2.5 if is_qwen else 1))),
        "--sampling-method", str(options.get("sampling_method", "euler")),
        "--steps", str(steps),
        "--clip-on-cpu",
    ]
    negative_prompt = str(options.get("negative_prompt", "")).strip()
    if negative_prompt:
        command_options.extend(["--negative-prompt", negative_prompt])
    seed = options.get("seed")
    if seed is not None:
        command_options.extend(["--seed", str(int(seed))])
    for key, flag in (("width", "--width"), ("height", "--height")):
        value = options.get(key)
        if value is not None:
            command_options.extend([flag, str(max(64, min(int(value), 4096)))])
    image_base64 = options.get("image_base64")
    if image_base64:
        try:
            input_path = output_path.parent / "input.png"
            input_path.write_bytes(base64.b64decode(image_base64, validate=True))
            command_options.extend(["--init-img", str(input_path)])
        except (ValueError, OSError) as exc:
            raise ValueError("image_base64 must be valid base64 image data") from exc
    if is_qwen:
        command_options.extend([
            "--llm", dependencies["llm"],
            "--flow-shift", "3",
            "--offload-to-cpu",
            "--diffusion-fa",
        ])
        if _cli_supports(cli, "--qwen-image-zero-cond-t"):
            command_options.append("--qwen-image-zero-cond-t")
        if image_base64:
            command_options.extend(["-r", str(input_path)])
    command = [
        str(cli), "--diffusion-model", str(diffusion_model),
        "--vae", dependencies["vae"],
        *(["--clip_l", dependencies["clip_l"], "--t5xxl", dependencies["t5xxl"]]
          if not is_qwen else []),
        "-p", prompt,
        *command_options, "-o", str(output_path),
    ]
    completed = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if on_process:
        on_process(completed)
    stdout, stderr = completed.communicate(timeout=1800)
    if completed.returncode:
        raise RuntimeError(stderr.strip() or stdout.strip() or "stable-diffusion.cpp failed")
    if not output_path.is_file():
        raise RuntimeError("stable-diffusion.cpp completed without an image artifact")
    return {"path": str(output_path), "stdout": stdout[-2000:]}

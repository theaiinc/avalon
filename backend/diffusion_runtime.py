"""Small stable-diffusion.cpp runtime adapter for local image generation."""

import os
import platform
import subprocess
import zipfile
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


def generate_flux(model_path: str, prompt: str, output_path: Path, steps: int = 4) -> dict:
    cli = ensure_cli()
    dependencies = ensure_flux_files()
    diffusion_model = _resolve_diffusion_model(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli), "--diffusion-model", str(diffusion_model),
        "--vae", dependencies["vae"], "--clip_l", dependencies["clip_l"],
        "--t5xxl", dependencies["t5xxl"], "-p", prompt,
        "--cfg-scale", "1", "--sampling-method", "euler",
        "--steps", str(steps), "--clip-on-cpu", "-o", str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "stable-diffusion.cpp failed")
    if not output_path.is_file():
        raise RuntimeError("stable-diffusion.cpp completed without an image artifact")
    return {"path": str(output_path), "stdout": completed.stdout[-2000:]}

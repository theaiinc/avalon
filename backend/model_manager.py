import os
import json
import shutil
from typing import List, Dict, Optional, Callable
from huggingface_hub import HfApi
from config import MODELS_DIR


api = HfApi()

CRISPERWHISPER_FORMAT = "crisperwhisper"
CRISPERWHISPER_REPOS = {
    "nyralabs/CrisperWhisper2.0_large",
    "nyralabs/CrisperWhisper2.0_medium",
    "nyralabs/CrisperWhisper2.0_turbo",
    "nyralabs/CrisperWhisper2.0_small",
}


def _infer_capabilities(text: str, declared=None) -> List[str]:
    if declared:
        return sorted({str(value) for value in declared})
    value = text.lower()
    capabilities = []
    patterns = {
        "tts": ("tts", "text-to-speech", "speech-synthesis", "kokoro", "bark", "piper"),
        "stt": ("whisper", "speech-to-text", "speech_recognition", "automatic-speech-recognition", "wav2vec", "asr"),
        "imagegen": ("stable-diffusion", "stable_diffusion", "sdxl", "flux", "qwen-image", "qwen_image", "text-to-image", "imagegen", "diffusion"),
        "videogen": ("video-generation", "video_generation", "cogvideo", "hunyuan-video", "mochi", "animatediff", "text-to-video"),
    }
    for capability, markers in patterns.items():
        if any(marker in value for marker in markers):
            capabilities.append(capability)
    return capabilities


def search_models(query: str, limit: int = 20, model_format: str = "gguf") -> List[Dict]:
    if model_format == CRISPERWHISPER_FORMAT:
        models = [
            model for model in api.list_models(
                search=query or "CrisperWhisper",
                sort="downloads",
                limit=max(limit, len(CRISPERWHISPER_REPOS)),
            )
            if model.modelId in CRISPERWHISPER_REPOS
        ]
    else:
        tag = "gguf" if model_format == "gguf" else "openvino"
        models = api.list_models(
            search=query,
            filter=tag,
            sort="downloads",
            limit=limit,
        )
    return [
        {
            "id": m.modelId,
            "name": m.modelId.split("/")[-1],
            "author": m.modelId.split("/")[0],
            "downloads": getattr(m, "downloads", 0),
            "likes": getattr(m, "likes", 0),
            "pipeline_tag": getattr(m, "pipeline_tag", ""),
            "last_modified": m.lastModified.isoformat() if m.lastModified else "",
            "format": model_format,
            "supported": model_format != CRISPERWHISPER_FORMAT or m.modelId in CRISPERWHISPER_REPOS,
        }
        for m in models
    ]


def list_files(repo_id: str, model_format: str = "gguf") -> List[Dict]:
    try:
        files = api.list_repo_files(repo_id)
        if model_format == "gguf":
            matched = [f for f in files if f.endswith(".gguf")]
        elif model_format == CRISPERWHISPER_FORMAT:
            matched = [f for f in files if not f.endswith("/") and f != "model.json"]
        else:
            matched = [f for f in files if f.endswith(".xml") or f.endswith(".bin")]
        path_infos = api.get_paths_info(repo_id, paths=matched)
        sizes = {p.path: p.size for p in path_infos}
        return [
            {"name": f, "size": sizes.get(f, 0), "path": f}
            for f in matched
        ]
    except Exception:
        return []


def get_local_models() -> List[Dict]:
    models = []
    for entry in os.listdir(MODELS_DIR):
        entry_path = MODELS_DIR / entry
        if entry_path.is_dir() and not entry.startswith("_"):
            meta_file = entry_path / "model.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            capabilities = _infer_capabilities(
                " ".join([entry, meta.get("repo_id", ""), *meta.get("files", [])]),
                meta.get("capabilities"),
            )

            gguf_files = list(entry_path.rglob("*.gguf"))
            ov_xmls = list(entry_path.glob("openvino_*.xml"))

            if meta.get("format") == CRISPERWHISPER_FORMAT:
                file_names = [
                    str(path.relative_to(entry_path))
                    for path in entry_path.rglob("*")
                    if path.is_file() and path.name != "model.json"
                ]
                models.append({
                    "id": entry,
                    "repo_id": meta.get("repo_id", entry),
                    "files": sorted(file_names),
                    "path": str(entry_path),
                    "size": sum(f.stat().st_size for f in entry_path.rglob("*") if f.is_file()),
                    "format": CRISPERWHISPER_FORMAT,
                    "openvino_path": None,
                    "serving_supported": False,
                    "capabilities": capabilities or ["stt"],
                })
            elif ov_xmls:
                models.append({
                    "id": entry,
                    "repo_id": meta.get("repo_id", entry),
                    "files": [],
                    "path": str(entry_path),
                    "size": sum(f.stat().st_size for f in entry_path.rglob("*") if f.is_file()),
                    "format": "openvino",
                    "openvino_path": str(entry_path),
                    "capabilities": capabilities,
                })
            elif gguf_files:
                file_names = [str(f.relative_to(entry_path)) for f in gguf_files]
                search_text = " ".join([entry, meta.get("repo_id", ""), *file_names]).lower()
                # DSpark/DFlash GGUFs are speculative draft heads, not
                # standalone chat targets. They require a separate target
                # model and must not be offered as directly completable.
                is_draft = any(marker in search_text for marker in ("dspark", "dflash")) and (
                    "draft" in search_text or "block" in search_text
                )
                models.append({
                    "id": entry,
                    "repo_id": meta.get("repo_id", entry),
                    "files": file_names,
                    "path": str(entry_path),
                    "size": sum(f.stat().st_size for f in gguf_files if f.exists()),
                    "format": "gguf",
                    "openvino_path": None,
                    "is_draft": is_draft,
                    "serving_supported": not is_draft,
                    "capabilities": capabilities,
                })
    return models


def download_model(repo_id: str, filename: str, on_progress: Optional[Callable] = None) -> Dict:
    import httpx
    from huggingface_hub import hf_hub_url, get_hf_file_metadata
    safe_name = repo_id.replace("/", "_")
    model_dir = MODELS_DIR / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress(status="starting", percent=0, stage="Starting...")

    url = hf_hub_url(repo_id, filename)
    meta = get_hf_file_metadata(url)
    total_size = meta.size or 0
    downloaded = 0
    partial_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with partial_path.open("wb") as f:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(300, connect=30)) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and total_size > 0:
                            pct = int(downloaded / total_size * 95)
                            on_progress(status="downloading", percent=pct, stage=f"Downloading... ({downloaded//1024//1024} MB / {total_size//1024//1024} MB)")
        if total_size and downloaded != total_size:
            raise RuntimeError(f"model download incomplete: received {downloaded} of {total_size} bytes")
        partial_path.replace(dest_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    if on_progress:
        on_progress(status="done", percent=100, stage="Done")

    meta = {"repo_id": repo_id, "files": [filename]}
    with open(model_dir / "model.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "id": safe_name,
        "repo_id": repo_id,
        "filename": filename,
        "path": str(dest_path),
    }


def download_openvino_model(repo_id: str, on_progress: Optional[Callable] = None) -> Dict:
    from huggingface_hub import snapshot_download
    safe_name = repo_id.replace("/", "_")
    model_dir = MODELS_DIR / safe_name

    if on_progress:
        on_progress(status="downloading", percent=10, stage="Downloading OpenVINO IR model...")

    snapshot_download(repo_id=repo_id, local_dir=str(model_dir), local_dir_use_symlinks=False)

    if on_progress:
        on_progress(status="done", percent=100, stage="Done")

    meta = {"repo_id": repo_id, "files": [], "format": "openvino"}
    with open(model_dir / "model.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "id": safe_name,
        "repo_id": repo_id,
        "filename": "",
        "path": str(model_dir),
    }


def download_crisperwhisper_model(repo_id: str, on_progress: Optional[Callable] = None) -> Dict:
    if repo_id not in CRISPERWHISPER_REPOS:
        raise ValueError("Only official Nyra Labs CrisperWhisper 2.0 repositories are supported")
    from huggingface_hub import snapshot_download

    safe_name = repo_id.replace("/", "_")
    model_dir = MODELS_DIR / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(status="downloading", percent=5, stage="Downloading CrisperWhisper Transformers model...")
    # Do not selectively fetch files: the official Transformers loader needs
    # the complete snapshot, including tokenizer/configuration sidecars.
    snapshot_download(repo_id=repo_id, local_dir=str(model_dir))
    files = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file()
    )
    metadata = {
        "repo_id": repo_id,
        "files": files,
        "format": CRISPERWHISPER_FORMAT,
        "capabilities": ["stt"],
        "serving_supported": False,
    }
    with open(model_dir / "model.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if on_progress:
        on_progress(status="done", percent=100, stage="Done")
    return {
        "id": safe_name,
        "repo_id": repo_id,
        "filename": "",
        "path": str(model_dir),
        "format": CRISPERWHISPER_FORMAT,
    }


def remove_model(model_id: str):
    model_dir = MODELS_DIR / model_id
    if model_dir.exists():
        shutil.rmtree(model_dir)

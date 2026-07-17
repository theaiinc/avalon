import os
import json
import shutil
import subprocess
import httpx
import zipfile
import tarfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable
from config import DRIVERS_DIR, LLAMA_CPP_REPO

RELEASES_CACHE_FILE = DRIVERS_DIR / "releases_cache.json"

BACKEND_MAP = {
    "cuda": {"pattern": r"llama-b.*-bin-win-cuda-\d+\.\d+-x64\.zip", "ggml_dll": "ggml-cuda.dll"},
    "vulkan": {"pattern": r"llama-b.*-bin-win-vulkan-x64\.zip", "ggml_dll": "ggml-vulkan.dll"},
    "sycl": {"pattern": r"llama-b.*-bin-win-sycl-x64\.zip", "ggml_dll": "ggml-sycl.dll"},
    "openvino": {"pattern": r"llama-b.*-bin-win-openvino-.*-x64\.zip", "ggml_dll": "ggml-openvino.dll"},
    "directml": {"pattern": r"llama-cpp-directml.*\.zip", "ggml_dll": "ggml-directml.dll"},
    "hip": {"pattern": r"llama-b.*-bin-win-hip-.*-x64\.zip", "ggml_dll": "ggml-hip.dll"},
    "cpu": {"pattern": r"llama-b.*-bin-win-cpu-x64\.zip", "ggml_dll": "ggml.dll"},
    "metal": {"pattern": r"llama-b.*-bin-macos-.*\.tar\.gz", "ggml_dll": "ggml-metal.dylib"},
}

FORK_SOURCES = {
    "directml": {
        "repo": "Caerleus/llama-cpp-directml",
        "label": "Caerleus/llama-cpp-directml (community fork)",
        "per_page": 3,
    },
}


async def fetch_releases() -> List[Dict]:
    if RELEASES_CACHE_FILE.exists():
        with open(RELEASES_CACHE_FILE) as f:
            return json.load(f)
    data = []
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=10",
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        resp.raise_for_status()
        releases = resp.json()
        for r in releases:
            entry = {
                "id": r["id"],
                "tag_name": r["tag_name"],
                "name": r["name"],
                "published_at": r["published_at"],
                "source": "official",
                "source_repo": LLAMA_CPP_REPO,
                "backends": [],
                "assets": [
                    {
                        "id": a["id"],
                        "name": a["name"],
                        "size": a["size"],
                        "browser_download_url": a["browser_download_url"],
                        "content_type": a.get("content_type", ""),
                    }
                    for a in r.get("assets", [])
                ],
            }
            for b, info in BACKEND_MAP.items():
                if any(re.match(info["pattern"], a["name"]) for a in r.get("assets", [])):
                    entry["backends"].append(b)
            data.append(entry)

        for backend, fork in FORK_SOURCES.items():
            try:
                fork_r = await client.get(
                    f"https://api.github.com/repos/{fork['repo']}/releases?per_page={fork['per_page']}",
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=10,
                )
                if fork_r.status_code == 200:
                    for fr in fork_r.json():
                        fe = {
                            "id": f"fork_{fr['id']}",
                            "tag_name": fr["tag_name"],
                            "name": f"{fr['name']} [{fork['label']}]",
                            "published_at": fr["published_at"],
                            "source": "community",
                            "source_repo": fork["repo"],
                            "source_label": fork["label"],
                            "backends": [],
                            "assets": [
                                {
                                    "id": a["id"],
                                    "name": a["name"],
                                    "size": a["size"],
                                    "browser_download_url": a["browser_download_url"],
                                    "content_type": a.get("content_type", ""),
                                }
                                for a in fr.get("assets", [])
                            ],
                        }
                        if any(re.match(BACKEND_MAP[backend]["pattern"], a["name"]) for a in fr.get("assets", [])):
                            fe["backends"].append(backend)
                        data.append(fe)
            except Exception:
                pass
    with open(RELEASES_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _parse_tag_num(tag: str) -> int:
    m = re.search(r'b(\d+)', tag)
    return int(m.group(1)) if m else 0


def get_local_drivers(releases: Optional[List[Dict]] = None) -> List[Dict]:
    drivers = []
    for entry in sorted(os.listdir(DRIVERS_DIR)):
        entry_path = DRIVERS_DIR / entry
        if entry_path.is_dir():
            meta_file = entry_path / "driver.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            backend = meta.get("backend", "unknown")
            tag = meta.get("tag", entry)
            item = {
                "id": entry,
                "tag": tag,
                "backend": backend,
                "path": str(entry_path),
                "active": meta.get("active", False),
                "size": _get_dir_size(entry_path),
                "llama_bench_path": find_llama_bench(entry),
                "source_label": meta.get("source_label"),
                "has_update": False,
                "latest_tag": None,
            }
            if releases:
                current_num = _parse_tag_num(tag)
                for rel in releases:
                    if backend in rel.get("backends", []):
                        rel_num = _parse_tag_num(rel["tag_name"])
                        if rel_num > current_num:
                            item["has_update"] = True
                            if item["latest_tag"] is None or _parse_tag_num(item["latest_tag"]) < rel_num:
                                item["latest_tag"] = rel["tag_name"]
            drivers.append(item)

    # Synthetic "npu" driver using openvino-genai (not llama-bench)
    _add_npu_driver(drivers)

    return drivers


def _add_npu_driver(drivers: List[Dict]):
    try:
        from npu_runner import check_npu_available
        if check_npu_available():
            existing = any(d["backend"] == "npu" for d in drivers)
            if not existing:
                drivers.append({
                    "id": "npu_openvino_genai",
                    "tag": "openvino-genai",
                    "backend": "npu",
                    "path": "",
                    "active": True,
                    "size": 0,
                    "llama_bench_path": None,
                    "source_label": "OpenVINO GenAI (system package)",
                    "has_update": False,
                    "latest_tag": None,
                })
    except Exception:
        pass


async def download_driver(tag: str, backend: str, on_progress: Optional[Callable] = None) -> Dict:
    if on_progress:
        on_progress(status="downloading", percent=0, stage="Fetching release info...")
    releases = await fetch_releases()
    release = next(
        (r for r in releases if r["tag_name"] == tag and backend in r.get("backends", [])),
        None,
    )
    if not release:
        release = next((r for r in releases if r["tag_name"] == tag), None)
    if not release:
        raise ValueError(f"Release {tag} not found")

    pattern_info = BACKEND_MAP.get(backend)
    if not pattern_info:
        raise ValueError(f"Unknown backend: {backend}")

    pattern = pattern_info["pattern"]
    asset = None
    for a in release["assets"]:
        if re.match(pattern, a["name"]):
            asset = a
            break
    if not asset:
        raise ValueError(f"No asset matching '{pattern}' in release {tag}")

    driver_id = f"{backend}_{tag}"
    driver_dir = DRIVERS_DIR / driver_id
    driver_dir.mkdir(parents=True, exist_ok=True)

    dest_path = driver_dir / asset["name"]
    if not dest_path.exists():
        if on_progress:
            on_progress(status="downloading", percent=5, stage="Downloading...")
        total_size = asset["size"]
        downloaded = 0
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", asset["browser_download_url"]) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and total_size > 0:
                            pct = 5 + int(downloaded / total_size * 70)
                            on_progress(status="downloading", percent=min(pct, 75), stage=f"Downloading... ({downloaded//1024//1024} MB / {total_size//1024//1024} MB)")

    extract_dir = driver_dir / "extracted"
    if not extract_dir.exists():
        if on_progress:
            on_progress(status="extracting", percent=80, stage="Extracting...")
        extract_dir.mkdir(exist_ok=True)
        if dest_path.suffix == ".zip":
            with zipfile.ZipFile(dest_path, "r") as zf:
                zf.extractall(extract_dir)
        elif dest_path.suffix in (".tar", ".gz", ".xz"):
            with tarfile.open(dest_path, "r:*") as tf:
                tf.extractall(extract_dir)
        if on_progress:
            on_progress(status="done", percent=100, stage="Done")

    meta = {
        "backend": backend,
        "tag": tag,
        "active": True,
        "source_label": release.get("source_label", LLAMA_CPP_REPO),
    }
    with open(driver_dir / "driver.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "id": driver_id,
        "tag": tag,
        "backend": backend,
        "path": str(driver_dir),
        "active": True,
        "llama_bench_path": find_llama_bench(driver_id),
    }


def set_driver_active(driver_id: str, active: bool) -> Dict:
    driver_dir = DRIVERS_DIR / driver_id
    if not driver_dir.exists():
        raise ValueError(f"Driver {driver_id} not found")
    meta_file = driver_dir / "driver.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
    meta["active"] = active
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    return {
        "id": driver_id,
        "tag": meta.get("tag", driver_id),
        "backend": meta.get("backend", "unknown"),
        "active": active,
        "llama_bench_path": find_llama_bench(driver_id),
    }


def remove_driver(driver_id: str):
    driver_dir = DRIVERS_DIR / driver_id
    if driver_dir.exists():
        shutil.rmtree(driver_dir)


def get_active_drivers() -> List[Dict]:
    result = []
    for entry in os.listdir(DRIVERS_DIR):
        meta_file = DRIVERS_DIR / entry / "driver.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("active"):
                result.append({
                    "id": entry,
                    "tag": meta.get("tag", ""),
                    "backend": meta.get("backend", ""),
                    "path": str(DRIVERS_DIR / entry),
                    "llama_bench_path": find_llama_bench(entry),
                })
    return result


def find_driver_executable(driver_id: str, names: List[str]) -> Optional[str]:
    base = DRIVERS_DIR / driver_id / "extracted"
    if not base.exists():
        return None
    wanted = {name.lower() for name in names}
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower() in wanted:
                path = os.path.join(root, f)
                if os.name != "nt":
                    try:
                        mode = os.stat(path).st_mode
                        os.chmod(path, mode | 0o755)
                    except OSError:
                        pass
                return path
    return None


def find_llama_bench(driver_id: str) -> Optional[str]:
    return find_driver_executable(driver_id, ["llama-bench", "llama-bench.exe"])


def _get_dir_size(path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

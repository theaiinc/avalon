import os
import re
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

from config import MODELS_DIR, RESULTS_DIR

OV_CACHE_DIR = MODELS_DIR / "_openvino_cache"
OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def check_npu_available() -> bool:
    try:
        import openvino as ov
        core = ov.Core()
        return "NPU" in [d.upper() for d in core.available_devices]
    except Exception:
        return False


def ensure_ov_model(gguf_path: str, bench_params: Optional[Dict] = None) -> Optional[str]:
    path = Path(gguf_path)

    def _is_ov_dir(p: Path) -> bool:
        if not p.is_dir():
            return False
        return any(p.glob("openvino_*.xml"))

    # Already an OpenVINO IR model directory
    if _is_ov_dir(path):
        return str(path)

    # Check if parent is OpenVINO IR (e.g. if user selected a file inside the dir)
    if path.parent and _is_ov_dir(path.parent):
        return str(path.parent)

    # GGUF file — check size
    if path.suffix.lower() == ".gguf":
        gguf_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        if gguf_size_mb > 4096:
            return None

        model_dir_name = path.parent.name
        cache_key = re.sub(r'[^a-zA-Z0-9_-]', '_', model_dir_name)
        ov_dir = OV_CACHE_DIR / cache_key
        ov_dir.mkdir(parents=True, exist_ok=True)

        ir_xml = ov_dir / "openvino_model.xml"
        if ir_xml.exists():
            return str(ov_dir)

        hf_repo = _infer_hf_repo(path)
        if not hf_repo:
            hf_repo = _infer_hf_repo_from_path(path)

        if hf_repo:
            ov_repo = _find_openvino_variant(hf_repo)
            if ov_repo:
                _download_ov_model(ov_repo, ov_dir)
                if ir_xml.exists():
                    return str(ov_dir)

        if hf_repo:
            _convert_hf_to_openvino(hf_repo, ov_dir)
            if ir_xml.exists():
                return str(ov_dir)

    return None


def _infer_hf_repo(gguf_path: Path) -> Optional[str]:
    try:
        with open(gguf_path, "rb") as f:
            header = f.read(8)
            if header[:4] != b"GGUF":
                return None
            import struct
            n_kv = struct.unpack("<Q", header[4:8])[0]
            f.read(8)
            f.read(8)
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                key = f.read(klen).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:
                    vlen = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(vlen).decode("utf-8", errors="replace")
                    if "source" in key.lower() and "hf" in key.lower():
                        return val.strip()
                elif vtype in (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12):
                    pass
                else:
                    break
    except Exception:
        pass
    return None


def _infer_hf_repo_from_path(gguf_path: Path) -> Optional[str]:
    name = gguf_path.parent.name
    parts = name.split("_", 1)
    if len(parts) >= 2 and parts[0].isalnum():
        repo = f"{parts[0]}/{parts[1]}"
        for suffix in ["_GGUF", "-GGUF", "_gguf", "-gguf"]:
            repo = repo.replace(suffix, "")
        return repo
    return None


def _find_openvino_variant(hf_repo: str) -> Optional[str]:
    candidates = [
        f"{hf_repo}-OpenVINO",
        f"{hf_repo}-openvino",
        f"{hf_repo}-OV",
    ]
    author = hf_repo.split("/")[0]
    model_name = hf_repo.split("/")[-1]
    candidates.extend([
        f"{author}/{model_name}-OpenVINO",
        f"{author}/{model_name}-openvino",
        f"Intel/{model_name}-OpenVINO",
    ])
    from huggingface_hub import HfApi
    api = HfApi()
    for c in candidates:
        try:
            api.model_info(c, timeout=5)
            return c
        except Exception:
            continue
    try:
        for m in api.list_models(search="openvino", pipeline_tag="text-generation", sort="downloads", limit=50):
            if m.modelId and model_name.lower() in m.modelId.lower() and "openvino" in m.modelId.lower():
                return m.modelId
    except Exception:
        pass
    return None


def _download_ov_model(repo_id: str, dest: Path):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(dest), local_dir_use_symlinks=False)
    except Exception as e:
        pass


def _convert_hf_to_openvino(hf_repo: str, dest: Path):
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer
        model = OVModelForCausalLM.from_pretrained(hf_repo, export=True, load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        model.save_pretrained(str(dest))
        tokenizer.save_pretrained(str(dest))
        model._save_config(str(dest))
    except Exception as e:
        pass


def run_npu_benchmark(
    ov_model_path: str,
    backend: str,
    task_id: str,
    bench_params: Optional[Dict] = None,
) -> Dict:
    bp = bench_params or {}
    n_prompt = bp.get("n_prompt", 512)
    n_gen = bp.get("n_gen", 128)
    n_ctx = bp.get("n_ctx", 2048)
    n_batch = bp.get("n_batch", 512)
    repetitions = bp.get("repetitions", 3)

    try:
        from openvino_genai import LLMPipeline, GenerationConfig

        start = time.time()
        pipe = LLMPipeline(ov_model_path, "NPU", {"MAX_PROMPT_LEN": n_ctx})

        prompt_text = "The meaning of life is" * (n_prompt // 5 + 1)

        prompt_timings = []
        gen_timings = []
        prompt_tps_list = []
        gen_tps_list = []

        for rep in range(repetitions):
            pipe.start_chat()
            p_start = time.time()
            _ = pipe.generate(prompt_text, max_new_tokens=1, do_sample=False)
            p_elapsed = time.time() - p_start
            prompt_tps_list.append(n_prompt / p_elapsed if p_elapsed > 0 else 0)
            prompt_timings.append(p_elapsed)

            gen_config = GenerationConfig(max_new_tokens=n_gen, do_sample=False)
            g_start = time.time()
            output = pipe.generate(prompt_text, gen_config)
            g_elapsed = time.time() - g_start
            gen_tps_list.append(n_gen / g_elapsed if g_elapsed > 0 else 0)
            gen_timings.append(g_elapsed)
            pipe.finish_chat()

        elapsed = time.time() - start

        avg_prompt_tps = sum(prompt_tps_list) / len(prompt_tps_list) if prompt_tps_list else 0
        avg_gen_tps = sum(gen_tps_list) / len(gen_tps_list) if gen_tps_list else 0

        return {
            "backend": backend,
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "npu": True,
            "results": [{
                "tokens_per_sec": {
                    "prompt_tps": round(avg_prompt_tps, 2),
                    "generation_tps": round(avg_gen_tps, 2),
                },
                "timing_ms": {
                    "prompt_tps": round(avg_prompt_tps, 2),
                    "generation_tps": round(avg_gen_tps, 2),
                },
            }],
        }
    except Exception as e:
        return {
            "backend": backend,
            "status": "error",
            "error": str(e),
        }

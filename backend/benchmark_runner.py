import subprocess
import json
import os
import time
import uuid
import re
import signal
from datetime import datetime
from typing import List, Dict, Optional
from config import RESULTS_DIR
from driver_manager import get_local_drivers
from npu_runner import check_npu_available, ensure_ov_model, run_npu_benchmark

_running: Dict[str, subprocess.Popen] = {}


def cancel_benchmark(task_id: str) -> bool:
    proc = _running.pop(task_id, None)
    if proc is not None:
        try:
            proc.terminate()
            return True
        except Exception:
            pass
    return False


def list_active() -> List[Dict]:
    now = time.time()
    dead = [k for k, v in list(_running.items()) if v is None or v.poll() is not None]
    for k in dead:
        _running.pop(k, None)
    return [{"id": k} for k in _running]


def run_benchmark(
    model_path: str,
    backends: List[str],
    task_id: str,
    model_params: Optional[Dict] = None,
    bench_params: Optional[Dict] = None,
) -> Dict:
    _running[task_id] = None  # mark as starting
    bench_params = bench_params or {}
    try:
        result = {
            "id": task_id,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "backends": backends,
            "model_params": model_params or {},
            "bench_params": bench_params,
            "runs": [],
        }

        for backend in backends:
            if task_id not in _running:
                break
            run_results = _run_single(model_path, backend, task_id, bench_params)
            if isinstance(run_results, list):
                result["runs"].extend(run_results)
            else:
                result["runs"].append(run_results)
    finally:
        _running.pop(task_id, None)

    if result["runs"]:
        result_file = RESULTS_DIR / f"{task_id}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    return result


def _find_driver_for_backend(backend: str) -> Optional[Dict]:
    for d in get_local_drivers():
        if d["backend"] == backend and d.get("llama_bench_path"):
            driver_dir = os.path.dirname(d["llama_bench_path"])
            bench_path = d["llama_bench_path"]
            cli_path = os.path.join(driver_dir, "llama-cli.exe")
            return {
                "bench_path": bench_path,
                "cli_path": cli_path if os.path.exists(cli_path) else None,
            }
    return None


def _find_mtp_head(model_path: str) -> Optional[str]:
    model_dir = os.path.dirname(model_path)
    mtp_dir = os.path.join(model_dir, "MTP")
    if os.path.isdir(mtp_dir):
        heads = [
            os.path.join(mtp_dir, f)
            for f in os.listdir(mtp_dir)
            if f.endswith(".gguf")
        ]
        if heads:
            heads.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
            return heads[0]
    return None


def _run_single(model_path: str, backend: str, task_id: str, bench_params: Dict) -> list:
    if backend == "npu":
        if not check_npu_available():
            return [{
                "backend": "npu",
                "status": "error",
                "error": "NPU device not available via OpenVINO. Check that openvino-genai is installed and NPU driver is up to date.",
            }]
        ov_path = ensure_ov_model(model_path, bench_params)
        if not ov_path:
            from pathlib import Path
            p = Path(model_path)
            is_ov_ir = any(p.glob("openvino_*.xml")) or any(p.parent.glob("openvino_*.xml"))
            if is_ov_ir:
                extra = " Could not load OpenVINO IR model on NPU device."
            elif p.suffix.lower() == ".gguf":
                gguf_size = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
                if gguf_size > 4096:
                    extra = f" Model file is too large ({gguf_size:.0f} MB) — NPU has limited memory."
                else:
                    extra = " Could not convert GGUF to OpenVINO IR. Download an NPU-compatible model from the Models page (select 'OpenVINO' tab) instead."
            else:
                extra = " Unsupported model format for NPU."
            return [{
                "backend": "npu",
                "status": "error",
                "error": "Cannot run on NPU." + extra,
            }]
        _running[task_id] = None
        result = run_npu_benchmark(ov_path, backend, task_id, bench_params)
        return [result]

    driver = _find_driver_for_backend(backend)
    if not driver:
        return [{
            "backend": backend,
            "status": "error",
            "error": f"No downloaded driver found with backend '{backend}' and llama-bench.exe.",
        }]

    mtp_head = _find_mtp_head(model_path)
    can_mtp = mtp_head is not None and driver["cli_path"] is not None
    mtp_mode = bench_params.get("mtp_mode", "auto")

    if mtp_mode == "both":
        results = []
        if can_mtp:
            results.append(_run_with_cli(driver["cli_path"], model_path, mtp_head, backend, task_id, bench_params))
        else:
            results.append({
                "backend": backend,
                "status": "error",
                "error": "MTP head not found or no llama-cli.exe — cannot run MTP benchmark",
            })
        results.append(_run_with_bench(driver["bench_path"], model_path, backend, task_id, bench_params))
        return results

    if mtp_mode == "force":
        if not can_mtp:
            return [{
                "backend": backend,
                "status": "error",
                "error": "MTP head not found or no llama-cli.exe — cannot force MTP mode",
            }]
        return [_run_with_cli(driver["cli_path"], model_path, mtp_head, backend, task_id, bench_params)]

    if mtp_mode == "off":
        return [_run_with_bench(driver["bench_path"], model_path, backend, task_id, bench_params)]

    # auto (default)
    if can_mtp:
        return [_run_with_cli(driver["cli_path"], model_path, mtp_head, backend, task_id, bench_params)]
    return [_run_with_bench(driver["bench_path"], model_path, backend, task_id, bench_params)]


def _run_with_bench(bench_path: str, model_path: str, backend: str, task_id: str, bench_params: Optional[Dict] = None) -> Dict:
    cmd = [bench_path, "-m", model_path, "-o", "json", "--no-warmup"]

    if bench_params:
        if "n_prompt" in bench_params:
            cmd.extend(["-p", str(bench_params["n_prompt"])])
        if "n_gen" in bench_params:
            cmd.extend(["-n", str(bench_params["n_gen"])])
        if "n_batch" in bench_params:
            cmd.extend(["-b", str(bench_params["n_batch"])])
        if "n_ubatch" in bench_params:
            cmd.extend(["-ub", str(bench_params["n_ubatch"])])
        if "n_threads" in bench_params:
            cmd.extend(["-t", str(bench_params["n_threads"])])
        if "repetitions" in bench_params:
            cmd.extend(["-r", str(bench_params["repetitions"])])

    extra_args = bench_params.get("extra_args", "") if bench_params else ""
    if extra_args:
        cmd.extend(extra_args.split())

    try:
        start = time.time()
        env = os.environ.copy()
        env["GGML_CUDA_ENABLE"] = "1"
        env["GGML_VULKAN_ENABLE"] = "1"

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        _running[task_id] = proc
        stdout, stderr = proc.communicate(timeout=120)
        elapsed = time.time() - start

        if proc.returncode != 0:
            return {
                "backend": backend,
                "status": "error",
                "error": stderr.strip() or f"exit code {proc.returncode}",
                "elapsed_sec": round(elapsed, 2),
            }

        parsed = _parse_bench_json(stdout)
        return {
            "backend": backend,
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "results": parsed,
        }
    except subprocess.TimeoutExpired:
        return {"backend": backend, "status": "error", "error": "timeout (120s)"}
    except Exception as e:
        return {"backend": backend, "status": "error", "error": str(e)}


def _run_with_cli(cli_path: str, model_path: str, mtp_head: str, backend: str, task_id: str, bench_params: Optional[Dict] = None) -> Dict:
    n_ctx = bench_params.get("n_ctx", 2048) if bench_params else 2048
    n_prompt = bench_params.get("n_prompt", 512) if bench_params else 512
    n_gen = bench_params.get("n_gen", 128) if bench_params else 128
    n_batch = bench_params.get("n_batch", 512) if bench_params else 512

    cmd = [
        cli_path,
        "-m", model_path,
        "--spec-draft-model", mtp_head,
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3",
        "--single-turn",
        "-p", "Hello",
        "-ngl", "999",
        "-c", str(n_ctx),
        "-n", str(n_gen),
        "-b", str(n_batch),
    ]

    try:
        start = time.time()
        env = os.environ.copy()
        env["GGML_CUDA_ENABLE"] = "1"
        env["GGML_VULKAN_ENABLE"] = "1"

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        _running[task_id] = proc
        stdout, stderr = proc.communicate(timeout=120)
        elapsed = time.time() - start

        if proc.returncode != 0:
            return {
                "backend": backend,
                "status": "error",
                "error": stderr.strip() or f"exit code {proc.returncode}",
                "elapsed_sec": round(elapsed, 2),
            }

        timing = _parse_cli_timing(stdout)
        tokens_per_sec = None
        if timing:
            tokens_per_sec = timing

        return {
            "backend": backend,
            "status": "success",
            "elapsed_sec": round(elapsed, 2),
            "mtp": True,
            "draft_model": mtp_head,
            "results": [{
                "tokens_per_sec": tokens_per_sec,
                "timing_ms": timing,
            }] if tokens_per_sec else [],
        }
    except subprocess.TimeoutExpired:
        return {"backend": backend, "status": "error", "error": "timeout (120s)"}
    except Exception as e:
        return {"backend": backend, "status": "error", "error": str(e)}


_BENCH_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_bench_json(stdout: str) -> List[Dict]:
    try:
        data = json.loads(stdout)
        results = data.get("results", []) if isinstance(data, dict) else data
        return results
    except json.JSONDecodeError:
        m = _BENCH_JSON_RE.search(stdout)
        if m:
            try:
                data = json.loads(m.group())
                return data.get("results", []) if isinstance(data, dict) else data
            except json.JSONDecodeError:
                pass
        return [{"raw": stdout}]


def _parse_cli_timing(stderr: str) -> Dict:
    timing = {}
    m = re.search(r"\[ Prompt:\s+([\d.]+)\s+t/s\s+\|\s+Generation:\s+([\d.]+)\s+t/s\s+\]", stderr)
    if m:
        timing["prompt_tps"] = float(m.group(1))
        timing["generation_tps"] = float(m.group(2))
    return timing


def list_results() -> List[Dict]:
    results = []
    for f in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if f.endswith(".json"):
            with open(RESULTS_DIR / f) as fh:
                data = json.load(fh)
            results.append({
                "id": data["id"],
                "timestamp": data["timestamp"],
                "model_name": data["model_name"],
                "backends": data["backends"],
                "n_runs": len(data.get("runs", [])),
            })
    return results


def get_result(result_id: str) -> Optional[Dict]:
    result_file = RESULTS_DIR / f"{result_id}.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def delete_result(result_id: str):
    result_file = RESULTS_DIR / f"{result_id}.json"
    if result_file.exists():
        result_file.unlink()

import psutil
import subprocess
import time
import re
import threading
from gpu_detector import detect_gpus

_known_gpus = None
_known_gpus_lock = threading.Lock()
_cache = None
_cache_lock = threading.Lock()
_cache_ready = threading.Event()


def _ensure_gpus():
    global _known_gpus
    if _known_gpus is None:
        with _known_gpus_lock:
            if _known_gpus is None:
                _known_gpus = detect_gpus()


def _worker():
    global _cache
    _ensure_gpus()
    while True:
        stats = _do_collect()
        with _cache_lock:
            _cache = stats
        _cache_ready.set()
        time.sleep(5)


_cache_thread = threading.Thread(target=_worker, daemon=True)
_cache_thread.start()


def collect() -> dict:
    with _cache_lock:
        cached = dict(_cache) if _cache else None
    result = {
        "cpu": _collect_cpu(),
        "ram": _collect_ram(),
        "gpus": cached["gpus"] if cached else _fallback_gpus(),
        "timestamp": int(time.time() * 1000),
    }
    return result


def _fallback_gpus() -> list:
    _ensure_gpus()
    return [
        {
            "name": g["name"],
            "backend": g.get("backend", ""),
            "utilization_percent": 0,
            "memory_used_mb": 0,
            "memory_total_mb": g.get("memory_mb", 0) + g.get("shared_memory_mb", 0),
            "temperature_c": 0,
        }
        for g in _known_gpus
    ]


def _do_collect() -> dict:
    _ensure_gpus()
    cpu = _collect_cpu()
    ram = _collect_ram()
    gpus = _collect_gpu_stats()
    return {
        "cpu": cpu,
        "ram": ram,
        "gpus": gpus,
        "timestamp": int(time.time() * 1000),
    }


def _collect_cpu() -> dict:
    return {
        "percent": psutil.cpu_percent(interval=0),
        "count": psutil.cpu_count(),
    }


def _collect_ram() -> dict:
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 1),
        "used_gb": round(mem.used / (1024 ** 3), 1),
        "percent": mem.percent,
    }


def _collect_gpu_stats() -> list:
    nv_result = {}
    intel_3d = 0
    intel_compute = 0
    npu_3d = 0
    npu_compute = 0
    intel_dedicated = 0
    intel_shared = 0

    def run_nvidia():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8
            )
            for line in out.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    nv_result[parts[1]] = {
                        "util_percent": int(parts[2]),
                        "mem_used_mb": int(parts[3]),
                        "mem_total_mb": int(parts[4]),
                        "temp_c": int(parts[5]),
                    }
        except Exception:
            pass

    def run_powershell():
        nonlocal intel_3d, intel_compute, npu_3d, npu_compute, intel_dedicated, intel_shared
        ps = (
            "$e = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples;"
            "$m = (Get-Counter '\\GPU Adapter Memory(*)\\*' -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples;"
            "Write-Output '___ENGINES___';"
            "if ($e) { $e | ForEach-Object { $_.InstanceName + '|' + [math]::Round($_.CookedValue, 1) } };"
            "Write-Output '___MEMORY___';"
            "if ($m) { $m | ForEach-Object { $_.InstanceName + '|' + $_.Path.Split('\\')[-1] + '|' + [math]::Round($_.CookedValue) } }"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=15
            )
            lines = out.stdout.strip().split("\n")
            section = None
            eng_lines = []
            mem_lines = []
            for line in lines:
                line = line.strip()
                if line == "___ENGINES___": section = "eng"; continue
                if line == "___MEMORY___": section = "mem"; continue
                if section == "eng" and line: eng_lines.append(line)
                if section == "mem" and line: mem_lines.append(line)

            for line in eng_lines:
                if "|" not in line:
                    continue
                inst, val_str = line.split("|", 1)
                val = float(val_str)
                is_intel = "0x00000002" in inst
                is_npu = "0x000142ad" in inst
                if "engtype_3d" in inst:
                    if is_intel: intel_3d += val
                    if is_npu: npu_3d += val
                if "engtype_compute" in inst:
                    if is_intel: intel_compute += val
                    if is_npu: npu_compute += val

            for line in mem_lines:
                if "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                inst = parts[0]
                metric = parts[1].lower().replace(" ", "_")
                raw_val = int(float(parts[2]))
                val_mb = raw_val // (1024 * 1024)
                if "0x00000002" in inst:
                    if "dedicated" in metric: intel_dedicated += val_mb
                    if "shared" in metric: intel_shared += val_mb
        except Exception:
            pass

    t1 = threading.Thread(target=run_nvidia, daemon=True)
    t2 = threading.Thread(target=run_powershell, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    result = []
    for gpu in _known_gpus:
        name = gpu["name"]
        backend = gpu.get("backend", "")
        entry = {
            "name": name,
            "backend": backend,
            "utilization_percent": 0,
            "memory_used_mb": 0,
            "memory_total_mb": gpu.get("memory_mb", 0),
            "temperature_c": 0,
        }

        if "nvidia" in name.lower():
            nv = nv_result.get(name, {})
            entry["utilization_percent"] = nv.get("util_percent", 0)
            entry["memory_used_mb"] = nv.get("mem_used_mb", 0)
            entry["memory_total_mb"] = nv.get("mem_total_mb", gpu.get("memory_mb", 0))
            entry["temperature_c"] = nv.get("temp_c", 0)
        elif backend == "openvino":
            entry["utilization_percent"] = round(npu_3d + npu_compute, 1)
        else:
            entry["utilization_percent"] = round(intel_3d + intel_compute, 1)
            dedicated = gpu.get("memory_mb", 0)
            shared = gpu.get("shared_memory_mb", 0)
            entry["memory_total_mb"] = dedicated + shared
            used = intel_dedicated + intel_shared
            entry["memory_used_mb"] = min(used, entry["memory_total_mb"]) if entry["memory_total_mb"] > 0 else used

        result.append(entry)

    return result

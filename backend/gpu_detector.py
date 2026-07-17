import subprocess
import json
import re
import shutil
import sys
from typing import List, Dict


def detect_gpus() -> List[Dict]:
    seen: set = set()
    gpus = []

    def add(gpu: Dict):
        key = (gpu["name"].lower(), gpu["backend"])
        if key not in seen:
            seen.add(key)
            gpus.append(gpu)

    for gpu in _detect_nvidia_gpus():
        add(gpu)
    for gpu in _detect_macos_metal_gpus():
        add(gpu)
    for gpu in _detect_intel_gpus():
        native = gpu["backend"]
        if native == "sycl":
            # Intel Arc/Xe supports SYCL, OpenVINO, and Vulkan — emit all three
            base = {k: v for k, v in gpu.items() if k != "backend"}
            for b in ["sycl", "openvino", "vulkan"]:
                add({**base, "backend": b})
        else:
            add(gpu)
    for gpu in _detect_vulkan_gpus():
        # Vulkan entries that aren't already covered (e.g. non-Intel/non-NVIDIA GPUs)
        add(gpu)
    for gpu in _detect_npu():
        add(gpu)
    if not gpus:
        for gpu in _detect_wmi_gpus():
            add(gpu)
    return gpus


def _detect_nvidia_gpus() -> List[Dict]:
    result = []
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return result
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30
        )
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                base = {
                    "vendor": "NVIDIA",
                    "index": parts[0],
                    "name": parts[1],
                    "memory_mb": int(float(parts[2])) if parts[2].replace('.', '').isdigit() else 0,
                    "driver_version": parts[3] if len(parts) > 3 else "",
                }
                result.append({**base, "backend": "cuda"})
                result.append({**base, "backend": "vulkan"})
    except Exception:
        pass
    return result


def _detect_macos_metal_gpus() -> List[Dict]:
    result = []
    if sys.platform != "darwin":
        return result
    try:
        unified_memory_mb = _get_macos_unified_memory_mb()
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=30
        )
        if out.returncode != 0:
            return result

        current = {}
        for raw_line in out.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Chipset Model:"):
                if current:
                    result.append(current)
                name = line.split(":", 1)[1].strip()
                current = {
                    "vendor": "Apple" if "apple" in name.lower() else "Unknown",
                    "index": str(len(result)),
                    "name": name,
                    "memory_mb": unified_memory_mb,
                    "driver_version": "",
                    "backend": "metal",
                }
            elif current and line.startswith("Metal Support:"):
                current["driver_version"] = line.split(":", 1)[1].strip()
            elif current and line.startswith("Total Number of Cores:"):
                cores = line.split(":", 1)[1].strip()
                current["core_count"] = int(cores) if cores.isdigit() else cores

        if current:
            result.append(current)
    except Exception:
        pass
    return result


def _get_macos_unified_memory_mb() -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=10
        )
        return int(out.stdout.strip()) // (1024 * 1024)
    except Exception:
        return 0


def _detect_intel_gpus() -> List[Dict]:
    result = []
    try:
        ps = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"
        out = subprocess.run(
            ["powershell", "-Command", ps],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(out.stdout)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            name = entry.get("Name", "")
            if not name:
                continue
            is_intel = "intel" in name.lower()
            if is_intel:
                ram_bytes = entry.get("AdapterRAM") or 0
                dedicated_mb = ram_bytes // (1024 * 1024) if ram_bytes else 0
                shared_mb = _get_intel_shared_memory_mb()
                result.append({
                    "vendor": "Intel",
                    "index": str(len(result)),
                    "name": name,
                    "memory_mb": dedicated_mb,
                    "shared_memory_mb": shared_mb,
                    "driver_version": entry.get("DriverVersion", ""),
                    "backend": "sycl" if "arc" in name.lower() or "xe" in name.lower() else "cpu",
                })
    except Exception:
        pass
    return result


def _get_intel_shared_memory_mb() -> int:
    try:
        ps = "(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize"
        out = subprocess.run(
            ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=10
        )
        total_kb = int(out.stdout.strip())
        total_gb = total_kb / (1024 * 1024)
        return int(min(total_gb / 2, 32) * 1024)
    except Exception:
        return 0


def _detect_vulkan_gpus() -> List[Dict]:
    result = []
    try:
        vkinfo = shutil.which("vulkaninfo")
        if not vkinfo:
            return result
        out = subprocess.run(
            [vkinfo, "--summary"],
            capture_output=True, text=True, timeout=30
        )
        lines = out.stdout.splitlines()
        i = 0
        while i < len(lines):
            m = re.match(r'GPU(\d+):', lines[i])
            if m:
                idx = m.group(1)
                name = ""
                i += 1
                while i < len(lines) and lines[i].startswith("\t"):
                    if "deviceName" in lines[i]:
                        name = lines[i].split("=", 1)[-1].strip()
                        break
                    i += 1
                if not name:
                    i += 1
                    continue
                lower = name.lower()
                if "intel" in lower:
                    vendor = "Intel"
                elif "nvidia" in lower:
                    vendor = "NVIDIA"
                elif "amd" in lower or "advanced micro devices" in lower:
                    vendor = "AMD"
                else:
                    vendor = "Unknown"
                result.append({
                    "vendor": vendor,
                    "index": idx,
                    "name": name,
                    "memory_mb": 0,
                    "driver_version": "",
                    "backend": "vulkan",
                })
            i += 1
    except Exception:
        pass
    return result


def _detect_npu() -> List[Dict]:
    result = []
    try:
        ps_cmd = "Get-PnpDevice | Where-Object { $_.Class -eq 'ComputeAccelerator' -and $_.Status -eq 'OK' } | Select-Object FriendlyName, Class, Status | ConvertTo-Json"
        out = subprocess.run(
            ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(out.stdout)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            name = entry.get("FriendlyName", "")
            if not name:
                continue
            result.append({
                "vendor": "Intel" if "intel" in name.lower() else "Unknown",
                "index": str(len(result)),
                "name": name,
                "memory_mb": 0,
                "driver_version": "",
                "backend": "npu",
            })
    except Exception:
        pass
    return result


def _detect_wmi_gpus() -> List[Dict]:
    result = []
    try:
        out = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor | ConvertTo-Json"],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(out.stdout)
        entries = data if isinstance(data, list) else [data]
        for i, entry in enumerate(entries):
            name = entry.get("Name", "")
            if not name:
                continue
            ram_bytes = entry.get("AdapterRAM") or 0
            memory_mb = ram_bytes // (1024 * 1024) if ram_bytes else 0
            result.append({
                "vendor": "Unknown",
                "index": str(i),
                "name": name,
                "memory_mb": memory_mb,
                "driver_version": entry.get("DriverVersion", ""),
                "backend": "cpu",
            })
    except Exception:
        pass
    return result


_KNOWN_LATEST_DRIVERS = {
    "Intel Arc": {
        "version": "32.0.101.8860",
        "download_url": "https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html",
        "label": "Intel Arc Graphics Driver",
    },
    "Intel NPU": {
        "version": "32.0.100.4778",
        "download_url": "https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html",
        "label": "Intel NPU Driver",
    },
}


def _parse_intel_driver_version(v: str) -> tuple:
    parts = v.strip().split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 4:
        nums.append(0)
    return tuple(nums[:4])


def check_driver_updates(gpus: List[Dict]) -> List[Dict]:
    updates = []
    for gpu in gpus:
        name = gpu.get("name", "")
        vendor = gpu.get("vendor", "")
        driver = gpu.get("driver_version", "")
        if vendor != "Intel" or not driver:
            continue
        key = None
        if "Arc" in name or "Xe" in name:
            key = "Intel Arc"
        elif "AI Boost" in name or "NPU" in name:
            key = "Intel NPU"
        if not key or key not in _KNOWN_LATEST_DRIVERS:
            continue
        latest = _KNOWN_LATEST_DRIVERS[key]
        installed_ver = _parse_intel_driver_version(driver)
        latest_ver = _parse_intel_driver_version(latest["version"])
        if installed_ver < latest_ver:
            updates.append({
                "device_name": name,
                "installed": driver,
                "latest": latest["version"],
                "label": latest["label"],
                "download_url": latest["download_url"],
            })
    return updates

"""Conservative local-model admission control.

The gateway uses this module before starting a native inference subprocess.
It intentionally favors refusing or delaying work over allowing unified-memory
pressure to make the host unresponsive.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


DEFAULT_RESERVE_BYTES = 3 * 1024**3
DEFAULT_MAX_WAIT_SECONDS = 300
DEFAULT_POLL_SECONDS = 1.0


@dataclass
class Reservation:
    manager: "ResourceManager"
    model_id: str
    estimate_bytes: int
    reservation_id: str
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.manager.release(self.reservation_id)


class ResourcePressureError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503, retry_after: int = 5, snapshot: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.snapshot = snapshot or {}


class ResourceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reservations: Dict[str, Dict[str, Any]] = {}
        self._sequence = 0
        self._last_pressure = False
        self._events: list[Dict[str, Any]] = []

    @property
    def reserve_bytes(self) -> int:
        configured = os.environ.get("AVALON_MEMORY_RESERVE_MB")
        try:
            return max(512, int(configured)) * 1024**2 if configured else DEFAULT_RESERVE_BYTES
        except ValueError:
            return DEFAULT_RESERVE_BYTES

    def snapshot(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        with self._lock:
            reserved = sum(item["estimate_bytes"] for item in self._reservations.values())
            active = len(self._reservations)
            pressure = vm.available <= self.reserve_bytes or reserved + self.reserve_bytes > vm.available
            if pressure != self._last_pressure:
                self._last_pressure = pressure
                self._record_event("pressure_started" if pressure else "pressure_cleared", {
                    "available_bytes": vm.available,
                    "reserved_bytes": reserved,
                })
            return {
                "pressure": pressure,
                "available_bytes": vm.available,
                "available_mb": round(vm.available / 1024**2),
                "total_bytes": vm.total,
                "total_mb": round(vm.total / 1024**2),
                "safety_reserve_bytes": self.reserve_bytes,
                "safety_reserve_mb": round(self.reserve_bytes / 1024**2),
                "reserved_bytes": reserved,
                "reserved_mb": round(reserved / 1024**2),
                "active_reservations": active,
                "events": list(self._events[-20:]),
            }

    def estimate_model(self, model_path: str, max_tokens: int = 512, mtp_head_path: str = "") -> Dict[str, Any]:
        path = Path(model_path)
        if path.is_dir():
            weight_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            runtime = "openvino"
        else:
            weight_bytes = path.stat().st_size if path.exists() else 0
            runtime = "gguf"
        head_bytes = 0
        if mtp_head_path and Path(mtp_head_path).exists():
            head_bytes = Path(mtp_head_path).stat().st_size
        # File size is a lower bound: native runtimes add tensors, allocator
        # overhead, and KV cache. These multipliers are deliberately conservative.
        multiplier = 1.55 if runtime == "openvino" else 1.35
        runtime_bytes = int((weight_bytes + head_bytes) * multiplier)
        kv_cache_bytes = max(256 * 1024**2, min(2 * 1024**3, max(1, max_tokens) * 1024**2))
        estimate = runtime_bytes + kv_cache_bytes
        return {
            "runtime": runtime,
            "weight_bytes": weight_bytes,
            "weight_mb": round(weight_bytes / 1024**2),
            "mtp_head_bytes": head_bytes,
            "runtime_overhead_bytes": runtime_bytes - weight_bytes - head_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "estimate_bytes": estimate,
            "estimate_mb": round(estimate / 1024**2),
        }

    def acquire(self, model_id: str, model_path: str, max_tokens: int = 512, mtp_head_path: str = "") -> Reservation:
        estimate = self.estimate_model(model_path, max_tokens, mtp_head_path)
        wait_seconds = int(os.environ.get("AVALON_RESOURCE_MAX_WAIT_SECONDS", DEFAULT_MAX_WAIT_SECONDS))
        deadline = time.monotonic() + max(0, wait_seconds)
        queued = False
        while True:
            state = self.snapshot()
            available_after_reserve = state["available_bytes"] - state["reserved_bytes"] - self.reserve_bytes
            if not state["pressure"] and estimate["estimate_bytes"] <= available_after_reserve:
                with self._lock:
                    self._sequence += 1
                    reservation_id = f"reservation-{self._sequence}"
                    self._reservations[reservation_id] = {
                        "model_id": model_id,
                        **estimate,
                        "started_at": int(time.time() * 1000),
                    }
                    if queued:
                        self._record_event("request_admitted", {
                            "model_id": model_id,
                            "estimate_mb": estimate["estimate_mb"],
                        })
                    return Reservation(self, model_id, estimate["estimate_bytes"], reservation_id)
            if not queued:
                queued = True
                self._record_event("request_delayed", {
                    "model_id": model_id,
                    "estimate_mb": estimate["estimate_mb"],
                    "available_mb": state["available_mb"],
                })
            if time.monotonic() >= deadline:
                self._record_event("request_rejected", {"model_id": model_id, **estimate})
                raise ResourcePressureError(
                    f"Model request delayed: estimated {estimate['estimate_mb']} MB is not safe with "
                    f"{state['available_mb']} MB available and a {state['safety_reserve_mb']} MB safety reserve.",
                    status_code=503,
                    retry_after=5,
                    snapshot={**state, **estimate},
                )
            time.sleep(DEFAULT_POLL_SECONDS)

    def release(self, reservation_id: str) -> None:
        with self._lock:
            item = self._reservations.pop(reservation_id, None)
            if item:
                self._record_event("request_released", {"model_id": item["model_id"]})

    def _record_event(self, event: str, details: Dict[str, Any]) -> None:
        self._events.append({"event": event, "at_ms": int(time.time() * 1000), **details})
        del self._events[:-20]


resource_manager = ResourceManager()

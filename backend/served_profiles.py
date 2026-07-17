import json
import re
import threading
from pathlib import Path
from typing import Any, Dict

from config import DATA_DIR


STORE_PATH = DATA_DIR / "served_profiles.json"
_lock = threading.Lock()


def _ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        STORE_PATH.write_text("{}")


def _read() -> Dict[str, Dict[str, Any]]:
    _ensure_store()
    try:
        data = json.loads(STORE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(data: Dict[str, Dict[str, Any]]) -> None:
    _ensure_store()
    tmp = STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(STORE_PATH)


def _clean_served_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        raise ValueError("served_id cannot be empty")
    return cleaned


def _clean_bool(value: Any) -> bool:
    return bool(value)


def _clean_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if parsed < 0:
        raise ValueError("numeric values must be non-negative")
    return parsed


def _clean_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError("numeric values must be non-negative")
    return parsed


def sanitize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    if "served_id" in profile:
        cleaned["served_id"] = _clean_served_id(str(profile["served_id"]))
    if "driver" in profile:
        cleaned["driver"] = str(profile["driver"]).strip()
    if "model_file" in profile:
        cleaned["model_file"] = str(profile["model_file"]).strip()
    if "mtp" in profile:
        cleaned["mtp"] = _clean_bool(profile["mtp"])
    if "dspark" in profile:
        cleaned["dspark"] = _clean_bool(profile["dspark"])

    serving_config = profile.get("serving_config")
    if isinstance(serving_config, dict):
        config: Dict[str, Any] = {}
        if "max_tokens_default" in serving_config:
            config["max_tokens_default"] = _clean_int(serving_config["max_tokens_default"], 512)
        if "temperature_default" in serving_config:
            config["temperature_default"] = _clean_float(serving_config["temperature_default"], 0.7)
        if config:
            cleaned["serving_config"] = config
    return cleaned


def list_overrides() -> Dict[str, Dict[str, Any]]:
    with _lock:
        return _read()


def get_override(base_id: str) -> Dict[str, Any]:
    return list_overrides().get(base_id, {})


def upsert_override(base_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if not base_id:
        raise ValueError("base_id is required")
    cleaned = sanitize_profile(profile)
    with _lock:
        data = _read()
        current = data.get(base_id, {})
        current.update(cleaned)
        data[base_id] = current
        _write(data)
        return current


def reset_override(base_id: str) -> bool:
    if not base_id:
        raise ValueError("base_id is required")
    with _lock:
        data = _read()
        existed = base_id in data
        data.pop(base_id, None)
        _write(data)
        return existed

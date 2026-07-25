import json
import uuid
from typing import Dict, Optional
from config import DOWNLOADS_DIR

def _path(download_id: str):
    return DOWNLOADS_DIR / f"{download_id}.json"

def create_download(**metadata) -> str:
    download_id = str(uuid.uuid4())[:8]
    update(download_id, status="starting", percent=0, stage="", **metadata)
    return download_id

def update(download_id: str, **kwargs):
    path = _path(download_id)
    value = get(download_id) or {"id": download_id}
    value.update(kwargs)
    temporary = path.with_suffix(".json.part")
    with temporary.open("w") as fh:
        json.dump(value, fh)
    temporary.replace(path)

def get(download_id: str) -> Optional[dict]:
    try:
        with _path(download_id).open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

def remove(download_id: str):
    _path(download_id).unlink(missing_ok=True)

def list_active() -> list:
    active = []
    for path in DOWNLOADS_DIR.glob("*.json"):
        value = get(path.stem)
        if value and value.get("status") not in {"done", "error"}:
            active.append(value)
    return active

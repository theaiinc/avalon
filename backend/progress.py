import uuid
from typing import Dict, Optional

_progress: Dict[str, dict] = {}

def create_download() -> str:
    download_id = str(uuid.uuid4())[:8]
    _progress[download_id] = {"status": "starting", "percent": 0, "stage": ""}
    return download_id

def update(download_id: str, **kwargs):
    if download_id in _progress:
        _progress[download_id].update(kwargs)

def get(download_id: str) -> Optional[dict]:
    return _progress.get(download_id)

def remove(download_id: str):
    _progress.pop(download_id, None)

def list_active() -> list:
    return [{"id": k, **v} for k, v in _progress.items()]

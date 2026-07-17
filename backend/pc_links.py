import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from config import DATA_DIR
from pairing import peer_credential


LINKS_FILE = DATA_DIR / "pc_links.json"


def _ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LINKS_FILE.exists():
        LINKS_FILE.write_text("[]")


def _read_links() -> List[Dict[str, Any]]:
    _ensure_store()
    try:
        data = json.loads(LINKS_FILE.read_text())
    except json.JSONDecodeError:
        data = []
    return data if isinstance(data, list) else []


def _write_links(links: List[Dict[str, Any]]) -> None:
    _ensure_store()
    LINKS_FILE.write_text(json.dumps(links, indent=2))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "pc"


def normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"http://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url must be an http(s) URL")
    return url.rstrip("/")


def list_links() -> List[Dict[str, Any]]:
    return _read_links()


def get_link(pc_id: str) -> Optional[Dict[str, Any]]:
    return next((link for link in _read_links() if link["id"] == pc_id), None)


def upsert_link(name: str, base_url: str, pc_id: str = "", peer_id: str = "") -> Dict[str, Any]:
    normalized = normalize_base_url(base_url)
    now = int(time.time())
    links = _read_links()
    link_id = _slugify(pc_id or name or urlparse(normalized).netloc)
    link = {
        "id": link_id,
        "name": name.strip() or link_id,
        "base_url": normalized,
        "updated_at": now,
    }
    if peer_id:
        link["peer_id"] = peer_id

    existing = next((i for i, item in enumerate(links) if item["id"] == link_id), None)
    if existing is None:
        link["created_at"] = now
        links.append(link)
    else:
        link["created_at"] = links[existing].get("created_at", now)
        links[existing] = link

    _write_links(links)
    return link


def remove_link(pc_id: str) -> bool:
    links = _read_links()
    kept = [link for link in links if link["id"] != pc_id]
    _write_links(kept)
    return len(kept) != len(links)


def fetch_remote_models(link: Dict[str, Any], timeout: float = 10.0) -> List[Dict[str, Any]]:
    headers = {}
    if link.get("peer_id"):
        credential = peer_credential(link["peer_id"])
        if credential:
            headers["X-API-Key"] = credential
    with httpx.Client(timeout=timeout, headers=headers) as client:
        response = client.get(f"{link['base_url']}/v1/models")
        response.raise_for_status()
        data = response.json()
    models = data.get("data", []) if isinstance(data, dict) else []
    return models if isinstance(models, list) else []


def test_link(base_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    normalized = normalize_base_url(base_url)
    try:
        models = fetch_remote_models({"base_url": normalized}, timeout=timeout)
        return {"ok": True, "base_url": normalized, "model_count": len(models), "models": models}
    except Exception as e:
        return {"ok": False, "base_url": normalized, "error": str(e)}


def list_remote_models() -> List[Dict[str, Any]]:
    remote_models = []
    for link in _read_links():
        try:
            for model in fetch_remote_models(link):
                remote_id = str(model.get("id", ""))
                if not remote_id:
                    continue
                remote_models.append({
                    **model,
                    "id": remote_model_id(link["id"], remote_id),
                    "remote_model_id": remote_id,
                    "pc_id": link["id"],
                    "pc_name": link["name"],
                    "owned_by": f"pc:{link['id']}",
                })
        except Exception:
            continue
    return remote_models


def remote_model_id(pc_id: str, model_id: str) -> str:
    return f"pc:{pc_id}:{model_id}"


def parse_remote_model_id(model_id: str) -> Optional[Dict[str, str]]:
    if not model_id.startswith("pc:"):
        return None
    parts = model_id.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return {"pc_id": parts[1], "model_id": parts[2]}


def resolve_remote_model(model_id: str) -> Optional[Dict[str, Any]]:
    parsed = parse_remote_model_id(model_id)
    if not parsed:
        return None
    link = get_link(parsed["pc_id"])
    if not link:
        return None
    return {**link, "remote_model_id": parsed["model_id"], "local_model_id": model_id}

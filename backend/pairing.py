"""Accountless local device pairing and peer credentials."""

from __future__ import annotations

import base64
import hashlib
import secrets
import socket
import threading
import time
from typing import Any, Dict, List, Optional

import keyring
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


SERVICE_NAME = "avalon.device"
IDENTITY_KEY = "identity"
PAIRING_TTL_SECONDS = 300
CODE_LENGTH = 8

_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}
_peers: Dict[str, Dict[str, Any]] = {}
_zeroconf = None
_service_name = ""


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _load_identity() -> Ed25519PrivateKey:
    encoded = keyring.get_password(SERVICE_NAME, IDENTITY_KEY)
    if encoded:
        return Ed25519PrivateKey.from_private_bytes(_unb64(encoded))
    private = Ed25519PrivateKey.generate()
    keyring.set_password(SERVICE_NAME, IDENTITY_KEY, _b64(private.private_bytes_raw()))
    return private


def device_info() -> Dict[str, str]:
    private = _load_identity()
    public = private.public_key().public_bytes_raw()
    return {
        "device_id": hashlib.sha256(public).hexdigest()[:16],
        "device_name": socket.gethostname(),
        "public_key": _b64(public),
    }


def sign_pairing(session_id: str, code: str) -> str:
    return _b64(_load_identity().sign(f"avalon-pair:{session_id}:{code.strip().upper()}".encode()))


def _new_code() -> str:
    # Avoid ambiguous characters while keeping the code easy to read aloud.
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(secrets.choice(alphabet) for _ in range(CODE_LENGTH))


def create_pairing_code() -> Dict[str, Any]:
    info = device_info()
    session_id = secrets.token_urlsafe(18)
    expires_at = int(time.time()) + PAIRING_TTL_SECONDS
    with _lock:
        _sessions[session_id] = {
            "code": _new_code(),
            "expires_at": expires_at,
            "used": False,
        }
    return {
        "session_id": session_id,
        "code": _sessions[session_id]["code"],
        "expires_at": expires_at,
        **info,
    }


def _peer_token(peer_id: str) -> str:
    token = secrets.token_urlsafe(32)
    keyring.set_password(SERVICE_NAME, f"peer:{peer_id}", token)
    return token


def store_peer_credential(peer_id: str, token: str) -> None:
    if not peer_id or not token:
        raise ValueError("Peer credential is required")
    keyring.set_password(SERVICE_NAME, f"remote:{peer_id}", token)


def peer_credential(peer_id: str) -> str:
    return keyring.get_password(SERVICE_NAME, f"remote:{peer_id}") or ""


def validate_peer_token(token: str) -> bool:
    if not token:
        return False
    with _lock:
        peer_ids = list(_peers)
    return any(keyring.get_password(SERVICE_NAME, f"peer:{peer_id}") == token for peer_id in peer_ids)


def accept_pairing(
    session_id: str,
    code: str,
    peer_name: str,
    peer_public_key: str,
    peer_signature: str,
) -> Dict[str, Any]:
    now = int(time.time())
    with _lock:
        session = _sessions.get(session_id) if session_id else None
        if session is None:
            session = next(
                (candidate for candidate in _sessions.values() if candidate["code"] == code.strip().upper()),
                None,
            )
        if not session or session["used"] or now >= session["expires_at"]:
            raise ValueError("Pairing code is invalid or expired")
        if not secrets.compare_digest(session["code"], code.strip().upper()):
            raise ValueError("Pairing code is invalid or expired")
        session["used"] = True

    try:
        public_key = _unb64(peer_public_key)
        if len(public_key) != 32:
            raise ValueError
        signature = _unb64(peer_signature)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            f"avalon-pair:{session_id}:{code.strip().upper()}".encode(),
        )
    except Exception as exc:
        raise ValueError("Invalid peer identity proof") from exc

    peer_id = hashlib.sha256(public_key).hexdigest()[:16]
    token = _peer_token(peer_id)
    with _lock:
        _peers[peer_id] = {
            "id": peer_id,
            "name": peer_name.strip() or peer_id,
            "public_key": peer_public_key,
            "paired_at": now,
        }
    return {
        "peer": _peers[peer_id],
        "api_key": token,
        "device": device_info(),
    }


def list_peers() -> List[Dict[str, Any]]:
    with _lock:
        return [dict(peer) for peer in _peers.values()]


def remove_peer(peer_id: str) -> bool:
    with _lock:
        existed = _peers.pop(peer_id, None) is not None
    if existed:
        for key in (f"peer:{peer_id}", f"remote:{peer_id}"):
            try:
                keyring.delete_password(SERVICE_NAME, key)
            except keyring.errors.PasswordDeleteError:
                pass
    return existed


def reset_for_tests() -> None:
    with _lock:
        _sessions.clear()
        _peers.clear()


def start_discovery(port: int) -> None:
    """Advertise this instance and keep discovery optional on unsupported hosts."""
    global _zeroconf, _service_name
    try:
        from zeroconf import ServiceInfo, Zeroconf
        import ipaddress

        info = device_info()
        host = socket.gethostbyname(socket.gethostname())
        _service_name = f"Avalon-{info['device_id']}._avalon._tcp.local."
        service = ServiceInfo(
            "_avalon._tcp.local.",
            _service_name,
            addresses=[ipaddress.ip_address(host).packed],
            port=port,
            properties={
                b"device_id": info["device_id"].encode(),
                b"device_name": info["device_name"].encode(),
                b"public_key": info["public_key"].encode(),
            },
        )
        _zeroconf = Zeroconf()
        _zeroconf.register_service(service)
    except Exception:
        _zeroconf = None


def stop_discovery() -> None:
    global _zeroconf
    if _zeroconf is not None:
        try:
            if _service_name:
                _zeroconf.unregister_service(_service_name)
            _zeroconf.close()
        except Exception:
            pass
        _zeroconf = None


def discover_devices(timeout_seconds: float = 1.5) -> List[Dict[str, Any]]:
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except Exception:
        return []

    found: List[Dict[str, Any]] = []

    class Listener(ServiceListener):
        def add_service(self, zc, service_type, name):
            info = zc.get_service_info(service_type, name)
            if not info:
                return
            properties = {
                (key.decode() if isinstance(key, bytes) else str(key)):
                (value.decode() if isinstance(value, bytes) else str(value))
                for key, value in info.properties.items()
            }
            addresses = info.parsed_addresses()
            if addresses:
                found.append({
                    "name": properties.get("device_name", name),
                    "device_id": properties.get("device_id", ""),
                    "public_key": properties.get("public_key", ""),
                    "base_url": f"http://{addresses[0]}:{info.port}",
                })

        def update_service(self, zc, service_type, name):
            self.add_service(zc, service_type, name)

        def remove_service(self, zc, service_type, name):
            return None

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, "_avalon._tcp.local.", Listener())
        time.sleep(max(0.1, min(timeout_seconds, 5.0)))
    finally:
        zc.close()
    unique = {item["device_id"]: item for item in found if item["device_id"]}
    return list(unique.values())

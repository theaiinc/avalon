import json
import sys

import main


def test_source_download_worker_uses_external_script(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    command = main._download_worker_command("abc123", "model", {"repo_id": "org/model"})

    assert command[0] == sys.executable
    assert command[1].endswith("download_worker.py")
    assert command[2:4] == ["abc123", "model"]
    assert json.loads(command[4]) == {"repo_id": "org/model"}


def test_frozen_download_worker_reenters_backend_sidecar(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/app/avalon-backend")

    command = main._download_worker_command("abc123", "model", {"repo_id": "org/model"})

    assert command[:4] == [
        "/app/avalon-backend",
        main.DOWNLOAD_WORKER_FLAG,
        "abc123",
        "model",
    ]
    assert json.loads(command[4]) == {"repo_id": "org/model"}

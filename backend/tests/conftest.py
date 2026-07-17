import sys
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class FakeReservation:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


@pytest.fixture
def fake_reservation():
    return FakeReservation()


@pytest.fixture
def model_catalog(tmp_path):
    gguf_dir = tmp_path / "gguf-model"
    gguf_dir.mkdir()
    (gguf_dir / "model.gguf").touch()

    openvino_dir = tmp_path / "openvino-model"
    openvino_dir.mkdir()
    (openvino_dir / "openvino_model.xml").touch()

    return [
        {
            "id": "gguf-model",
            "repo_id": "test/gguf-model",
            "files": ["model.gguf"],
            "path": str(gguf_dir),
            "size": 1,
            "format": "gguf",
            "openvino_path": None,
        },
        {
            "id": "openvino-model",
            "repo_id": "test/openvino-model",
            "files": [],
            "path": str(openvino_dir),
            "size": 1,
            "format": "openvino",
            "openvino_path": str(openvino_dir),
        },
    ]

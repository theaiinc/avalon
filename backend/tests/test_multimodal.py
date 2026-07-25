import base64
import json

import pytest

import multimodal
import progress
from model_manager import _infer_capabilities


def test_profile_requires_allowlisted_local_executable(monkeypatch):
    monkeypatch.setenv("AVALON_PLUGIN_ALLOWLIST", json.dumps({"demo": ["/usr/bin/demo"]}))
    profile = multimodal.validate_profile({
        "name": "demo", "modality": "tts", "mode": "local", "executable_id": "demo",
    })
    assert profile["protocol_version"] == multimodal.PROTOCOL_VERSION
    with pytest.raises(ValueError, match="approved executable"):
        multimodal.validate_profile({
            "modality": "tts", "mode": "local", "executable_id": "sh -c evil",
        })


def test_model_capabilities_are_conservative_and_explicit():
    assert "imagegen" in _infer_capabilities("black-forest-labs/FLUX.1-dev")
    assert "stt" in _infer_capabilities("openai/whisper-large-v3")
    assert _infer_capabilities("qwen2.5-7b-instruct") == []
    assert _infer_capabilities("anything", ["videogen"]) == ["videogen"]


def test_download_progress_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "DOWNLOADS_DIR", tmp_path)
    download_id = progress.create_download(kind="model", repo_id="org/model", filename="model.gguf")
    progress.update(download_id, status="downloading", percent=42)
    assert progress.get(download_id)["percent"] == 42
    assert progress.list_active()[0]["filename"] == "model.gguf"


def test_http_profile_rejects_private_target_without_explicit_opt_in():
    with pytest.raises(ValueError, match="private"):
        multimodal.validate_profile({
            "modality": "stt", "mode": "http", "url": "http://127.0.0.1:9999/test",
        })
    profile = multimodal.validate_profile({
        "modality": "stt", "mode": "http",
        "url": "http://127.0.0.1:9999/test", "allow_private_network": True,
    })
    assert profile["allow_private_network"] is True


def test_builtin_image_profile_is_allowed_but_other_modalities_are_rejected():
    profile = multimodal.validate_profile({
        "modality": "imagegen", "mode": "builtin", "model_path": "/tmp/flux.gguf",
    })
    assert profile["protocol_version"] == multimodal.PROTOCOL_VERSION
    with pytest.raises(ValueError, match="imagegen"):
        multimodal.validate_profile({"modality": "tts", "mode": "builtin"})


def test_artifact_ingestion_validates_image_and_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(multimodal, "ARTIFACT_DIR", tmp_path)
    artifact = multimodal._store_artifact("run", {
        "mime": "image/png",
        "data_base64": base64.b64encode(b"\x89PNG\r\n\x1a\npayload").decode(),
    }, 0)
    assert artifact["mime"] == "image/png"
    assert (tmp_path / "run" / artifact["filename"]).exists()
    with pytest.raises(ValueError, match="magic"):
        multimodal._store_artifact("run", {
            "mime": "image/png",
            "data_base64": base64.b64encode(b"not an image").decode(),
        }, 0)


@pytest.mark.parametrize("modality, payload, expected", [
    ("tts", {"characters": 100}, "characters_per_sec"),
    ("stt", {"audio_duration_sec": 2}, "real_time_factor"),
    ("imagegen", {"image_count": 2}, "images_per_sec"),
    ("videogen", {"video_duration_sec": 2, "frames": 24}, "frames_per_sec"),
])
def test_modality_metrics_are_normalized(modality, payload, expected):
    metrics = multimodal._metric_payload(modality, payload, 1.0, [])
    assert expected in metrics
    assert metrics["latency_ms"] == 1000


def test_artifact_path_cannot_escape_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(multimodal, "ARTIFACT_DIR", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ok.txt").write_text("ok")
    assert multimodal.artifact_path("run", "ok.txt") == run_dir / "ok.txt"
    assert multimodal.artifact_path("run", "../ok.txt") is None

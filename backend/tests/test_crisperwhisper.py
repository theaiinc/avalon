import base64
import json
import sys
import types
import wave
from pathlib import Path

import pytest

import crisperwhisper_runtime
import model_manager


def test_crisper_search_is_restricted_to_official_repositories(monkeypatch):
    class Model:
        def __init__(self, model_id):
            self.modelId = model_id
            self.downloads = 1
            self.likes = 2
            self.pipeline_tag = "automatic-speech-recognition"
            self.lastModified = None

    class FakeApi:
        def list_models(self, **kwargs):
            assert kwargs["search"] == "CrisperWhisper"
            return [
                Model("nyralabs/CrisperWhisper2.0_turbo"),
                Model("someone/arbitrary-transformers-model"),
            ]

    monkeypatch.setattr(model_manager, "api", FakeApi())
    models = model_manager.search_models("", model_format=model_manager.CRISPERWHISPER_FORMAT)
    assert [item["id"] for item in models] == ["nyralabs/CrisperWhisper2.0_turbo"]
    assert models[0]["format"] == model_manager.CRISPERWHISPER_FORMAT
    assert models[0]["supported"] is True


def test_custom_build_file_listing_is_available_without_download_support(monkeypatch):
    class FakeApi:
        def list_repo_files(self, _repo_id):
            return ["config.json", "weights.bin", "README.md", "model.json"]

        def get_paths_info(self, _repo_id, paths):
            return [types.SimpleNamespace(path=path, size=len(path)) for path in paths]

    monkeypatch.setattr(model_manager, "api", FakeApi())
    files = model_manager.list_files(
        "someone/unsupported-custom-build",
        model_manager.CRISPERWHISPER_FORMAT,
    )

    assert [item["name"] for item in files] == ["config.json", "weights.bin", "README.md"]
    assert all(item["size"] > 0 for item in files)


def test_crisper_download_writes_snapshot_metadata(tmp_path, monkeypatch):
    repo_id = "nyralabs/CrisperWhisper2.0_small"
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"weights")
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = model_manager.download_crisperwhisper_model(repo_id)
    model_dir = tmp_path / repo_id.replace("/", "_")
    metadata = json.loads((model_dir / "model.json").read_text())
    assert calls["repo_id"] == repo_id
    assert "allow_patterns" not in calls
    assert metadata["format"] == "crisperwhisper"
    assert metadata["capabilities"] == ["stt"]
    assert result["format"] == "crisperwhisper"


def _wav_base64() -> str:
    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    return base64.b64encode(output.getvalue()).decode()


def test_transformers_runtime_returns_transcript_words_and_duration(tmp_path, monkeypatch):
    model_path = tmp_path / "crisper"
    model_path.mkdir()
    (model_path / "model.json").write_text(json.dumps({"format": "crisperwhisper"}))
    calls = {}

    class Word:
        word = " hello"
        start = 0.0
        end = 0.5

    class Result:
        text = " hello"
        words = [Word()]
        duration = None

    class FakeModel:
        def __init__(self, path, backend):
            calls["constructor"] = (path, backend)

        def transcribe(self, path, **kwargs):
            calls["transcribe"] = kwargs
            assert Path(path).suffix == ".wav"
            return Result()

    monkeypatch.setitem(sys.modules, "crisperwhisper", types.SimpleNamespace(CrisperWhisperModel=FakeModel))
    response = crisperwhisper_runtime.transcribe_audio(
        str(model_path),
        _wav_base64(),
        audio_name="sample.wav",
        mode="intended",
    )
    assert calls["constructor"] == (str(model_path), "transformers")
    assert calls["transcribe"]["mode"] == "intended"
    assert calls["transcribe"]["hallucination_mitigation"] is False
    assert calls["transcribe"]["temperature_fallback"] is False
    assert response["transcript"] == " hello"
    assert response["words"][0]["start"] == 0.0
    assert response["audio_duration_sec"] == pytest.approx(0.1)


def test_webm_input_is_decoded_to_wav_before_transcription(tmp_path, monkeypatch):
    model_path = tmp_path / "crisper"
    model_path.mkdir()
    (model_path / "model.json").write_text(json.dumps({"format": "crisperwhisper"}))
    calls = {}

    def fake_decode(source, target):
        calls["source"] = source
        calls["target"] = target
        target.write_bytes(base64.b64decode(_wav_base64()))

    class Result:
        text = "decoded"
        words = []
        duration = 0.1

    class FakeModel:
        def __init__(self, path, backend):
            pass

        def transcribe(self, path, **kwargs):
            calls["transcribe_path"] = Path(path)
            return Result()

    monkeypatch.setattr(crisperwhisper_runtime, "_decode_webm_to_wav", fake_decode)
    monkeypatch.setitem(sys.modules, "crisperwhisper", types.SimpleNamespace(CrisperWhisperModel=FakeModel))
    response = crisperwhisper_runtime.transcribe_audio(
        str(model_path),
        "d2VicQ==",
        audio_name="recording.webm",
        audio_mime="audio/webm;codecs=opus",
    )

    assert calls["source"].suffix == ".webm"
    assert calls["target"].suffix == ".wav"
    assert calls["transcribe_path"].suffix == ".wav"
    assert response["transcript"] == "decoded"


def test_missing_webm_decoder_reports_bundled_runtime_error(monkeypatch):
    def fail_decoder():
        raise crisperwhisper_runtime.CrisperWhisperUnavailable("decoder missing")

    monkeypatch.setattr(crisperwhisper_runtime, "_load_audio_decoder", fail_decoder)
    with pytest.raises(
        crisperwhisper_runtime.CrisperWhisperUnavailable,
        match="decoder missing",
    ):
        crisperwhisper_runtime._decode_webm_to_wav(Path("input.webm"), Path("output.wav"))


@pytest.mark.parametrize("audio_name", ["sample.wav", "sample.mp3", "sample.flac", "sample.ogg"])
def test_existing_audio_formats_are_passed_through(audio_name, tmp_path, monkeypatch):
    model_path = tmp_path / "crisper"
    model_path.mkdir()
    (model_path / "model.json").write_text(json.dumps({"format": "crisperwhisper"}))
    seen = {}

    class Result:
        text = ""
        words = []
        duration = None

    class FakeModel:
        def __init__(self, path, backend):
            pass

        def transcribe(self, path, **kwargs):
            seen["suffix"] = Path(path).suffix
            return Result()

    def unexpected_webm_decode(source, target):
        raise AssertionError("non-WebM audio must not use the WebM decoder")

    monkeypatch.setattr(crisperwhisper_runtime, "_decode_webm_to_wav", unexpected_webm_decode)
    monkeypatch.setitem(sys.modules, "crisperwhisper", types.SimpleNamespace(CrisperWhisperModel=FakeModel))
    crisperwhisper_runtime.transcribe_audio(
        str(model_path),
        _wav_base64(),
        audio_name=audio_name,
        audio_mime=f"audio/{Path(audio_name).suffix[1:]}",
    )

    assert seen["suffix"] == Path(audio_name).suffix


def test_runtime_import_failure_preserves_dependency_error(tmp_path, monkeypatch):
    model_path = tmp_path / "crisper"
    model_path.mkdir()
    (model_path / "model.json").write_text(json.dumps({"format": "crisperwhisper"}))
    real_import_module = crisperwhisper_runtime.importlib.import_module

    def fail_transformers(name):
        if name == "transformers":
            raise OSError("mach-o, but wrong architecture (arm64)")
        return real_import_module(name)

    monkeypatch.setattr(
        crisperwhisper_runtime.importlib,
        "import_module",
        fail_transformers,
    )
    with pytest.raises(
        crisperwhisper_runtime.CrisperWhisperUnavailable,
        match="OSError: mach-o, but wrong architecture",
    ):
        crisperwhisper_runtime.transcribe_audio(str(model_path), _wav_base64())


def test_runtime_diagnostics_reports_each_module_failure(monkeypatch):
    def fail_torch(name):
        if name == "torch":
            raise ImportError("No module named torch._C")
        return types.SimpleNamespace(__file__="", __version__="")

    monkeypatch.setattr(
        crisperwhisper_runtime.importlib,
        "import_module",
        fail_torch,
    )
    diagnostics = crisperwhisper_runtime.runtime_diagnostics()
    assert diagnostics["available"] is False
    assert diagnostics["modules"]["torch"]["error"] == (
        "ImportError: No module named torch._C"
    )
    assert "torch._C" in diagnostics["error"]


def test_runtime_rejects_untagged_model(tmp_path):
    model_path = tmp_path / "not-crisper"
    model_path.mkdir()
    (model_path / "model.json").write_text(json.dumps({"format": "gguf"}))
    with pytest.raises(ValueError, match="crisperwhisper"):
        crisperwhisper_runtime.transcribe_audio(str(model_path), _wav_base64())

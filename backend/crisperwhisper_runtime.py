"""Optional CrisperWhisper Transformers runtime.

The import is intentionally lazy so the dashboard sidecar does not pull the
PyTorch/Transformers stack into its packaged executable.  Only model
directories explicitly tagged as ``crisperwhisper`` are accepted here.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import json
import mimetypes
import platform
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional


class CrisperWhisperUnavailable(RuntimeError):
    """Raised when the optional STT runtime is not installed."""


def _load_audio_decoder():
    """Load Avalon’s self-contained container decoder.

    ``soundfile``/libsndfile does not decode browser-produced WebM/Opus.
    PyAV wheels carry the FFmpeg libraries they need, so the packaged app
    does not depend on a user-installed ``ffmpeg`` executable.
    """
    try:
        return importlib.import_module("av")
    except Exception as exc:
        raise CrisperWhisperUnavailable(
            "Browser WebM/Opus audio requires Avalon’s bundled PyAV decoder, "
            f"which could not be imported ({_exception_details(exc)})."
        ) from exc


def _exception_details(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _load_transformers_runtime():
    """Load the complete runtime and preserve the real import failure.

    ``crisperwhisper`` itself imports without Torch or Transformers because
    those dependencies are loaded lazily by its model class.  Checking only
    the package import therefore reports a false positive in a frozen app
    whose native Torch extension or Transformers dependency was not bundled.
    """
    try:
        module = importlib.import_module("crisperwhisper")
        importlib.import_module("torch")
        importlib.import_module("transformers")
        importlib.import_module("accelerate")
        return module.CrisperWhisperModel
    except Exception as exc:
        raise CrisperWhisperUnavailable(
            "CrisperWhisper Transformers runtime could not be imported "
            f"({_exception_details(exc)}); install crisperwhisper[transformers]"
        ) from exc


def runtime_diagnostics() -> Dict[str, Any]:
    """Return import diagnostics suitable for a packaged-sidecar smoke test."""
    diagnostics: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "modules": {},
    }
    for name in ("crisperwhisper", "torch", "transformers", "accelerate", "av"):
        try:
            module = importlib.import_module(name)
            diagnostics["modules"][name] = {
                "available": True,
                "file": str(getattr(module, "__file__", "") or ""),
                "version": str(getattr(module, "__version__", "") or ""),
            }
        except Exception as exc:
            diagnostics["modules"][name] = {
                "available": False,
                "error": _exception_details(exc),
            }
    try:
        _load_transformers_runtime()
    except CrisperWhisperUnavailable as exc:
        diagnostics["available"] = False
        diagnostics["error"] = str(exc)
    else:
        diagnostics["available"] = True
    try:
        decoder = _load_audio_decoder()
    except CrisperWhisperUnavailable as exc:
        diagnostics["audio_decoder"] = {"available": False, "error": str(exc)}
    else:
        diagnostics["audio_decoder"] = {
            "available": True,
            "file": str(getattr(decoder, "__file__", "") or ""),
            "version": str(getattr(decoder, "__version__", "") or ""),
        }
    return diagnostics


def _model_is_supported(model_path: str | Path) -> bool:
    path = Path(model_path)
    if not path.is_dir():
        return False
    try:
        with (path / "model.json").open() as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    return metadata.get("format") == "crisperwhisper"


def _suffix(audio_name: str, audio_mime: str) -> str:
    name_suffix = Path(audio_name).suffix.lower()
    if name_suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}:
        return name_suffix
    mime = (audio_mime or "").split(";", 1)[0].strip().lower()
    return mimetypes.guess_extension(mime) or ".wav"


def _is_webm(audio_name: str, audio_mime: str) -> bool:
    return (
        Path(audio_name).suffix.lower() == ".webm"
        or (audio_mime or "").split(";", 1)[0].strip().lower() == "audio/webm"
    )


def _decode_webm_to_wav(source_path: Path, target_path: Path) -> None:
    """Decode browser WebM/Opus into a mono 16 kHz PCM WAV file."""
    av = _load_audio_decoder()
    try:
        import numpy as np

        with av.open(str(source_path), mode="r") as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise ValueError("the WebM container has no audio stream")
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=16_000,
            )
            frame_count = 0
            with wave.open(str(target_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        for converted in resampler.resample(frame):
                            samples = np.asarray(converted.to_ndarray())
                            output.writeframes(
                                np.ascontiguousarray(samples).astype(np.int16, copy=False).tobytes()
                            )
                            frame_count += int(converted.samples)
                for converted in resampler.resample(None):
                    samples = np.asarray(converted.to_ndarray())
                    output.writeframes(
                        np.ascontiguousarray(samples).astype(np.int16, copy=False).tobytes()
                    )
                    frame_count += int(converted.samples)
            if frame_count == 0:
                raise ValueError("the WebM container has no decodable audio frames")
    except CrisperWhisperUnavailable:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not decode browser WebM/Opus audio with the bundled decoder "
            f"({_exception_details(exc)})"
        ) from exc


def _decode_audio(audio_base64: str) -> bytes:
    value = str(audio_base64 or "")
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("audio_base64 is not valid base64") from exc
    if not payload:
        raise ValueError("audio input is empty")
    return payload


def _duration_seconds(path: Path) -> Optional[float]:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path)) as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        return None


def _word_payload(word: Any) -> Dict[str, Any]:
    if isinstance(word, dict):
        get = word.get
    else:
        get = lambda key, default=None: getattr(word, key, default)
    text = get("word", get("text", ""))
    start = get("start")
    end = get("end")
    item: Dict[str, Any] = {"word": str(text)}
    if start is not None:
        item["start"] = float(start)
    if end is not None:
        item["end"] = float(end)
    return item


def transcribe_audio(
    model_path: str,
    audio_base64: str,
    *,
    audio_name: str = "",
    audio_mime: str = "audio/wav",
    language: str = "en",
    mode: str = "verbatim",
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    """Transcribe one base64 audio input using the official Transformers backend."""
    if not _model_is_supported(model_path):
        raise ValueError("built-in CrisperWhisper requires a downloaded crisperwhisper model")
    if mode not in {"verbatim", "intended"}:
        raise ValueError("STT mode must be verbatim or intended")

    CrisperWhisperModel = _load_transformers_runtime()

    payload = _decode_audio(audio_base64)
    suffix = _suffix(audio_name, audio_mime)
    with tempfile.TemporaryDirectory(prefix="avalon-stt-") as temp_dir:
        source_path = Path(temp_dir) / f"input{suffix}"
        source_path.write_bytes(payload)
        audio_path = source_path
        if _is_webm(audio_name, audio_mime):
            audio_path = Path(temp_dir) / "decoded.wav"
            _decode_webm_to_wav(source_path, audio_path)
        transcribe_options = {
            "language": language or "en",
            "mode": mode,
            "word_timestamps": word_timestamps,
        }
        try:
            model = CrisperWhisperModel(str(model_path), backend="transformers")
            if getattr(model, "backend", "transformers") == "transformers":
                # CrisperWhisper 2.0's optional hallucination repair and
                # temperature fallback import CT2-only helpers. The official
                # macOS Transformers extra intentionally does not install
                # ctranslate2, so keep the pure-Torch path self-contained.
                transcribe_options.update({
                    "hallucination_mitigation": False,
                    "temperature_fallback": False,
                })
            result = model.transcribe(str(audio_path), **transcribe_options)
        except TypeError:
            # Older 2.0 releases accept the same backend but omit one newer
            # keyword; keep the official backend usable across patch releases.
            model = CrisperWhisperModel(str(model_path), backend="transformers")
            result = model.transcribe(
                str(audio_path),
                language=language or "en",
                mode=mode,
            )
        duration_from_file = _duration_seconds(audio_path)

    words = [_word_payload(word) for word in (getattr(result, "words", None) or [])]
    duration = getattr(result, "duration", None) or duration_from_file
    return {
        "transcript": str(getattr(result, "text", "") or ""),
        "words": words,
        "audio_duration_sec": float(duration) if duration is not None else None,
        "transcript_mode": mode,
    }

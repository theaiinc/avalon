"""Build native Avalon Python sidecars for Electron packaging.

Run this from the repository root after installing backend requirements and
PyInstaller on the target runner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OUTPUT = ROOT / "frontend" / "runtime" / "sidecars"
WORK = ROOT / ".build" / "pyinstaller"


def build(name: str, script: str) -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        name,
        "--paths",
        str(BACKEND),
        "--distpath",
        str(OUTPUT),
        "--workpath",
        str(WORK / name),
        "--specpath",
        str(WORK),
        str(BACKEND / script),
    ]
    if sys.platform == "darwin":
        # OpenVINO's arm64 macOS wheel contains signed dylibs whose Mach-O
        # load-command layout cannot be rewritten by PyInstaller's
        # install_name_tool step. Avalon has no supported NPU/OpenVINO target
        # on macOS, so keep that optional runtime out of the macOS sidecar.
        command[command.index(str(BACKEND / script)):command.index(str(BACKEND / script))] = [
            "--exclude-module",
            "openvino",
            "--exclude-module",
            "openvino_genai",
        ]
    if script == "main.py":
        # Keep the optional OpenVINO inference stack out of this process.
        # Built-in multimodal STT runs in the dashboard sidecar, so its
        # separate Transformers/Torch runtime must remain bundled here.
        command[command.index(str(BACKEND / script)):command.index(str(BACKEND / script))] = [
            "--exclude-module",
            "openvino",
            "--exclude-module",
            "openvino_genai",
            "--exclude-module",
            "optimum",
        ]
    if script in {"main.py", "api_server.py"}:
        # multimodal.py imports the STT runtime lazily. Collect it explicitly
        # in every sidecar that can execute a built-in multimodal run.
        command[command.index(str(BACKEND / script)):command.index(str(BACKEND / script))] += [
            "--collect-all",
            "crisperwhisper",
            "--collect-all",
            "av",
            "--hidden-import",
            "crisperwhisper.transformers_engine",
            "--hidden-import",
            "crisperwhisper_runtime",
            "--hidden-import",
            "av",
            "--hidden-import",
            "torch",
            "--hidden-import",
            "transformers",
            "--hidden-import",
            "accelerate",
        ]
    subprocess.run(command, cwd=ROOT, check=True)


def verify_stt_runtime() -> None:
    """Fail the macOS build if either frozen sidecar cannot import STT."""
    if sys.platform != "darwin":
        return
    for name in ("avalon-backend", "avalon-gateway"):
        executable = OUTPUT / name
        result = subprocess.run(
            [str(executable), "--diagnose-crisperwhisper"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Packaged {name} CrisperWhisper diagnostic exited "
                f"with code {result.returncode}: {result.stderr.strip()}"
            )
        try:
            diagnostics = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Packaged {name} CrisperWhisper diagnostic did not return JSON: "
                f"{result.stdout.strip()}"
            ) from exc
        if not diagnostics.get("available"):
            raise RuntimeError(
                f"Packaged {name} CrisperWhisper runtime is unavailable: "
                f"{diagnostics.get('error', diagnostics)}"
            )
        if not diagnostics.get("audio_decoder", {}).get("available"):
            raise RuntimeError(
                f"Packaged {name} WebM/Opus decoder is unavailable: "
                f"{diagnostics.get('audio_decoder', {}).get('error', diagnostics)}"
            )
        print(
            f"Verified packaged {name} CrisperWhisper Transformers runtime "
            f"({diagnostics.get('machine', 'unknown')})",
            flush=True,
        )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("avalon-backend*"):
        if old.is_file():
            old.unlink()
    for old in OUTPUT.glob("avalon-gateway*"):
        if old.is_file():
            old.unlink()
    build("avalon-backend", "main.py")
    build("avalon-gateway", "api_server.py")
    verify_stt_runtime()
    readme = OUTPUT / "README.md"
    if readme.exists():
        readme.unlink()
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()

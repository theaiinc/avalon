"""Build native Avalon Python sidecars for Electron packaging.

Run this from the repository root after installing backend requirements and
PyInstaller on the target runner.
"""

from __future__ import annotations

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
    elif script == "main.py":
        # The dashboard sidecar delegates inference to avalon-gateway. Keep
        # optional inference stacks out of this process so Windows/Linux do
        # not analyze and bundle OpenVINO twice.
        command[command.index(str(BACKEND / script)):command.index(str(BACKEND / script))] = [
            "--exclude-module",
            "openvino",
            "--exclude-module",
            "openvino_genai",
            "--exclude-module",
            "optimum",
            "--exclude-module",
            "transformers",
        ]
    subprocess.run(command, cwd=ROOT, check=True)


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
    readme = OUTPUT / "README.md"
    if readme.exists():
        readme.unlink()
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()

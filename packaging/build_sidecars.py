"""Build native Avalon Python sidecars for Electron packaging.

Run this from the repository root after installing backend requirements and
PyInstaller on the target runner.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OUTPUT = ROOT / "frontend" / "runtime" / "sidecars"
WORK = ROOT / ".build" / "pyinstaller"


def build(name: str, script: str) -> None:
    if sys.platform == "darwin":
        # OpenVINO wheels ship dylibs with signatures that Apple's codesign
        # subsystem may reject when PyInstaller re-signs the collected copy.
        # Remove those wheel signatures first; PyInstaller will apply its
        # ad-hoc signature to the clean copies it embeds.
        openvino_spec = find_spec("openvino")
        if openvino_spec and openvino_spec.submodule_search_locations:
            for root in openvino_spec.submodule_search_locations:
                for library in Path(root).rglob("*.dylib"):
                    subprocess.run(
                        ["codesign", "--remove-signature", str(library)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

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

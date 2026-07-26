"""Ensure native Python sidecars exist before Electron packaging."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend" / "runtime" / "sidecars"
EXPECTED = [
    OUTPUT / "avalon-backend",
    OUTPUT / "avalon-gateway",
]
if sys.platform == "win32":
    EXPECTED = [path.with_suffix(".exe") for path in EXPECTED]


def main() -> None:
    missing = [path.name for path in EXPECTED if not path.is_file()]
    if not missing:
        return
    print(f"Building missing Electron sidecars: {', '.join(missing)}", flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_sidecars.py"))],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

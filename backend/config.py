import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AVALON_DATA_DIR", str(ROOT_DIR / "data"))).expanduser()
DRIVERS_DIR = DATA_DIR / "drivers"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
LLAMA_BENCH_EXE = "llama-bench.exe"

os.makedirs(DRIVERS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

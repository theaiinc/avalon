# Avalon

Download, serve, and benchmark local LLMs across multiple GPU backends (CUDA, Vulkan, SYCL, Metal, CPU).

## Quick Start

```bash
# One command — installs deps + starts both backend & frontend
cd benchmark
.\start.bat        # CMD
.\start.ps1        # PowerShell (preferred)
```

Or manually:

```bash
# Terminal 1 - Backend
cd benchmark/backend
pip install -r requirements.txt
python main.py

# Terminal 2 - Frontend
cd benchmark/frontend
npm install
npm run dev
```

## Electron desktop app

The desktop app keeps dashboard controls on loopback and exposes only the
authenticated LLM gateway to the LAN:

```bash
cd frontend
npm run dev:app       # Vite renderer + Electron shell
npm run pack          # unsigned unpacked desktop executable
npm run dist          # unsigned platform installer and portable archive
```

The packaged app stores models and its generated API key under the platform
application-data directory. The API page displays the LAN gateway URL and
key. Use the `Authorization: Bearer <key>` or `X-API-Key: <key>` header for
`/v1/models`, `/v1/chat/completions`, and `/v1/messages`; dashboard controls
remain local-only.

Tagged releases (`v*`) are built by GitHub Actions for macOS arm64/x64,
Windows x64, and Linux x64. Release artifacts include installers, portable
archives, and SHA-256 manifests; model weights are intentionally not bundled.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/gpu/list | List detected GPUs |
| GET | /api/drivers | List local drivers |
| GET | /api/drivers/releases | List GitHub releases |
| POST | /api/drivers/download | Download a driver |
| POST | /api/drivers/activate | Activate a driver |
| POST | /api/drivers/remove | Remove a driver |
| GET | /api/models | List local models |
| GET | /api/models/search | Search HuggingFace |
| POST | /api/models/download | Download a model |
| POST | /api/models/remove | Remove a model |
| POST | /api/benchmark/run | Run benchmark |
| GET | /api/benchmark/results | List benchmark results |
| GET | /api/benchmark/results/:id | Get benchmark result |

## Agent Interfaces

The dashboard backend exposes HTTP APIs on `http://127.0.0.1:8771`.

CLI:

```bash
cd backend
.venv/bin/python agent_cli.py gpu-list
.venv/bin/python agent_cli.py models
.venv/bin/python agent_cli.py model-search "qwen 0.5b" --limit 5
.venv/bin/python agent_cli.py model-files <repo-id>
.venv/bin/python agent_cli.py model-download <repo-id> <file.gguf> --wait
.venv/bin/python agent_cli.py benchmark-run --model-path /path/to/model.gguf --backend metal
.venv/bin/python agent_cli.py pc-link-add "Gaming PC" http://192.168.1.50:8787
.venv/bin/python agent_cli.py pc-links
.venv/bin/python agent_cli.py api-start --device metal
.venv/bin/python agent_cli.py chat --model <local-model-id> --message "Hello"
```

MCP stdio server:

```bash
cd backend
LLAMA_DASH_URL=http://127.0.0.1:8771 .venv/bin/python mcp_server.py
```

The MCP server provides tools for GPUs, model search/download/progress, local models, drivers, benchmark runs/results, inference API lifecycle, and OpenAI-compatible chat against the local inference API.

The local inference API is a multi-model gateway: every model under `data/models/` appears in `GET /v1/models`, and chat requests select one with the standard `model` field.
Linked PC models also appear in `GET /v1/models` with IDs like `pc:<pc-id>:<remote-model-id>`.

## Accountless device pairing

Avalon can pair two installations directly over the local network without an
account or cloud service. On the remote device, open **PC Links** and generate
a pairing code. On the initiating device, enter the remote device URL and that
short-lived code, then choose **Pair Device**.

The device identity and per-peer credentials are stored through the operating
system keyring. Pairing codes expire after five minutes and can be used only
once. Avalon advertises local devices with Bonjour/mDNS as `_avalon._tcp`; the
URL field remains a fallback for networks where multicast discovery is blocked.
Only the pairing acceptance endpoint is reachable from the LAN; dashboard
controls remain local-only.

## Completion tests

The fast suite uses fake model runtimes and covers every supported local model
format (GGUF and OpenVINO), OpenAI chat completions, OpenAI streaming SSE, and
Anthropic messages:

```bash
cd backend
python -m pip install -r requirements-dev.txt
pytest tests -q
```

To run the same flows against every model currently downloaded on the machine,
run the opt-in real inference smoke test sequentially:

```bash
AVALON_RUN_REAL_MODEL_TESTS=1 pytest tests/test_real_model_completions.py -q
```

Set `AVALON_TEST_MAX_TOKENS` to reduce or increase the generated response
length. Real-model tests require an active compatible driver and can use
substantial memory.

## Data Directories

- `data/drivers/` - Downloaded llama.cpp driver packages
- `data/models/` - Downloaded GGUF model files
- `data/results/` - Benchmark result JSON files

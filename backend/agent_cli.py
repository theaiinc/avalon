import argparse
import json
from typing import Any

from agent_api import (
    DEFAULT_DASHBOARD_URL,
    DEFAULT_INFERENCE_URL,
    DashboardClient,
    chat_completion,
    parse_json_object,
)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _client(args: argparse.Namespace) -> DashboardClient:
    return DashboardClient(args.base_url, timeout=args.timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI for llama.cpp Benchmark Dashboard")
    parser.add_argument("--base-url", default=DEFAULT_DASHBOARD_URL, help="Dashboard backend URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gpu-list", help="List detected GPUs")
    subparsers.add_parser("models", help="List local models")
    subparsers.add_parser("drivers", help="List local and active drivers")
    subparsers.add_parser("pc-links", help="List linked PCs")
    subparsers.add_parser("results", help="List benchmark results")
    subparsers.add_parser("active-benchmarks", help="List active benchmarks")
    subparsers.add_parser("active-downloads", help="List active model/driver downloads")
    subparsers.add_parser("api-status", help="Show local inference API status")
    subparsers.add_parser("api-stop", help="Stop the local inference API")

    search = subparsers.add_parser("model-search", help="Search Hugging Face models")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--format", default="gguf", choices=["gguf", "openvino"])

    files = subparsers.add_parser("model-files", help="List files for a Hugging Face model")
    files.add_argument("repo_id")
    files.add_argument("--format", default="gguf", choices=["gguf", "openvino"])

    pc_test = subparsers.add_parser("pc-link-test", help="Test a linked PC URL")
    pc_test.add_argument("pc_base_url")

    pc_add = subparsers.add_parser("pc-link-add", help="Add or update a linked PC")
    pc_add.add_argument("name")
    pc_add.add_argument("pc_base_url")
    pc_add.add_argument("--id", default="")

    pc_remove = subparsers.add_parser("pc-link-remove", help="Remove a linked PC")
    pc_remove.add_argument("id")

    download = subparsers.add_parser("model-download", help="Download a GGUF model file")
    download.add_argument("repo_id")
    download.add_argument("filename")
    download.add_argument("--wait", action="store_true", help="Wait until the download finishes")
    download.add_argument("--wait-interval", type=float, default=2.0)
    download.add_argument("--wait-timeout", type=float, default=3600.0)

    download_ov = subparsers.add_parser("model-download-openvino", help="Download an OpenVINO model snapshot")
    download_ov.add_argument("repo_id")
    download_ov.add_argument("--wait", action="store_true", help="Wait until the download finishes")
    download_ov.add_argument("--wait-interval", type=float, default=2.0)
    download_ov.add_argument("--wait-timeout", type=float, default=3600.0)

    progress = subparsers.add_parser("download-progress", help="Get download progress")
    progress.add_argument("download_id")

    result = subparsers.add_parser("result", help="Get a benchmark result by id")
    result.add_argument("result_id")

    run = subparsers.add_parser("benchmark-run", help="Start a benchmark")
    run.add_argument("--model-path", required=True)
    run.add_argument("--backend", action="append", required=True, dest="backends")
    run.add_argument("--bench-params-json", help="JSON object for benchmark parameters")
    run.add_argument("--model-params-json", help="JSON object for model parameters")

    status = subparsers.add_parser("benchmark-status", help="Get benchmark task status")
    status.add_argument("task_id")

    cancel = subparsers.add_parser("benchmark-cancel", help="Cancel an active benchmark")
    cancel.add_argument("task_id")

    start = subparsers.add_parser("api-start", help="Start the local inference API")
    start.add_argument("--model-id", default="")
    start.add_argument("--port", type=int, default=8787)
    start.add_argument("--mode", default="openai", choices=["openai", "anthropic", "both"])
    start.add_argument("--device", default="")
    start.add_argument("--gpu-index", default="")

    chat = subparsers.add_parser("chat", help="Call the OpenAI-compatible local inference API")
    chat.add_argument("--inference-url", default=DEFAULT_INFERENCE_URL)
    chat.add_argument("--model", default="")
    chat.add_argument("--message", required=True)
    chat.add_argument("--system")
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    client = _client(args)

    if args.command == "gpu-list":
        _print_json(client.list_gpus())
    elif args.command == "models":
        _print_json(client.list_models())
    elif args.command == "drivers":
        _print_json(client.list_drivers())
    elif args.command == "pc-links":
        _print_json(client.list_pc_links())
    elif args.command == "results":
        _print_json(client.list_benchmark_results())
    elif args.command == "active-benchmarks":
        _print_json(client.list_active_benchmarks())
    elif args.command == "active-downloads":
        _print_json(client.list_active_downloads())
    elif args.command == "api-status":
        _print_json(client.inference_status())
    elif args.command == "api-stop":
        _print_json(client.stop_inference_api())
    elif args.command == "model-search":
        _print_json(client.search_models(args.query, args.limit, args.format))
    elif args.command == "model-files":
        _print_json(client.list_model_files(args.repo_id, args.format))
    elif args.command == "pc-link-test":
        _print_json(client.test_pc_link(args.pc_base_url))
    elif args.command == "pc-link-add":
        _print_json(client.save_pc_link(args.name, args.pc_base_url, args.id))
    elif args.command == "pc-link-remove":
        _print_json(client.remove_pc_link(args.id))
    elif args.command == "model-download":
        result = client.download_model(args.repo_id, args.filename)
        if args.wait:
            result["final_progress"] = client.wait_for_download(
                result["download_id"],
                interval=args.wait_interval,
                timeout=args.wait_timeout,
            )
        _print_json(result)
    elif args.command == "model-download-openvino":
        result = client.download_openvino_model(args.repo_id)
        if args.wait:
            result["final_progress"] = client.wait_for_download(
                result["download_id"],
                interval=args.wait_interval,
                timeout=args.wait_timeout,
            )
        _print_json(result)
    elif args.command == "download-progress":
        _print_json(client.download_progress(args.download_id))
    elif args.command == "result":
        _print_json(client.get_benchmark_result(args.result_id))
    elif args.command == "benchmark-run":
        _print_json(
            client.run_benchmark(
                args.model_path,
                args.backends,
                parse_json_object(args.bench_params_json),
                parse_json_object(args.model_params_json),
            )
        )
    elif args.command == "benchmark-status":
        _print_json(client.benchmark_status(args.task_id))
    elif args.command == "benchmark-cancel":
        _print_json(client.cancel_benchmark(args.task_id))
    elif args.command == "api-start":
        _print_json(
            client.start_inference_api(
                args.model_id,
                args.port,
                args.mode,
                args.device,
                args.gpu_index,
            )
        )
    elif args.command == "chat":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.message})
        _print_json(
            chat_completion(
                args.inference_url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
            )
        )


if __name__ == "__main__":
    main()

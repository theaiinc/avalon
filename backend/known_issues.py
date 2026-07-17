# Known compatibility issues between models and backends.
# Rules can match by model name pattern, architecture prefix, or parameter count.
# When a check matches, a warning/error is shown on the Benchmark page.

KNOWN_ISSUES = [
    {
        "id": "openvino-gemma4-matmul-shape",
        "backends": ["openvino"],
        "match": {
            "model_name_contains": ["gemma-4", "gemma4", "Gemma4"],
        },
        "severity": "error",
        "title": "OpenVINO + Gemma 4 incompatible",
        "detail": (
            "llama.cpp's OpenVINO backend has a hardcoded matmul shape constraint "
            "that expects 256-dim token embeddings, but Gemma 4 uses 2560-dim embeddings. "
            "The graph compilation fails with 'Incompatible MatMul matrix dimension'."
        ),
        "workaround": "Use the Vulkan backend instead on Intel Arc hardware.",
        "ref": "https://github.com/ggml-ai/llama.cpp/issues/23945",
    },
    {
        "id": "sycl-intel-arc-deadlock",
        "backends": ["sycl"],
        "match": {
            "gpu_name_contains": ["Arc"],
            "driver_version_below": "32.0.101.8860",
        },
        "severity": "error",
        "title": "SYCL deadlock on Intel Arc with driver < 8860",
        "detail": (
            "llama.cpp SYCL backend deadlocks in ggml-sycl.dll during model initialization "
            "on Intel Arc GPUs with driver version below 32.0.101.8860. "
            "The subprocess hangs for >120s and times out."
        ),
        "workaround": "Update Intel GPU driver to 32.0.101.8860 or later, or use the Vulkan backend instead.",
        "ref": "https://github.com/ggml-ai/llama.cpp/issues/23945",
    },
    {
        "id": "npu-model-conversion",
        "backends": ["npu"],
        "match": {},
        "severity": "info",
        "title": "NPU requires OpenVINO IR model format",
        "detail": (
            "The NPU backend converts GGUF models to OpenVINO IR format. "
            "It first tries to find a pre-converted OpenVINO IR model on HuggingFace, "
            "and falls back to on-the-fly conversion from the original HuggingFace source model. "
            "Only small models (<3B params) perform well on NPU."
        ),
        "workaround": "For best results, use models under 3B parameters on the NPU.",
    },
]

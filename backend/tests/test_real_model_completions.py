"""Opt-in smoke tests for every model currently onboarded on this machine.

Run with:
    AVALON_RUN_REAL_MODEL_TESTS=1 pytest backend/tests/test_real_model_completions.py -q
"""

import asyncio
import os

import httpx
import pytest

import api_server
from model_manager import get_local_models


COMPLETION_MODELS = [
    model for model in get_local_models()
    if not model.get("is_draft") and model.get("serving_supported", True)
]


pytestmark = pytest.mark.skipif(
    os.environ.get("AVALON_RUN_REAL_MODEL_TESTS") != "1",
    reason="set AVALON_RUN_REAL_MODEL_TESTS=1 to run local inference",
)


@pytest.mark.parametrize(
    "model",
    COMPLETION_MODELS,
    ids=lambda model: model["id"],
)
def test_onboarded_model_supports_chat_messages_and_streaming(model):
    """Run all completion forms for exactly one model.

    Parametrizing by model is intentional: a failure in one large model must
    not prevent the remaining onboarded models from being tested and reported.
    """

    async def run():
        failures = []

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app),
            base_url="http://127.0.0.1",
            timeout=300,
        ) as client:
            payload = {
                "model": model["id"],
                "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                "max_tokens": int(os.environ.get("AVALON_TEST_MAX_TOKENS", "16")),
                "temperature": 0,
            }

            response = await client.post("/v1/chat/completions", json=payload)
            if response.status_code != 200:
                failures.append(f"chat completion: HTTP {response.status_code} {response.text}")
            else:
                try:
                    body = response.json()
                    if not body["choices"][0]["message"]["content"].strip():
                        failures.append("chat completion: empty response")
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    failures.append(f"chat completion: invalid response ({exc})")

            anthropic_response = await client.post("/v1/messages", json=payload)
            if anthropic_response.status_code != 200:
                failures.append(
                    f"Anthropic messages: HTTP {anthropic_response.status_code} "
                    f"{anthropic_response.text}"
                )
            else:
                try:
                    anthropic_body = anthropic_response.json()
                    if not anthropic_body["content"][0]["text"].strip():
                        failures.append("Anthropic messages: empty response")
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    failures.append(f"Anthropic messages: invalid response ({exc})")

            stream_response = await client.post(
                "/v1/chat/completions",
                json={**payload, "stream": True},
            )
            if stream_response.status_code != 200:
                failures.append(
                    f"streaming completion: HTTP {stream_response.status_code} "
                    f"{stream_response.text}"
                )
            elif not stream_response.headers["content-type"].startswith("text/event-stream"):
                failures.append(
                    "streaming completion: response was not text/event-stream"
                )
            elif "data: [DONE]" not in stream_response.text:
                failures.append("streaming completion: missing [DONE] event")

        assert not failures, f"{model['id']} failures:\n" + "\n".join(failures)

    asyncio.run(run())

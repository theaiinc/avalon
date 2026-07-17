import asyncio
import json

import httpx
import pytest
from fastapi.responses import StreamingResponse

import api_server


def request_json(client, method, url, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app),
            base_url="http://127.0.0.1",
        ) as async_client:
            return await async_client.request(method, url, **kwargs)

    return asyncio.run(run())


@pytest.fixture
def isolated_gateway(monkeypatch, model_catalog, fake_reservation):
    monkeypatch.setattr(api_server, "get_local_models", lambda: model_catalog)
    monkeypatch.setattr(api_server, "list_remote_models", lambda: [])
    monkeypatch.setattr(api_server, "resolve_remote_model", lambda _model_id: None)
    monkeypatch.setattr(api_server, "get_override", lambda _model_id: {})
    monkeypatch.setattr(api_server, "_default_gguf_backend", lambda: "metal")
    monkeypatch.setattr(
        api_server.resource_manager,
        "acquire",
        lambda *args, **kwargs: fake_reservation,
    )
    api_server._server_config.clear()
    api_server._server_config.update({"port": 8787, "mode": "both"})
    api_server._request_log.clear()
    return model_catalog


@pytest.mark.parametrize("model_index", [0, 1], ids=["gguf", "openvino"])
def test_each_local_model_supports_openai_chat_completion(
    monkeypatch, isolated_gateway, model_index
):
    model = isolated_gateway[model_index]

    def fake_completion(model_path, req, reservation=None):
        assert model_path == api_server._find_model_path(model["id"])
        assert req.model == model["id"]
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"hello from {req.model}"},
                "finish_reason": "stop",
            }],
        }

    target = "_chat_ov" if model["format"] == "openvino" else "_chat_llama"
    monkeypatch.setattr(api_server, target, fake_completion)

    response = request_json(
        None,
        "POST",
        "/v1/chat/completions",
        json={
            "model": model["id"],
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == model["id"]
    assert body["choices"][0]["message"]["content"] == f"hello from {model['id']}"


@pytest.mark.parametrize("model_index", [0, 1], ids=["gguf", "openvino"])
def test_each_local_model_supports_anthropic_messages(
    monkeypatch, isolated_gateway, model_index
):
    model = isolated_gateway[model_index]

    def fake_messages(model_path, req, reservation=None):
        assert model_path == api_server._find_model_path(model["id"])
        return {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": [{"type": "text", "text": f"hello from {req.model}"}],
            "stop_reason": "end_turn",
        }

    target = "_messages_ov" if model["format"] == "openvino" else "_messages_llama"
    monkeypatch.setattr(api_server, target, fake_messages)

    response = request_json(
        None,
        "POST",
        "/v1/messages",
        json={
            "model": model["id"],
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == f"hello from {model['id']}"


@pytest.mark.parametrize("model_index", [0, 1], ids=["gguf", "openvino"])
def test_each_local_model_supports_openai_streaming(
    monkeypatch, isolated_gateway, model_index
):
    model = isolated_gateway[model_index]

    def fake_stream(model_path, req, reservation=None):
        assert model["id"] in model_path

        def events():
            yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"delta":{"content":"hello"}}]}\n\n'
            yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        if model["format"] == "gguf":
            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": model["id"],
            "created": 1,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }],
        }

    target = "_chat_ov" if model["format"] == "openvino" else "_chat_llama"
    monkeypatch.setattr(api_server, target, fake_stream)

    response = request_json(
        None,
        "POST",
        "/v1/chat/completions",
        json={
            "model": model["id"],
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0][6:])["choices"][0]["delta"].get("content")
    assert events[-1] == "data: [DONE]"

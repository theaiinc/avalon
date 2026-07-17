import api_server


def test_chat_prompt_preserves_message_order_and_roles():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello."},
    ]

    assert api_server._format_chat_prompt(messages) == (
        "system: You are concise.\nuser: Say hello.\nassistant:"
    )


def test_thinking_filter_handles_markers_split_across_chunks():
    stream_filter = api_server._ThinkingBlockFilter()

    visible = "".join(
        stream_filter.feed(chunk)
        for chunk in ["answer ", "[Start", " thinking]internal", "[End thinking]", " done"]
    )
    visible += stream_filter.flush()

    assert visible == "answer  done"


def test_mtp_head_is_required_for_mtp_ready(tmp_path):
    model_file = tmp_path / "base.gguf"
    model_file.touch()
    (tmp_path / "MTP").mkdir()
    draft = tmp_path / "MTP" / "draft.gguf"
    draft.touch()

    assert api_server._find_mtp_head(str(model_file)) == str(draft)
    assert api_server._is_mtp_head_artifact(str(model_file), "base model") is False
    assert api_server._is_mtp_head_artifact(str(draft), "mtp") is True


def test_catalog_profiles_distinguish_gguf_and_openvino(monkeypatch, model_catalog):
    monkeypatch.setattr(api_server, "get_local_models", lambda: model_catalog)
    monkeypatch.setattr(api_server, "list_remote_models", lambda: [])
    monkeypatch.setattr(api_server, "get_override", lambda _model_id: {})
    monkeypatch.setattr(api_server, "_default_gguf_backend", lambda: "metal")
    api_server._server_config.clear()
    api_server._server_config.update({"port": 8787, "mode": "openai"})

    catalog = api_server._served_model_catalog()

    assert [model["runtime"] for model in catalog] == ["gguf", "openvino"]
    assert catalog[0]["model_file"].endswith("model.gguf")
    assert catalog[1]["model_file"] == str(model_catalog[1]["path"])
    assert catalog[0]["driver"] == "metal"
    assert catalog[1]["driver"] == "NPU"


def test_model_path_resolves_served_id_for_each_format(monkeypatch, model_catalog):
    monkeypatch.setattr(api_server, "get_local_models", lambda: model_catalog)
    monkeypatch.setattr(api_server, "get_override", lambda _model_id: {})
    monkeypatch.setattr(api_server, "_default_gguf_backend", lambda: "metal")

    assert api_server._find_model_path("gguf-model").endswith("model.gguf")
    assert api_server._find_model_path("openvino-model") == str(model_catalog[1]["path"])


def test_stop_server_refuses_to_interrupt_active_requests(monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(api_server, "_server_process", process)
    monkeypatch.setattr(api_server, "_server_config", {"port": 8787})
    monkeypatch.setattr(
        api_server,
        "_active_gateway_requests",
        lambda _port: [{"model": "busy-model", "status": "streaming"}],
    )

    result = api_server.stop_server()

    assert result["status"] == "busy"
    assert result["active_requests"][0]["model"] == "busy-model"
    assert process.terminated is False

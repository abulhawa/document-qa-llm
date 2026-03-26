import json
import requests
import pytest
from core import llm


@pytest.fixture(autouse=True)
def _force_local_llm_mode(monkeypatch):
    monkeypatch.setattr(llm, "USE_GROQ", False)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")


class DummyResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("boom")


def test_check_llm_status_variants(monkeypatch):
    # 200 with no model
    def get_info(url, timeout):
        return DummyResponse({"model_name": None})
    monkeypatch.setattr(requests, "get", get_info)
    status = llm.check_llm_status()
    assert status["server_online"] is True
    assert status["model_loaded"] is False
    assert status["active"] is False
    assert "No LLM model" in (status.get("status_message") or "")

    # 200 with model
    def get_info_loaded(url, timeout):
        return DummyResponse({"model_name": "m1"})
    monkeypatch.setattr(requests, "get", get_info_loaded)
    status = llm.check_llm_status()
    assert status["server_online"] and status["model_loaded"]
    assert status["active"] is True

    # timeout / error
    def get_fail(url, timeout):
        raise requests.Timeout()
    monkeypatch.setattr(requests, "get", get_fail)
    status = llm.check_llm_status()
    assert status["server_online"] is False
    assert status["active"] is False


def test_ask_llm_payload_and_error(monkeypatch):
    captured = {}

    def fake_post(endpoint, json=None, timeout=None):
        captured["endpoint"] = endpoint
        captured["json"] = json
        if endpoint == llm.LLM_CHAT_ENDPOINT:
            return DummyResponse({"choices": [{"message": {"content": "hi"}}]})
        if endpoint == llm.LLM_COMPLETION_ENDPOINT:
            return DummyResponse({"choices": [{"text": "done"}]})
        return DummyResponse({}, status_code=500)
    monkeypatch.setattr(requests, "post", fake_post)

    chat = llm.ask_llm(
        [{"role": "user", "content": "h"}],
        mode="chat",
        temperature=0.1,
        max_tokens=5,
        use_cache=False,
    )
    assert chat == "hi"
    assert captured["endpoint"] == llm.LLM_CHAT_ENDPOINT
    assert captured["json"]["messages"]
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["max_tokens"] == 5
    assert captured["json"]["stop"] == llm.STOP_TOKENS

    comp = llm.ask_llm("prompt", mode="completion", model="m1", use_cache=False)
    assert comp == "done"
    assert captured["endpoint"] == llm.LLM_COMPLETION_ENDPOINT
    assert captured["json"]["model"] == "m1"
    assert captured["json"]["stop"] == llm.STOP_TOKENS

    def fail_post(*args, **kwargs):
        raise requests.RequestException("x")
    monkeypatch.setattr(requests, "post", fail_post)
    err = llm.ask_llm("p", use_cache=False)
    assert err == "[LLM Error]"


def test_stop_tokens_and_warn_threshold_defaults():
    assert llm.STOP_TOKENS == ["</s>"]
    assert "###" not in llm.STOP_TOKENS
    assert "---" not in llm.STOP_TOKENS
    assert llm.PROMPT_LENGTH_WARN_THRESHOLD == 2000


def test_ask_llm_uses_groq_headers_and_model_selection(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "USE_GROQ", True)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(llm, "GROQ_MODEL", "groq-default")
    monkeypatch.setattr(
        llm, "LLM_CHAT_ENDPOINT", "https://api.groq.test/openai/v1/chat/completions"
    )
    monkeypatch.setattr(
        llm, "LLM_COMPLETION_ENDPOINT", "https://api.groq.test/openai/v1/completions"
    )

    def fake_post(endpoint, json=None, timeout=None, headers=None):
        captured["endpoint"] = endpoint
        captured["json"] = json
        captured["headers"] = headers
        if endpoint == llm.LLM_CHAT_ENDPOINT:
            if json.get("model") == "groq-override-model":
                return DummyResponse(
                    {"choices": [{"message": {"content": "completion-ok"}}]}
                )
            return DummyResponse({"choices": [{"message": {"content": "chat-ok"}}]})
        return DummyResponse({"choices": [{"text": "unexpected"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    chat = llm.ask_llm(
        [{"role": "user", "content": "hello"}], mode="chat", use_cache=False
    )
    assert chat == "chat-ok"
    assert captured["endpoint"] == llm.LLM_CHAT_ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer test-groq-key"
    assert captured["json"]["model"] == "groq-default"

    completion = llm.ask_llm(
        "hello", mode="completion", model="groq-override-model", use_cache=False
    )
    assert completion == "completion-ok"
    assert captured["endpoint"] == llm.LLM_CHAT_ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer test-groq-key"
    assert captured["json"]["model"] == "groq-override-model"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_groq_model_listing_and_status_use_auth_header(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "USE_GROQ", True)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(llm, "GROQ_MODEL", "groq-default")
    monkeypatch.setattr(
        llm, "LLM_MODEL_LIST_ENDPOINT", "https://api.groq.test/openai/v1/models"
    )

    def fake_get(url, timeout, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return DummyResponse({"data": [{"id": "groq-model-1"}, {"id": "groq-model-2"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    models = llm.get_available_models()
    assert models == ["groq-model-1", "groq-model-2"]
    assert captured["url"] == llm.LLM_MODEL_LIST_ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer test-groq-key"

    status = llm.check_llm_status()
    assert status["server_online"] is True
    assert status["active"] is True
    assert status["current_model"] == "groq-default"

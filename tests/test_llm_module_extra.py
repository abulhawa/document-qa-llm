import json
import requests
from core import llm

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

    chat = llm.ask_llm([{"role": "user", "content": "h"}], mode="chat", temperature=0.1, max_tokens=5)
    assert chat == "hi"
    assert captured["endpoint"] == llm.LLM_CHAT_ENDPOINT
    assert captured["json"]["messages"]
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["max_tokens"] == 5

    comp = llm.ask_llm("prompt", mode="completion", model="m1")
    assert comp == "done"
    assert captured["endpoint"] == llm.LLM_COMPLETION_ENDPOINT
    assert captured["json"]["model"] == "m1"

    def fail_post(*args, **kwargs):
        raise requests.RequestException("x")
    monkeypatch.setattr(requests, "post", fail_post)
    err = llm.ask_llm("p")
    assert err == "[LLM Error]"

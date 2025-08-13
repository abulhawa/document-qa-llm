import json
import requests
import pytest
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
            raise requests.HTTPError("error")

def test_get_available_models(monkeypatch):
    def fake_get(url, timeout):
        return DummyResponse({"model_names": ["m1", "m2"]})
    monkeypatch.setattr(requests, "get", fake_get)
    models = llm.get_available_models()
    assert models == ["m1", "m2"]

def test_get_available_models_error(monkeypatch):
    def fake_get(url, timeout):
        raise requests.RequestException("boom")
    monkeypatch.setattr(requests, "get", fake_get)
    models = llm.get_available_models()
    assert models == []

def test_ask_llm_chat_and_completion(monkeypatch):
    def fake_post(endpoint, json=None, timeout=None):
        if endpoint == llm.LLM_CHAT_ENDPOINT:
            return DummyResponse({"choices": [{"message": {"content": "hi</s>"}}]})
        return DummyResponse({"choices": [{"text": "hello"}]})
    monkeypatch.setattr(requests, "post", fake_post)
    chat = llm.ask_llm([{"role": "user", "content": "hi"}], mode="chat")
    assert chat.startswith("hi")
    comp = llm.ask_llm("test prompt", mode="completion")
    assert comp == "hello"

def test_ask_llm_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.RequestException("fail")
    monkeypatch.setattr(requests, "post", fake_post)
    result = llm.ask_llm("prompt")
    assert result == "[LLM Error]"

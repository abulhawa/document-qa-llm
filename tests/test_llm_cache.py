import json
from typing import Any, Dict

import requests

from core import llm
from core import llm_cache
from opensearchpy import exceptions


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


class FakeIndices:
    def __init__(self, store: Dict[str, Any]):
        self._store = store

    def exists(self, index):
        return index in self._store["indices"]

    def create(self, index, body=None, params=None):
        self._store["indices"].add(index)
        return {"acknowledged": True}


class FakeOpenSearch:
    def __init__(self):
        self._store: Dict[str, Any] = {"indices": set(), "docs": {}}
        self.indices = FakeIndices(self._store)

    def get(self, index, id):
        try:
            doc = self._store["docs"][(index, id)]
        except KeyError:
            raise exceptions.NotFoundError(404, "not_found", {}) from None
        return {"_source": doc}

    def index(self, index, id, body):
        self._store["docs"][(index, id)] = body
        return {"result": "created"}

    def update(self, index, id, body):
        key = (index, id)
        if key not in self._store["docs"]:
            raise exceptions.NotFoundError(404, "not_found", {})
        source = self._store["docs"][key]
        params = body.get("script", {}).get("params", {})
        source["hit_count"] = source.get("hit_count", 0) + params.get("count", 0)
        source["last_access_at"] = params.get("ts", source.get("last_access_at"))
        return {"result": "updated"}


def _enable_cache(monkeypatch, fake_client):
    monkeypatch.setattr(llm_cache, "get_client", lambda: fake_client)
    monkeypatch.setattr(llm_cache, "LLM_CACHE_BACKEND", "opensearch")
    monkeypatch.setattr(llm_cache, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(llm_cache, "LLM_CACHE_INDEX", "llm_cache_v1")
    monkeypatch.setattr(llm_cache, "LLM_CACHE_STORE_PROMPT_TEXT", False)
    monkeypatch.setattr(llm_cache, "LLM_CACHE_TTL_DAYS", 30)
    llm_cache._cache_unavailable = False
    llm_cache._warned_unavailable = False
    llm_cache._index_ready = False


def test_llm_cache_hit(monkeypatch):
    fake_client = FakeOpenSearch()
    _enable_cache(monkeypatch, fake_client)

    calls = {"count": 0}

    def fake_post(endpoint, json=None, timeout=None):
        calls["count"] += 1
        return DummyResponse({"choices": [{"text": "cached"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    first = llm.ask_llm("hello", mode="completion", temperature=0.3, max_tokens=10)
    second = llm.ask_llm("hello", mode="completion", temperature=0.3, max_tokens=10)

    assert first == "cached"
    assert second == "cached"
    assert calls["count"] == 1


def test_llm_cache_miss_on_param_change(monkeypatch):
    fake_client = FakeOpenSearch()
    _enable_cache(monkeypatch, fake_client)

    calls = {"count": 0}

    def fake_post(endpoint, json=None, timeout=None):
        calls["count"] += 1
        return DummyResponse({"choices": [{"text": f"resp{calls['count']}"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    first = llm.ask_llm("hello", mode="completion", temperature=0.2, max_tokens=10)
    second = llm.ask_llm("hello", mode="completion", temperature=0.4, max_tokens=10)

    assert first == "resp1"
    assert second == "resp2"
    assert calls["count"] == 2


def test_llm_cache_disabled(monkeypatch):
    fake_client = FakeOpenSearch()
    _enable_cache(monkeypatch, fake_client)
    monkeypatch.setattr(llm_cache, "LLM_CACHE_ENABLED", False)

    calls = {"count": 0}

    def fake_post(endpoint, json=None, timeout=None):
        calls["count"] += 1
        return DummyResponse({"choices": [{"text": "nocache"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    llm.ask_llm("hello", mode="completion", temperature=0.3, max_tokens=10)
    llm.ask_llm("hello", mode="completion", temperature=0.3, max_tokens=10)

    assert calls["count"] == 2


def test_llm_cache_opensearch_failure(monkeypatch):
    def fail_client():
        raise exceptions.OpenSearchException("down")

    monkeypatch.setattr(llm_cache, "get_client", fail_client)
    monkeypatch.setattr(llm_cache, "LLM_CACHE_BACKEND", "opensearch")
    monkeypatch.setattr(llm_cache, "LLM_CACHE_ENABLED", True)
    llm_cache._cache_unavailable = False
    llm_cache._warned_unavailable = False
    llm_cache._index_ready = False

    calls = {"count": 0}

    def fake_post(endpoint, json=None, timeout=None):
        calls["count"] += 1
        return DummyResponse({"choices": [{"text": "ok"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    result = llm.ask_llm("hello", mode="completion", temperature=0.3, max_tokens=10)
    assert result == "ok"
    assert calls["count"] == 1

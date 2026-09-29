import pytest
import requests

import core.embeddings as embeddings
from core.embeddings import embed_texts


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def test_embed_texts_defaults_to_passage_and_sends_role(monkeypatch):
    captured = {}

    class DummySession:
        def post(self, url, json, timeout):
            captured["json"] = json
            return DummyResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr(embeddings, "_session", DummySession())
    result = embed_texts(["hello"], batch_size=1)
    assert result == [[0.1, 0.2]]
    assert captured["json"] == {
        "texts": ["hello"],
        "batch_size": 1,
        "input_type": "passage",
    }


def test_embed_texts_can_request_query_role(monkeypatch):
    captured = {}

    class DummySession:
        def post(self, url, json, timeout):
            captured["json"] = json
            return DummyResponse({"embeddings": [[0.3, 0.4]]})

    monkeypatch.setattr(embeddings, "_session", DummySession())
    result = embed_texts(["question"], input_type="query")
    assert result == [[0.3, 0.4]]
    assert captured["json"]["input_type"] == "query"


def test_embed_texts_rejects_unknown_role():
    with pytest.raises(ValueError):
        embed_texts(["hello"], input_type="other")  # type: ignore[arg-type]


def test_embed_texts_failure(monkeypatch):
    class FailingSession:
        def post(self, url, json, timeout):
            raise requests.RequestException("boom")

    monkeypatch.setattr(embeddings, "_session", FailingSession())
    with pytest.raises(requests.RequestException):
        embed_texts(["hi"])

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


def test_embed_texts_success(monkeypatch):
    class DummySession:
        def post(self, url, json, timeout):
            return DummyResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr(embeddings, "_session", DummySession())
    result = embed_texts(["hello"], batch_size=1)
    assert result == [[0.1, 0.2]]


def test_embed_texts_failure(monkeypatch):
    class FailingSession:
        def post(self, url, json, timeout):
            raise requests.RequestException("boom")

    monkeypatch.setattr(embeddings, "_session", FailingSession())
    with pytest.raises(requests.RequestException):
        embed_texts(["hi"])

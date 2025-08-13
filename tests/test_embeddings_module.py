import pytest
from core.embeddings import embed_texts
import requests

class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data
    def raise_for_status(self):
        pass


def test_embed_texts_success(monkeypatch):
    def fake_post(url, json, timeout):
        return DummyResponse({"embeddings": [[0.1, 0.2]]})
    monkeypatch.setattr(requests, "post", fake_post)
    result = embed_texts(["hello"], batch_size=1)
    assert result == [[0.1, 0.2]]


def test_embed_texts_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.RequestException("boom")
    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(RuntimeError):
        embed_texts(["hi"])

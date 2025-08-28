import types
import importlib
import sys
from unittest.mock import MagicMock
import pytest

# Remove stubbed qdrant_client if present and import the real one
if isinstance(sys.modules.get("qdrant_client"), types.SimpleNamespace):
    del sys.modules["qdrant_client"]
    sys.modules.pop("qdrant_client.http", None)
    sys.modules.pop("qdrant_client.http.models", None)
import qdrant_client
sys.modules["qdrant_client"] = qdrant_client
sys.modules["qdrant_client.http"] = qdrant_client.http
sys.modules["qdrant_client.http.models"] = qdrant_client.http.models

from utils import qdrant_utils as qdu
importlib.reload(qdu)


class DummyResult:
    def __init__(self, count):
        self.count = count

def test_ensure_collection_exists_creates(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_collections.return_value = types.SimpleNamespace(
        collections=[types.SimpleNamespace(name="other")]
    )
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "QDRANT_COLLECTION", "col")

    qdu.ensure_collection_exists()

    mock_client.create_collection.assert_called_once()


def test_ensure_collection_exists_noop(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_collections.return_value = types.SimpleNamespace(
        collections=[types.SimpleNamespace(name="col")]
    )
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "QDRANT_COLLECTION", "col")

    qdu.ensure_collection_exists()

    mock_client.create_collection.assert_not_called()


def test_index_chunks_success(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    def fake_embed(texts):
        return [[i, i + 1, i + 2] for i, _ in enumerate(texts)]

    monkeypatch.setattr(qdu, "embed_texts", fake_embed)

    chunks = [
        {"id": 1, "text": "a", "extra": "x"},
        {"id": 2, "text": "b"},
    ]

    assert qdu.index_chunks(chunks) is True
    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 2


def test_index_chunks_embedding_failure(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    def fail_embed(texts):
        raise RuntimeError("fail")

    monkeypatch.setattr(qdu, "embed_texts", fail_embed)

    chunks = [{"id": 1, "text": "a"}]

    with pytest.raises(RuntimeError):
        qdu.index_chunks(chunks)
    mock_client.upsert.assert_not_called()


def test_index_chunks_upsert_failure(monkeypatch):
    mock_client = MagicMock()
    mock_client.upsert.side_effect = Exception("boom")
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "embed_texts", lambda texts: [[0, 0, 0] for _ in texts])

    chunks = [{"id": 1, "text": "a"}]

    with pytest.raises(Exception):
        qdu.index_chunks(chunks)


def test_count_qdrant_chunks_by_path(monkeypatch):
    mock_client = MagicMock()
    mock_client.count.return_value = DummyResult(3)
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "QDRANT_COLLECTION", "col")

    assert qdu.count_qdrant_chunks_by_path("/p") == 3
    call = mock_client.count.call_args
    assert call.kwargs["collection_name"] == "col"
    flt = call.kwargs["count_filter"]
    cond = flt.must[0]
    assert cond.key == "path"
    assert cond.match.value == "/p"


def test_count_qdrant_chunks_by_path_failure(monkeypatch):
    mock_client = MagicMock()
    mock_client.count.side_effect = Exception("fail")
    monkeypatch.setattr(qdu, "client", mock_client)

    assert qdu.count_qdrant_chunks_by_path("/p") is None


def test_delete_vectors_by_ids(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    ids = ["a", "b", "c"]
    assert qdu.delete_vectors_by_ids(ids) == 3

    call = mock_client.delete.call_args
    assert call.kwargs["collection_name"] == qdu.QDRANT_COLLECTION
    assert list(call.kwargs["points_selector"].points) == ids


def test_delete_vectors_by_ids_empty(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    assert qdu.delete_vectors_by_ids([]) == 0
    mock_client.delete.assert_not_called()


def test_delete_vectors_by_ids_failure(monkeypatch):
    mock_client = MagicMock()
    mock_client.delete.side_effect = Exception("boom")
    monkeypatch.setattr(qdu, "client", mock_client)

    with pytest.raises(Exception):
        qdu.delete_vectors_by_ids(["a"])

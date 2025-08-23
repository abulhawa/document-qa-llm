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
    ec = MagicMock()
    monkeypatch.setattr(qdu, "ensure_collection_exists", ec)
    monkeypatch.setattr(qdu, "EMBEDDING_SIZE", 3)

    def fake_embed(texts):
        return [[i, i + 1, i + 2] for i, _ in enumerate(texts)]

    monkeypatch.setattr(qdu, "embed_texts", fake_embed)

    chunks = [
        {"id": 1, "text": "a", "has_embedding": True, "extra": "x"},
        {"id": 2, "text": "b", "has_embedding": False},
    ]

    resp = qdu.index_chunks(chunks)
    assert resp["upserted"] == 2
    ec.assert_called_once()
    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert all("has_embedding" not in p.payload for p in points)


def test_index_chunks_embedding_failure(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "ensure_collection_exists", MagicMock())

    def fail_embed(texts):
        raise RuntimeError("fail")

    monkeypatch.setattr(qdu, "embed_texts", fail_embed)

    chunks = [{"id": 1, "text": "a"}]

    with pytest.raises(Exception):
        qdu.index_chunks(chunks)
    mock_client.upsert.assert_not_called()


def test_index_chunks_upsert_failure(monkeypatch):
    mock_client = MagicMock()
    mock_client.upsert.side_effect = Exception("boom")
    monkeypatch.setattr(qdu, "client", mock_client)
    monkeypatch.setattr(qdu, "ensure_collection_exists", MagicMock())
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


def test_delete_vectors_by_checksum(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    qdu.delete_vectors_by_checksum("abc")

    call = mock_client.delete.call_args
    assert call.kwargs["collection_name"] == qdu.QDRANT_COLLECTION
    filt = call.kwargs["points_selector"].filter
    cond = filt.must[0]
    assert cond.key == "checksum"
    assert cond.match.value == "abc"


def test_delete_vectors_by_checksum_handles_error(monkeypatch):
    mock_client = MagicMock()
    mock_client.delete.side_effect = Exception("boom")
    monkeypatch.setattr(qdu, "client", mock_client)

    qdu.delete_vectors_by_checksum("abc")  # should not raise


def test_delete_vectors_many_by_checksum(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    checksums = [f"c{i}" for i in range(70)] + [None, "", "c1"]
    qdu.delete_vectors_many_by_checksum(checksums)

    assert mock_client.delete.call_count == 2
    lengths = [
        len(call.kwargs["points_selector"].filter.must[0].match.any)
        for call in mock_client.delete.call_args_list
    ]
    assert sorted(lengths) == [6, 64]


def test_delete_vectors_many_by_checksum_empty(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    qdu.delete_vectors_many_by_checksum([None, ""])

    mock_client.delete.assert_not_called()


def test_delete_vectors_by_path_checksum(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdu, "client", mock_client)

    pairs = [
        ("/a", "1"),
        ("/a", "1"),  # duplicate
        ("/b", "2"),
        ("", "3"),
        ("/c", None),
    ]

    qdu.delete_vectors_by_path_checksum(pairs)

    assert mock_client.delete.call_count == 2
    called = [
        {c.key: c.match.value for c in call.kwargs["points_selector"].filter.must}
        for call in mock_client.delete.call_args_list
    ]
    expected = [{"path": "/a", "checksum": "1"}, {"path": "/b", "checksum": "2"}]
    assert {tuple(sorted(d.items())) for d in called} == {
        tuple(sorted(e.items())) for e in expected
    }


def test_delete_vectors_by_path_checksum_handles_error(monkeypatch):
    mock_client = MagicMock()
    mock_client.delete.side_effect = Exception("boom")
    monkeypatch.setattr(qdu, "client", mock_client)

    pairs = [("/a", "1"), ("/b", "2")]
    qdu.delete_vectors_by_path_checksum(pairs)  # should not raise

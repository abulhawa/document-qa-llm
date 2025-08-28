import types
import math
import pytest

from core import vector_store
from utils import qdrant_utils


class FakeQdrant:
    def __init__(self):
        self.points = {}
        self.collections = set()

    # Collection management
    def get_collections(self):
        cols = [types.SimpleNamespace(name=n) for n in self.collections]
        return types.SimpleNamespace(collections=cols)

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    # Upsert points
    def upsert(self, collection_name, points, **kwargs):
        for p in points:
            self.points[p.id] = p

    # Search using cosine similarity
    def search(self, collection_name, query_vector, limit, score_threshold, with_payload):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = []
        for p in self.points.values():
            score = cosine(query_vector, p.vector)
            if score >= score_threshold:
                scored.append(types.SimpleNamespace(payload=p.payload, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    def delete(self, collection_name, points_selector, **kwargs):
        if hasattr(points_selector, "filter"):
            flt = points_selector.filter.must[0]
            checksum = flt.match.value
            to_del = [
                pid for pid, p in self.points.items() if p.payload.get("checksum") == checksum
            ]
        else:
            to_del = list(points_selector.points)
        for pid in to_del:
            self.points.pop(pid, None)


# Fixture to set up fake Qdrant and embedding
@pytest.fixture(autouse=True)
def setup_fake_qdrant(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(qdrant_utils, "client", fake)
    monkeypatch.setattr(vector_store, "client", fake)
    monkeypatch.setattr(qdrant_utils, "ensure_collection_exists", lambda: None)
    # simple PointStruct replacement (qdrant_client may be stubbed by other tests)
    class P:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    monkeypatch.setattr(qdrant_utils, "PointStruct", P)

    mapping = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "c": [-1.0, 0.0],
        "q": [1.0, 0.0],
    }

    def fake_embed(texts):
        return [mapping.get(t[0].lower(), [0.0, 0.0]) for t in texts]

    monkeypatch.setattr(qdrant_utils, "embed_texts", fake_embed)
    monkeypatch.setattr(vector_store, "embed_texts", fake_embed)
    yield fake


# Helper to build chunk

def make_chunk(text, idx, checksum=None):
    return {
        "id": f"id{idx}",
        "text": text,
        "checksum": checksum or f"cs{idx}",
        "path": f"p{idx}",
        "chunk_index": idx,
        "modified_at": "2024-01-01",
    }


# Test 10: index + retrieve happy path

def test_vector_index_and_retrieve(setup_fake_qdrant):
    chunks = [make_chunk("a", 0), make_chunk("b", 1), make_chunk("c", 2)]
    assert qdrant_utils.index_chunks(chunks) is True
    # lower threshold so second vector is included
    vector_store.CHUNK_SCORE_THRESHOLD = 0.0
    results = vector_store.retrieve_top_k("q", top_k=2)
    assert [r["id"] for r in results] == ["id0", "id1"]


# Test 11: scores are in non-increasing order

def test_vector_score_order(setup_fake_qdrant):
    chunks = [make_chunk("a", 0), make_chunk("b", 1)]
    qdrant_utils.index_chunks(chunks)
    results = vector_store.retrieve_top_k("q", top_k=2)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# Test 12: threshold filtering excludes low scores

def test_vector_threshold_filter(monkeypatch, setup_fake_qdrant):
    chunks = [make_chunk("a", 0), make_chunk("b", 1)]
    qdrant_utils.index_chunks(chunks)
    vector_store.CHUNK_SCORE_THRESHOLD = 0.9
    results = vector_store.retrieve_top_k("q", top_k=5)
    # Only exact match 'a' should remain
    assert [r["id"] for r in results] == ["id0"]


# Test 13: delete by ids removes points

def test_vector_delete_by_ids(setup_fake_qdrant):
    chunks = [make_chunk("a", 0), make_chunk("b", 1)]
    qdrant_utils.index_chunks(chunks)
    qdrant_utils.delete_vectors_by_ids(["id0", "id1"])
    results = vector_store.retrieve_top_k("q", top_k=5)
    assert results == []


# Test 14: retrieving from empty collection returns empty list

def test_vector_empty_collection(monkeypatch, setup_fake_qdrant):
    results = vector_store.retrieve_top_k("q", top_k=5)
    assert results == []


# Test 15: search exception handled gracefully

def test_vector_search_exception(monkeypatch, setup_fake_qdrant):
    class Boom(FakeQdrant):
        def search(self, *args, **kwargs):
            raise RuntimeError("fail")

    boom = Boom()
    monkeypatch.setattr(vector_store, "client", boom)
    results = vector_store.retrieve_top_k("q", top_k=5)
    assert results == []

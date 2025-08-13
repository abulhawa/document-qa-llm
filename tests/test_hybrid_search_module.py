import pytest

from core import hybrid_search

# Test 1: Fusion with both sources present and ranking by fused score

def test_hybrid_fusion_ranking(monkeypatch):
    vector_results = [
        {
            "id": "v1",
            "text": "doc1",
            "score": 0.9,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "c1",
        },
        {
            "id": "v2",
            "text": "doc2",
            "score": 0.9,
            "path": "p2",
            "chunk_index": 0,
            "modified_at": "2024-01-02",
            "checksum": "c2",
        },
    ]

    bm25_results = [
        {
            "_id": "v2",
            "text": "doc2",
            "score": 0.4,
            "path": "p2",
            "chunk_index": 0,
            "modified_at": "2024-01-02",
            "checksum": "c2",
        },
        {
            "_id": "b3",
            "text": "doc3",
            "score": 0.7,
            "path": "p3",
            "chunk_index": 0,
            "modified_at": "2024-01-03",
            "checksum": "c3",
        },
    ]

    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(
        hybrid_search, "keyword_retriever", lambda q, top_k: bm25_results
    )

    results = hybrid_search.retrieve_hybrid("query", top_k_each=2, final_k=3)
    # Expected order: doc2 (has both scores), doc1 (semantic only), doc3 (bm25 only)
    assert [r["path"] for r in results] == ["p2", "p1", "p3"]
    assert len(results) == 3


# Test 2: Only BM25 available

def test_hybrid_bm25_only(monkeypatch):
    bm25_results = [
        {
            "_id": "b1",
            "text": "bm25",
            "score": 1.0,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "c1",
        }
    ]

    monkeypatch.setattr(hybrid_search, "semantic_retriever", lambda q, top_k: [])
    monkeypatch.setattr(
        hybrid_search, "keyword_retriever", lambda q, top_k: bm25_results
    )
    results = hybrid_search.retrieve_hybrid("query", top_k_each=1, final_k=5)
    assert [r["path"] for r in results] == ["p1"]


# Test 3: Only vector available

def test_hybrid_vector_only(monkeypatch):
    vector_results = [
        {
            "id": "v1",
            "text": "vec",
            "score": 1.0,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "c1",
        }
    ]

    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(hybrid_search, "keyword_retriever", lambda q, top_k: [])
    results = hybrid_search.retrieve_hybrid("query", top_k_each=1, final_k=5)
    assert [r["id"] for r in results] == ["v1"]


# Test 4: Checksum dedup keeps highest score

def test_hybrid_checksum_dedup(monkeypatch):
    vector_results = [
        {
            "id": "v1",
            "text": "doc",
            "score": 0.9,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "dup",
        }
    ]
    bm25_results = [
        {
            "_id": "b1",
            "text": "doc",
            "score": 0.8,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "dup",
        }
    ]
    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(
        hybrid_search, "keyword_retriever", lambda q, top_k: bm25_results
    )
    results = hybrid_search.retrieve_hybrid("query", top_k_each=1, final_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "v1"


# Test 5: Secondary sort by modified_at when scores tie

def test_hybrid_sort_modified_at(monkeypatch):
    vector_results = [
        {
            "id": "v1",
            "text": "old",
            "score": 1.0,
            "path": "p1",
            "chunk_index": 0,
            "modified_at": "2023-01-01",
            "checksum": "c1",
        },
        {
            "id": "v2",
            "text": "new",
            "score": 1.0,
            "path": "p2",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "c2",
        },
    ]
    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(hybrid_search, "keyword_retriever", lambda q, top_k: [])
    results = hybrid_search.retrieve_hybrid("query", top_k_each=2, final_k=2)
    assert [r["id"] for r in results] == ["v2", "v1"]


# Test 7: Empty or whitespace query returns empty list

def test_hybrid_empty_query(monkeypatch):
    calls = {"vec": 0, "bm25": 0}

    def vec(q, top_k):
        calls["vec"] += 1
        return []

    def bm25(q, top_k):
        calls["bm25"] += 1
        return []

    monkeypatch.setattr(hybrid_search, "semantic_retriever", vec)
    monkeypatch.setattr(hybrid_search, "keyword_retriever", bm25)

    results = hybrid_search.retrieve_hybrid("   ")
    assert results == []
    # retrieval functions still called but return empty
    assert calls["vec"] == 1 and calls["bm25"] == 1


# Test 8: BM25 receives original query tokens

def test_hybrid_bm25_guardrail(monkeypatch):
    received = {}

    def bm25(q, top_k):
        received["query"] = q
        return []

    monkeypatch.setattr(hybrid_search, "semantic_retriever", lambda q, top_k: [])
    monkeypatch.setattr(hybrid_search, "keyword_retriever", bm25)
    hybrid_search.retrieve_hybrid("original terms")
    assert received["query"] == "original terms"


# Test 9: final_k enforcement after dedup

def test_hybrid_final_k_after_dedup(monkeypatch):
    vector_results = [
        {
            "id": f"v{i}",
            "text": f"doc{i}",
            "score": 1.0,
            "path": f"p{i}",
            "chunk_index": 0,
            "modified_at": "2024-01-01",
            "checksum": "dup" if i % 2 == 0 else f"c{i}",
        }
        for i in range(4)
    ]
    bm25_results = []
    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(hybrid_search, "keyword_retriever", lambda q, top_k: [])
    results = hybrid_search.retrieve_hybrid("q", top_k_each=4, final_k=2)
    assert len(results) <= 2

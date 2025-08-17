import pytest

from core import hybrid_search

# Test 1: Fusion with both sources present and ranking by fused score

def test_hybrid_fusion_ranking(monkeypatch):
    vector_results = [
        {
            "doc_id": "d1",
            "text": "doc1",
            "score": 0.9,
            "path": "p1",
            "checksum": "c1",
        },
        {
            "doc_id": "d2",
            "text": "doc2",
            "score": 0.9,
            "path": "p2",
            "checksum": "c2",
        },
    ]

    bm25_results = [
        {
            "doc_id": "d2",
            "path": "p2",
            "score": 0.4,
            "checksum": "c2",
            "chunks": [],
        },
        {
            "doc_id": "d3",
            "path": "p3",
            "score": 0.5,
            "checksum": "c3",
            "chunks": [],
        },
    ]

    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(
        hybrid_search, "keyword_retriever", lambda q, top_k: bm25_results
    )

    results = hybrid_search.retrieve_hybrid("query", top_k_each=2, final_k=3)
    # Expected order: doc2 (both), doc1 (semantic), doc3 (bm25)
    assert [r["path"] for r in results] == ["p2", "p1", "p3"]
    assert len(results) == 3


# Test 2: Only BM25 available

def test_hybrid_bm25_only(monkeypatch):
    bm25_results = [
        {
            "doc_id": "d1",
            "path": "p1",
            "score": 1.0,
            "checksum": "c1",
            "chunks": [],
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
            "doc_id": "d1",
            "text": "vec",
            "score": 1.0,
            "path": "p1",
            "checksum": "c1",
        }
    ]

    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(hybrid_search, "keyword_retriever", lambda q, top_k: [])
    results = hybrid_search.retrieve_hybrid("query", top_k_each=1, final_k=5)
    assert [r["doc_id"] for r in results] == ["d1"]


# Test 4: Checksum dedup keeps highest score

def test_hybrid_checksum_dedup(monkeypatch):
    vector_results = [
        {
            "doc_id": "d1",
            "text": "doc",
            "score": 0.9,
            "path": "p1",
            "checksum": "dup",
        }
    ]
    bm25_results = [
        {
            "doc_id": "d1",
            "path": "p1",
            "score": 0.8,
            "checksum": "dup",
            "chunks": [],
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
    assert results[0]["doc_id"] == "d1"


# Test 5: Secondary sort by modified_at when scores tie

def test_hybrid_sort_modified_at(monkeypatch):
    vector_results = [
        {
            "doc_id": "d1",
            "text": "old",
            "score": 1.0,
            "path": "p1",
            "modified_at": "2023-01-01",
            "checksum": "c1",
        },
        {
            "doc_id": "d2",
            "text": "new",
            "score": 1.0,
            "path": "p2",
            "modified_at": "2024-01-01",
            "checksum": "c2",
        },
    ]
    monkeypatch.setattr(
        hybrid_search, "semantic_retriever", lambda q, top_k: vector_results
    )
    monkeypatch.setattr(hybrid_search, "keyword_retriever", lambda q, top_k: [])
    results = hybrid_search.retrieve_hybrid("query", top_k_each=2, final_k=2)
    assert [r["doc_id"] for r in results] == ["d2", "d1"]


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
    # guard: empty query short-circuits without calling retrievers
    assert calls["vec"] == 0 and calls["bm25"] == 0


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
            "doc_id": f"d{i}",
            "text": f"doc{i}",
            "score": 1.0,
            "path": f"p{i}",
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

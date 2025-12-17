import sys
import sys
import types
from typing import List, Dict

import pytest


class _DummySession:
    def mount(self, *args, **kwargs):
        return None

    def post(self, *args, **kwargs):
        return types.SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"embeddings": []}
        )


sys.modules.setdefault("requests", types.SimpleNamespace(Session=_DummySession))
sys.modules.setdefault(
    "requests.adapters",
    types.SimpleNamespace(
        HTTPAdapter=type("HTTPAdapter", (), {"__init__": lambda self, *args, **kwargs: None})
    ),
)
sys.modules.setdefault(
    "urllib3.util.retry", types.SimpleNamespace(Retry=lambda **kwargs: None)
)
sys.modules.setdefault(
    "opensearchpy",
    types.SimpleNamespace(OpenSearch=type("OpenSearch", (), {}), RequestsHttpConnection=object),
)
sys.modules.setdefault(
    "qdrant_client",
    types.SimpleNamespace(
        QdrantClient=type(
            "QdrantClient",
            (),
            {
                "__init__": lambda self, *args, **kwargs: None,
                "get_collections": lambda self: types.SimpleNamespace(collections=[]),
                "create_collection": lambda self, *args, **kwargs: None,
                "search": lambda self, **kwargs: [],
            },
        ),
        models=types.SimpleNamespace(),
    ),
)
sys.modules.setdefault(
    "qdrant_client.http.models",
    types.SimpleNamespace(
        PointStruct=type("PointStruct", (), {}),
        PointIdsList=type("PointIdsList", (), {}),
        VectorParams=type("VectorParams", (), {}),
        Distance=types.SimpleNamespace(COSINE="cosine"),
    ),
)
def _mmr_select(query, docs, embed, k=1, lambda_mult=0.5):
    if embed:
        try:
            embed([query] + [d.get("text", "") for d in docs])
        except Exception:
            pass
    return docs[:k]


sys.modules.setdefault(
    "core.retrieval.mmr", types.SimpleNamespace(mmr_select=_mmr_select)
)
sys.modules.setdefault(
    "core.retrieval.dedup",
    types.SimpleNamespace(
        collapse_near_duplicates=lambda docs, embed_texts, sim_threshold=0.9, keep_limit=64: (docs, []),
    ),
)


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def set_attribute(self, *_, **__):
        return None

    def set_status(self, *_, **__):
        return None


tracing_module = types.ModuleType("tracing")
tracing_module.start_span = lambda *_, **__: _Span()
tracing_module.record_span_error = lambda *_, **__: None
tracing_module.STATUS_OK = "OK"
tracing_module.EMBEDDING = "EMBEDDING"
tracing_module.RETRIEVER = "RETRIEVER"
tracing_module.INPUT_VALUE = "INPUT"
tracing_module.OUTPUT_VALUE = "OUTPUT"
tracing_module.LLM = "LLM"
tracing_module.CHAIN = "CHAIN"
tracing_module.TOOL = "TOOL"
tracing_module.get_current_span = lambda *_, **__: _Span()
sys.modules["tracing"] = tracing_module


class _DummySession:
    def mount(self, *args, **kwargs):
        return None

    def post(self, *args, **kwargs):
        return types.SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"embeddings": []}
        )


sys.modules.setdefault("requests", types.SimpleNamespace(Session=_DummySession))
sys.modules.setdefault(
    "requests.adapters",
    types.SimpleNamespace(
        HTTPAdapter=type("HTTPAdapter", (), {"__init__": lambda self, *args, **kwargs: None})
    ),
)
sys.modules.setdefault(
    "urllib3.util.retry", types.SimpleNamespace(Retry=lambda **kwargs: None)
)

from core.retrieval import pipeline
from core.retrieval.types import RetrievalConfig, RetrievalDeps


@pytest.fixture(autouse=True)
def isolate_variants(monkeypatch):
    # Prevent real rewriting side-effects
    monkeypatch.setattr(pipeline, "generate_variants", lambda q: {"variants": [(q, 1.0)]})


def _build_deps(vector_hits: List[Dict], bm25_hits: List[Dict], embedder=None, reranker=None):
    return RetrievalDeps(
        semantic_retriever=lambda q, top_k: vector_hits,
        keyword_retriever=lambda q, top_k: bm25_hits,
        embed_texts=embedder,
        cross_encoder=reranker,
    )


def test_retrieval_respects_top_k_and_weights():
    vector_hits = [
        {"id": "v1", "path": "a", "text": "a", "score": 1.0, "checksum": "c1"},
        {"id": "v2", "path": "b", "text": "b", "score": 0.5, "checksum": "c2"},
    ]
    bm25_hits = [
        {"_id": "b1", "path": "c", "text": "c", "score": 1.0, "checksum": "b1"},
        {"_id": "v1", "path": "a", "text": "a", "score": 0.8, "checksum": "c1"},
    ]
    cfg = RetrievalConfig(top_k=1, top_k_each=2, fusion_weight_vector=0.5, fusion_weight_bm25=0.5)
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, bm25_hits))

    assert result.clarify is None
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc["path"] == "a"
    assert doc["retrieval_score"] > 0


def test_retrieval_handles_clarify(monkeypatch):
    monkeypatch.setattr(pipeline, "generate_variants", lambda q: {"clarify": "need more"})
    cfg = RetrievalConfig(top_k=2)
    result = pipeline.retrieve("question", cfg=cfg, deps=_build_deps([], []))

    assert result.clarify == "need more"
    assert result.documents == []


def test_retrieval_uses_mmr_when_enabled():
    calls = {"embed": 0}

    def fake_embed(texts):
        calls["embed"] += 1
        # deterministic 2d vectors
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 0.9, "checksum": "m1"},
        {"id": "v2", "text": "doc2", "score": 0.8, "checksum": "m2"},
        {"id": "v3", "text": "doc3", "score": 0.7, "checksum": "m3"},
    ]
    cfg = RetrievalConfig(top_k=2, enable_mmr=True)
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, [], embedder=fake_embed))

    assert len(result.documents) == 2


def test_retrieval_supports_rerank_hook():
    class DummyReranker:
        def __init__(self):
            self.seen = []

        def rerank(self, query, docs, top_n=None):  # noqa: D401 - simple stub
            self.seen.append((query, len(docs), top_n))
            return list(reversed(docs))

    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 0.9, "checksum": "r1"},
        {"id": "v2", "text": "doc2", "score": 0.8, "checksum": "r2"},
    ]
    cfg = RetrievalConfig(top_k=2, enable_rerank=True, rerank_top_n=1)
    reranker = DummyReranker()
    result = pipeline.retrieve(
        "query", cfg=cfg, deps=_build_deps(vector_hits, [], reranker=reranker)
    )

    assert reranker.seen
    assert result.documents[0]["id"] == "v2"

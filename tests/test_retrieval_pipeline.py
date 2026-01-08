import sys
import types
from types import ModuleType
from typing import Callable, Dict, List

import pytest


class _DummySession:
    def mount(self, *args, **kwargs):
        return None

    def post(self, *args, **kwargs):
        return types.SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"embeddings": []}
        )


class _RequestsModule(ModuleType):
    Session: type[_DummySession]


class _RequestsAdaptersModule(ModuleType):
    HTTPAdapter: type


class _Urllib3RetryModule(ModuleType):
    Retry: Callable[..., None]


class _OpenSearchModule(ModuleType):
    OpenSearch: type
    RequestsHttpConnection: type


class _QdrantModule(ModuleType):
    QdrantClient: type
    models: ModuleType


class _QdrantHttpModelsModule(ModuleType):
    PointStruct: type
    PointIdsList: type
    VectorParams: type
    Distance: ModuleType


class _QdrantDistanceModule(ModuleType):
    COSINE: str


class _MmrModule(ModuleType):
    mmr_select: Callable[..., list]


class _DedupModule(ModuleType):
    collapse_near_duplicates: Callable[..., tuple]


requests_module = _RequestsModule("requests")
requests_module.Session = _DummySession
sys.modules.setdefault("requests", requests_module)
requests_adapters_module = _RequestsAdaptersModule("requests.adapters")
requests_adapters_module.HTTPAdapter = type(
    "HTTPAdapter", (), {"__init__": lambda self, *args, **kwargs: None}
)
sys.modules.setdefault("requests.adapters", requests_adapters_module)
urllib3_retry_module = _Urllib3RetryModule("urllib3.util.retry")
urllib3_retry_module.Retry = lambda **kwargs: None
sys.modules.setdefault("urllib3.util.retry", urllib3_retry_module)
opensearch_module = _OpenSearchModule("opensearchpy")
opensearch_module.OpenSearch = type("OpenSearch", (), {})
opensearch_module.RequestsHttpConnection = object
sys.modules.setdefault("opensearchpy", opensearch_module)
qdrant_module = _QdrantModule("qdrant_client")
qdrant_module.QdrantClient = type(
    "QdrantClient",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "get_collections": lambda self: types.SimpleNamespace(collections=[]),
        "create_collection": lambda self, *args, **kwargs: None,
        "search": lambda self, **kwargs: [],
    },
)
qdrant_module.models = ModuleType("qdrant_client.models")
sys.modules.setdefault("qdrant_client", qdrant_module)
qdrant_http_models = _QdrantHttpModelsModule("qdrant_client.http.models")
qdrant_http_models.PointStruct = type("PointStruct", (), {})
qdrant_http_models.PointIdsList = type("PointIdsList", (), {})
qdrant_http_models.VectorParams = type("VectorParams", (), {})
distance_module = _QdrantDistanceModule("qdrant_client.http.models.Distance")
distance_module.COSINE = "cosine"
qdrant_http_models.Distance = distance_module
sys.modules.setdefault("qdrant_client.http.models", qdrant_http_models)
def _mmr_select(query, docs, embed, k=1, lambda_mult=0.5):
    if embed:
        try:
            embed([query] + [d.get("text", "") for d in docs])
        except Exception:
            pass
    return docs[:k]


mmr_module = _MmrModule("core.retrieval.mmr")
mmr_module.mmr_select = _mmr_select
sys.modules.setdefault("core.retrieval.mmr", mmr_module)
dedup_module = _DedupModule("core.retrieval.dedup")
dedup_module.collapse_near_duplicates = (
    lambda docs, embed_texts, sim_threshold=0.9, keep_limit=64: (docs, [])
)
sys.modules.setdefault("core.retrieval.dedup", dedup_module)


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def set_attribute(self, *_, **__):
        return None

    def set_status(self, *_, **__):
        return None


def _build_tracing_module():
    class _StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class _Status:
        def __init__(self, status_code, description=""):
            self.status_code = status_code
            self.description = description

    class TracingModule(ModuleType):
        start_span: Callable[..., _Span]
        record_span_error: Callable[..., None]
        STATUS_OK: str
        EMBEDDING: str
        RETRIEVER: str
        INPUT_VALUE: str
        OUTPUT_VALUE: str
        LLM: str
        CHAIN: str
        TOOL: str
        get_current_span: Callable[..., _Span]
        StatusCode: type[_StatusCode]
        Status: type[_Status]

    tracing_module = TracingModule("tracing")
    tracing_module.start_span = lambda *_, **__: _Span()
    tracing_module.record_span_error = (
        lambda span, err: span.set_status(_Status(_StatusCode.ERROR, str(err)))
    )
    tracing_module.STATUS_OK = "OK"
    tracing_module.EMBEDDING = "EMBEDDING"
    tracing_module.RETRIEVER = "RETRIEVER"
    tracing_module.INPUT_VALUE = "INPUT"
    tracing_module.OUTPUT_VALUE = "OUTPUT"
    tracing_module.LLM = "LLM"
    tracing_module.CHAIN = "CHAIN"
    tracing_module.TOOL = "TOOL"
    tracing_module.get_current_span = lambda *_, **__: _Span()
    tracing_module.StatusCode = _StatusCode
    tracing_module.Status = _Status
    return tracing_module


@pytest.fixture(autouse=True)
def _stub_tracing(monkeypatch):
    """Provide a scoped tracing stub so other tests can use the real module."""

    monkeypatch.setitem(sys.modules, "tracing", _build_tracing_module())


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
    assert doc.get("path") == "a"
    retrieval_score = doc.get("retrieval_score")
    assert retrieval_score is not None
    assert retrieval_score > 0


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
    assert result.documents[0].get("id") == "v2"

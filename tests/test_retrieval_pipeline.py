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
    exceptions: object


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
opensearch_module.exceptions = types.SimpleNamespace(
    OpenSearchException=Exception,
    NotFoundError=Exception,
)
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


def test_retrieval_skips_variants_for_anchored_query_when_enabled(monkeypatch):
    calls = {"variants": 0, "semantic": [], "keyword": []}

    def fake_generate_variants(query):
        calls["variants"] += 1
        return {"variants": [("rewritten-query", 0.6)]}

    def fake_semantic_retriever(query, top_k):
        calls["semantic"].append(query)
        return []

    def fake_keyword_retriever(query, top_k):
        calls["keyword"].append(query)
        return [{"_id": "k1", "path": "a", "text": "a", "score": 1.0, "checksum": "k1"}]

    monkeypatch.setattr(pipeline, "generate_variants", fake_generate_variants)

    cfg = RetrievalConfig(
        top_k=1,
        top_k_each=1,
        enable_mmr=False,
        anchored_exact_only=True,
        fusion_weight_vector=0.0,
        fusion_weight_bm25=1.0,
    )
    deps = RetrievalDeps(
        semantic_retriever=fake_semantic_retriever,
        keyword_retriever=fake_keyword_retriever,
        embed_texts=None,
        cross_encoder=None,
    )
    query = "In Ali's latest CV, what is his most recent job title?"
    result = pipeline.retrieve(query, cfg=cfg, deps=deps)

    assert calls["variants"] == 0
    assert calls["semantic"] == [query]
    assert calls["keyword"] == [query]
    assert len(result.documents) == 1


def test_retrieval_uses_variants_for_non_anchored_query(monkeypatch):
    calls = {"variants": 0, "semantic": [], "keyword": []}

    def fake_generate_variants(query):
        calls["variants"] += 1
        return {"variants": [("exact-query", 1.0), ("rewrite-query", 0.6)]}

    def fake_semantic_retriever(query, top_k):
        calls["semantic"].append(query)
        return []

    def fake_keyword_retriever(query, top_k):
        calls["keyword"].append(query)
        return [{"_id": query, "path": query, "text": query, "score": 1.0, "checksum": query}]

    monkeypatch.setattr(pipeline, "generate_variants", fake_generate_variants)

    cfg = RetrievalConfig(
        top_k=1,
        top_k_each=1,
        enable_mmr=False,
        anchored_exact_only=True,
        fusion_weight_vector=0.0,
        fusion_weight_bm25=1.0,
    )
    deps = RetrievalDeps(
        semantic_retriever=fake_semantic_retriever,
        keyword_retriever=fake_keyword_retriever,
        embed_texts=None,
        cross_encoder=None,
    )
    result = pipeline.retrieve("summarize this", cfg=cfg, deps=deps)

    assert calls["variants"] == 1
    assert calls["semantic"] == ["exact-query", "rewrite-query"]
    assert calls["keyword"] == ["exact-query", "rewrite-query"]
    assert len(result.documents) == 1


def test_retrieval_prefers_bm25_for_anchored_query_when_lexical_bias_enabled():
    vector_hits = [{"id": "v1", "text": "vector doc", "score": 1.0, "checksum": "v1"}]
    bm25_hits = [{"_id": "b1", "text": "keyword doc", "score": 1.0, "checksum": "b1"}]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=0.7,
        fusion_weight_bm25=0.3,
        anchored_lexical_bias_enabled=True,
        anchored_fusion_weight_vector=0.4,
        anchored_fusion_weight_bm25=0.6,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "In Ali's latest CV, what is his most recent job title?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "b1"


def test_retrieval_keeps_default_fusion_for_non_anchored_query():
    vector_hits = [{"id": "v1", "text": "vector doc", "score": 1.0, "checksum": "v1"}]
    bm25_hits = [{"_id": "b1", "text": "keyword doc", "score": 1.0, "checksum": "b1"}]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=0.7,
        fusion_weight_bm25=0.3,
        anchored_lexical_bias_enabled=True,
        anchored_fusion_weight_vector=0.4,
        anchored_fusion_weight_bm25=0.6,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "summarize this",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "v1"


def test_retrieval_rescues_lexical_title_match_for_canonical_anchored_query():
    vector_hits = [
        {
            "id": "v1",
            "text": "vector winner",
            "score": 1.0,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "id": "l1",
            "text": "lexical candidate",
            "score": 0.75,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "v1",
            "text": "vector winner",
            "score": 0.9,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "_id": "l1",
            "text": "lexical candidate",
            "score": 1.0,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=0.7,
        fusion_weight_bm25=0.3,
        anchored_lexical_bias_enabled=True,
        anchored_fusion_weight_vector=0.4,
        anchored_fusion_weight_bm25=0.6,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "In Ali's latest CV contact section, which city is listed?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "l1"
    assert result.documents[0].get("_canonical_lexical_rescue") is not None


def test_retrieval_rescues_lexical_match_for_canonical_semi_anchored_query():
    vector_hits = [
        {
            "id": "v1",
            "text": "vector winner",
            "score": 1.0,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "id": "l1",
            "text": "lexical candidate",
            "score": 0.75,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "v1",
            "text": "vector winner",
            "score": 0.9,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "_id": "l1",
            "text": "lexical candidate",
            "score": 1.0,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "What city is listed in his CV contact section?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "l1"
    assert result.documents[0].get("_canonical_lexical_rescue") is not None


def test_retrieval_does_not_apply_lexical_rescue_for_non_canonical_query():
    vector_hits = [
        {
            "id": "v1",
            "text": "vector winner",
            "score": 1.0,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "id": "l1",
            "text": "lexical candidate",
            "score": 0.75,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "v1",
            "text": "vector winner",
            "score": 0.9,
            "checksum": "v1",
            "filename": "random_notes.txt",
        },
        {
            "_id": "l1",
            "text": "lexical candidate",
            "score": 1.0,
            "checksum": "l1",
            "filename": "ali_latest_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "Where did Ali do his PhD studies?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "v1"
    assert result.documents[0].get("_canonical_lexical_rescue") is None


def test_retrieval_suppresses_generic_hard_negative_for_canonical_query():
    vector_hits = [
        {
            "id": "g1",
            "text": "generic list winner",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "c1",
            "text": "relevant cv doc",
            "score": 0.74,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "g1",
            "text": "generic list winner",
            "score": 0.95,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "_id": "c1",
            "text": "relevant cv doc",
            "score": 0.9,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "What city is listed in his CV contact section?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "c1"
    assert result.documents[0].get("_canonical_hard_negative_suppression") is not None


def test_retrieval_does_not_suppress_generic_hard_negative_for_non_canonical_query():
    vector_hits = [
        {
            "id": "g1",
            "text": "generic list winner",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "c1",
            "text": "relevant cv doc",
            "score": 0.74,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "g1",
            "text": "generic list winner",
            "score": 0.95,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "_id": "c1",
            "text": "relevant cv doc",
            "score": 0.9,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
    )
    result = pipeline.retrieve(
        "Where did Ali do his PhD studies?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 1
    assert result.documents[0].get("checksum") == "g1"
    assert result.documents[0].get("_canonical_hard_negative_suppression") is None


def test_retrieval_control_abstention_unchanged_with_hard_negative_suppression():
    vector_hits = [
        {
            "id": "g1",
            "text": "generic list winner",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "c1",
            "text": "possible corpus doc",
            "score": 0.74,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "g1",
            "text": "generic list winner",
            "score": 0.95,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "_id": "c1",
            "text": "possible corpus doc",
            "score": 0.9,
            "checksum": "c1",
            "filename": "ali_cv_contact_section.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
    )
    result = pipeline.retrieve(
        "What is Bitcoin price today?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert result.documents == []
    assert result.clarify is None


def test_retrieval_demotes_list_artifacts_for_content_question_evidence_order():
    vector_hits = [
        {
            "id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "g2",
            "text": "drive file list includes sliding mode control of PEM fuel cells",
            "score": 0.99,
            "checksum": "g2",
            "filename": "gdrive-file-list.txt",
        },
        {
            "id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.96,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "_id": "g2",
            "text": "drive file list includes sliding mode control of PEM fuel cells",
            "score": 0.98,
            "checksum": "g2",
            "filename": "gdrive-file-list.txt",
        },
        {
            "_id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.97,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=3,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        canonical_lexical_rescue_enabled=False,
        canonical_hard_negative_suppression_enabled=False,
        content_evidence_guard_enabled=True,
        content_evidence_guard_max_score_gap=0.08,
    )
    result = pipeline.retrieve(
        "In the PEM fuel-cell sliding mode control research paper, what control approach is used?",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 3
    assert result.documents[0].get("checksum") == "p1"
    assert result.documents[0].get("_evidence_quality_class") == "content_or_unknown"
    assert result.documents[1].get("_evidence_quality_class") == "non_evidence_artifact"
    assert result.documents[2].get("_evidence_quality_class") == "non_evidence_artifact"
    assert result.documents[1].get("_content_evidence_quality_guard") is not None


def test_retrieval_keeps_list_artifacts_for_discovery_style_query():
    vector_hits = [
        {
            "id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.96,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    bm25_hits = [
        {
            "_id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "_id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.97,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_variants=False,
        enable_mmr=False,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        canonical_lexical_rescue_enabled=False,
        canonical_hard_negative_suppression_enabled=False,
        content_evidence_guard_enabled=True,
        content_evidence_guard_max_score_gap=0.08,
    )
    result = pipeline.retrieve(
        "List the files that mention the PEM fuel-cell sliding mode control paper.",
        cfg=cfg,
        deps=_build_deps(vector_hits, bm25_hits),
    )

    assert len(result.documents) == 2
    assert result.documents[0].get("checksum") == "g1"
    assert result.documents[0].get("_content_evidence_quality_guard") is None


def test_retrieval_anchored_near_tie_prefers_content_over_artifact():
    vector_hits = [
        {
            "id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.999995,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
        {
            "id": "g2",
            "text": "drive file list includes sliding mode control of PEM fuel cells",
            "score": 0.999994,
            "checksum": "g2",
            "filename": "gdrive-file-list.txt",
        },
    ]
    cfg = RetrievalConfig(
        top_k=3,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        canonical_lexical_rescue_enabled=False,
        canonical_hard_negative_suppression_enabled=False,
        content_evidence_guard_enabled=False,
        anchored_content_near_tie_score_epsilon=1e-5,
    )
    result = pipeline.retrieve(
        "In the PEM fuel-cell sliding mode control research paper, what control approach is used?",
        cfg=cfg,
        deps=_build_deps(vector_hits, []),
    )

    assert len(result.documents) == 3
    assert result.documents[0].get("checksum") == "p1"
    g1 = next(doc for doc in result.documents if doc.get("checksum") == "g1")
    assert g1.get("_anchored_content_near_tie_break") is not None
    assert g1.get("retrieval_score") < result.documents[0].get("retrieval_score")


def test_retrieval_anchored_near_tie_break_does_not_apply_to_discovery_query():
    vector_hits = [
        {
            "id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.999995,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        canonical_lexical_rescue_enabled=False,
        canonical_hard_negative_suppression_enabled=False,
        content_evidence_guard_enabled=False,
        anchored_content_near_tie_score_epsilon=1e-5,
    )
    result = pipeline.retrieve(
        "List the files that mention the PEM fuel-cell sliding mode control paper.",
        cfg=cfg,
        deps=_build_deps(vector_hits, []),
    )

    assert len(result.documents) == 2
    assert result.documents[0].get("checksum") == "g1"
    assert result.documents[0].get("_anchored_content_near_tie_break") is None


def test_retrieval_anchored_near_tie_break_does_not_apply_to_non_anchored_query():
    vector_hits = [
        {
            "id": "g1",
            "text": "filtered drive list mentioning sliding mode control of PEM fuel cells paper",
            "score": 1.0,
            "checksum": "g1",
            "filename": "filtered_gdrive_list.txt",
        },
        {
            "id": "p1",
            "text": "Control approach used: Sliding Mode Control.",
            "score": 0.999995,
            "checksum": "p1",
            "filename": "Sliding Mode Control of PEM Fuel Cells.pdf",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_variants=False,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        canonical_lexical_rescue_enabled=False,
        canonical_hard_negative_suppression_enabled=False,
        content_evidence_guard_enabled=False,
        anchored_content_near_tie_score_epsilon=1e-5,
    )
    result = pipeline.retrieve(
        "What control approach is used in it?",
        cfg=cfg,
        deps=_build_deps(vector_hits, []),
    )

    assert len(result.documents) == 2
    assert result.documents[0].get("checksum") == "g1"
    assert result.documents[0].get("_anchored_content_near_tie_break") is None


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


def test_retrieval_keeps_docs_when_checksum_missing():
    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 1.0},
        {"id": "v2", "text": "doc2", "score": 0.9},
    ]
    cfg = RetrievalConfig(top_k=2, enable_mmr=False)
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["v1", "v2"]


def test_retrieval_dedups_bm25_variant_hits_by_id(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_variants",
        lambda q: {"variants": [("exact-query", 1.0), ("rewrite-query", 0.1)]},
    )

    def fake_keyword_retriever(query, top_k):
        if query == "exact-query":
            return [
                {
                    "_id": "shared-id",
                    "path": "a",
                    "text": "doc",
                    "score": 10.0,
                    "checksum": "c1",
                }
            ]
        return [
            {
                "_id": "shared-id",
                "path": "a",
                "text": "doc",
                "score": 1.0,
                "checksum": "c1",
            }
        ]

    deps = RetrievalDeps(
        semantic_retriever=lambda q, top_k: [],
        keyword_retriever=fake_keyword_retriever,
        embed_texts=None,
        cross_encoder=None,
    )
    cfg = RetrievalConfig(
        top_k=1,
        top_k_each=1,
        enable_mmr=False,
        fusion_weight_vector=0.0,
        fusion_weight_bm25=1.0,
    )
    result = pipeline.retrieve("query", cfg=cfg, deps=deps)

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.get("_variant_rank") == 0
    assert doc.get("_bm25_variant_weight") == pytest.approx(1.0)
    assert doc.get("retrieval_score") == pytest.approx(1.0)


def test_retrieval_tops_up_from_unique_docs_before_duplicates(monkeypatch):
    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 1.0, "checksum": "u1"},
        {"id": "v2", "text": "doc2", "score": 0.9, "checksum": "u2"},
        {"id": "v3", "text": "doc3", "score": 0.8, "checksum": "u3"},
    ]
    duplicate_pool = [
        {"id": "d1", "text": "dup1", "score": 0.7, "checksum": "u1"},
        {"id": "d2", "text": "dup2", "score": 0.6, "checksum": "u2"},
    ]

    monkeypatch.setattr(
        pipeline,
        "collapse_near_duplicates",
        lambda docs, embed_texts, sim_threshold=0.9, keep_limit=64: (
            list(docs),
            list(duplicate_pool),
        ),
    )

    cfg = RetrievalConfig(top_k=3, mmr_k=1, enable_mmr=True, include_dups_if_needed=True)
    result = pipeline.retrieve(
        "query",
        cfg=cfg,
        deps=_build_deps(
            vector_hits,
            [],
            embedder=lambda texts: [[1.0, float(i)] for i, _ in enumerate(texts)],
        ),
    )

    assert [doc.get("id") for doc in result.documents] == ["v1", "v2", "v3"]


def test_retrieval_applies_authority_boost_when_metadata_exists():
    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 1.0, "checksum": "a1"},
        {
            "id": "v2",
            "text": "doc2",
            "score": 0.95,
            "checksum": "a2",
            "authority_rank": 1.0,
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_weight=0.08,
        authority_boost_max_fraction=0.2,
    )
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["v2", "v1"]
    assert result.documents[0].get("retrieval_score") == pytest.approx(1.03)


def test_retrieval_authority_boost_respects_bound():
    vector_hits = [
        {"id": "v1", "text": "doc1", "score": 1.0, "checksum": "b1"},
        {
            "id": "v2",
            "text": "doc2",
            "score": 0.5,
            "checksum": "b2",
            "authority_rank": 1.0,
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_weight=0.5,
        authority_boost_max_fraction=0.1,
    )
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["v1", "v2"]
    assert result.documents[1].get("retrieval_score") == pytest.approx(0.55)


def test_retrieval_applies_recency_boost_from_modified_at():
    vector_hits = [
        {
            "id": "old",
            "text": "old doc",
            "score": 1.0,
            "checksum": "r1",
            "modified_at": "2023-01-01T00:00:00+00:00",
        },
        {
            "id": "new",
            "text": "new doc",
            "score": 0.97,
            "checksum": "r2",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_weight=0.08,
        recency_boost_half_life_days=90,
        recency_boost_max_fraction=0.2,
    )
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["new", "old"]
    assert result.documents[0].get("retrieval_score") == pytest.approx(1.05)


def test_retrieval_recency_boost_respects_bound():
    vector_hits = [
        {
            "id": "older",
            "text": "older doc",
            "score": 1.0,
            "checksum": "s1",
            "modified_at": "2020-01-01T00:00:00+00:00",
        },
        {
            "id": "newer",
            "text": "newer doc",
            "score": 0.5,
            "checksum": "s2",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_weight=0.5,
        recency_boost_half_life_days=365,
        recency_boost_max_fraction=0.1,
    )
    result = pipeline.retrieve("query", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["older", "newer"]
    assert result.documents[1].get("retrieval_score") == pytest.approx(0.55)


def test_retrieval_collapses_cv_family_prefers_newer_when_relevance_close():
    vector_hits = [
        {
            "id": "cv_old",
            "text": "Ali old CV",
            "score": 1.0,
            "checksum": "cv-old",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2017-01-01T00:00:00+00:00",
        },
        {
            "id": "cv_new",
            "text": "Ali new CV",
            "score": 0.95,
            "checksum": "cv-new",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "contract",
            "text": "contract",
            "score": 0.8,
            "checksum": "contract",
            "doc_type": "contract",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    ]
    cfg = RetrievalConfig(
        top_k=3,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        cv_family_relevance_margin=0.1,
    )
    result = pipeline.retrieve("where did ali do his phd studies", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["cv_new", "contract"]
    assert result.documents[0].get("_cv_family_choice_reason") == "newest_within_margin"
    assert result.documents[0].get("_cv_family_suppressed") == 1


def test_retrieval_cv_family_keeps_older_when_clearly_more_relevant():
    vector_hits = [
        {
            "id": "cv_old",
            "text": "Ali old CV",
            "score": 1.0,
            "checksum": "cv-old",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2017-01-01T00:00:00+00:00",
        },
        {
            "id": "cv_new",
            "text": "Ali new CV",
            "score": 0.7,
            "checksum": "cv-new",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "contract",
            "text": "contract",
            "score": 0.8,
            "checksum": "contract",
            "doc_type": "contract",
            "modified_at": "2025-01-01T00:00:00+00:00",
        },
    ]
    cfg = RetrievalConfig(
        top_k=3,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        cv_family_relevance_margin=0.1,
    )
    result = pipeline.retrieve("where did ali do his phd studies", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["cv_old", "contract"]
    assert result.documents[0].get("_cv_family_choice_reason") == "older_higher_relevance"


def test_profile_intent_boost_prioritizes_profile_docs_for_profile_query():
    vector_hits = [
        {
            "id": "cv_old",
            "text": "Ali old CV",
            "score": 1.0,
            "checksum": "cv-old",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2017-01-01T00:00:00+00:00",
        },
        {
            "id": "cv_new",
            "text": "Ali new CV",
            "score": 0.92,
            "checksum": "cv-new",
            "doc_type": "cv",
            "person_name": "Ali A",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "contract",
            "text": "contract",
            "score": 0.99,
            "checksum": "contract",
            "doc_type": "contract",
            "modified_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    cfg = RetrievalConfig(
        top_k=3,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        cv_family_relevance_margin=0.1,
        profile_intent_boost_enabled=True,
        profile_intent_boost_weight=0.1,
        profile_intent_boost_max_fraction=0.2,
    )
    result = pipeline.retrieve("where did ali do his phd studies", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert [doc.get("id") for doc in result.documents] == ["cv_new", "contract"]
    assert result.documents[0].get("_profile_intent_adjustment", 0.0) > 0
    assert result.documents[1].get("_profile_intent_adjustment", 0.0) < 0


def test_retrieval_abstains_for_out_of_corpus_style_query_with_low_overlap():
    vector_hits = [
        {
            "id": "v1",
            "text": "Ali curriculum vitae and project details",
            "path": "C:/docs/ali_cv.pdf",
            "score": 1.0,
            "checksum": "o1",
        },
        {
            "id": "v2",
            "text": "Insurance statement for annual premium",
            "path": "C:/docs/insurance.pdf",
            "score": 0.95,
            "checksum": "o2",
        },
    ]
    cfg = RetrievalConfig(
        top_k=2,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
    )
    result = pipeline.retrieve("What is Bitcoin price today?", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert result.documents == []
    assert result.clarify is None


def test_retrieval_does_not_abstain_for_domain_anchored_live_query():
    vector_hits = [
        {
            "id": "cv1",
            "text": "Ali latest CV includes Senior Engineer role",
            "path": "C:/docs/ali_latest_cv.pdf",
            "score": 1.0,
            "checksum": "d1",
        }
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
    )
    result = pipeline.retrieve("In Ali latest CV today, what is his role?", cfg=cfg, deps=_build_deps(vector_hits, []))

    assert len(result.documents) == 1
    assert result.documents[0].get("id") == "cv1"


def test_retrieval_abstains_for_live_out_of_corpus_query_even_with_partial_overlap():
    vector_hits = [
        {
            "id": "w1",
            "text": "weather forecast for tomorrow in training example text",
            "path": "C:/docs/ml_notes.pdf",
            "score": 1.0,
            "checksum": "w1",
        }
    ]
    cfg = RetrievalConfig(
        top_k=1,
        enable_mmr=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
    )
    result = pipeline.retrieve(
        "What is the weather in Berlin tomorrow?",
        cfg=cfg,
        deps=_build_deps(vector_hits, []),
    )

    assert result.documents == []
    assert result.clarify is None


def test_retrieval_config_sim_threshold_default():
    assert RetrievalConfig().sim_threshold == pytest.approx(0.82)
    cfg = RetrievalConfig()
    assert cfg.anchored_exact_only is True
    assert cfg.anchored_lexical_bias_enabled is True
    assert cfg.anchored_fusion_weight_vector == pytest.approx(0.4)
    assert cfg.anchored_fusion_weight_bm25 == pytest.approx(0.6)
    assert cfg.canonical_lexical_rescue_enabled is True
    assert cfg.canonical_lexical_rescue_max_score_gap == pytest.approx(0.20)
    assert cfg.canonical_lexical_rescue_min_bm25 == pytest.approx(0.85)
    assert cfg.canonical_lexical_rescue_min_bm25_advantage == pytest.approx(0.08)
    assert cfg.canonical_lexical_rescue_strong_bm25_advantage == pytest.approx(0.25)
    assert cfg.canonical_lexical_rescue_min_title_overlap_ratio == pytest.approx(0.40)
    assert cfg.canonical_lexical_rescue_min_title_overlap_count == 2
    assert cfg.canonical_hard_negative_suppression_enabled is True
    assert cfg.canonical_hard_negative_max_score_gap == pytest.approx(0.55)
    assert cfg.canonical_hard_negative_min_winner_vector_score == pytest.approx(0.80)
    assert cfg.canonical_hard_negative_max_winner_title_overlap_count == 0
    assert cfg.canonical_hard_negative_min_candidate_title_overlap_count == 1
    assert cfg.content_evidence_guard_enabled is True
    assert cfg.content_evidence_guard_max_score_gap == pytest.approx(0.08)
    assert cfg.anchored_content_near_tie_score_epsilon == pytest.approx(1e-5)
    assert cfg.recency_boost_enabled is True
    assert cfg.recency_boost_weight == pytest.approx(0.06)
    assert cfg.cv_family_collapse_enabled is True
    assert cfg.cv_family_relevance_margin == pytest.approx(0.10)
    assert cfg.profile_intent_boost_enabled is True
    assert cfg.profile_intent_boost_weight == pytest.approx(0.10)
    assert cfg.abstention_enabled is True
    assert cfg.abstention_min_overlap_terms == 2

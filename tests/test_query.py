import sys
import types
from types import ModuleType
from typing import Callable

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


class _RetrievalPipelineModule(ModuleType):
    retrieve: Callable[..., list]


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


class _Span:
    def __init__(self):
        self.attrs = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, status):
        self.attrs["status"] = status


def _start_span(*_, **__):
    return _Span()


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
        get_current_span: Callable[..., _Span]
        STATUS_OK: str
        EMBEDDING: str
        RETRIEVER: str
        LLM: str
        INPUT_VALUE: str
        OUTPUT_VALUE: str
        CHAIN: str
        TOOL: str
        StatusCode: type[_StatusCode]
        Status: type[_Status]

    tracing_module = TracingModule("tracing")
    tracing_module.start_span = _start_span
    tracing_module.record_span_error = (
        lambda span, err: span.set_status(_Status(_StatusCode.ERROR, str(err)))
    )
    tracing_module.get_current_span = lambda *_, **__: _Span()
    tracing_module.STATUS_OK = "OK"
    tracing_module.EMBEDDING = "EMBEDDING"
    tracing_module.RETRIEVER = "RETRIEVER"
    tracing_module.LLM = "LLM"
    tracing_module.INPUT_VALUE = "INPUT"
    tracing_module.OUTPUT_VALUE = "OUTPUT"
    tracing_module.CHAIN = "CHAIN"
    tracing_module.TOOL = "TOOL"
    tracing_module.StatusCode = _StatusCode
    tracing_module.Status = _Status
    return tracing_module


@pytest.fixture(autouse=True)
def _stub_tracing(monkeypatch):
    """Provide a scoped tracing stub so other tests can import real tracing."""

    monkeypatch.setitem(sys.modules, "tracing", _build_tracing_module())
    pipeline_module = _RetrievalPipelineModule("core.retrieval.pipeline")
    pipeline_module.retrieve = lambda *_, **__: []
    monkeypatch.setitem(sys.modules, "core.retrieval.pipeline", pipeline_module)

from qa_pipeline.coordinator import answer_question
from qa_pipeline.types import QueryRewrite, RetrievalResult, RetrievedDocument
from qa_pipeline.retrieve import retrieve_context


def test_retrieval_limit_matches_top_k(monkeypatch):
    spans = []

    class DummySpan:
        def __init__(self):
            self.attrs = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def set_attribute(self, key, value):
            self.attrs[key] = value

        def set_status(self, status):
            self.attrs["status"] = status

    def dummy_start_span(name, kind):
        span = DummySpan()
        spans.append(span)
        return span

    call_args = {}

    def mock_retrieve(query, top_k, retrieval_cfg=None):
        call_args["final_k"] = top_k
        documents = [
            RetrievedDocument(
                text=f"doc{i}",
                path=f"path{i}",
                chunk_index=i,
                score=1.0,
                page=i + 1,
                location_percent=i * 10,
            )
            for i in range(top_k)
        ]
        return RetrievalResult(query=query, documents=documents)

    def mock_rewrite(question, temperature=0.15, use_cache=True):
        return QueryRewrite(rewritten=question)

    def mock_generate(*args, **kwargs):
        return "answer"

    monkeypatch.setattr("qa_pipeline.coordinator.start_span", dummy_start_span)
    monkeypatch.setattr("qa_pipeline.coordinator.retrieve_context", mock_retrieve)
    monkeypatch.setattr("qa_pipeline.coordinator.rewrite_question", mock_rewrite)
    monkeypatch.setattr("qa_pipeline.coordinator.generate_answer", mock_generate)

    result = answer_question("question", top_k=4)

    retrieval_span = next(span for span in spans if "results_found" in span.attrs)
    assert call_args["final_k"] == 4
    assert retrieval_span.attrs["results_found"] == 4
    assert result.retrieval is not None


def test_answer_question_uses_low_default_temperature(monkeypatch):
    observed = {}

    def mock_retrieve(query, top_k, retrieval_cfg=None):
        return RetrievalResult(
            query=query,
            documents=[RetrievedDocument(text="doc", path="path", score=1.0)],
        )

    def mock_rewrite(question, temperature=0.15, use_cache=True):
        return QueryRewrite(rewritten=question)

    def mock_generate(*args, **kwargs):
        observed["temperature"] = kwargs.get("temperature")
        return "answer"

    monkeypatch.setattr("qa_pipeline.coordinator.retrieve_context", mock_retrieve)
    monkeypatch.setattr("qa_pipeline.coordinator.rewrite_question", mock_rewrite)
    monkeypatch.setattr("qa_pipeline.coordinator.generate_answer", mock_generate)

    result = answer_question("question")

    assert observed["temperature"] == 0.1
    assert result.temperature == 0.1


def test_retrieve_context_prefers_normalized_retrieval_score(monkeypatch):
    fake_output = types.SimpleNamespace(
        clarify=None,
        documents=[
            {
                "text": "doc",
                "path": "path",
                "score": 10.0,
                "retrieval_score": 0.91,
            }
        ],
    )
    monkeypatch.setattr("qa_pipeline.retrieve.retrieve", lambda query, cfg, deps: fake_output)

    result = retrieve_context(
        "question",
        top_k=1,
        deps=types.SimpleNamespace(
            semantic_retriever=lambda q, top_k: [],
            keyword_retriever=lambda q, top_k: [],
            embed_texts=None,
            cross_encoder=None,
        ),
    )

    assert len(result.documents) == 1
    assert result.documents[0].score == pytest.approx(0.91)


def test_retrieve_context_passes_identity_metadata(monkeypatch):
    fake_output = types.SimpleNamespace(
        clarify=None,
        documents=[
            {
                "text": "doc",
                "path": "path",
                "score": 10.0,
                "retrieval_score": 0.91,
                "doc_type": "cv",
                "person_name": "Jane Doe",
                "authority_rank": 0.85,
            }
        ],
    )
    monkeypatch.setattr("qa_pipeline.retrieve.retrieve", lambda query, cfg, deps: fake_output)

    result = retrieve_context(
        "question",
        top_k=1,
        deps=types.SimpleNamespace(
            semantic_retriever=lambda q, top_k: [],
            keyword_retriever=lambda q, top_k: [],
            embed_texts=None,
            cross_encoder=None,
        ),
    )

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_type == "cv"
    assert doc.person_name == "Jane Doe"
    assert doc.authority_rank == pytest.approx(0.85)

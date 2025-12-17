import sys
import types

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

    tracing_module = types.ModuleType("tracing")
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
    monkeypatch.setitem(
        sys.modules,
        "core.retrieval.pipeline",
        types.SimpleNamespace(retrieve=lambda *_, **__: []),
    )

from qa_pipeline.coordinator import answer_question
from qa_pipeline.types import QueryRewrite, RetrievalResult, RetrievedDocument


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

    def mock_rewrite(question, temperature=0.15):
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

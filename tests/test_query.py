import sys
import types

dummy_tracing = types.SimpleNamespace()


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


dummy_tracing.start_span = _start_span
dummy_tracing.record_span_error = lambda *_, **__: None
dummy_tracing.get_current_span = lambda *_, **__: _Span()
dummy_tracing.STATUS_OK = "OK"
dummy_tracing.RETRIEVER = "RETRIEVER"
dummy_tracing.LLM = "LLM"
dummy_tracing.INPUT_VALUE = "INPUT"
dummy_tracing.OUTPUT_VALUE = "OUTPUT"
dummy_tracing.CHAIN = "CHAIN"
dummy_tracing.TOOL = "TOOL"

sys.modules.setdefault("tracing", dummy_tracing)
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault(
    "core.hybrid.pipeline", types.SimpleNamespace(retrieve_hybrid=lambda *_, **__: [])
)

from core.query import answer_question
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

    def mock_retrieve(query, top_k):
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

    monkeypatch.setattr("core.query.start_span", dummy_start_span)
    monkeypatch.setattr("core.query.retrieve_context", mock_retrieve)
    monkeypatch.setattr("core.query.rewrite_question", mock_rewrite)
    monkeypatch.setattr("core.query.generate_answer", mock_generate)

    result = answer_question("question", top_k=4)

    retrieval_span = next(span for span in spans if "results_found" in span.attrs)
    assert call_args["final_k"] == 4
    assert retrieval_span.attrs["results_found"] == 4
    assert result.retrieval is not None

import sys
import os

from core.query import answer_question


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
            pass

    def dummy_start_span(name, kind):
        span = DummySpan()
        spans.append(span)
        return span

    call_args = {}

    def mock_retrieve(query, top_k_each=20, final_k=5):
        call_args["final_k"] = final_k
        return [
            {
                "text": f"doc{i}",
                "path": f"path{i}",
                "chunk_index": i,
                "score": 1.0,
                "page": i + 1,
                "location_percent": i * 10,
            }
            for i in range(final_k)
        ]

    def mock_rewrite(question, temperature=0.15):
        return {"rewritten": question}

    def mock_ask_llm(*args, **kwargs):
        return "answer"

    monkeypatch.setattr("core.query.start_span", dummy_start_span)
    monkeypatch.setattr("core.query.retrieve_hybrid", mock_retrieve)
    monkeypatch.setattr("core.query.rewrite_query", mock_rewrite)
    monkeypatch.setattr("core.query.ask_llm", mock_ask_llm)

    answer_question("question", top_k=4)

    retrieval_span = spans[0]
    assert call_args["final_k"] == 4
    assert retrieval_span.attrs["top_k"] == 4
    assert retrieval_span.attrs["results_found"] == 4

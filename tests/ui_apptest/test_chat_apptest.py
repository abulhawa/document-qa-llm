import types
import sys
from contextlib import contextmanager

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _stub_tracing(monkeypatch):
    class _Span:
        def set_attribute(self, *args, **kwargs):
            pass

        def set_status(self, *args, **kwargs):
            pass

    @contextmanager
    def start_span(name: str, kind: str):
        yield _Span()

    def get_current_span():
        return _Span()

    def record_span_error(span, error):
        pass

    dummy = types.SimpleNamespace(
        start_span=start_span,
        CHAIN="CHAIN",
        INPUT_VALUE="INPUT",
        OUTPUT_VALUE="OUTPUT",
        STATUS_OK="OK",
        LLM="LLM",
        RETRIEVER="RETRIEVER",
        EMBEDDING="EMBEDDING",
        TOOL="TOOL",
        get_current_span=get_current_span,
        record_span_error=record_span_error,
    )
    monkeypatch.setitem(sys.modules, "tracing", dummy)


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(
        "core.llm.check_llm_status",
        lambda: {
            "active": True,
            "server_online": True,
            "model_loaded": True,
            "status_message": "",
            "current_model": "model",
        },
    )
    monkeypatch.setattr("core.llm.get_available_models", lambda: ["model"])
    monkeypatch.setattr("core.llm.load_model", lambda model: True)


def _run_query(at, question: str):
    at.run()
    at.text_input[0].input(question).run()
    at.button[0].click().run()


def _get_sources(at):
    return [m.value for m in at.markdown if m.value.startswith("- ")]


def test_query_returns_answer_with_dedup_sources(monkeypatch):
    monkeypatch.setattr(
        "core.query.answer_question",
        lambda **_: (
            "fixed answer",
            ["doc (Page 1)", "doc (Page 1)", "other (Page 2)"],
        ),
    )

    at = AppTest.from_file("pages/0_chat.py", default_timeout=10)
    _run_query(at, "question")

    markdown_vals = [m.value for m in at.markdown]
    assert "fixed answer" in markdown_vals

    sources = _get_sources(at)
    assert list(dict.fromkeys(sources)) == ["- doc (Page 1)", "- other (Page 2)"]


def test_no_results_shows_message(monkeypatch):
    monkeypatch.setattr(
        "core.query.answer_question",
        lambda **_: ("No relevant context found to answer the question.", []),
    )

    at = AppTest.from_file("pages/0_chat.py", default_timeout=10)
    _run_query(at, "question")

    assert any(
        "No relevant context found" in m.value for m in at.markdown
    )

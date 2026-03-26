import pytest
from core.query_rewriter import rewrite_query, has_strong_query_anchors


def test_rewrite_query_rewritten(monkeypatch):
    def fake_ask_llm(**kwargs):
        return '{"rewritten": "Ali BSc study location"}'
    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    result = rewrite_query("where did ali do his bsc studies")
    assert result == {"rewritten": "Ali BSc study location"}


def test_rewrite_query_clarify(monkeypatch):
    def fake_ask_llm(**kwargs):
        return '{"clarify": "Who are you referring to?"}'
    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    result = rewrite_query("where did he do his bsc studies")
    assert result == {"clarify": "Who are you referring to?"}


def test_rewrite_query_error(monkeypatch):
    def fake_ask_llm(**kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    result = rewrite_query("bad query")
    assert "Error" in result


def test_rewrite_query_clarify_bypassed_for_strong_anchor_query(monkeypatch):
    def fake_ask_llm(**kwargs):
        return '{"clarify": "Can you clarify?"}'

    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    result = rewrite_query("In Ali's latest CV, what is his most recent role?")
    assert result == {"rewritten": "In Ali's latest CV, what is his most recent role?"}


def test_has_strong_query_anchors_blocks_ambiguous_pronoun_query():
    assert has_strong_query_anchors("where did he do his bsc studies") is False

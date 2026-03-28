import pytest
from core.query_rewriter import build_query_plan, rewrite_query, has_strong_query_anchors


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


def test_build_query_plan_generates_typed_queries_without_hyde(monkeypatch):
    def fake_ask_llm(**kwargs):
        return '{"rewritten": "Ali BSc study location"}'

    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    plan = build_query_plan(
        "best time to visit paris",
        enable_hyde=False,
    )

    assert plan.raw_query == "best time to visit paris"
    assert plan.semantic_query == "Ali BSc study location"
    assert plan.bm25_query == "Ali BSc study location"
    assert plan.hyde_passage is None
    assert plan.clarify is None


def test_build_query_plan_emits_hyde_only_when_enabled(monkeypatch):
    calls = {"count": 0}

    def fake_ask_llm(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"rewritten": "Ali BSc study location"}'
        return "Ali completed BSc studies at a university."

    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    no_hyde = build_query_plan("best time to visit paris", enable_hyde=False)
    with_hyde = build_query_plan("best time to visit paris", enable_hyde=True)

    assert no_hyde.hyde_passage is None
    assert with_hyde.hyde_passage == "Ali completed BSc studies at a university."
    assert calls["count"] == 3


def test_build_query_plan_preserves_exact_for_anchored_query(monkeypatch):
    def fake_ask_llm(**kwargs):
        return '{"rewritten": "latest job title Ali CV"}'

    monkeypatch.setattr("core.query_rewriter.ask_llm", fake_ask_llm)
    query = "In Ali's latest CV, what is his most recent role?"
    plan = build_query_plan(query, enable_hyde=False)

    assert plan.raw_query == query
    assert plan.semantic_query == query
    assert plan.bm25_query == query

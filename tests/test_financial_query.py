from __future__ import annotations

import sys
from types import ModuleType

class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def set_attribute(self, *_, **__):
        return None

    def set_status(self, *_, **__):
        return None


class _TracingModule(ModuleType):
    get_current_span: object
    start_span: object
    record_span_error: object
    STATUS_OK: str
    EMBEDDING: str
    RETRIEVER: str
    INPUT_VALUE: str
    OUTPUT_VALUE: str
    LLM: str
    CHAIN: str
    TOOL: str


tracing_module = _TracingModule("tracing")
tracing_module.get_current_span = lambda *_, **__: _Span()
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
sys.modules.setdefault("tracing", tracing_module)

from core.financial_query import detect_financial_query
from core.query_rewriter import build_query_plan


def test_detect_financial_query_extracts_year_entity_and_concept():
    intent = detect_financial_query(
        "What expenses did Ali make in 2022 that help for tax returns?"
    )

    assert intent.financial_query_mode is True
    assert intent.target_entity == "Ali"
    assert intent.target_year == 2022
    assert intent.target_concept == "expenses"


def test_detect_financial_query_non_finance_query_is_disabled():
    intent = detect_financial_query("Where did Ali do his MSc studies?")
    assert intent.financial_query_mode is False
    assert intent.target_entity is None
    assert intent.target_year is None
    assert intent.target_concept is None


def test_build_query_plan_includes_financial_fields(monkeypatch):
    monkeypatch.setattr(
        "core.query_rewriter.rewrite_query",
        lambda *args, **kwargs: {"rewritten": "Ali tax expenses 2022"},
    )
    monkeypatch.setattr(
        "core.query_rewriter._generate_hyde_passage",
        lambda *args, **kwargs: None,
    )

    plan = build_query_plan(
        "What expenses did Ali make in 2022 for tax?",
        enable_hyde=False,
    )

    assert plan.financial_query_mode is True
    assert plan.target_entity == "Ali"
    assert plan.target_year == 2022
    assert plan.target_concept == "expenses"

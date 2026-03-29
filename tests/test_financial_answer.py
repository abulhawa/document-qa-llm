from __future__ import annotations

from qa_pipeline.financial_answer import build_financial_answer
from qa_pipeline.types import RetrievalResult, RetrievedDocument


def _retrieval(financial_fallback: bool = False) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        documents=[
            RetrievedDocument(
                text="Invoice paid in 2022 amount EUR 123.45",
                path="C:/docs/invoice.pdf",
                checksum="chk-1",
                is_financial_document=True,
            )
        ],
        stage_metadata={
            "financial_query_mode": True,
            "fallback_used": financial_fallback,
            "target_year": 2022,
            "target_entity": "Ali",
            "target_concept": "expenses",
        },
    )


def test_build_financial_answer_uses_sidecar_records(monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.financial_answer.get_financial_records_for_checksums",
        lambda checksums, year=None, size=200: [
            {
                "record_type": "expense",
                "date": "2022-03-10",
                "amount": 123.45,
                "currency": "EUR",
                "counterparty": "ACME",
                "confidence": 0.9,
                "checksum": "chk-1",
                "source_links": [
                    {
                        "checksum": "chk-1",
                        "chunk_id": "chk-1:0",
                    }
                ],
                "year": 2022,
            }
        ],
    )

    answer, meta = build_financial_answer(
        retrieval=_retrieval(financial_fallback=True),
        target_year=2022,
        target_entity="Ali",
        target_concept="expenses",
    )

    assert "Clearly supported 2022 expenses" in answer
    assert "123.45 EUR" in answer
    assert "fallback retrieval stages were used" in answer
    assert meta["financial_query_mode"] is True
    assert meta["sidecar_records_found"] == 1


def test_build_financial_answer_returns_idk_when_no_evidence(monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.financial_answer.get_financial_records_for_checksums",
        lambda checksums, year=None, size=200: [],
    )
    empty_retrieval = RetrievalResult(
        query="q",
        documents=[],
        stage_metadata={"financial_query_mode": True},
    )

    answer, meta = build_financial_answer(
        retrieval=empty_retrieval,
        target_year=2022,
        target_entity=None,
        target_concept="expenses",
    )

    assert answer == "I don't know."
    assert meta["financial_query_mode"] is True


def test_build_financial_answer_uses_chunk_fallback_when_sidecar_incomplete(monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.financial_answer.get_financial_records_for_checksums",
        lambda checksums, year=None, size=200: [],
    )

    answer, meta = build_financial_answer(
        retrieval=_retrieval(financial_fallback=False),
        target_year=2022,
        target_entity="Ali",
        target_concept="expenses",
    )

    assert "Coverage disclosure: normalized financial-record coverage is incomplete" in answer
    assert "Mentioned items not confirmed as paid in 2022:" in answer
    assert "invoice.pdf" in answer
    assert "C:/docs/invoice.pdf" not in answer
    assert meta["normalized_record_coverage_incomplete"] is True


def test_build_financial_answer_suppresses_weak_mention_only_fallback_docs(monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.financial_answer.get_financial_records_for_checksums",
        lambda checksums, year=None, size=200: [],
    )

    retrieval = RetrievalResult(
        query="What expenses did Ali make in 2022?",
        documents=[
            RetrievedDocument(
                text="Invoice paid on 2022-05-10 amount EUR 230.00",
                path="C:/finance/tax/ali_invoice_2022.pdf",
                checksum="inv-2022",
                is_financial_document=True,
                mentioned_years=[2022],
                financial_record_type="expense",
                source_family="invoice",
            ),
            RetrievedDocument(
                text="Course notes mention tax expenses in 2022 and include EUR 10.00 as an example.",
                path="C:/notes/course_material_tax_mention.txt",
                checksum="weak-mention",
                is_financial_document=False,
                mentioned_years=[2022],
                source_family="official_letter",
            ),
        ],
        stage_metadata={"financial_query_mode": True},
    )

    answer, meta = build_financial_answer(
        retrieval=retrieval,
        target_year=2022,
        target_entity="Ali",
        target_concept="expenses",
    )

    assert "ali_invoice_2022.pdf" in answer
    assert "course_material_tax_mention.txt" not in answer
    assert "weak mention-only candidates were suppressed" in answer
    assert "unknown unknown" not in answer
    fallback_metrics = meta.get("fallback_item_metrics") or {}
    assert fallback_metrics.get("strong_items") == 1
    assert fallback_metrics.get("weak_suppressed", 0) >= 1


def test_build_financial_answer_formats_missing_amount_without_unknown_unknown(monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.financial_answer.get_financial_records_for_checksums",
        lambda checksums, year=None, size=200: [
            {
                "record_type": "expense",
                "date": "2022-03-10",
                "amount": None,
                "currency": None,
                "counterparty": "ACME",
                "confidence": 0.61,
                "checksum": "chk-1",
                "source_links": [
                    {
                        "checksum": "chk-1",
                        "chunk_id": "chk-1:0",
                    }
                ],
                "year": 2022,
            }
        ],
    )

    answer, _meta = build_financial_answer(
        retrieval=_retrieval(financial_fallback=False),
        target_year=2022,
        target_entity="Ali",
        target_concept="expenses",
    )

    assert "Amount: not normalized" in answer
    assert "unknown unknown" not in answer

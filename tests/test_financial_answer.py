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
    assert "C:/docs/invoice.pdf" in answer
    assert meta["normalized_record_coverage_incomplete"] is True

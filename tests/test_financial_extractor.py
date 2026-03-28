from __future__ import annotations

import ingestion.financial_extractor as fx


def test_extract_financial_enrichment_deterministic_invoice():
    full_text = (
        "Invoice #4491\n"
        "Invoice Date: 2022-04-12\n"
        "Payee: ACME Supplies GmbH\n"
        "Total EUR 123.45\n"
        "VAT included."
    )
    chunks = [
        {
            "id": "chk-1:0",
            "text": "Invoice Date: 2022-04-12 Payee: ACME Supplies GmbH Total EUR 123.45",
        }
    ]

    result = fx.extract_financial_enrichment(
        path="C:/docs/invoice_2022_04.pdf",
        full_text=full_text,
        chunks=chunks,
        doc_type="invoice",
        checksum="chk-1",
        document_id="chk-1",
        enable_llm_fallback=False,
    )

    assert result.source_family == "invoice"
    assert result.used_llm_fallback is False
    assert result.document_metadata["is_financial_document"] is True
    assert result.document_metadata["document_date"] == "2022-04-12"
    assert 2022 in result.document_metadata["mentioned_years"]
    assert "2022-04-12" in result.document_metadata["transaction_dates"]
    assert 123.45 in result.document_metadata["amounts"]
    assert "acme supplies gmbh" in str(result.document_metadata["counterparties"]).lower()
    assert result.document_metadata["financial_metadata_source"] == "deterministic"
    assert len(result.records) == 1
    record = result.records[0]
    assert record["date"] == "2022-04-12"
    assert record["amount"] == 123.45
    assert record["currency"] == "EUR"
    assert record["record_type"] == "expense"
    assert record["checksum"] == "chk-1"
    assert record["chunk_id"] == "chk-1:0"
    assert record["source_count"] == 1


def test_merge_duplicate_records_preserves_links_and_caps_confidence():
    record_a = {
        "record_type": "expense",
        "date": "2022-01-10",
        "amount": 90.0,
        "currency": "EUR",
        "counterparty": "Store A",
        "confidence": 0.62,
        "extraction_method": "deterministic",
        "source_links": [
            {
                "document_id": "doc-1",
                "checksum": "chk-1",
                "chunk_id": "chk-1:0",
                "source_text_span": "EUR 90.00",
                "extraction_method": "deterministic",
                "confidence": 0.62,
            }
        ],
    }
    record_b = {
        "record_type": "expense",
        "date": "2022-01-10",
        "amount": 90.0,
        "currency": "EUR",
        "counterparty": "Store A",
        "confidence": 0.91,
        "extraction_method": "llm",
        "source_links": [
            {
                "document_id": "doc-2",
                "checksum": "chk-2",
                "chunk_id": "chk-2:1",
                "source_text_span": "90 EUR",
                "extraction_method": "llm",
                "confidence": 0.91,
            }
        ],
    }

    merged = fx.merge_duplicate_records([record_a, record_b])

    assert len(merged) == 1
    item = merged[0]
    assert item["confidence"] == 0.91
    assert item["source_count"] == 2
    assert item["extraction_method"] == "hybrid"
    links = item["source_links"]
    assert len(links) == 2

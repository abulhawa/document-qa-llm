from __future__ import annotations

import types

from langchain_core.documents import Document

from ingestion.orchestrator import ingest_one


class DummyLog:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set(self, **kwargs):
        pass

    def done(self, **kwargs):
        pass

    def fail(self, **kwargs):
        pass


def test_ingest_one_persists_financial_metadata_and_sidecar_records(tmp_path, monkeypatch):
    f = tmp_path / "invoice.txt"
    f.write_text("hello")

    monkeypatch.setattr("ingestion.io_loader.compute_checksum", lambda p: "finance-chk")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)
    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="full", metadata={})],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: docs_like,
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents",
        lambda docs: [{"text": "Invoice Date 2022-04-12 Total EUR 123.45"}],
    )
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", lambda chunks: (len(chunks), [])
    )
    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches",
        lambda chunks, os_index_batch=None: True,
    )
    monkeypatch.setattr("ingestion.orchestrator.ensure_financial_metadata_mappings", lambda: None)

    captured = {}

    def fake_index_fulltext(doc):
        captured["fulltext"] = dict(doc)

    monkeypatch.setattr("ingestion.storage.index_fulltext", fake_index_fulltext)

    def fake_extract(**kwargs):
        return types.SimpleNamespace(
            document_metadata={
                "is_financial_document": True,
                "document_date": "2022-04-12",
                "mentioned_years": [2022],
                "transaction_dates": ["2022-04-12"],
                "tax_years_referenced": [2022],
                "amounts": [123.45],
                "counterparties": ["ACME Supplies"],
                "tax_relevance_signals": ["invoice"],
                "expense_category": "professional",
                "financial_record_type": "expense",
                "financial_metadata_version": "v1",
                "financial_metadata_source": "deterministic",
            },
            records=[
                {
                    "record_type": "expense",
                    "date": "2022-04-12",
                    "amount": 123.45,
                    "currency": "EUR",
                    "counterparty": "ACME Supplies",
                    "confidence": 0.8,
                    "document_id": "finance-chk",
                    "checksum": "finance-chk",
                    "chunk_id": "finance-chk:0",
                    "source_links": [],
                }
            ],
        )

    monkeypatch.setattr("ingestion.orchestrator.extract_financial_enrichment", fake_extract)

    sidecar_stats = {}

    def fake_sidecar(records):
        sidecar_stats["count"] = len(records)
        return {"processed": len(records), "created": len(records), "updated": 0, "errors": 0}

    monkeypatch.setattr("ingestion.orchestrator.upsert_financial_records", fake_sidecar)

    result = ingest_one(str(f))

    assert result["success"] is True
    assert captured["fulltext"]["is_financial_document"] is True
    assert captured["fulltext"]["document_date"] == "2022-04-12"
    assert captured["fulltext"]["financial_metadata_source"] == "deterministic"
    assert sidecar_stats["count"] == 1

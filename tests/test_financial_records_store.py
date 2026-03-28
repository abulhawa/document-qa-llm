from __future__ import annotations

from typing import Any, Dict

import ingestion.financial_records_store as frs


class _DummyNotFoundError(Exception):
    pass


class _FakeClient:
    def __init__(self) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.index_calls = []
        self.search_calls = []

    def get(self, index: str, id: str):
        if id not in self.docs:
            raise _DummyNotFoundError("missing")
        return {"_source": dict(self.docs[id])}

    def index(self, index: str, id: str, body: Dict[str, Any], op_type: str, refresh: bool):
        self.docs[id] = dict(body)
        self.index_calls.append(
            {
                "index": index,
                "id": id,
                "body": dict(body),
                "op_type": op_type,
                "refresh": refresh,
            }
        )
        return {"result": "created"}

    def search(self, index: str, body: Dict[str, Any]):
        self.search_calls.append({"index": index, "body": body})
        hits = [{"_source": {"record_type": "expense", "year": 2022}}]
        return {"hits": {"hits": hits}}


def test_upsert_financial_records_merges_duplicate_sources(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(frs, "get_client", lambda: fake_client)
    monkeypatch.setattr(frs, "ensure_financial_records_index", lambda: None)
    monkeypatch.setattr(frs.exceptions, "NotFoundError", _DummyNotFoundError)

    base_record = {
        "record_type": "expense",
        "date": "2022-03-01",
        "amount": 45.0,
        "currency": "EUR",
        "counterparty": "Store A",
        "confidence": 0.6,
        "extraction_method": "deterministic",
        "source_links": [
            {
                "document_id": "doc-1",
                "checksum": "chk-1",
                "chunk_id": "chk-1:0",
                "source_text_span": "EUR 45",
                "extraction_method": "deterministic",
                "confidence": 0.6,
            }
        ],
    }
    second_record = {
        **base_record,
        "confidence": 0.9,
        "extraction_method": "llm",
        "source_links": [
            {
                "document_id": "doc-2",
                "checksum": "chk-2",
                "chunk_id": "chk-2:1",
                "source_text_span": "45 EUR",
                "extraction_method": "llm",
                "confidence": 0.9,
            }
        ],
    }

    stats = frs.upsert_financial_records([base_record, second_record])

    assert stats["processed"] == 2
    assert stats["created"] == 1
    assert stats["updated"] == 1
    assert stats["errors"] == 0
    assert len(fake_client.index_calls) == 2
    stored = fake_client.index_calls[-1]["body"]
    assert stored["confidence"] == 0.9
    assert stored["source_count"] == 2
    assert stored["extraction_method"] == "hybrid"


def test_fetch_financial_records_uses_checksum_and_year_filters(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(frs, "get_client", lambda: fake_client)
    monkeypatch.setattr(frs, "ensure_financial_records_index", lambda: None)

    rows = frs.fetch_financial_records(checksums=["chk-1", "chk-2"], year=2022, size=10)

    assert rows == [{"record_type": "expense", "year": 2022}]
    assert len(fake_client.search_calls) == 1
    body = fake_client.search_calls[0]["body"]
    filters = body["query"]["bool"]["filter"]
    assert any("bool" in item for item in filters)
    assert any(item.get("term", {}).get("year", {}).get("value") == 2022 for item in filters)

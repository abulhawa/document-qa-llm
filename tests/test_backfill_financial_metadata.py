from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any, Dict, List

import pytest


_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "backfill_financial_metadata.py"


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    if "opensearchpy" not in sys.modules:
        opensearch_module = types.ModuleType("opensearchpy")
        opensearch_module.OpenSearch = type("OpenSearch", (), {})
        opensearch_module.helpers = types.SimpleNamespace()
        opensearch_module.exceptions = types.SimpleNamespace(
            OpenSearchException=Exception,
            NotFoundError=Exception,
        )
        monkeypatch.setitem(sys.modules, "opensearchpy", opensearch_module)

    if "langchain_core" not in sys.modules and "langchain_core.documents" not in sys.modules:
        langchain_docs = types.ModuleType("langchain_core.documents")
        langchain_docs.Document = type("Document", (), {})
        monkeypatch.setitem(sys.modules, "langchain_core.documents", langchain_docs)

    if "langchain_core" not in sys.modules and "langchain_core.documents" in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        langchain_core.documents = sys.modules["langchain_core.documents"]
        monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)

    if "tracing" not in sys.modules:
        tracing_module = types.ModuleType("tracing")
        _span = types.SimpleNamespace(
            set_attribute=lambda *a, **k: None,
            set_status=lambda *a, **k: None,
            record_exception=lambda *a, **k: None,
        )
        tracing_module.get_current_span = lambda *args, **kwargs: _span
        tracing_module.start_span = lambda *args, **kwargs: types.SimpleNamespace(
            __enter__=lambda: _span,
            __exit__=lambda exc_type, exc, tb: None,
        )
        tracing_module.record_span_error = lambda *args, **kwargs: None
        tracing_module.STATUS_OK = "OK"
        tracing_module.RETRIEVER = "RETRIEVER"
        tracing_module.INPUT_VALUE = "INPUT_VALUE"
        tracing_module.OUTPUT_VALUE = "OUTPUT_VALUE"
        tracing_module.CHAIN = "CHAIN"
        tracing_module.TOOL = "TOOL"
        tracing_module.LLM = "LLM"
        tracing_module.EMBEDDING = "EMBEDDING"
        monkeypatch.setitem(sys.modules, "tracing", tracing_module)


@pytest.fixture
def backfill_financial_metadata(monkeypatch: pytest.MonkeyPatch):
    _install_runtime_stubs(monkeypatch)
    module_name = "_test_backfill_financial_metadata"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load backfill_financial_metadata.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeOpenSearchClient:
    def __init__(
        self,
        fulltext_hits: List[Dict[str, Any]],
        chunk_hits: Dict[str, List[Dict[str, Any]]],
        *,
        fulltext_index: str,
    ) -> None:
        self._fulltext_hits = fulltext_hits
        self._chunk_hits = chunk_hits
        self._fulltext_index = fulltext_index
        self._scroll_reads = 0
        self.updated_docs: List[Dict[str, Any]] = []
        self.chunk_updates: List[Dict[str, Any]] = []
        self.cleared_scroll_ids: List[str] = []

    def search(self, index: str, body: Dict[str, Any], params: Dict[str, Any] | None = None):
        if index == self._fulltext_index:
            return {"_scroll_id": "scroll-1", "hits": {"hits": list(self._fulltext_hits)}}

        checksum = body.get("query", {}).get("term", {}).get("checksum", {}).get("value")
        hits = self._chunk_hits.get(str(checksum), [])
        return {"hits": {"hits": hits}}

    def scroll(self, scroll_id: str, params: Dict[str, Any]):
        self._scroll_reads += 1
        return {"_scroll_id": scroll_id, "hits": {"hits": []}}

    def clear_scroll(self, scroll_id: str):
        self.cleared_scroll_ids.append(scroll_id)

    def update(self, index: str, id: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.updated_docs.append({"index": index, "id": id, "body": body, "params": params})

    def update_by_query(self, index: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.chunk_updates.append({"index": index, "body": body, "params": params})
        return {"updated": 4}


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.count_calls: List[Dict[str, Any]] = []
        self.set_payload_calls: List[Dict[str, Any]] = []

    def count(self, collection_name: str, count_filter: Any, exact: bool):
        self.count_calls.append(
            {
                "collection_name": collection_name,
                "count_filter": count_filter,
                "exact": exact,
            }
        )
        return types.SimpleNamespace(count=3)

    def set_payload(self, collection_name: str, points: Any, payload: Dict[str, Any], wait: bool):
        self.set_payload_calls.append(
            {
                "collection_name": collection_name,
                "points": points,
                "payload": payload,
                "wait": wait,
            }
        )
        return {"result": {"count": 3}}


class _FakeQdrantModels:
    class MatchValue:
        def __init__(self, value):
            self.value = value

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must):
            self.must = must


def _fake_extraction():
    return types.SimpleNamespace(
        source_family="invoice",
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
                "document_id": "chk-1",
                "checksum": "chk-1",
                "chunk_id": "chk-1:0",
                "source_links": [],
            }
        ],
    )


def test_backfill_financial_metadata_dry_run(backfill_financial_metadata, monkeypatch):
    fulltext_hits = [
        {
            "_id": "doc-1",
            "_source": {
                "checksum": "chk-1",
                "path": "C:/docs/invoice.pdf",
                "filetype": "pdf",
                "doc_type": "invoice",
                "text_full": "Invoice Date 2022-04-12 Total EUR 123.45",
            },
        }
    ]
    chunk_hits = {
        "chk-1": [
            {
                "_id": "chk-1:0",
                "_source": {"id": "chk-1:0", "text": "Invoice Date 2022-04-12 Total EUR 123.45"},
            }
        ]
    }
    fake_client = _FakeOpenSearchClient(
        fulltext_hits,
        chunk_hits,
        fulltext_index=backfill_financial_metadata.FULLTEXT_INDEX,
    )
    fake_qdrant = _FakeQdrantClient()

    monkeypatch.setattr(backfill_financial_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_financial_metadata,
        "_get_qdrant_components",
        lambda: (fake_qdrant, _FakeQdrantModels),
    )
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_metadata_mappings", lambda: None)
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_records_index", lambda: None)
    monkeypatch.setattr(backfill_financial_metadata, "extract_financial_enrichment", lambda **kwargs: _fake_extraction())

    stats = backfill_financial_metadata.backfill_financial_metadata(
        batch_size=50,
        dry_run=True,
        overwrite=False,
    )

    assert stats["scanned_fulltext_docs"] == 1
    assert stats["processed_docs"] == 1
    assert stats["fulltext_updates"] == 0
    assert stats["fulltext_would_update"] == 1
    assert stats["chunk_update_calls"] == 0
    assert stats["chunk_would_update_calls"] == 1
    assert stats["qdrant_payload_would_update_points"] == 3
    assert stats["sidecar_records_processed"] == 0
    assert fake_client.updated_docs == []
    assert fake_client.chunk_updates == []
    assert fake_qdrant.set_payload_calls == []
    assert fake_client.cleared_scroll_ids == ["scroll-1"]


def test_backfill_financial_metadata_apply_updates_indices_and_sidecar(backfill_financial_metadata, monkeypatch):
    fulltext_hits = [
        {
            "_id": "doc-1",
            "_source": {
                "checksum": "chk-1",
                "path": "C:/docs/invoice.pdf",
                "filetype": "pdf",
                "doc_type": "invoice",
                "text_full": "Invoice Date 2022-04-12 Total EUR 123.45",
            },
        }
    ]
    chunk_hits = {
        "chk-1": [
            {
                "_id": "chk-1:0",
                "_source": {"id": "chk-1:0", "text": "Invoice Date 2022-04-12 Total EUR 123.45"},
            }
        ]
    }
    fake_client = _FakeOpenSearchClient(
        fulltext_hits,
        chunk_hits,
        fulltext_index=backfill_financial_metadata.FULLTEXT_INDEX,
    )
    fake_qdrant = _FakeQdrantClient()

    monkeypatch.setattr(backfill_financial_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_financial_metadata,
        "_get_qdrant_components",
        lambda: (fake_qdrant, _FakeQdrantModels),
    )
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_metadata_mappings", lambda: None)
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_records_index", lambda: None)
    monkeypatch.setattr(backfill_financial_metadata, "extract_financial_enrichment", lambda **kwargs: _fake_extraction())
    monkeypatch.setattr(
        backfill_financial_metadata,
        "upsert_financial_records",
        lambda records: {"processed": len(records), "created": len(records), "updated": 0, "errors": 0},
    )

    stats = backfill_financial_metadata.backfill_financial_metadata(
        batch_size=10,
        dry_run=False,
        overwrite=False,
    )

    assert stats["fulltext_updates"] == 1
    assert stats["chunk_update_calls"] == 1
    assert stats["chunk_docs_updated"] == 4
    assert stats["qdrant_payload_updated_points"] == 3
    assert stats["sidecar_records_processed"] == 1
    assert stats["sidecar_records_created"] == 1
    assert stats["sidecar_records_updated"] == 0
    assert stats["sidecar_records_errors"] == 0
    assert len(fake_client.updated_docs) == 1
    assert len(fake_client.chunk_updates) == 1
    assert len(fake_qdrant.set_payload_calls) == 1


def test_backfill_financial_metadata_skip_and_limit(backfill_financial_metadata, monkeypatch):
    fulltext_hits = [
        {
            "_id": "doc-1",
            "_source": {
                "checksum": "chk-1",
                "path": "C:/docs/first.pdf",
                "filetype": "pdf",
                "doc_type": "invoice",
                "text_full": "Invoice 1",
            },
        },
        {
            "_id": "doc-2",
            "_source": {
                "checksum": "chk-2",
                "path": "C:/docs/second.pdf",
                "filetype": "pdf",
                "doc_type": "invoice",
                "text_full": "Invoice 2",
            },
        },
        {
            "_id": "doc-3",
            "_source": {
                "checksum": "chk-3",
                "path": "C:/docs/third.pdf",
                "filetype": "pdf",
                "doc_type": "invoice",
                "text_full": "Invoice 3",
            },
        },
    ]
    chunk_hits = {
        "chk-3": [
            {
                "_id": "chk-3:0",
                "_source": {"id": "chk-3:0", "text": "Invoice 3"},
            }
        ]
    }
    fake_client = _FakeOpenSearchClient(
        fulltext_hits,
        chunk_hits,
        fulltext_index=backfill_financial_metadata.FULLTEXT_INDEX,
    )
    fake_qdrant = _FakeQdrantClient()
    extracted_paths: List[str] = []

    monkeypatch.setattr(backfill_financial_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_financial_metadata,
        "_get_qdrant_components",
        lambda: (fake_qdrant, _FakeQdrantModels),
    )
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_metadata_mappings", lambda: None)
    monkeypatch.setattr(backfill_financial_metadata, "ensure_financial_records_index", lambda: None)

    def _extractor(**kwargs):
        extracted_paths.append(str(kwargs.get("path") or ""))
        out = _fake_extraction()
        record = dict(out.records[0])
        record["checksum"] = "chk-3"
        record["document_id"] = "doc-3"
        return types.SimpleNamespace(
            source_family=out.source_family,
            document_metadata=out.document_metadata,
            records=[record],
        )

    monkeypatch.setattr(backfill_financial_metadata, "extract_financial_enrichment", _extractor)
    monkeypatch.setattr(
        backfill_financial_metadata,
        "upsert_financial_records",
        lambda records: {"processed": len(records), "created": len(records), "updated": 0, "errors": 0},
    )

    stats = backfill_financial_metadata.backfill_financial_metadata(
        batch_size=10,
        skip=2,
        limit=1,
        dry_run=False,
        overwrite=False,
    )

    assert stats["scanned_fulltext_docs"] == 1
    assert stats["processed_docs"] == 1
    assert stats["fulltext_updates"] == 1
    assert stats["sidecar_records_processed"] == 1
    assert extracted_paths == ["C:/docs/third.pdf"]
    assert fake_client._scroll_reads == 0

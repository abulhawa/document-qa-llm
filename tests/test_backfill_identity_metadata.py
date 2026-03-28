from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any, Dict, List

import pytest


_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "backfill_identity_metadata.py"


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


@pytest.fixture
def backfill_identity_metadata(monkeypatch: pytest.MonkeyPatch):
    _install_runtime_stubs(monkeypatch)
    module_name = "_test_backfill_identity_metadata"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load backfill_identity_metadata.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeOpenSearchClient:
    def __init__(self, hits: List[Dict[str, Any]]) -> None:
        self._hits = hits
        self._scroll_reads = 0
        self.updated_docs: List[Dict[str, Any]] = []
        self.chunk_updates: List[Dict[str, Any]] = []
        self.cleared_scroll_ids: List[str] = []

    def search(self, index: str, body: Dict[str, Any], params: Dict[str, Any]):
        return {
            "_scroll_id": "scroll-1",
            "hits": {"hits": list(self._hits)},
        }

    def scroll(self, scroll_id: str, params: Dict[str, Any]):
        self._scroll_reads += 1
        return {
            "_scroll_id": scroll_id,
            "hits": {"hits": []},
        }

    def clear_scroll(self, scroll_id: str):
        self.cleared_scroll_ids.append(scroll_id)

    def update(self, index: str, id: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.updated_docs.append({"index": index, "id": id, "body": body, "params": params})

    def update_by_query(self, index: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.chunk_updates.append({"index": index, "body": body, "params": params})
        return {"updated": 3}


def test_build_fulltext_patch_respects_overwrite(backfill_identity_metadata):
    source = {"doc_type": "cv", "person_name": None}
    classified = {"doc_type": "cover_letter", "person_name": "Jane Doe", "authority_rank": 0.9}

    patch_default = backfill_identity_metadata._build_fulltext_patch(
        source,
        classified,
        overwrite=False,
    )
    assert patch_default == {"person_name": "Jane Doe", "authority_rank": 0.9}

    patch_overwrite = backfill_identity_metadata._build_fulltext_patch(
        source,
        classified,
        overwrite=True,
    )
    assert patch_overwrite == {
        "doc_type": "cover_letter",
        "person_name": "Jane Doe",
        "authority_rank": 0.9,
    }


def test_backfill_identity_metadata_updates_fulltext_and_chunks(backfill_identity_metadata, monkeypatch):
    hits = [
        {
            "_id": "doc-1",
            "_source": {
                "checksum": "abc123",
                "path": "C:/docs/jane_resume.pdf",
                "filetype": "pdf",
                "text_full": "Jane Doe\nCurriculum Vitae",
            },
        }
    ]
    fake_client = _FakeOpenSearchClient(hits)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "ensure_identity_metadata_mappings",
        lambda: None,
    )
    monkeypatch.setattr(backfill_identity_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "classify_document",
        lambda path, filetype, full_text: {
            "doc_type": "cv",
            "doc_type_confidence": 0.97,
            "doc_type_source": "rule",
            "person_name": "Jane Doe",
            "authority_rank": 1.0,
        },
    )

    stats = backfill_identity_metadata.backfill_identity_metadata(
        batch_size=50,
        dry_run=False,
        overwrite=False,
    )

    assert stats["scanned_fulltext_docs"] == 1
    assert stats["classified_docs"] == 1
    assert stats["fulltext_updates"] == 1
    assert stats["chunk_update_calls"] == 1
    assert stats["chunk_docs_updated"] == 3
    assert fake_client.updated_docs[0]["id"] == "doc-1"
    assert fake_client.updated_docs[0]["body"]["doc"] == {
        "doc_type": "cv",
        "doc_type_confidence": 0.97,
        "doc_type_source": "rule",
        "person_name": "Jane Doe",
        "authority_rank": 1.0,
    }
    chunk_body = fake_client.chunk_updates[0]["body"]
    assert chunk_body["query"] == {"term": {"checksum": {"value": "abc123"}}}
    assert "ctx._source.doc_type" in chunk_body["script"]["source"]
    assert fake_client.cleared_scroll_ids == ["scroll-1"]


def test_backfill_identity_metadata_dry_run_skips_writes(backfill_identity_metadata, monkeypatch):
    hits = [
        {
            "_id": "doc-2",
            "_source": {
                "checksum": "xyz999",
                "path": "C:/docs/resume.pdf",
                "filetype": "pdf",
                "text_full": "John Doe\nWork Experience",
            },
        }
    ]
    fake_client = _FakeOpenSearchClient(hits)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "ensure_identity_metadata_mappings",
        lambda: None,
    )
    monkeypatch.setattr(backfill_identity_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "classify_document",
        lambda path, filetype, full_text: {
            "doc_type": "cv",
            "doc_type_confidence": 0.97,
            "doc_type_source": "rule",
            "person_name": "John Doe",
            "authority_rank": 1.0,
        },
    )

    stats = backfill_identity_metadata.backfill_identity_metadata(
        batch_size=10,
        dry_run=True,
        overwrite=False,
    )

    assert stats["fulltext_updates"] == 0
    assert stats["fulltext_would_update"] == 1
    assert stats["chunk_update_calls"] == 0
    assert stats["chunk_would_update_calls"] == 1
    assert fake_client.updated_docs == []
    assert fake_client.chunk_updates == []


def test_backfill_identity_metadata_respects_target_doc_type_cohort(backfill_identity_metadata, monkeypatch):
    hits = [
        {
            "_id": "doc-1",
            "_source": {
                "checksum": "abc123",
                "path": "C:/docs/keep.pdf",
                "filetype": "pdf",
                "doc_type": "__missing__",
                "text_full": "document one",
            },
        },
        {
            "_id": "doc-2",
            "_source": {
                "checksum": "xyz999",
                "path": "C:/docs/skip.pdf",
                "filetype": "pdf",
                "doc_type": "cv",
                "text_full": "document two",
            },
        },
    ]
    fake_client = _FakeOpenSearchClient(hits)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "ensure_identity_metadata_mappings",
        lambda: None,
    )
    monkeypatch.setattr(backfill_identity_metadata, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        backfill_identity_metadata,
        "classify_document",
        lambda path, filetype, full_text: {
            "doc_type": "contract",
            "doc_type_confidence": 0.95,
            "doc_type_source": "rule",
            "person_name": None,
            "authority_rank": None,
        },
    )

    stats = backfill_identity_metadata.backfill_identity_metadata(
        batch_size=10,
        dry_run=False,
        overwrite=False,
        target_doc_types=("__missing__", "other"),
    )

    assert stats["scanned_fulltext_docs"] == 2
    assert stats["skipped_not_in_target_cohort"] == 1
    assert stats["classified_docs"] == 1
    assert stats["fulltext_updates"] == 1
    assert stats["chunk_update_calls"] == 1

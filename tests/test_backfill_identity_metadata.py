from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List


_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "backfill_identity_metadata.py"
_SPEC = importlib.util.spec_from_file_location("backfill_identity_metadata", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load backfill_identity_metadata.py")
backfill_identity_metadata = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("backfill_identity_metadata", backfill_identity_metadata)
_SPEC.loader.exec_module(backfill_identity_metadata)


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


def test_build_fulltext_patch_respects_overwrite():
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


def test_backfill_identity_metadata_updates_fulltext_and_chunks(monkeypatch):
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
        "person_name": "Jane Doe",
        "authority_rank": 1.0,
    }
    chunk_body = fake_client.chunk_updates[0]["body"]
    assert chunk_body["query"] == {"term": {"checksum": {"value": "abc123"}}}
    assert "ctx._source.doc_type" in chunk_body["script"]["source"]
    assert fake_client.cleared_scroll_ids == ["scroll-1"]


def test_backfill_identity_metadata_dry_run_skips_writes(monkeypatch):
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

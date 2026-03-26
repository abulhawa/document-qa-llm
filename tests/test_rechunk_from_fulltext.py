from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List


_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rechunk_from_fulltext.py"
_SPEC = importlib.util.spec_from_file_location("rechunk_from_fulltext", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load rechunk_from_fulltext.py")
rechunk_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("rechunk_from_fulltext", rechunk_script)
_SPEC.loader.exec_module(rechunk_script)


class _FakeOpenSearchClient:
    def __init__(self, pages: List[List[Dict[str, Any]]], counts: Dict[str, int] | None = None) -> None:
        self._pages = pages
        self._scroll_idx = 1
        self._counts = counts or {}
        self.search_calls: List[Dict[str, Any]] = []
        self.scroll_calls: List[Dict[str, Any]] = []
        self.cleared_scroll_ids: List[str] = []

    def search(self, index: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.search_calls.append({"index": index, "body": body, "params": params})
        if index == rechunk_script.CHUNKS_INDEX and "count" not in body:
            return {"hits": {"hits": []}}
        first = self._pages[0] if self._pages else []
        return {"_scroll_id": "scroll-1", "hits": {"hits": first}}

    def scroll(self, scroll_id: str, params: Dict[str, Any]):
        self.scroll_calls.append({"scroll_id": scroll_id, "params": params})
        if self._scroll_idx >= len(self._pages):
            return {"_scroll_id": scroll_id, "hits": {"hits": []}}
        page = self._pages[self._scroll_idx]
        self._scroll_idx += 1
        return {"_scroll_id": scroll_id, "hits": {"hits": page}}

    def clear_scroll(self, scroll_id: str):
        self.cleared_scroll_ids.append(scroll_id)

    def count(self, index: str, body: Dict[str, Any]):
        checksum = body["query"]["term"]["checksum"]["value"]
        return {"count": int(self._counts.get(checksum, 0))}


def test_resolve_policy_rules():
    pol_identity = rechunk_script._resolve_policy("cv", "a" * 100)
    assert pol_identity.profile == "profile_identity_native_400_50"
    assert pol_identity.chunk_size == 400
    assert pol_identity.chunk_overlap == 50

    pol_short = rechunk_script._resolve_policy(None, "a" * 1000)
    assert pol_short.profile == "profile_native_short_600_80"
    assert pol_short.chunk_size == 600
    assert pol_short.chunk_overlap == 80
    assert pol_short.length_bucket == "short"

    pol_long = rechunk_script._resolve_policy(None, "a" * 50000)
    assert pol_long.profile == "profile_native_default_800_100"
    assert pol_long.chunk_size == 800
    assert pol_long.chunk_overlap == 100
    assert pol_long.length_bucket == "long"


def test_select_candidates_filters_empty_missing_and_limits():
    pages = [
        [
            {"_source": {"checksum": "a", "path": "C:/a.pdf", "text_full": "hello", "filetype": "pdf"}},
            {"_source": {"checksum": "", "path": "C:/b.pdf", "text_full": "x", "filetype": "pdf"}},
            {"_source": {"checksum": "c", "path": "", "text_full": "x", "filetype": "pdf"}},
            {"_source": {"checksum": "d", "path": "C:/d.pdf", "text_full": "   ", "filetype": "pdf"}},
        ],
        [
            {"_source": {"checksum": "e", "path": "C:/e.pdf", "text_full": "world", "filetype": "pdf"}},
        ],
    ]
    fake = _FakeOpenSearchClient(pages)
    candidates, stats = rechunk_script._select_candidates(
        prefix="C:/",
        checksums=None,
        batch_size=10,
        limit=2,
        client=fake,
    )

    assert len(candidates) == 2
    assert candidates[0].checksum == "a"
    assert candidates[1].checksum == "e"
    assert stats["selected_docs"] == 2
    assert stats["skipped_missing_checksum"] == 1
    assert stats["skipped_missing_path"] == 1
    assert stats["skipped_empty_text"] == 1
    assert fake.cleared_scroll_ids == ["scroll-1"]


def test_build_chunks_sets_expected_fields():
    cand = rechunk_script.FulltextCandidate(
        checksum="abc123",
        path="C:/doc.txt",
        filetype="txt",
        text_full="alpha beta gamma",
        created_at="2026-01-01T00:00:00Z",
        modified_at="2026-01-02T00:00:00Z",
        size_bytes=1234,
        doc_type="cv",
        person_name="Jane Doe",
        authority_rank=1.0,
    )
    policy = rechunk_script.ChunkPolicy(
        profile="profile_identity_native_400_50",
        chunk_size=400,
        chunk_overlap=50,
        length_bucket="short",
    )
    chunks = rechunk_script._build_chunks(cand, policy)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["checksum"] == "abc123"
    assert chunk["path"] == "C:/doc.txt"
    assert chunk["filetype"] == "txt"
    assert chunk["chunk_index"] == 0
    assert chunk["location_percent"] == 0.0
    assert chunk["doc_type"] == "cv"
    assert chunk["person_name"] == "Jane Doe"
    assert chunk["authority_rank"] == 1.0
    assert chunk["chunk_profile"] == "profile_identity_native_400_50"
    assert chunk["chunk_policy_version"] == "v1"
    assert chunk["chunk_size"] == 400
    assert chunk["chunk_overlap"] == 50


def test_apply_uses_predelete_qdrant_count_when_delete_metric_missing(monkeypatch):
    cand = rechunk_script.FulltextCandidate(
        checksum="abc123",
        path="C:/doc.txt",
        filetype="txt",
        text_full="alpha beta gamma",
        created_at=None,
        modified_at=None,
        size_bytes=0,
        doc_type=None,
        person_name=None,
        authority_rank=None,
    )
    monkeypatch.setattr(
        rechunk_script,
        "_select_candidates",
        lambda **kwargs: (
            [cand],
            {
                "scanned_docs": 1,
                "selected_docs": 1,
                "skipped_empty_text": 0,
                "skipped_missing_checksum": 0,
                "skipped_missing_path": 0,
            },
        ),
    )
    monkeypatch.setattr(
        rechunk_script,
        "_build_chunks",
        lambda candidate, policy: [{"id": "1", "text": "x", "checksum": "abc123"}],
    )
    monkeypatch.setattr(rechunk_script, "_count_os_chunks_by_checksum", lambda client, checksum: 5)
    monkeypatch.setattr(rechunk_script, "count_qdrant_chunks_by_checksum", lambda checksum: 7)
    monkeypatch.setattr(rechunk_script, "delete_vectors_by_checksum", lambda checksum: 0)
    monkeypatch.setattr(rechunk_script, "delete_chunks_by_checksum", lambda checksum: 5)
    monkeypatch.setattr(rechunk_script, "_index_new_chunks", lambda chunks: (1, 0))

    summary = rechunk_script.rechunk_from_fulltext(
        prefix=None,
        checksums=["abc123"],
        limit=1,
        batch_size=1,
        apply=True,
        client=object(),
    )

    assert summary["rebuilt_docs"] == 1
    assert summary["failed_docs"] == 0
    assert summary["deleted_old_vectors"] == 7
    assert summary["deleted_old_chunks"] == 5
    assert summary["indexed_new_chunks"] == 1

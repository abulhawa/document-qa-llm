from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_fulltext_rechunk_candidates.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_fulltext_rechunk_candidates",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load audit_fulltext_rechunk_candidates.py")
audit_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("audit_fulltext_rechunk_candidates", audit_script)
_SPEC.loader.exec_module(audit_script)


class _FakeOpenSearchClient:
    def __init__(self, pages: List[List[Dict[str, Any]]]) -> None:
        self._pages = pages
        self._scroll_idx = 1
        self.search_calls: List[Dict[str, Any]] = []
        self.scroll_calls: List[Dict[str, Any]] = []
        self.cleared_scroll_ids: List[str] = []

    def search(self, index: str, body: Dict[str, Any], params: Dict[str, Any]):
        self.search_calls.append({"index": index, "body": body, "params": params})
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


def test_length_bucket_and_profile_rules():
    assert audit_script._length_bucket(100) == "short"
    assert audit_script._length_bucket(3000) == "short"
    assert audit_script._length_bucket(3001) == "medium"
    assert audit_script._length_bucket(20001) == "long"

    assert (
        audit_script._recommended_profile("cv", "short")
        == "profile_identity_native_400_50"
    )
    assert (
        audit_script._recommended_profile("__missing__", "short")
        == "profile_native_short_600_80"
    )
    assert (
        audit_script._recommended_profile("__missing__", "medium")
        == "profile_native_default_800_100"
    )


def test_audit_counts_eligibility_buckets_and_scroll_cleanup():
    pages = [
        [
            {
                "_id": "doc-1",
                "_source": {
                    "checksum": "aaa",
                    "filetype": "pdf",
                    "doc_type": "cv",
                    "text_full": "alpha",
                },
            },
            {
                "_id": "doc-2",
                "_source": {
                    "checksum": "bbb",
                    "filetype": "pdf",
                    "doc_type": "cv",
                    "text_full": "   ",
                },
            },
        ],
        [
            {
                "_id": "doc-3",
                "_source": {
                    "filetype": "docx",
                    "text_full": "x" * 5001,
                },
            }
        ],
    ]
    fake = _FakeOpenSearchClient(pages)

    stats = audit_script.audit_fulltext_rechunk_candidates(
        client=fake,
        batch_size=2,
        sample_size=5,
    )

    assert stats["scanned_docs"] == 3
    assert stats["eligible_docs"] == 2
    assert stats["skipped_empty_text"] == 1
    assert stats["missing_checksum"] == 1
    assert stats["by_filetype"] == {"pdf": 1, "docx": 1}
    assert stats["by_doc_type"] == {"cv": 1, "__missing__": 1}
    assert stats["by_length_bucket"] == {"short": 1, "medium": 1}
    assert stats["by_profile"] == {
        "profile_identity_native_400_50": 1,
        "profile_native_default_800_100": 1,
    }
    assert stats["sample_checksums"] == ["aaa", "doc-3"]
    assert fake.cleared_scroll_ids == ["scroll-1"]


def test_search_body_uses_normalized_prefix():
    body = audit_script._search_body(batch_size=100, prefix="C:\\Users\\ali_a\\My Drive")
    assert body["query"]["prefix"]["path"] == "C:/Users/ali_a/My Drive"


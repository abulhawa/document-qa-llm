#!/usr/bin/env python
from __future__ import annotations

"""
Audit existing full-text docs for fulltext-only rechunk migration.

What it does
- Scans FULLTEXT_INDEX docs (optionally under a path prefix)
- Marks docs with non-empty text_full as rechunk-eligible
- Produces deterministic buckets for:
  - filetype
  - doc_type
  - length_bucket
  - recommended chunk profile

Notes
- Dry-run only: does not modify OpenSearch or Qdrant
- Intended as step 1 before any chunk/vector migration work

Usage
- Audit all docs:
    python scripts/audit_fulltext_rechunk_candidates.py
- Audit specific prefix:
    python scripts/audit_fulltext_rechunk_candidates.py --prefix "C:/Users/ali_a/My Drive"
- Audit first 500 docs:
    python scripts/audit_fulltext_rechunk_candidates.py --limit 500
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FULLTEXT_INDEX  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from utils.file_utils import normalize_path  # noqa: E402

_IDENTITY_DOC_TYPES = {"cv", "cover_letter", "reference_letter"}
_MISSING = "__missing__"
_SHORT_MAX_CHARS = 3000
_MEDIUM_MAX_CHARS = 20000


def _length_bucket(char_len: int) -> str:
    if char_len <= _SHORT_MAX_CHARS:
        return "short"
    if char_len <= _MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


def _recommended_profile(doc_type: str, length_bucket: str) -> str:
    if doc_type in _IDENTITY_DOC_TYPES:
        return "profile_identity_native_400_50"
    if length_bucket == "short":
        return "profile_native_short_600_80"
    return "profile_native_default_800_100"


def _search_body(*, batch_size: int, prefix: Optional[str]) -> Dict[str, Any]:
    query: Dict[str, Any] = {"match_all": {}}
    if prefix:
        query = {"prefix": {"path": normalize_path(prefix)}}

    return {
        "size": batch_size,
        "query": query,
        "_source": ["checksum", "path", "filetype", "doc_type", "text_full"],
        "sort": [{"_id": "asc"}],
    }


def _increment(counter: Counter[str], key: Optional[str]) -> None:
    if key is None:
        counter[_MISSING] += 1
        return
    text = str(key).strip()
    counter[text or _MISSING] += 1


def audit_fulltext_rechunk_candidates(
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
    prefix: Optional[str] = None,
    sample_size: int = 20,
    client: Any = None,
) -> Dict[str, Any]:
    os_client = client or get_client()

    stats: Dict[str, Any] = {
        "scanned_docs": 0,
        "eligible_docs": 0,
        "skipped_empty_text": 0,
        "missing_checksum": 0,
        "sample_checksums": [],
    }
    by_filetype: Counter[str] = Counter()
    by_doc_type: Counter[str] = Counter()
    by_length_bucket: Counter[str] = Counter()
    by_profile: Counter[str] = Counter()

    scroll_id = ""

    try:
        response = os_client.search(
            index=FULLTEXT_INDEX,
            body=_search_body(batch_size=max(1, batch_size), prefix=prefix),
            params={"scroll": "2m"},
        )
        scroll_id = response.get("_scroll_id", "")

        while True:
            hits = response.get("hits", {}).get("hits", []) or []
            if not hits:
                break

            for hit in hits:
                if limit is not None and stats["scanned_docs"] >= limit:
                    break

                stats["scanned_docs"] += 1
                source = hit.get("_source") or {}

                text_full = str(source.get("text_full") or "")
                if not text_full.strip():
                    stats["skipped_empty_text"] += 1
                    continue

                stats["eligible_docs"] += 1

                checksum = source.get("checksum")
                if not checksum:
                    stats["missing_checksum"] += 1
                    checksum = hit.get("_id")
                if checksum and len(stats["sample_checksums"]) < max(0, sample_size):
                    stats["sample_checksums"].append(str(checksum))

                doc_type = str(source.get("doc_type") or "").strip() or _MISSING
                filetype = str(source.get("filetype") or "").strip() or _MISSING
                length_bucket = _length_bucket(len(text_full))
                profile = _recommended_profile(doc_type, length_bucket)

                _increment(by_filetype, filetype)
                _increment(by_doc_type, doc_type)
                _increment(by_length_bucket, length_bucket)
                _increment(by_profile, profile)

            if limit is not None and stats["scanned_docs"] >= limit:
                break

            if not scroll_id:
                break
            response = os_client.scroll(scroll_id=scroll_id, params={"scroll": "2m"})
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                os_client.clear_scroll(scroll_id=scroll_id)
            except Exception:  # noqa: BLE001
                pass

    stats["by_filetype"] = dict(by_filetype)
    stats["by_doc_type"] = dict(by_doc_type)
    stats["by_length_bucket"] = dict(by_length_bucket)
    stats["by_profile"] = dict(by_profile)
    return stats


def _print_counter(title: str, values: Dict[str, int]) -> None:
    print(title)
    if not values:
        print("  (none)")
        return
    for key in sorted(values.keys(), key=lambda k: (-values[k], k)):
        print(f"  {key}: {values[key]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step-1 dry-run audit for fulltext-only rechunk migration."
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of eligible checksum samples to print (default 20)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output only",
    )
    args = ap.parse_args()

    stats = audit_fulltext_rechunk_candidates(
        batch_size=max(1, args.batch_size),
        limit=args.limit,
        prefix=args.prefix,
        sample_size=max(0, args.sample_size),
    )

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    scanned = int(stats["scanned_docs"])
    eligible = int(stats["eligible_docs"])
    skipped_empty = int(stats["skipped_empty_text"])
    pct = round((eligible * 100.0) / max(scanned, 1), 2)
    print(
        "Audit complete: "
        f"scanned_docs={scanned} "
        f"eligible_docs={eligible} "
        f"eligible_pct={pct} "
        f"skipped_empty_text={skipped_empty} "
        f"missing_checksum={stats['missing_checksum']}"
    )
    _print_counter("Eligible by filetype:", stats["by_filetype"])
    _print_counter("Eligible by doc_type:", stats["by_doc_type"])
    _print_counter("Eligible by length_bucket:", stats["by_length_bucket"])
    _print_counter("Recommended profile counts:", stats["by_profile"])
    sample_checksums = stats.get("sample_checksums", [])
    print(f"Sample checksums ({len(sample_checksums)}):")
    if sample_checksums:
        for item in sample_checksums:
            print(f"  {item}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()


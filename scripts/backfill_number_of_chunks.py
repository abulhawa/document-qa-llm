#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import WATCH_INVENTORY_INDEX  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from utils.opensearch.indexes import ensure_index_exists  # noqa: E402
from utils.file_utils import normalize_path  # noqa: E402
from utils.inventory import set_inventory_number_of_chunks  # noqa: E402

# Use the same pipeline pieces as ingest to compute the true, expected chunk count
from core.file_loader import load_documents  # noqa: E402
from core.document_preprocessor import preprocess_to_documents, PreprocessConfig  # noqa: E402
from core.chunking import split_documents  # noqa: E402


def _search_missing_number_of_chunks(prefix: str, batch: int = 500) -> Tuple[str, List[str]]:
    client = get_client()
    npref = normalize_path(prefix)
    body: Dict[str, Any] = {
        "size": batch,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"exists_now": True}},
                    {"prefix": {"path": npref}},
                ],
                "must_not": [{"exists": {"field": "number_of_chunks"}}],
            }
        },
        "_source": ["path"],
        "sort": [{"path": "asc"}],
    }
    resp = client.search(index=WATCH_INVENTORY_INDEX, body=body, scroll="2m")
    return resp.get("_scroll_id", ""), [
        (h.get("_source", {}) or {}).get("path") for h in resp.get("hits", {}).get("hits", [])
        if (h.get("_source", {}) or {}).get("path")
    ]


def _scroll_next(scroll_id: str) -> Tuple[str, List[str]]:
    client = get_client()
    resp = client.scroll(scroll_id=scroll_id, scroll="2m")
    return resp.get("_scroll_id", ""), [
        (h.get("_source", {}) or {}).get("path") for h in resp.get("hits", {}).get("hits", [])
        if (h.get("_source", {}) or {}).get("path")
    ]


def _compute_expected_chunks(path: str) -> int:
    io_path = normalize_path(path)
    # Bail if not present locally
    if not os.path.exists(io_path):
        return 0
    try:
        docs = load_documents(io_path)
    except Exception:
        return 0
    try:
        ext = os.path.splitext(io_path)[1].lower().lstrip(".")
        docs_list = preprocess_to_documents(
            docs_like=docs, source_path=io_path, cfg=PreprocessConfig(), doc_type=ext
        )
    except Exception:
        docs_list = docs
    try:
        chunks = split_documents(docs_list)
        return int(len(chunks))
    except Exception:
        return 0


def backfill(prefix: str, *, limit: int, max_workers: int, dry_run: bool) -> Dict[str, int]:
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    scroll_id, batch = _search_missing_number_of_chunks(prefix, batch=min(limit, 500))
    attempted = 0
    updated = 0
    missing_local = 0
    errors = 0

    def _work(p: str) -> Tuple[str, int]:
        n = _compute_expected_chunks(p)
        return p, n

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Process first batch
        futures = {ex.submit(_work, p): p for p in batch}
        while futures and attempted < limit:
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    path, n = fut.result()
                except Exception:
                    errors += 1
                    continue
                attempted += 1
                if n <= 0:
                    # 0 either means no chunks or file not accessible
                    if not os.path.exists(normalize_path(p)):
                        missing_local += 1
                    continue
                if not dry_run:
                    try:
                        set_inventory_number_of_chunks(path, n)
                        updated += 1
                    except Exception:
                        errors += 1
                else:
                    updated += 1  # count as would-update
                if attempted >= limit:
                    break
            if attempted >= limit:
                break

            # fetch next page
            futures.clear()
            if not scroll_id:
                break
            scroll_id, batch = _scroll_next(scroll_id)
            if not batch:
                break
            for p in batch:
                if attempted >= limit:
                    break
                futures[ex.submit(_work, p)] = p

    # Try to clear the scroll context
    try:
        if scroll_id:
            get_client().clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return {
        "attempted": attempted,
        "updated": updated,
        "missing_local": missing_local,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "One-off backfill: compute true number_of_chunks by loading and splitting files."
        )
    )
    ap.add_argument("prefix", help="Path prefix/folder to target (e.g., C:/Docs)")
    ap.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of inventory entries to process (default 2000)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for local file loading (default 8)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts but do not write to inventory",
    )
    args = ap.parse_args()

    ensure_index_exists(WATCH_INVENTORY_INDEX)
    summary = backfill(
        args.prefix, limit=max(1, args.limit), max_workers=max(1, args.workers), dry_run=args.dry_run
    )
    print(
        "Backfill finished: attempted={attempted}, updated={updated}, missing_local={missing_local}, errors={errors}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()

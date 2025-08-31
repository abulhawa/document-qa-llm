#!/usr/bin/env python
from __future__ import annotations
"""
Mark zero‑chunk files as indexed in the inventory.

What it does
- Finds WATCH_INVENTORY_INDEX docs under a prefix where:
  exists_now == true AND number_of_chunks == 0 AND last_indexed is missing
- Sets last_indexed and last_seen to now (UTC), ensuring exists_now == true

Notes
- Does NOT delete any data
- Use this after backfilling number_of_chunks so zero‑content files don’t show up as “remaining”

Usage
- Preview only:
    python scripts/mark_zero_chunk_indexed.py "C:/data/folder" --dry-run
- Apply updates:
    python scripts/mark_zero_chunk_indexed.py "C:/data/folder"
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Ensure repo root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import WATCH_INVENTORY_INDEX  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from utils.opensearch.indexes import ensure_index_exists  # noqa: E402
from utils.file_utils import normalize_path  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_zero_chunk_as_indexed(prefix: str, *, dry_run: bool = False) -> Dict[str, int]:
    """Set last_indexed for inventory docs under prefix with number_of_chunks == 0 and missing last_indexed.

    Returns a summary dict with counts.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    n_pref = normalize_path(prefix)

    # Count matching docs
    body_match: Dict[str, Any] = {
        "size": 0,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"exists_now": True}},
                    {"prefix": {"path": n_pref}},
                    {"term": {"number_of_chunks": 0}},
                ],
                "must_not": [{"exists": {"field": "last_indexed"}}],
            }
        },
    }
    resp = client.search(index=WATCH_INVENTORY_INDEX, body=body_match)
    total = int(resp.get("hits", {}).get("total", {}).get("value", 0))

    updated = 0
    if not dry_run and total > 0:
        upd = client.update_by_query(
            index=WATCH_INVENTORY_INDEX,
            body={
                "script": {
                    "source": "ctx._source.last_indexed = params.t; ctx._source.last_seen = params.t; ctx._source.exists_now = true",
                    "lang": "painless",
                    "params": {"t": _now_iso()},
                },
                "query": body_match["query"],
            },
            params={"refresh": "true", "conflicts": "proceed"},
        )
        updated = int(upd.get("updated", 0))

    return {"matched": total, "updated": updated}


def main() -> None:
    ap = argparse.ArgumentParser(description="Mark zero-chunk files as indexed in the inventory")
    ap.add_argument("prefix", help="Path prefix/folder to target (e.g., C:/Docs)")
    ap.add_argument("--dry-run", action="store_true", help="Count matches without updating")
    args = ap.parse_args()

    summary = mark_zero_chunk_as_indexed(args.prefix, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would update {summary['matched']} doc(s)")
    else:
        print(f"Updated {summary['updated']} of {summary['matched']} matching doc(s)")


if __name__ == "__main__":
    main()

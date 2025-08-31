#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple

# Make repo importable when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    WATCH_INVENTORY_INDEX,
    INGEST_LOG_INDEX,
)
from core.opensearch_client import get_client  # noqa: E402
from utils.file_utils import normalize_path  # noqa: E402
from utils.opensearch.indexes import ensure_index_exists  # noqa: E402
from utils.opensearch_utils import (  # noqa: E402
    get_chunk_ids_by_path,
    delete_chunks_by_path,
    delete_fulltext_by_path,
)
from utils.qdrant_utils import delete_vectors_by_ids  # noqa: E402


def iter_paths_by_prefix(prefix: str, page_size: int = 1000) -> Iterator[Tuple[str, int]]:
    """Yield (path, doc_count) for all files in CHUNKS_INDEX under prefix.

    Uses composite aggregation to page through unique paths efficiently.
    """
    client = get_client()
    after_key = None
    pref = normalize_path(prefix)
    while True:
        body: Dict[str, Any] = {
            "size": 0,
            "query": {"wildcard": {"path.keyword": f"{pref}*"}},
            "aggs": {
                "paths": {
                    "composite": {
                        "size": page_size,
                        "sources": [{"p": {"terms": {"field": "path.keyword"}}}],
                    }
                }
            },
        }
        if after_key:
            body["aggs"]["paths"]["composite"]["after"] = after_key
        resp = client.search(index=CHUNKS_INDEX, body=body)
        agg = resp.get("aggregations", {}).get("paths", {})
        buckets = agg.get("buckets", [])
        if not buckets:
            break
        for b in buckets:
            yield b["key"]["p"], int(b.get("doc_count", 0))
        after_key = agg.get("after_key")


def purge_by_prefix(
    prefix: str,
    *,
    cap_paths: int | None,
    batch_ids: int,
    delete_inventory: bool,
    delete_logs: bool,
    dry_run: bool,
) -> Dict[str, int]:
    ensure_index_exists(CHUNKS_INDEX)
    ensure_index_exists(FULLTEXT_INDEX)
    ensure_index_exists(WATCH_INVENTORY_INDEX)

    client = get_client()
    total_paths = 0
    total_vec_deleted = 0
    total_chunks_deleted = 0
    total_fulltext_deleted = 0
    total_inventory_deleted = 0

    pending_ids: List[str] = []

    for path, doc_count in iter_paths_by_prefix(prefix):
        total_paths += 1
        if cap_paths and total_paths > cap_paths:
            break

        # Gather Qdrant IDs before deleting from OpenSearch
        ids = [] if dry_run else get_chunk_ids_by_path(path, size=max(10000, doc_count))
        if not dry_run:
            pending_ids.extend(ids)
            # Flush in batches
            while len(pending_ids) >= batch_ids:
                batch = pending_ids[:batch_ids]
                del pending_ids[:batch_ids]
                total_vec_deleted += delete_vectors_by_ids(batch)

        # Delete from OpenSearch indices (chunks + fulltext)
        if dry_run:
            total_chunks_deleted += int(doc_count)
        else:
            total_chunks_deleted += delete_chunks_by_path(path)
        if dry_run:
            total_fulltext_deleted += 1  # approximate; 1 doc per file in full-text
        else:
            total_fulltext_deleted += delete_fulltext_by_path(path)

        # Inventory: delete (or keep if requested)
        if delete_inventory:
            if dry_run:
                total_inventory_deleted += 1
            else:
                resp = client.delete_by_query(
                    index=WATCH_INVENTORY_INDEX,
                    body={"query": {"term": {"path.keyword": path}}},
                    params={"refresh": "true", "conflicts": "proceed"},
                )
                total_inventory_deleted += int(resp.get("deleted", 0))

    # Flush remaining vector IDs
    if not dry_run and pending_ids:
        total_vec_deleted += delete_vectors_by_ids(pending_ids)
        pending_ids.clear()

    # Optional: delete ingest logs by prefix (best-effort)
    if delete_logs and not dry_run:
        try:
            ensure_index_exists(INGEST_LOG_INDEX)
            client.delete_by_query(
                index=INGEST_LOG_INDEX,
                body={"query": {"wildcard": {"path.keyword": f"{normalize_path(prefix)}*"}}},
                params={"refresh": "true", "conflicts": "proceed"},
            )
        except Exception:
            pass

    return {
        "paths": total_paths,
        "vectors": total_vec_deleted,
        "chunks": total_chunks_deleted,
        "fulltext": total_fulltext_deleted,
        "inventory": total_inventory_deleted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete OS docs and Qdrant vectors by path prefix.")
    ap.add_argument("prefix", help="Path prefix to purge (e.g., C:/data/folder)")
    ap.add_argument("--cap-paths", type=int, default=None, help="Max number of distinct files to process")
    ap.add_argument("--batch-ids", type=int, default=1000, help="Batch size for Qdrant deletes by ID")
    ap.add_argument("--keep-inventory", action="store_true", help="Do not delete WATCH_INVENTORY docs")
    ap.add_argument("--also-logs", action="store_true", help="Also delete entries from INGEST_LOG_INDEX if path is stored")
    ap.add_argument("--dry-run", action="store_true", help="Print summary without deleting")
    args = ap.parse_args()

    summary = purge_by_prefix(
        args.prefix,
        cap_paths=args.cap_paths,
        batch_ids=max(1, args.batch_ids),
        delete_inventory=not args.keep_inventory,
        delete_logs=args.also_logs,
        dry_run=args.dry_run,
    )
    print(
        "Done. paths={paths}, vectors={vectors}, chunks={chunks}, fulltext={fulltext}, inventory={inventory}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()


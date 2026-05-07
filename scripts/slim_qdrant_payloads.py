#!/usr/bin/env python
"""
scripts/slim_qdrant_payloads.py
================================
One-time migration: strip bloated payloads from existing Qdrant points down
to the minimal set (id, checksum, path).

Background
----------
Before the payload-slim change in utils/qdrant_utils.py, every ingested chunk
stored its full metadata in Qdrant (doc_type, financial fields, chunk_index,
etc.). This was redundant because OpenSearch is the source of truth for all
metadata — Qdrant is only used for ANN search.

This script cleans up existing points WITHOUT re-embedding. Vectors are never
touched. It uses Qdrant's set_payload() + clear_payload() to overwrite each
point's payload with only the three keys we actually need.

Usage
-----
    # Preview what would change (no writes):
    python scripts/slim_qdrant_payloads.py --dry-run

    # Run the migration:
    python scripts/slim_qdrant_payloads.py

    # Run with explicit batch size (default 256):
    python scripts/slim_qdrant_payloads.py --batch-size 512

    # Limit to N points (useful for testing):
    python scripts/slim_qdrant_payloads.py --limit 100 --dry-run

What it does per point
----------------------
1. Scroll all points with payloads (no vectors fetched — faster).
2. For each point, check if its payload contains any keys beyond the
   allowed set {id, checksum, path}.
3. If bloated: call set_payload() with only the slim keys, then call
   clear_payload() for all excess keys.
4. If already slim: skip (no write).

Output
------
Prints a summary at the end:
    scanned=N  already_slim=N  slimmed=N  errors=N  dry_run=True/False
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QDRANT_COLLECTION, QDRANT_URL, logger  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models  # noqa: E402

# Keys we want to keep — everything else is excess and will be removed.
_SLIM_KEYS = {"id", "checksum", "path"}


def slim_qdrant_payloads(
    *,
    batch_size: int = 256,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    client = QdrantClient(url=QDRANT_URL)

    stats = {
        "scanned": 0,
        "already_slim": 0,
        "slimmed": 0,
        "errors": 0,
        "dry_run": dry_run,
    }

    offset = None  # Qdrant scroll pagination token

    while True:
        try:
            result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # never fetch vectors — not needed
            )
        except Exception as e:
            logger.error("Qdrant scroll failed at offset=%s: %s", offset, e)
            stats["errors"] += 1
            break

        points, next_offset = result

        if not points:
            break

        for point in points:
            stats["scanned"] += 1

            payload = point.payload or {}
            excess_keys = set(payload.keys()) - _SLIM_KEYS

            if not excess_keys:
                stats["already_slim"] += 1
                continue

            # Build the slim payload from whatever slim keys exist on this point
            slim_payload = {k: payload[k] for k in _SLIM_KEYS if k in payload}

            if dry_run:
                logger.info(
                    "[DRY RUN] Would slim point id=%s — removing keys: %s",
                    point.id,
                    sorted(excess_keys),
                )
                stats["slimmed"] += 1
                continue

            try:
                # Step 1: overwrite with only the slim keys
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload=slim_payload,
                    points=[point.id],
                    wait=True,
                )
                # Step 2: explicitly delete the excess keys
                # (set_payload merges, it does not replace — so we must clear excess)
                client.delete_payload(
                    collection_name=QDRANT_COLLECTION,
                    keys=list(excess_keys),
                    points=[point.id],
                    wait=True,
                )
                stats["slimmed"] += 1
            except Exception as e:
                logger.error("Failed to slim point id=%s: %s", point.id, e)
                stats["errors"] += 1

            if limit and stats["scanned"] >= limit:
                break

        if limit and stats["scanned"] >= limit:
            logger.info("Reached --limit %d, stopping.", limit)
            break

        if next_offset is None:
            break

        offset = next_offset

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip bloated Qdrant payloads down to {id, checksum, path}."
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="Number of points to scroll per batch (default: 256).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Stop after scanning this many points (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would change without writing anything.",
    )
    args = parser.parse_args()

    stats = slim_qdrant_payloads(
        batch_size=max(1, args.batch_size),
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print(
        "slim_qdrant_payloads complete: "
        "scanned={scanned} "
        "already_slim={already_slim} "
        "slimmed={slimmed} "
        "errors={errors} "
        "dry_run={dry_run}".format(**stats)
    )


if __name__ == "__main__":
    main()

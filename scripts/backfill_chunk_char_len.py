from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from core.opensearch_client import get_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList

from config import CHUNKS_INDEX, QDRANT_COLLECTION, QDRANT_URL, logger
from utils.opensearch_utils import ensure_chunk_char_len_mapping


def _resolve_checksum_sort_field(os_client) -> str:
    mapping = os_client.indices.get_mapping(index=CHUNKS_INDEX)
    props = (
        mapping.get(CHUNKS_INDEX, {})
        .get("mappings", {})
        .get("properties", {})
        or {}
    )
    checksum_props = props.get("checksum")
    if isinstance(checksum_props, dict):
        fields = checksum_props.get("fields", {})
        if isinstance(fields, dict) and "keyword" in fields:
            return "checksum.keyword"
    return "checksum"


def _iter_chunks(
    os_client,
    *,
    batch_size: int,
    limit: Optional[int],
    checksum_sort_field: str,
) -> Iterable[list[dict[str, Any]]]:
    after = None
    processed = 0
    while True:
        if limit is not None:
            remaining = limit - processed
            if remaining <= 0:
                break
            size = min(batch_size, remaining)
        else:
            size = batch_size

        body: Dict[str, Any] = {
            "size": size,
            "_source": ["text"],
            "sort": [
                {checksum_sort_field: "asc"},
                {"chunk_index": "asc"},
                {"_id": "asc"},
            ],
        }
        if after:
            body["search_after"] = after
        resp = os_client.search(index=CHUNKS_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", []) or []
        if not hits:
            break
        yield hits
        processed += len(hits)
        after = hits[-1].get("sort")


def backfill_chunk_char_len(
    *,
    batch_size: int = 500,
    qdrant_batch_size: int = 200,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    os_client = get_client()
    qdrant_client = QdrantClient(url=QDRANT_URL)

    if dry_run:
        logger.info("Dry run enabled: skipping OpenSearch mapping update.")
    else:
        ensure_chunk_char_len_mapping()

    checksum_sort_field = _resolve_checksum_sort_field(os_client)
    stats = {"processed": 0, "updated": 0, "qdrant_not_found": 0, "errors": 0}

    for hits in _iter_chunks(
        os_client,
        batch_size=batch_size,
        limit=limit,
        checksum_sort_field=checksum_sort_field,
    ):
        id_to_len: dict[str, int] = {}
        for hit in hits:
            chunk_id = hit.get("_id")
            if not chunk_id:
                continue
            text = (hit.get("_source") or {}).get("text")
            id_to_len[str(chunk_id)] = len(text or "")

        if not id_to_len:
            continue

        ids = list(id_to_len.keys())
        stats["processed"] += len(ids)

        try:
            points = qdrant_client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=ids,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to retrieve Qdrant points: %s", exc)
            stats["errors"] += len(ids)
            continue

        found_ids = {str(point.id) for point in points}
        stats["qdrant_not_found"] += len(ids) - len(found_ids)

        if dry_run:
            logger.info(
                "Dry run: would update %s Qdrant points in this batch.",
                len(found_ids),
            )
            continue

        payload_groups: dict[int, list[str]] = defaultdict(list)
        for chunk_id in found_ids:
            payload_groups[id_to_len[chunk_id]].append(chunk_id)

        for chunk_char_len, group in payload_groups.items():
            for i in range(0, len(group), qdrant_batch_size):
                batch_ids = group[i : i + qdrant_batch_size]
                try:
                    qdrant_client.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={"chunk_char_len": chunk_char_len},
                        points=PointIdsList(points=batch_ids),
                        wait=True,
                    )
                    stats["updated"] += len(batch_ids)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to set payload for %s points: %s",
                        len(batch_ids),
                        exc,
                    )
                    stats["errors"] += len(batch_ids)

    logger.info(
        "Backfill complete. processed=%s updated=%s qdrant_not_found=%s errors=%s",
        stats["processed"],
        stats["updated"],
        stats["qdrant_not_found"],
        stats["errors"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chunk_char_len into Qdrant payloads."
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--qdrant-batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backfill_chunk_char_len(
        batch_size=args.batch_size,
        qdrant_batch_size=args.qdrant_batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

from typing import List, Dict, Any, Optional, Tuple
from core.opensearch_store import client, get_client, INDEX_NAME
from typing import Iterable
from opensearchpy import exceptions
from config import OPENSEARCH_DELETE_BATCH, OPENSEARCH_REQUEST_TIMEOUT, logger


def list_files_from_opensearch(
    size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Retrieve a list of unique files indexed in OpenSearch, grouped by checksum.

    Args:
        client: OpenSearch client instance
        index_name: OpenSearch index where chunks are stored
        size: maximum number of unique files to return

    Returns:
        List of file metadata dicts
    """
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "files": {
                    "terms": {"field": "checksum", "size": size},
                    "aggs": {
                        "top_chunk": {
                            "top_hits": {
                                "size": 1,
                                "_source": [
                                    "path",
                                    "created_at",
                                    "modified_at",
                                    "indexed_at",
                                    "filetype",
                                ],
                                "sort": [{"indexed_at": "desc"}],
                            }
                        }
                    },
                }
            },
        },
    )

    results = []
    for bucket in response["aggregations"]["files"]["buckets"]:
        checksum = bucket["key"]
        doc_count = bucket["doc_count"]
        top_hit = bucket["top_chunk"]["hits"]["hits"][0]
        top_source = top_hit["_source"]
        top_chunk_id = top_hit["_id"]

        results.append(
            {
                "checksum": checksum,
                "path": top_source.get("path"),
                "filename": top_source.get("path", "").split("/")[-1],
                "created_at": top_source.get("created_at"),
                "modified_at": top_source.get("modified_at"),
                "indexed_at": top_source.get("indexed_at"),
                "filetype": top_source.get("filetype"),
                "num_chunks": doc_count,
                "first_chunk_id": top_chunk_id,
            }
        )

    return results


def delete_files_by_checksum(checksums: Iterable[str]) -> int:
    """Delete all OpenSearch docs that match any of the given checksums.
    Uses batched `terms` delete_by_query for speed. Returns total deleted count.
    """
    total_deleted = 0
    unique = [c for c in {c for c in checksums if c}]
    if not unique:
        return 0

    # Chunk to avoid overly large queries (safe default 1024)
    CHUNK = OPENSEARCH_DELETE_BATCH

    for i in range(0, len(unique), CHUNK):
        batch = unique[i : i + CHUNK]
        try:
            resp = client.delete_by_query(
                index=INDEX_NAME,
                body={"query": {"terms": {"checksum": batch}}},
                params={
                    "refresh": "true",
                    "conflicts": "proceed",
                    "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                },
            )
            deleted = int(resp.get("deleted", 0))
            total_deleted += deleted
            logger.info(
                f"🗑️ OpenSearch deleted {deleted} docs for {len(batch)} checksum(s)."
            )
        except exceptions.OpenSearchException as e:
            logger.exception(
                f"OpenSearch delete failed for a batch of {len(batch)} checksum(s): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error deleting a batch: {e}")

    return total_deleted


def get_duplicate_checksums(limit: int = 10000) -> List[str]:
    """Return checksums that appear under more than one distinct path.
    Uses an aggregation over `checksum` with a cardinality sub-agg on `path`.
    """
    try:
        body = {
            "size": 0,
            "aggs": {
                "by_checksum": {
                    "terms": {"field": "checksum", "size": limit},
                    "aggs": {
                        "distinct_paths": {"cardinality": {"field": "path"}},
                    },
                }
            },
        }
        resp = client.search(index=INDEX_NAME, body=body)
        buckets = resp.get("aggregations", {}).get("by_checksum", {}).get("buckets", [])
        dups = [
            b["key"] for b in buckets if b.get("distinct_paths", {}).get("value", 0) > 1
        ]
        return dups
    except Exception as e:
        # Fallback: if mapping uses 'path.keyword'
        try:
            body = {
                "size": 0,
                "aggs": {
                    "by_checksum": {
                        "terms": {"field": "checksum", "size": limit},
                        "aggs": {
                            "distinct_paths": {
                                "cardinality": {"field": "path.keyword"}
                            },
                        },
                    }
                },
            }
            resp = client.search(index=INDEX_NAME, body=body)
            buckets = (
                resp.get("aggregations", {}).get("by_checksum", {}).get("buckets", [])
            )
            dups = [
                b["key"]
                for b in buckets
                if b.get("distinct_paths", {}).get("value", 0) > 1
            ]
            return dups
        except Exception:
            return []


def set_has_embedding_true_by_ids(ids: Iterable[str]) -> Tuple[int, int]:
    """
    Bulk-update docs by _id to set has_embedding=True.
    Returns (updated_or_noop_count, error_count).
    Idempotent: running it twice is safe.
    """
    ids = [i for i in dict.fromkeys(ids) if i]  # dedupe, keep order, drop falsy
    if not ids:
        return (0, 0)

    ops = []
    for doc_id in ids:
        ops.append({"update": {"_index": INDEX_NAME, "_id": doc_id}})
        ops.append({"doc": {"has_embedding": True}})

    client = get_client()
    resp = client.bulk(
        body=ops,
        params={"refresh": "true", "timeout": OPENSEARCH_REQUEST_TIMEOUT},
    )

    updated = 0
    errors = 0
    for item in resp.get("items", []):
        upd = item.get("update", {})
        if upd.get("error"):
            errors += 1
        else:
            # result can be "updated" or "noop" (already true)
            if upd.get("result") in ("updated", "noop"):
                updated += 1
    logger.info(
        f"🔖 OpenSearch flip has_embedding: updated/noop={updated}, errors={errors}"
    )
    return updated, errors

from typing import List, Dict, Any, Optional, Tuple
from core.opensearch_client import get_client
from typing import Iterable
from opensearchpy import helpers, exceptions

from config import (
    OPENSEARCH_INDEX,
    OPENSEARCH_DELETE_BATCH,
    OPENSEARCH_REQUEST_TIMEOUT,
    logger,
)

# Analyzer/mapping config (optional: can also be created manually in advance)
INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "custom_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "asciifolding"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "custom_text_analyzer"},
            "path": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}},
            },
            "chunk_index": {"type": "integer"},
            "checksum": {"type": "keyword"},
            "filetype": {"type": "keyword"},
            "indexed_at": {"type": "date"},
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},
            "page": {"type": "integer"},
            "location_percent": {"type": "float"},
        }
    },
}


def ensure_index_exists():
    client = get_client()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        logger.info(f"Creating OpenSearch index: {OPENSEARCH_INDEX}")
        client.indices.create(index=OPENSEARCH_INDEX, body=INDEX_SETTINGS)


def index_documents(chunks: List[Dict[str, Any]]) -> None:
    """Index a list of chunks into OpenSearch."""

    client = get_client()
    ensure_index_exists()
    actions = [
        {
            "_index": OPENSEARCH_INDEX,
            "_id": chunk["id"],
            "_source": {k: v for k, v in chunk.items() if k != "id"},
        }
        for chunk in chunks
    ]
    success_count, errors = helpers.bulk(client, actions)
    if errors:
        logger.error(f"❌ OpenSearch indexing failed for {len(errors)} chunks")
    else:
        logger.info(f"✅ OpenSearch successfully indexed {success_count} chunks")


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
    client = get_client()
    response = client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": 0,
            "aggs": {
                "files": {
                    "terms": {"field": "path.keyword", "size": size},
                    "aggs": {
                        "top_chunk": {
                            "top_hits": {
                                "size": 1,
                                "_source": [
                                    "checksum",
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
        path = bucket["key"]
        doc_count = bucket["doc_count"]
        top_hit = bucket["top_chunk"]["hits"]["hits"][0]
        top_source = top_hit["_source"]
        top_chunk_id = top_hit["_id"]

        results.append(
            {
                "path": path,
                "checksum": top_source.get("checksum"),
                "filename": path.split("/")[-1],
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
    client = get_client()
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
                index=OPENSEARCH_INDEX,
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


def delete_files_by_path_checksum(pairs: Iterable[Tuple[str, str]]) -> int:
    """Delete docs matching both path and checksum for each provided pair.

    Args:
        pairs: iterable of (path, checksum) tuples.

    Returns:
        Total number of deleted documents.
    """
    client = get_client()
    total_deleted = 0
    # remove duplicates / ignore falsy
    unique = {(p, c) for p, c in pairs if p and c}
    if not unique:
        return 0

    for path, checksum in unique:
        try:
            resp = client.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"checksum": checksum}},
                                {"match_phrase": {"path": path}},
                            ]
                        }
                    }
                },
                params={
                    "refresh": "true",
                    "conflicts": "proceed",
                    "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                },
            )
            deleted = int(resp.get("deleted", 0))
            total_deleted += deleted
            logger.info(
                f"🗑️ OpenSearch deleted {deleted} docs for path={path} checksum={checksum}."
            )
        except exceptions.OpenSearchException as e:
            logger.exception(
                f"OpenSearch delete failed for path={path} checksum={checksum}: {e}"
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error deleting path={path} checksum={checksum}: {e}"
            )

    return total_deleted


def get_duplicate_checksums(limit: int = 10000) -> List[str]:
    """Return checksums that appear under more than one distinct path.
    Uses an aggregation over `checksum` with a cardinality sub-agg on `path`.
    """
    client = get_client()
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
        resp = client.search(index=OPENSEARCH_INDEX, body=body)
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
            resp = client.search(index=OPENSEARCH_INDEX, body=body)
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


def get_files_by_checksum(checksum: str) -> List[Dict[str, Any]]:
    """Return a list of files (unique paths) associated with a checksum."""
    client = get_client()
    resp = client.search(
        index=OPENSEARCH_INDEX,
        body={"size": 10000, "query": {"term": {"checksum": checksum}}},
    )
    hits = resp.get("hits", {}).get("hits", [])
    files: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        src = hit.get("_source", {})
        path = src.get("path")
        if not path:
            continue
        info = files.setdefault(
            path,
            {
                "path": path,
                "filetype": src.get("filetype"),
                "created_at": src.get("created_at"),
                "modified_at": src.get("modified_at"),
                "indexed_at": src.get("indexed_at"),
                "checksum": checksum,
                "num_chunks": 0,
            },
        )
        info["num_chunks"] += 1
    return list(files.values())


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
        ops.append({"update": {"_index": OPENSEARCH_INDEX, "_id": doc_id}})
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


def is_file_up_to_date(checksum: str, path: str) -> bool:
    """Check if a file with the given checksum and path is already indexed."""
    try:
        client = get_client()
        response = client.count(
            index=OPENSEARCH_INDEX,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"checksum": checksum}},
                            {"match_phrase": {"path": path}},
                        ]
                    }
                }
            },
        )
        return response.get("count", 0) > 0
    except exceptions.OpenSearchException as e:
        logger.warning(f"OpenSearch checksum/path check failed: {e}")
        return False

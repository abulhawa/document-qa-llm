from typing import List, Dict, Any, Optional, Tuple, Iterable
from core.opensearch_client import get_client
from opensearchpy import helpers, exceptions

from config import (
    OPENSEARCH_INDEX,
    OPENSEARCH_DELETE_BATCH,
    OPENSEARCH_REQUEST_TIMEOUT,
    INGEST_LOG_INDEX,
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
            # join field describing parent/child relationship
            "join_field": {"type": "join", "relations": {"doc": "chunk"}},
            # parent fields
            "filename": {"type": "keyword"},
            "path": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}},
            },
            "file_type": {"type": "keyword"},
            # legacy name for compatibility
            "filetype": {"type": "keyword"},
            "lang": {"type": "keyword"},
            "size_bytes": {"type": "long"},
            # legacy name for compatibility
            "bytes": {"type": "long"},
            "checksum": {"type": "keyword"},
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},
            "indexed_at": {"type": "date"},
            "flags": {"type": "object"},
            # child fields
            "text": {"type": "text", "analyzer": "custom_text_analyzer"},
            "chunk_index": {"type": "integer"},
            "page": {"type": "integer"},
            "location_percent": {"type": "float"},
            "has_embedding": {"type": "boolean"},
        }
    },
}

INGEST_LOGS_SETTINGS = {
    "settings": {"index": {"number_of_shards": 1}},
    "mappings": {
        "properties": {
            "op": {"type": "keyword"},
            "source": {"type": "keyword"},
            "path": {"type": "keyword"},
            "path_hash": {"type": "keyword"},
            "checksum": {"type": "keyword"},
            "status": {"type": "keyword"},
            "stage": {"type": "keyword"},
            "reason": {"type": "text"},
            "error_type": {"type": "keyword"},
            "attempt_at": {"type": "date"},
            "duration_ms": {"type": "long"},
            "bytes": {"type": "long"},
            "size": {"type": "keyword"},
            "user": {"type": "keyword"},
            "host": {"type": "keyword"},
        }
    },
}


def ensure_index_exists():
    client = get_client()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        logger.info(f"Creating OpenSearch index: {OPENSEARCH_INDEX}")
        client.indices.create(index=OPENSEARCH_INDEX, body=INDEX_SETTINGS)


def ensure_ingest_log_index_exists():
    try:
        client = get_client()
        if not hasattr(client, "indices"):
            return
        if not client.indices.exists(index=INGEST_LOG_INDEX):
            logger.info(f"Creating OpenSearch index: {INGEST_LOG_INDEX}")
            client.indices.create(index=INGEST_LOG_INDEX, body=INGEST_LOGS_SETTINGS)
    except Exception as e:
        logger.warning(f"Ingest log index check failed: {e}")


def index_documents(docs: List[Dict[str, Any]]) -> None:
    """Index a list of documents (parents or children) into OpenSearch.

    Each document should contain an ``id`` field. Child documents must
    include ``parent_id`` so we can set the routing key required by the
    join mapping.
    """

    client = get_client()
    ensure_index_exists()
    actions = []
    for doc in docs:
        doc_id = doc.get("id")
        parent_id = doc.get("parent_id")
        src = {k: v for k, v in doc.items() if k not in {"id", "parent_id", "path"}}
        action: Dict[str, Any] = {"_index": OPENSEARCH_INDEX, "_id": doc_id, "_source": src}
        if parent_id:
            action["routing"] = parent_id
        else:
            # parent docs should still be routed to their own id for consistency
            action["routing"] = doc_id
        actions.append(action)

    success_count, errors = helpers.bulk(client, actions)
    if errors:
        logger.error(f"❌ OpenSearch indexing failed for {len(errors)} docs")
    else:
        logger.info(f"✅ OpenSearch successfully indexed {success_count} docs")


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
            "query": {"term": {"join_field": "doc"}},
            "aggs": {
                "files": {
                    "terms": {"field": "path.keyword", "size": size},
                    "aggs": {
                        "top_doc": {
                            "top_hits": {
                                "size": 1,
                                "_source": [
                                    "checksum",
                                    "created_at",
                                    "modified_at",
                                    "indexed_at",
                                    "file_type",
                                    "filetype",
                                    "size_bytes",
                                    "bytes",
                                    "size",
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
        top_hits_section = bucket.get("top_doc") or bucket.get("top_chunk")
        top_hit = top_hits_section["hits"]["hits"][0]
        top_source = top_hit["_source"]

        results.append(
            {
                "path": path,
                "checksum": top_source.get("checksum"),
                "filename": path.split("/")[-1],
                "created_at": top_source.get("created_at"),
                "modified_at": top_source.get("modified_at"),
                "indexed_at": top_source.get("indexed_at"),
                # return both new and legacy fields if available
                "filetype": top_source.get("file_type") or top_source.get("filetype"),
                "bytes": top_source.get("size_bytes") or top_source.get("bytes"),
                "size": top_source.get("size"),
                "num_chunks": doc_count,
            }
        )

    return results


def delete_files_by_checksum(checksums: Iterable[str]) -> int:
    """Delete parent documents (and their chunks) by checksum."""

    client = get_client()
    total_deleted = 0
    unique = [c for c in {c for c in checksums if c}]
    if not unique:
        return 0

    CHUNK = OPENSEARCH_DELETE_BATCH
    for i in range(0, len(unique), CHUNK):
        batch = unique[i : i + CHUNK]
        should = []
        for checksum in batch:
            should.append({
                "bool": {
                    "filter": [
                        {"term": {"checksum": checksum}},
                        {"term": {"join_field": "doc"}},
                    ]
                }
            })
            should.append({
                "has_parent": {
                    "parent_type": "doc",
                    "query": {"term": {"checksum": checksum}},
                }
            })
        try:
            client.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={"query": {"bool": {"should": should, "minimum_should_match": 1}}},
                params={
                    "refresh": "true",
                    "conflicts": "proceed",
                    "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                },
            )
            # Return number of unique checksums processed
            total_deleted += len(batch)
            logger.info(
                f"OpenSearch deleted {deleted} docs for {len(batch)} checksum(s).",
            )
        except exceptions.OpenSearchException as e:
            logger.exception(
                f"OpenSearch delete failed for a batch of {len(batch)} checksum(s): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error deleting a batch: {e}")

    return total_deleted



def delete_files_by_path_checksum(pairs: Iterable[Tuple[str, str]]) -> int:
    """Delete OpenSearch docs matching specific (path, checksum) pairs."""

    client = get_client()
    total_deleted = 0
    unique = [(p, c) for p, c in {(p, c) for p, c in pairs if p and c}]
    if not unique:
        return 0

    CHUNK = OPENSEARCH_DELETE_BATCH

    for i in range(0, len(unique), CHUNK):
        batch = unique[i : i + CHUNK]
        should_parent = []
        should_full = []
        for path, checksum in batch:
            clause = {
                "bool": {
                    "filter": [
                        {"term": {"path.keyword": path}},
                        {"term": {"checksum": checksum}},
                        {"term": {"join_field": "doc"}},
                    ]
                }
            }
            should_parent.append(clause)
            should_full.extend(
                [
                    clause,
                    {
                        "has_parent": {
                            "parent_type": "doc",
                            "query": {
                                "bool": {
                                    "filter": [
                                        {"term": {"path.keyword": path}},
                                        {"term": {"checksum": checksum}},
                                    ]
                                }
                            },
                        }
                    },
                ]
            )
        try:
            resp = client.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={"query": {"bool": {"should": should_full, "minimum_should_match": 1}}},
                params={
                    "refresh": "true",
                    "conflicts": "proceed",
                    "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                    "slices": "auto",
                },
            )
            failures = resp.get("failures", [])
            if failures:
                logger.warning("delete_by_query had failures: %s", failures)
        except Exception as e:
            logger.exception(
                f"OpenSearch delete failed for {len(batch)} path/checksum pair(s): {e}"
            )
            try:
                client.delete_by_query(
                    index=OPENSEARCH_INDEX,
                    body={"query": {"bool": {"should": should_parent, "minimum_should_match": 1}}},
                    params={
                        "refresh": "true",
                        "conflicts": "proceed",
                        "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                        "slices": "auto",
                    },
                )
            except Exception:
                pass
        total_deleted += len(batch)

    return total_deleted



def get_duplicate_checksums(limit: int = 10000) -> List[str]:
    """Return checksums that appear under more than one distinct path.
    Uses an aggregation over `checksum` with a cardinality sub-agg on `path`.
    """
    client = get_client()
    try:
        body = {
            "size": 0,
            "query": {"term": {"join_field": "doc"}},
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
                "query": {"term": {"join_field": "doc"}},
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
        body={
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"checksum": checksum}},
                        {"term": {"join_field": "doc"}},
                    ]
                }
            },
        },
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
                "bytes": src.get("bytes"),
                "size": src.get("size"),
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
                            {"term": {"path.keyword": path}},
                            {"term": {"join_field": "doc"}},
                        ]
                    }
                }
            },
        )
        return response.get("count", 0) > 0
    except exceptions.OpenSearchException as e:
        logger.warning(f"OpenSearch checksum/path check failed: {e}")
        return False


def is_duplicate_checksum(checksum: str, path: str) -> bool:
    """Check if checksum exists for a different path."""
    try:
        client = get_client()
        response = client.count(
            index=OPENSEARCH_INDEX,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"checksum": checksum}},
                            {"term": {"join_field": "doc"}},
                        ],
                        "must_not": [{"term": {"path.keyword": path}}],
                    }
                }
            },
        )
        return response.get("count", 0) > 0
    except exceptions.OpenSearchException as e:
        logger.warning(f"OpenSearch duplicate check failed: {e}")
        return False


def search_ingest_logs(
    *,
    status: str | None = None,
    path_query: str | None = None,
    start: str | None = None,
    end: str | None = None,
    size: int = 100,
) -> List[Dict[str, Any]]:
    """Search ingest_logs index with optional filters."""
    try:
        client = get_client()
        must: List[Dict[str, Any]] = []
        if status:
            must.append({"term": {"status": status}})
        if path_query:
            must.append({"wildcard": {"path": f"*{path_query}*"}})
        query: Dict[str, Any] = {"bool": {"must": must}}
        if start or end:
            rng: Dict[str, Any] = {}
            if start:
                rng["gte"] = start
            if end:
                rng["lte"] = end
            # ensure the range filter lives under the bool query
            query["bool"].setdefault("filter", []).append(
                {"range": {"attempt_at": rng}}
            )
        body = {"size": size, "sort": [{"attempt_at": "desc"}], "query": query}
        resp = client.search(index=INGEST_LOG_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [{"log_id": h.get("_id"), **h.get("_source", {})} for h in hits]
    except exceptions.OpenSearchException as e:
        logger.warning(f"Search ingest logs failed: {e}")
        return []

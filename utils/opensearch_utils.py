import os
from typing import List, Dict, Any, Optional, Tuple, Iterable
from core.opensearch_client import get_client
from opensearchpy import helpers, exceptions


from config import (
    OPENSEARCH_INDEX,
    OPENSEARCH_FULLTEXT_INDEX,
    OPENSEARCH_DELETE_BATCH,
    OPENSEARCH_REQUEST_TIMEOUT,
    INGEST_LOG_INDEX,
    logger,
)

from core.file_loader import load_documents
from core.document_preprocessor import preprocess_to_documents, PreprocessConfig
from utils.file_utils import (
    hash_path,
    compute_checksum,
    get_file_timestamps,
    get_file_size,
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
            "bytes": {"type": "long"},
            "size": {"type": "keyword"},
            "page": {"type": "integer"},
            "location_percent": {"type": "float"},
        }
    },
}

FULLTEXT_INDEX_SETTINGS = {
    "settings": {
        "index": {"max_ngram_diff": 15, "highlight.max_analyzed_offset": 5000000},
        "analysis": {
            "char_filter": {
                "alnum_only": {
                    "type": "pattern_replace",
                    "pattern": "[^\\p{L}\\p{Nd}]+",
                    "replacement": "",
                }
            },
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 15,
                    "token_chars": ["letter", "digit"],
                }
            },
            "analyzer": {
                "custom_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "asciifolding"],
                },
                "path_ngram": {
                    "type": "custom",
                    "char_filter": ["alnum_only"],
                    "tokenizer": "ngram_tokenizer",
                    "filter": ["lowercase"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "text_full": {"type": "text", "analyzer": "custom_text_analyzer"},
            "path": {
                "type": "keyword",
                "ignore_above": 2048,
                "fields": {"ngram": {"type": "text", "analyzer": "path_ngram"}},
            },
            "filename": {
                "type": "text",
                "analyzer": "custom_text_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "filetype": {"type": "keyword"},
            "modified_at": {"type": "date"},
            "created_at": {"type": "date"},
            "size_bytes": {"type": "long"},
            "checksum": {"type": "keyword"},
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
        client.indices.create(
            index=OPENSEARCH_INDEX,
            body=INDEX_SETTINGS,
            params={"wait_for_active_shards": "1"},
        )


def ensure_fulltext_index_exists():
    client = get_client()
    if not client.indices.exists(index=OPENSEARCH_FULLTEXT_INDEX):
        logger.info(f"Creating OpenSearch index: {OPENSEARCH_FULLTEXT_INDEX}")
        client.indices.create(
            index=OPENSEARCH_FULLTEXT_INDEX,
            body=FULLTEXT_INDEX_SETTINGS,
            params={"wait_for_active_shards": "1"},
        )


def ensure_ingest_log_index_exists():
    client = get_client()
    if not client.indices.exists(index=INGEST_LOG_INDEX):
        logger.info(f"Creating OpenSearch index: {INGEST_LOG_INDEX}")
        client.indices.create(
            index=INGEST_LOG_INDEX,
            body=INGEST_LOGS_SETTINGS,
            params={"wait_for_active_shards": "1"},
        )


def index_documents(chunks: List[Dict[str, Any]]) -> None:
    """Index a list of chunks into OpenSearch."""

    client = get_client()
    ensure_index_exists()
    actions = []
    for chunk in chunks:
        op_type = chunk.get("op_type", "create")
        action: Dict[str, Any] = {
            "_index": OPENSEARCH_INDEX,
            "_id": chunk["id"],
            "_op_type": op_type,
        }
        body = {k: v for k, v in chunk.items() if k not in {"id", "op_type"}}
        if op_type == "update":
            action["doc"] = body
        else:
            action["_source"] = body
        actions.append(action)
    success_count, errors = helpers.bulk(client, actions)
    if errors:
        logger.error(f"❌ OpenSearch indexing failed for {len(errors)} chunks")
    else:
        logger.info(f"✅ OpenSearch successfully indexed {success_count} chunks")


def index_fulltext_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Index a single full-text document into OpenSearch (op_type=create).
    Returns the OpenSearch response dict. Raises on fatal errors."""
    client = get_client()
    ensure_fulltext_index_exists()

    payload = {k: v for k, v in doc.items() if k != "id"}
    doc_id = doc["id"]

    try:
        resp = client.index(
            index=OPENSEARCH_FULLTEXT_INDEX,
            id=doc_id,
            body=payload,
            op_type="create",            # pyright: ignore[reportCallIssue]
            refresh=False,               # pyright: ignore[reportCallIssue]
            request_timeout=30,          # pyright: ignore[reportCallIssue]
        )
    except exceptions.ConflictError:
        # Document already exists (expected with op_type=create)
        logger.info(
            "Full-text indexing skipped (already exists).",
            extra={"index": OPENSEARCH_FULLTEXT_INDEX, "doc_id": doc_id}
        )
        return {"skipped": True, "reason": "conflict", "doc_id": doc_id}
    except exceptions.ConnectionTimeout as e:
        logger.error(
            "OpenSearch index timeout.",
            extra={"index": OPENSEARCH_FULLTEXT_INDEX, "doc_id": doc_id}
        )
        raise
    except exceptions.TransportError as e:
        # Surface status code/info if available
        status = getattr(e, "status_code", None)
        info = getattr(e, "info", None)
        logger.exception(
            "OpenSearch transport error during indexing.",
            extra={"index": OPENSEARCH_FULLTEXT_INDEX, "doc_id": doc_id, "status": status, "info": info}
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error during full-text indexing.",
            extra={"index": OPENSEARCH_FULLTEXT_INDEX, "doc_id": doc_id}
        )
        raise

    # Examine response and log accordingly
    result = resp.get("result")                # expected: "created"
    shards = resp.get("_shards", {}) or {}
    failed = int(shards.get("failed", 0))
    successful = int(shards.get("successful", 0))
    version = resp.get("_version")

    if failed > 0 or result != "created":
        logger.warning(
            "Indexing completed with non-ideal outcome.",
            extra={
                "index": OPENSEARCH_FULLTEXT_INDEX,
                "doc_id": doc_id,
                "result": result,
                "version": version,
                "shards_failed": failed,
                "shards_successful": successful,
                "resp": resp,  # keep raw for troubleshooting; remove if logs must stay minimal
            },
        )
    else:
        logger.info(
            "Indexed full-text doc.",
            extra={
                "index": OPENSEARCH_FULLTEXT_INDEX,
                "doc_id": doc_id,
                "result": result,
                "version": version,
                "shards_successful": successful,
            },
        )

    return resp


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
        top_hit = bucket["top_chunk"]["hits"]["hits"][0]
        top_source = top_hit["_source"]

        results.append(
            {
                "path": path,
                "checksum": top_source.get("checksum"),
                "filename": path.split("/")[-1],
                "created_at": top_source.get("created_at"),
                "modified_at": top_source.get("modified_at"),
                "indexed_at": top_source.get("indexed_at"),
                "filetype": top_source.get("filetype"),
                "bytes": top_source.get("bytes"),
                "size": top_source.get("size"),
                "num_chunks": doc_count,
            }
        )

    return results


def list_fulltext_paths(size: int = 1000) -> List[str]:
    """Return a list of file paths present in the full-text index."""

    client = get_client()
    resp = client.search(
        index=OPENSEARCH_FULLTEXT_INDEX,
        body={"size": size, "query": {"match_all": {}}, "_source": ["path"]},
    )
    hits = resp.get("hits", {}).get("hits", [])
    return [h.get("_source", {}).get("path") for h in hits if h.get("_source")]


def list_files_missing_fulltext(size: int = 1000) -> List[Dict[str, Any]]:
    """
    Return metadata for files that exist in the chunk index but are missing
    from the full-text index.
    """

    doc_files = list_files_from_opensearch(size=size)
    fulltext_paths = {p for p in list_fulltext_paths(size=size) if p}
    missing = [f for f in doc_files if f.get("path") not in fulltext_paths]
    return missing


def reindex_fulltext_from_chunks(paths: Iterable[str]) -> int:
    """Rebuild full-text documents for the given file paths by reloading and preprocessing the source files."""

    unique_paths = [p for p in {p for p in paths if p}]
    if not unique_paths:
        return 0

    total = 0
    for path in unique_paths:
        try:
            docs = load_documents(path)
            docs_list = preprocess_to_documents(
                docs_like=docs,
                source_path=path,
                cfg=PreprocessConfig(),
                doc_type=os.path.splitext(path)[1].lstrip(".").lower(),
            )
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

        full_text = "\n\n".join(getattr(d, "page_content", "") for d in docs_list)
        timestamps = get_file_timestamps(path)
        doc = {
            "id": hash_path(path),
            "path": path,
            "filename": os.path.basename(path),
            "filetype": os.path.splitext(path)[1].lstrip(".").lower(),
            "modified_at": timestamps.get("modified"),
            "created_at": timestamps.get("created"),
            "size_bytes": get_file_size(path),
            "checksum": compute_checksum(path),
            "text_full": full_text,
        }
        index_fulltext_document(doc)
        total += 1
    return total


def get_chunks_by_paths(paths: Iterable[str], batch_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all chunk documents from OpenSearch for the given file paths.

    Args:
        paths: Iterable of file paths to fetch chunks for.
        batch_size: Number of documents to request per query.

    Returns:
        List of chunk dictionaries including their ``id`` field.
    """
    client = get_client()
    results: List[Dict[str, Any]] = []
    unique_paths = [p for p in {p for p in paths if p}]
    for path in unique_paths:
        after: Optional[List[Any]] = None
        while True:
            body: Dict[str, Any] = {
                "size": batch_size,
                "query": {"term": {"path.keyword": path}},
                "sort": [{"chunk_index": "asc"}, {"_id": "asc"}],
            }
            if after:
                body["search_after"] = after
            resp = client.search(index=OPENSEARCH_INDEX, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                src["id"] = h.get("_id") or src.get("id")
                results.append(src)
            after = hits[-1].get("sort")
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
    """Delete OpenSearch docs matching specific (path, checksum) pairs.

    Each pair targets a unique file instance so duplicates with the same
    checksum but different paths can be removed individually. The deletion
    is batched for efficiency.
    """

    client = get_client()
    total_deleted = 0
    unique = [(p, c) for p, c in {(p, c) for p, c in pairs if p and c}]
    if not unique:
        return 0

    CHUNK = OPENSEARCH_DELETE_BATCH

    for i in range(0, len(unique), CHUNK):
        batch = unique[i : i + CHUNK]
        should = []
        for path, checksum in batch:
            should.append(
                {
                    "bool": {
                        "filter": [
                            {"term": {"path.keyword": path}},
                            {"term": {"checksum": checksum}},
                        ]
                    }
                }
            )
        try:
            resp = client.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={"query": {"bool": {"should": should, "minimum_should_match": 1}}},
                params={
                    "refresh": "true",
                    "conflicts": "proceed",
                    "timeout": OPENSEARCH_REQUEST_TIMEOUT,
                    "slices": "auto",
                },
            )
            deleted = int(resp.get("deleted", 0))
            conflicts = resp.get("version_conflicts", 0)
            failures = resp.get("failures", [])
            if failures:
                logger.warning("delete_by_query had failures: %s", failures)
            total_deleted += deleted
            logger.info(
                f"🗑️ OpenSearch deleted {deleted} docs for {len(batch)} path/checksum pair(s)."
            )
        except exceptions.OpenSearchException as e:
            logger.exception(
                f"OpenSearch delete failed for {len(batch)} path/checksum pair(s): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error deleting a batch: {e}")

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
                        "must": [{"term": {"checksum": checksum}}],
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

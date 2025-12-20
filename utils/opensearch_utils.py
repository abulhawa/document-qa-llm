import os
from typing import List, Dict, Any, Optional, Tuple, Iterable
from core.opensearch_client import get_client
from opensearchpy import helpers, exceptions


from config import (
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    OPENSEARCH_DELETE_BATCH,
    OPENSEARCH_REQUEST_TIMEOUT,
    INGEST_LOG_INDEX,
    logger,
)

from utils.file_utils import (
    hash_path,
    compute_checksum,
    get_file_timestamps,
    get_file_size,
)
from utils.file_utils import normalize_path

# Analyzer/mapping config
CHUNKS_INDEX_SETTINGS = {
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
            "aliases": {
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
            "indexed_at": {"type": "date"},
            "size_bytes": {"type": "long"},
            "checksum": {"type": "keyword"},
        }
    },
}


INGEST_LOGS_INDEX_SETTINGS = {
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


def index_documents(chunks: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """Index a list of chunks into OpenSearch using bulk API.

    Returns a tuple of ``(success_count, errors)`` and raises on fatal
    connection/transport errors. Conflicts and per-item errors are returned in
    the ``errors`` list for the caller to inspect.
    """

    client = get_client()
    actions = []
    for chunk in chunks:
        op_type = chunk.get("op_type", "index")
        action: Dict[str, Any] = {
            "_index": CHUNKS_INDEX,
            "_id": chunk["id"],
            "_op_type": op_type,
        }
        body = {k: v for k, v in chunk.items() if k not in {"id", "op_type"}}
        if op_type == "update":
            action["doc"] = body
        else:
            action["_source"] = body
        actions.append(action)

    conn_timeout_exc = getattr(exceptions, "ConnectionTimeout", Exception)
    transport_exc = getattr(exceptions, "TransportError", Exception)

    try:
        success_count, errors = helpers.bulk(
            client,
            actions,
            raise_on_error=False,
        )
    except conn_timeout_exc as e:  # type: ignore[misc]
        logger.error(
            "OpenSearch bulk index timeout.",
            extra={"index": CHUNKS_INDEX},
        )
        raise
    except transport_exc as e:  # type: ignore[misc]
        status = getattr(e, "status_code", None)
        info = getattr(e, "info", None)
        logger.exception(
            "OpenSearch transport error during bulk indexing.",
            extra={"index": CHUNKS_INDEX, "status": status, "info": info},
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error during chunk indexing.",
            extra={"index": CHUNKS_INDEX},
        )
        raise

    if errors:
        logger.error(
            f"❌ OpenSearch indexing failed for {len(errors)} chunks",
            extra={"index": CHUNKS_INDEX, "errors": errors},
        )
    else:
        logger.info(
            f"✅ OpenSearch successfully indexed {success_count} chunks",
            extra={"index": CHUNKS_INDEX},
        )

    return success_count, errors


def index_fulltext_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Index or update a single full-text document into OpenSearch.
    Returns the OpenSearch response dict. Raises on fatal errors."""
    client = get_client()

    payload = {k: v for k, v in doc.items() if k != "id"}
    doc_id = doc["id"]

    conn_timeout_exc = getattr(exceptions, "ConnectionTimeout", Exception)
    transport_exc = getattr(exceptions, "TransportError", Exception)

    try:
        resp = client.index(
            index=FULLTEXT_INDEX,
            id=doc_id,
            body=payload,
            op_type="index",  # pyright: ignore[reportCallIssue]
            refresh=False,  # pyright: ignore[reportCallIssue]
            request_timeout=30,  # pyright: ignore[reportCallIssue]
        )
    except conn_timeout_exc as e:  # type: ignore[misc]
        logger.error(
            "OpenSearch index timeout.",
            extra={"index": FULLTEXT_INDEX, "doc_id": doc_id},
        )
        raise
    except transport_exc as e:  # type: ignore[misc]
        # Surface status code/info if available
        status = getattr(e, "status_code", None)
        info = getattr(e, "info", None)
        logger.exception(
            "OpenSearch transport error during indexing.",
            extra={
                "index": FULLTEXT_INDEX,
                "doc_id": doc_id,
                "status": status,
                "info": info,
            },
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error during full-text indexing.",
            extra={"index": FULLTEXT_INDEX, "doc_id": doc_id},
        )
        raise

    # Examine response and log accordingly
    result = resp.get("result")
    shards = resp.get("_shards", {}) or {}
    failed = int(shards.get("failed", 0))
    successful = int(shards.get("successful", 0))
    version = resp.get("_version")

    if failed > 0 or result != "created":
        logger.warning(
            "Indexing completed with non-ideal outcome.",
            extra={
                "index": FULLTEXT_INDEX,
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
                "index": FULLTEXT_INDEX,
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
        index=CHUNKS_INDEX,
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


def get_chunk_ids_by_path(path: str, size: int = 10000) -> list[str]:
    """Fetch chunk IDs for a file path from the documents index."""
    client = get_client()
    resp = client.search(
        index=CHUNKS_INDEX,
        body={
            "size": size,
            "_source": False,
            "query": {"term": {"path.keyword": path}},
        },
    )
    return [h.get("_id") for h in resp.get("hits", {}).get("hits", []) if h.get("_id")]


def delete_chunks_by_path(path: str) -> int:
    """Delete all chunk docs for a file from the documents index."""
    client = get_client()
    resp = client.delete_by_query(
        index=CHUNKS_INDEX,
        body={"query": {"term": {"path.keyword": path}}},
        params={
            "refresh": "true",
            "conflicts": "proceed",
            "timeout": OPENSEARCH_REQUEST_TIMEOUT,
        },
    )
    return int(resp.get("deleted", 0))


def delete_fulltext_by_path(path: str) -> int:
    """Delete full-text doc(s) for a file from the full_text index."""
    client = get_client()
    resp = client.delete_by_query(
        index=FULLTEXT_INDEX,
        body={"query": {"term": {"path.keyword": path}}},
        params={
            "refresh": "true",
            "conflicts": "proceed",
            "timeout": OPENSEARCH_REQUEST_TIMEOUT,
        },
    )
    return int(resp.get("deleted", 0))


def list_fulltext_paths(size: int = 1000) -> List[str]:
    """Return a list of file paths present in the full-text index."""

    client = get_client()
    resp = client.search(
        index=FULLTEXT_INDEX,
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


def get_fulltext_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
    """Return full-text document for the checksum, if present."""
    client = get_client()
    try:
        resp = client.get(index=FULLTEXT_INDEX, id=checksum)
    except exceptions.NotFoundError:
        return None
    except Exception:
        logger.warning("Failed to fetch full-text doc for checksum=%s", checksum)
        return None

    source = resp.get("_source") or {}
    source["id"] = resp.get("_id") or checksum
    return source


def get_chunks_by_paths(
    paths: Iterable[str], batch_size: int = 1000
) -> List[Dict[str, Any]]:
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
            resp = client.search(index=CHUNKS_INDEX, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                src["id"] = h.get("_id") or src.get("id")
                results.append(src)
            after = hits[-1].get("sort")
    return results


def get_duplicate_checksums(
    page_size: int = 1000, max_results: int | None = None
) -> list[str]:
    """
    Scan for all checksums that appear under >1 distinct path.
    Uses composite agg for deterministic paging (no 'top N' bias).
    """
    client = get_client()
    dups: list[str] = []
    after = None

    while True:
        comp = {
            "size": page_size,
            "sources": [{"checksum": {"terms": {"field": "checksum"}}}],
        }
        if after:
            comp["after"] = after

        body = {
            "size": 0,
            "aggs": {
                "by_checksum": {
                    "composite": comp,
                    "aggregations": {
                        "paths": {"terms": {"field": "path.keyword", "size": 2}},
                    },
                }
            },
        }
        resp = client.search(index=CHUNKS_INDEX, body=body)
        agg = resp["aggregations"]["by_checksum"]
        for b in agg["buckets"]:
            if len(b["paths"]["buckets"]) >= 2:
                dups.append(b["key"]["checksum"])
                if max_results and len(dups) >= max_results:
                    return dups
        after = agg.get("after_key")
        if not after:
            break

    return dups


def get_files_by_checksum(checksum: str) -> List[Dict[str, Any]]:
    """Return a list of files (unique paths) associated with a checksum."""
    client = get_client()
    resp = client.search(
        index=CHUNKS_INDEX,
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


def is_file_up_to_date(checksum: str, path: str) -> bool:
    """Check if a file with the given checksum and path is already indexed."""
    try:
        client = get_client()
        response = client.count(
            index=CHUNKS_INDEX,
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
            index=CHUNKS_INDEX,
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

import os
from typing import List, Dict, Any, Optional, Tuple, Iterable
from core.opensearch_client import get_client
from opensearchpy import helpers, exceptions


from config import (
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    FINANCIAL_RECORDS_INDEX,
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
from utils.timing import timed_block

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
            "chunk_char_len": {"type": "integer"},
            "filetype": {"type": "keyword"},
            "indexed_at": {"type": "date"},
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},
            "bytes": {"type": "long"},
            "size": {"type": "keyword"},
            "page": {"type": "integer"},
            "location_percent": {"type": "float"},
            "doc_type": {"type": "keyword"},
            "doc_type_confidence": {"type": "float"},
            "doc_type_source": {"type": "keyword"},
            "person_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "authority_rank": {"type": "float"},
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
            "doc_type": {"type": "keyword"},
            "doc_type_confidence": {"type": "float"},
            "doc_type_source": {"type": "keyword"},
            "person_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "authority_rank": {"type": "float"},
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


def ensure_chunk_char_len_mapping() -> None:
    client = get_client()
    try:
        mapping = client.indices.get_mapping(index=CHUNKS_INDEX)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to fetch OpenSearch mappings for index=%s", CHUNKS_INDEX
        )
        raise

    index_mapping = mapping.get(CHUNKS_INDEX, {}).get("mappings", {})
    props = index_mapping.get("properties", {}) or {}
    if "chunk_char_len" in props:
        return

    try:
        client.indices.put_mapping(
            index=CHUNKS_INDEX,
            body={"properties": {"chunk_char_len": {"type": "integer"}}},
        )
        logger.info(
            "Added chunk_char_len mapping to OpenSearch index=%s", CHUNKS_INDEX
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to add chunk_char_len mapping to OpenSearch index=%s",
            CHUNKS_INDEX,
        )
        raise


_IDENTITY_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "doc_type": {"type": "keyword"},
    "doc_type_confidence": {"type": "float"},
    "doc_type_source": {"type": "keyword"},
    "person_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
    "authority_rank": {"type": "float"},
}


_NUMERIC_MAPPINGS = {
    "float",
    "half_float",
    "double",
    "scaled_float",
    "long",
    "integer",
    "short",
    "byte",
}


_INTEGER_MAPPINGS = {"long", "integer", "short", "byte"}


_FINANCIAL_METADATA_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "is_financial_document": {"type": "boolean"},
    "document_date": {"type": "date"},
    "mentioned_years": {"type": "integer"},
    "transaction_dates": {"type": "date"},
    "tax_years_referenced": {"type": "integer"},
    "amounts": {"type": "double"},
    "counterparties": {"type": "keyword"},
    "tax_relevance_signals": {"type": "keyword"},
    "expense_category": {"type": "keyword"},
    "financial_record_type": {"type": "keyword"},
    "financial_metadata_version": {"type": "keyword"},
    "financial_metadata_source": {"type": "keyword"},
}


FINANCIAL_RECORDS_INDEX_SETTINGS = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "record_type": {"type": "keyword"},
            "date": {"type": "date"},
            "amount": {"type": "double"},
            "currency": {"type": "keyword"},
            "counterparty": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "description": {"type": "text"},
            "confidence": {"type": "float"},
            "document_id": {"type": "keyword"},
            "checksum": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "source_text_span": {"type": "text"},
            "extraction_method": {"type": "keyword"},
            "merge_key": {"type": "keyword"},
            "source_count": {"type": "integer"},
            "financial_record_version": {"type": "keyword"},
            "source_family": {"type": "keyword"},
            "year": {"type": "integer"},
            "source_links": {
                "type": "nested",
                "properties": {
                    "document_id": {"type": "keyword"},
                    "checksum": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "source_text_span": {"type": "text"},
                    "extraction_method": {"type": "keyword"},
                    "confidence": {"type": "float"},
                },
            },
        }
    },
}


def _extract_properties_from_mapping_response(mapping: Dict[str, Any]) -> Dict[str, Any]:
    if not mapping:
        return {}
    first_index_mapping = next(iter(mapping.values()), {})
    return first_index_mapping.get("mappings", {}).get("properties", {}) or {}


def _is_identity_mapping_compatible(field: str, existing: Dict[str, Any]) -> bool:
    existing_type = existing.get("type")
    if field == "doc_type":
        # Accept either keyword or text for legacy dynamic mappings.
        return existing_type in {"keyword", "text"}
    if field == "doc_type_confidence":
        return existing_type in {"float", "half_float", "double", "scaled_float", "long", "integer"}
    if field == "doc_type_source":
        # Accept either keyword or text for legacy dynamic mappings.
        return existing_type in {"keyword", "text"}
    if field == "person_name":
        # Text or keyword are both safe for read-time pass-through.
        return existing_type in {"text", "keyword"}
    if field == "authority_rank":
        return existing_type in {"float", "half_float", "double", "scaled_float", "long", "integer"}
    return False


def _is_financial_metadata_mapping_compatible(field: str, existing: Dict[str, Any]) -> bool:
    existing_type = existing.get("type")
    if field == "is_financial_document":
        return existing_type == "boolean"
    if field in {"document_date", "transaction_dates"}:
        return existing_type == "date"
    if field in {"mentioned_years", "tax_years_referenced"}:
        return existing_type in _INTEGER_MAPPINGS
    if field == "amounts":
        return existing_type in _NUMERIC_MAPPINGS
    if field in {
        "counterparties",
        "tax_relevance_signals",
        "expense_category",
        "financial_record_type",
        "financial_metadata_version",
        "financial_metadata_source",
    }:
        return existing_type in {"keyword", "text"}
    return False


def _is_financial_record_mapping_compatible(field: str, existing: Dict[str, Any]) -> bool:
    existing_type = existing.get("type")
    if field in {
        "record_type",
        "currency",
        "document_id",
        "checksum",
        "chunk_id",
        "extraction_method",
        "merge_key",
        "financial_record_version",
        "source_family",
    }:
        return existing_type in {"keyword", "text"}
    if field == "date":
        return existing_type == "date"
    if field in {"amount", "confidence"}:
        return existing_type in _NUMERIC_MAPPINGS
    if field in {"description", "counterparty", "source_text_span"}:
        return existing_type in {"text", "keyword"}
    if field in {"source_count", "year"}:
        return existing_type in _INTEGER_MAPPINGS
    if field == "source_links":
        return existing_type in {"nested", "object"}
    return False


def ensure_identity_metadata_mappings() -> None:
    """Ensure identity metadata fields exist on both chunks/fulltext indices.

    This operation is non-destructive: it only adds missing fields via put_mapping.
    If an existing field has an incompatible type, it raises to avoid risky writes.
    """

    client = get_client()
    for index_name in (CHUNKS_INDEX, FULLTEXT_INDEX):
        try:
            mapping = client.indices.get_mapping(index=index_name)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch OpenSearch mappings for index=%s", index_name)
            raise

        properties = _extract_properties_from_mapping_response(mapping)
        additions: Dict[str, Dict[str, Any]] = {}
        conflicts: list[str] = []
        for field, expected_mapping in _IDENTITY_MAPPINGS.items():
            existing = properties.get(field)
            if existing is None:
                additions[field] = expected_mapping
                continue
            if not _is_identity_mapping_compatible(field, existing):
                conflicts.append(
                    f"{field}: existing_type={existing.get('type')} expected={expected_mapping.get('type')}"
                )

        if conflicts:
            conflict_message = "; ".join(conflicts)
            raise RuntimeError(
                f"Incompatible mapping detected for index '{index_name}'. "
                f"Refusing to continue to avoid data risk. Conflicts: {conflict_message}"
            )

        if not additions:
            continue

        try:
            client.indices.put_mapping(
                index=index_name,
                body={"properties": additions},
            )
            logger.info(
                "Added identity metadata mappings for index=%s fields=%s",
                index_name,
                sorted(additions.keys()),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to add identity metadata mappings for index=%s",
                index_name,
            )
            raise


def ensure_financial_metadata_mappings() -> None:
    """Ensure finance metadata fields exist on chunks/fulltext indices.

    This operation is non-destructive: only missing mappings are added.
    Incompatible existing mappings fail fast to avoid unsafe writes.
    """

    client = get_client()
    for index_name in (CHUNKS_INDEX, FULLTEXT_INDEX):
        try:
            mapping = client.indices.get_mapping(index=index_name)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to fetch OpenSearch mappings for index=%s",
                index_name,
            )
            raise

        properties = _extract_properties_from_mapping_response(mapping)
        additions: Dict[str, Dict[str, Any]] = {}
        conflicts: list[str] = []
        for field, expected_mapping in _FINANCIAL_METADATA_MAPPINGS.items():
            existing = properties.get(field)
            if existing is None:
                additions[field] = expected_mapping
                continue
            if not _is_financial_metadata_mapping_compatible(field, existing):
                conflicts.append(
                    f"{field}: existing_type={existing.get('type')} expected={expected_mapping.get('type')}"
                )

        if conflicts:
            conflict_message = "; ".join(conflicts)
            raise RuntimeError(
                f"Incompatible mapping detected for index '{index_name}'. "
                f"Refusing to continue to avoid data risk. Conflicts: {conflict_message}"
            )

        if not additions:
            continue

        try:
            client.indices.put_mapping(
                index=index_name,
                body={"properties": additions},
            )
            logger.info(
                "Added financial metadata mappings for index=%s fields=%s",
                index_name,
                sorted(additions.keys()),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to add financial metadata mappings for index=%s",
                index_name,
            )
            raise


def ensure_financial_records_index() -> None:
    """Ensure sidecar financial records index exists with compatible mappings."""

    client = get_client()
    expected_properties = (
        FINANCIAL_RECORDS_INDEX_SETTINGS.get("mappings", {}).get("properties", {}) or {}
    )

    if not client.indices.exists(index=FINANCIAL_RECORDS_INDEX):
        try:
            client.indices.create(
                index=FINANCIAL_RECORDS_INDEX,
                body=FINANCIAL_RECORDS_INDEX_SETTINGS,
                params={"wait_for_active_shards": "1"},
            )
            logger.info("Created OpenSearch index: %s", FINANCIAL_RECORDS_INDEX)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to create OpenSearch index=%s",
                FINANCIAL_RECORDS_INDEX,
            )
            raise
        return

    try:
        mapping = client.indices.get_mapping(index=FINANCIAL_RECORDS_INDEX)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to fetch OpenSearch mappings for index=%s",
            FINANCIAL_RECORDS_INDEX,
        )
        raise

    properties = _extract_properties_from_mapping_response(mapping)
    additions: Dict[str, Dict[str, Any]] = {}
    conflicts: list[str] = []
    for field, expected_mapping in expected_properties.items():
        existing = properties.get(field)
        if existing is None:
            additions[field] = expected_mapping
            continue
        if not _is_financial_record_mapping_compatible(field, existing):
            conflicts.append(
                f"{field}: existing_type={existing.get('type')} expected={expected_mapping.get('type')}"
            )

    if conflicts:
        conflict_message = "; ".join(conflicts)
        raise RuntimeError(
            f"Incompatible mapping detected for index '{FINANCIAL_RECORDS_INDEX}'. "
            f"Refusing to continue to avoid data risk. Conflicts: {conflict_message}"
        )

    if not additions:
        return

    try:
        client.indices.put_mapping(
            index=FINANCIAL_RECORDS_INDEX,
            body={"properties": additions},
        )
        logger.info(
            "Added financial-record mappings for index=%s fields=%s",
            FINANCIAL_RECORDS_INDEX,
            sorted(additions.keys()),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to add financial-record mappings for index=%s",
            FINANCIAL_RECORDS_INDEX,
        )
        raise


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
        with timed_block(
            "step.opensearch.query",
            extra={
                "index": CHUNKS_INDEX,
                "operation": "bulk_index",
                "chunk_count": len(actions),
            },
            logger=logger,
        ):
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
        with timed_block(
            "step.opensearch.query",
            extra={"index": FULLTEXT_INDEX, "operation": "index", "doc_id": doc_id},
            logger=logger,
        ):
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
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "list_files", "size": size},
        logger=logger,
    ):
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
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "get_chunk_ids_by_path"},
        logger=logger,
    ):
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
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "delete_by_path"},
        logger=logger,
    ):
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


def delete_chunks_by_checksum(checksum: str) -> int:
    """Delete chunk documents by checksum across all paths."""
    client = get_client()
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "delete_by_checksum"},
        logger=logger,
    ):
        resp = client.delete_by_query(
            index=CHUNKS_INDEX,
            body={"query": {"term": {"checksum": {"value": checksum}}}},
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
    with timed_block(
        "step.opensearch.query",
        extra={"index": FULLTEXT_INDEX, "operation": "delete_fulltext_by_path"},
        logger=logger,
    ):
        resp = client.delete_by_query(
            index=FULLTEXT_INDEX,
            body={"query": {"term": {"path": {"value": path}}}},
            params={
                "refresh": "true",
                "conflicts": "proceed",
                "timeout": OPENSEARCH_REQUEST_TIMEOUT,
            },
        )
    return int(resp.get("deleted", 0))


def delete_fulltext_by_checksum(checksum: str) -> int:
    """Delete full-text doc(s) by checksum (id or field match)."""
    client = get_client()
    # Try direct delete first; fall back to delete_by_query for legacy docs.
    deleted = 0
    try:
        client.delete(index=FULLTEXT_INDEX, id=checksum, params={"refresh": "true"})
        deleted += 1
    except exceptions.NotFoundError:
        pass
    except Exception:
        logger.warning("Full-text direct delete failed for checksum=%s", checksum, exc_info=True)
    with timed_block(
        "step.opensearch.query",
        extra={"index": FULLTEXT_INDEX, "operation": "delete_fulltext_by_checksum"},
        logger=logger,
    ):
        resp = client.delete_by_query(
            index=FULLTEXT_INDEX,
            body={"query": {"term": {"checksum": {"value": checksum}}}},
            params={
                "refresh": "true",
                "conflicts": "proceed",
                "timeout": OPENSEARCH_REQUEST_TIMEOUT,
            },
        )
    deleted += int(resp.get("deleted", 0))
    return deleted


def list_fulltext_paths(size: int = 1000) -> List[str]:
    """Return a list of file paths present in the full-text index."""

    client = get_client()
    with timed_block(
        "step.opensearch.query",
        extra={"index": FULLTEXT_INDEX, "operation": "list_fulltext_paths", "size": size},
        logger=logger,
    ):
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
        with timed_block(
            "step.opensearch.query",
            extra={"index": FULLTEXT_INDEX, "operation": "get_fulltext"},
            logger=logger,
        ):
            resp = client.get(index=FULLTEXT_INDEX, id=checksum)
        source = resp.get("_source") or {}
        source["id"] = resp.get("_id") or checksum
        return source
    except exceptions.NotFoundError:
        # Legacy docs may not use checksum as the document _id; fall back to a
        # term query on the checksum field.
        pass
    except Exception:
        logger.warning("Failed to fetch full-text doc for checksum=%s", checksum)
        return None

    try:
        with timed_block(
            "step.opensearch.query",
            extra={"index": FULLTEXT_INDEX, "operation": "search_fulltext_by_checksum"},
            logger=logger,
        ):
            search = client.search(
                index=FULLTEXT_INDEX,
                body={
                    "size": 1,
                    "query": {"term": {"checksum": {"value": checksum}}},
                },
            )
        hits = search.get("hits", {}).get("hits", [])
        if not hits:
            return None
        hit = hits[0]
        source = hit.get("_source") or {}
        source["id"] = hit.get("_id") or checksum
        return source
    except Exception:
        logger.warning(
            "Failed to search full-text doc by checksum=%s", checksum, exc_info=True
        )
        return None


def get_fulltext_by_path_or_alias(path: str) -> Optional[Dict[str, Any]]:
    """Return full-text document whose canonical path or aliases contain path."""
    client = get_client()
    try:
        with timed_block(
            "step.opensearch.query",
            extra={"index": FULLTEXT_INDEX, "operation": "search_by_path_or_alias"},
            logger=logger,
        ):
            search = client.search(
                index=FULLTEXT_INDEX,
                body={
                    "size": 1,
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"path": path}},
                                {"term": {"aliases": path}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                },
            )
        hits = search.get("hits", {}).get("hits", [])
        if not hits:
            return None
        hit = hits[0]
        source = hit.get("_source") or {}
        source["id"] = hit.get("_id")
        return source
    except Exception:
        logger.warning(
            "Failed to search full-text doc by path or alias=%s", path, exc_info=True
        )
        return None


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
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "get_chunks_by_paths", "path_count": len(unique_paths)},
        logger=logger,
    ):
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
    Scan for checksums that have multiple known locations (canonical path + aliases).
    Uses the full-text index, where canonical/alias paths are recorded even when chunk
    docs are deduplicated to a single path.
    """
    client = get_client()
    dups: list[str] = []
    after = None

    # Rely on aliases recorded in the full-text doc. Empty alias arrays are not stored,
    # so an ``exists`` filter surfaces only checksums with >0 alias paths.
    while True:
        body = {
            "size": page_size,
            "_source": ["checksum", "aliases"],
            "sort": [{"checksum": "asc"}, {"_id": "asc"}],
            "query": {"bool": {"must": [{"exists": {"field": "aliases"}}]}},
        }
        if after:
            body["search_after"] = after

        with timed_block(
            "step.opensearch.query",
            extra={"index": FULLTEXT_INDEX, "operation": "get_duplicate_checksums"},
            logger=logger,
        ):
            resp = client.search(index=FULLTEXT_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            src = hit.get("_source", {}) or {}
            aliases = src.get("aliases") or []
            if not aliases:
                continue
            checksum = src.get("checksum") or hit.get("_id")
            if checksum and checksum not in dups:
                dups.append(checksum)
                if max_results and len(dups) >= max_results:
                    return dups

        after = hits[-1].get("sort")
        if not after:
            break

    return dups


def get_files_by_checksum(checksum: str) -> List[Dict[str, Any]]:
    """
    Return all recorded file locations (canonical path + aliases) for a checksum.

    Chunk docs may only store the canonical path, so alias locations are sourced from
    the full-text document.
    """
    client = get_client()

    # Aggregate chunk metadata by path for this checksum (canonical path should exist).
    with timed_block(
        "step.opensearch.query",
        extra={"index": CHUNKS_INDEX, "operation": "get_files_by_checksum"},
        logger=logger,
    ):
        resp = client.search(
            index=CHUNKS_INDEX,
            body={
                "size": 0,
                "query": {"term": {"checksum": checksum}},
                "aggs": {
                    "by_path": {
                        "terms": {"field": "path.keyword", "size": 2000},
                        "aggs": {
                            "sample": {"top_hits": {"size": 1}},
                        },
                    }
                },
            },
        )
    buckets = resp.get("aggregations", {}).get("by_path", {}).get("buckets", []) or []
    chunk_meta: Dict[str, Dict[str, Any]] = {}
    for bucket in buckets:
        sample_hit = (
            bucket.get("sample", {})
            .get("hits", {})
            .get("hits", [{}])[0]
            .get("_source", {})
        )
        path = bucket.get("key")
        if not path:
            continue
        chunk_meta[path] = {
            "num_chunks": bucket.get("doc_count", 0),
            "filetype": sample_hit.get("filetype"),
            "created_at": sample_hit.get("created_at"),
            "modified_at": sample_hit.get("modified_at"),
            "indexed_at": sample_hit.get("indexed_at"),
            "bytes": sample_hit.get("bytes"),
            "size": sample_hit.get("size"),
        }

    fulltext_doc = get_fulltext_by_checksum(checksum) or {}
    canonical_path = fulltext_doc.get("path")
    aliases = fulltext_doc.get("aliases") or []

    files: list[Dict[str, Any]] = []
    base_meta = {
        "filetype": fulltext_doc.get("filetype"),
        "created_at": fulltext_doc.get("created_at"),
        "modified_at": fulltext_doc.get("modified_at"),
        "indexed_at": fulltext_doc.get("indexed_at"),
        "bytes": fulltext_doc.get("size_bytes"),
    }
    canonical_chunks = (
        chunk_meta.get(canonical_path, {}) if canonical_path else {}
    )

    def append_entry(path: str, is_alias: bool) -> None:
        path_meta = chunk_meta.get(path, {})
        meta = {**base_meta, **path_meta}
        files.append(
            {
                "path": path,
                "canonical_path": canonical_path or path,
                "location_type": "alias" if is_alias else "canonical",
                "filetype": meta.get("filetype"),
                "created_at": meta.get("created_at"),
                "modified_at": meta.get("modified_at"),
                "indexed_at": meta.get("indexed_at"),
                "bytes": meta.get("bytes"),
                "size": meta.get("size"),
                "checksum": checksum,
                "num_chunks": meta.get("num_chunks", canonical_chunks.get("num_chunks", 0)),
            }
        )

    if canonical_path:
        append_entry(canonical_path, False)
    else:
        for path in chunk_meta:
            append_entry(path, False)

    for alias in aliases:
        append_entry(alias, True)

    return files


def is_file_up_to_date(checksum: str, path: str) -> bool:
    """Check if a file with the given checksum and path is already indexed."""
    try:
        client = get_client()
        with timed_block(
            "step.opensearch.query",
            extra={"index": CHUNKS_INDEX, "operation": "count"},
            logger=logger,
        ):
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
        with timed_block(
            "step.opensearch.query",
            extra={"index": CHUNKS_INDEX, "operation": "count"},
            logger=logger,
        ):
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
        with timed_block(
            "step.opensearch.query",
            extra={"index": INGEST_LOG_INDEX, "operation": "search_ingest_logs"},
            logger=logger,
        ):
            resp = client.search(index=INGEST_LOG_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [{"log_id": h.get("_id"), **h.get("_source", {})} for h in hits]
    except exceptions.OpenSearchException as e:
        logger.warning(f"Search ingest logs failed: {e}")
        return []

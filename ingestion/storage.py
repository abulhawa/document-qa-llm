from typing import Any, Dict, Iterable, Sequence, Tuple

from config import logger
from utils import qdrant_utils
from utils.opensearch.chunks import (
    delete_chunks_by_path,
    get_chunk_ids_by_path,
    index_documents,
    is_duplicate_checksum,
    is_file_up_to_date,
)
from utils.opensearch.fulltext import (
    delete_fulltext_by_path,
    get_fulltext_by_checksum,
    index_fulltext_document,
)


def replace_existing_artifacts(normalized_path: str) -> None:
    """Delete prior vectors and OpenSearch docs for a path."""

    logger.info(
        "♻️ Reingest replace: deleting existing docs/vectors for file=%s",
        normalized_path,
    )

    try:
        ids = get_chunk_ids_by_path(normalized_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("List chunk IDs failed for %s: %s", normalized_path, e)
        ids = []

    if ids:
        try:
            qdrant_utils.delete_vectors_by_ids(ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("Qdrant delete failed for %s: %s", normalized_path, e)

    try:
        delete_chunks_by_path(normalized_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenSearch chunk delete failed for %s: %s", normalized_path, e)

    try:
        delete_fulltext_by_path(normalized_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenSearch full-text delete failed for %s: %s", normalized_path, e)


def embed_and_store(
    chunks: Sequence[Dict[str, Any]],
    *,
    os_index_batch,
) -> bool:
    """Embed and upsert chunks, invoking OpenSearch batch indexing callback."""

    return qdrant_utils.index_chunks_in_batches(chunks, os_index_batch=os_index_batch)


def index_fulltext(full_doc: Dict[str, Any]) -> None:
    index_fulltext_document(full_doc)


def get_existing_fulltext(checksum: str) -> Dict[str, Any] | None:
    return get_fulltext_by_checksum(checksum)


def index_chunk_batch(group: Sequence[Dict[str, Any]]) -> Tuple[int, Iterable]:
    return index_documents(list(group))


__all__ = [
    "delete_chunks_by_path",
    "delete_fulltext_by_path",
    "embed_and_store",
    "index_chunk_batch",
    "index_fulltext",
    "get_existing_fulltext",
    "is_duplicate_checksum",
    "is_file_up_to_date",
    "replace_existing_artifacts",
]

from utils.opensearch.fulltext import index_fulltext_document, delete_fulltext_by_path
from utils.inventory import set_inventory_number_of_chunks, set_inventory_last_indexed
import os
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Sequence, Dict, Any, Optional
from core.document_preprocessor import preprocess_to_documents, PreprocessConfig
from core.file_loader import load_documents
from core.chunking import split_documents
from config import logger, INGEST_IO_CONCURRENCY
from utils.file_utils import (
    compute_checksum,
    get_file_timestamps,
    hash_path,
    get_file_size,
    normalize_path,
    format_file_size,
)
from utils import qdrant_utils
from utils.opensearch.chunks import (
    is_file_up_to_date,
    is_duplicate_checksum,
    index_documents,
    get_chunk_ids_by_path,
    delete_chunks_by_path,
)
from utils.ingest_logging import IngestLogEmitter

# --- Concurrency from config.py ---
IO_CONCURRENCY: int = INGEST_IO_CONCURRENCY

# IO semaphore to limit simultaneous file reads (helps avoid too many open files)
_io_semaphore = threading.Semaphore(IO_CONCURRENCY)


@contextmanager
def _io_guard():
    _io_semaphore.acquire()
    try:
        yield
    finally:
        _io_semaphore.release()


def ingest_one(
    path: str,
    *,
    fs_path: Optional[str] = None,
    force: bool = False,
    replace: bool = True,
    total_files: int = 1,
    op: str = "ingest",
    source: str = "ingest_page",
) -> Dict[str, Any]:
    """
    Ingest a single file path:
      1) load + split locally
      2) build chunk metadata (UUIDv5 ids)
      3) vectors first (embed + Qdrant), then OpenSearch (chunks + full text)

    Returns:
      dict with keys: success, status, path, and optionally num_chunks
    """
    normalized_path = normalize_path(path)
    io_path = normalize_path(fs_path) if fs_path else normalized_path
    logger.info(f"📥 Starting ingestion for: {path}")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    log = IngestLogEmitter(path=normalized_path, op=op, source=source)
    with log:
        checksum = compute_checksum(io_path)
        size_bytes = get_file_size(io_path)
        log.set(
            checksum=checksum,
            path_hash=hash_path(normalized_path),
            bytes=size_bytes,
            size=format_file_size(size_bytes),
        )

        # Skip only if same path and checksum; allow duplicates across paths
        if not force and is_file_up_to_date(checksum, normalized_path):
            logger.info(f"✅ File already indexed and unchanged: {normalized_path}")
            log.done(status="Already indexed")
            return {
                "success": True,
                "status": "Already indexed",
                "path": normalized_path,
            }

        is_dup = False
        if not force and is_duplicate_checksum(checksum, normalized_path):
            logger.info(f"♻️ Duplicate file detected: {normalized_path}")
            is_dup = True

        timestamps = get_file_timestamps(io_path)
        created = timestamps.get("created")
        modified = timestamps.get("modified")
        # Use local timezone for indexing timestamp
        indexed_at = datetime.now().astimezone().isoformat()

        logger.info(f"📄 Loading: {normalized_path} (fs: {io_path})")
        try:
            with _io_guard():
                docs = load_documents(io_path)
        except Exception as e:
            logger.error(f"❌ Failed to load document: {e}")
            log.fail(stage="load", error_type=e.__class__.__name__, reason=str(e))
            raise RuntimeError(f"Failed to load document {normalized_path}: {e}") from e

        logger.info(f"🧼 Preprocessing {len(docs)} documents")
        try:
            docs_list = preprocess_to_documents(
                docs_like=docs,
                source_path=normalized_path,
                cfg=PreprocessConfig(),
                doc_type=ext,
            )
        except Exception as e:
            logger.warning(f"Preprocess step skipped due to error: {e}")
            docs_list = docs

        logger.info("📝 Indexing full document text")
        full_text = "\n\n".join(
            getattr(d, "page_content", "") for d in docs_list
        ).strip()
        if not full_text:
            logger.warning(f"⚠️ No valid content found in: {normalized_path}")
            log.done(status="No valid content found")
            return {
                "success": True,
                "status": "No valid content found",
                "path": normalized_path,
            }

        full_doc = {
            "id": hash_path(normalized_path),
            "path": normalized_path,
            "filename": os.path.basename(normalized_path),
            "filetype": ext,
            "modified_at": modified,
            "created_at": created,
            "indexed_at": indexed_at,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "text_full": full_text,
        }

        logger.info("✂️ Splitting document into chunks")
        try:
            chunks = split_documents(docs_list)
        except Exception as e:
            logger.error(f"❌ Failed to split document: {e}")
            log.fail(stage="extract", error_type=e.__class__.__name__, reason=str(e))
            raise RuntimeError(
                f"Failed to split document {normalized_path}: {e}"
            ) from e

        if not chunks:
            logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
            log.done(status="No valid content found")
            return {
                "success": True,
                "status": "No valid content found",
                "path": normalized_path,
                "num_chunks": 0,
            }

        logger.info(f"🧩 Split into {len(chunks)} chunks")

    # Build per-chunk metadata; UUIDv5 id uses path+chunk index so duplicates across paths get unique ids
    for i, chunk in enumerate(chunks):
        chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{normalized_path}-{i}"))
        chunk["chunk_index"] = i
        chunk["path"] = normalized_path
        chunk["checksum"] = checksum
        chunk["filetype"] = ext
        chunk["indexed_at"] = indexed_at
        chunk["created_at"] = created
        chunk["modified_at"] = modified
        chunk["bytes"] = size_bytes
        chunk["size"] = format_file_size(size_bytes)
        chunk["page"] = chunk.get("page", None)
        # position approx. (0..100)
        chunk["location_percent"] = round((i / max(len(chunks) - 1, 1)) * 100)

    # Optional: on force+replace, purge existing entries
    if force and replace:
        from utils.qdrant_utils import delete_vectors_by_ids

        logger.info(
            f"♻️ Reingest replace: deleting existing docs/vectors for file={normalized_path}"
        )

        # 1) Fetch existing chunk IDs from OS for this file (to delete vectors precisely)
        try:
            ids = get_chunk_ids_by_path(normalized_path)
        except Exception as e:
            logger.warning(f"List chunk IDs failed for {normalized_path}: {e}")
            ids = []
        # 2) Delete vectors in Qdrant by IDs (safe if ids is empty)
        try:
            if ids:
                delete_vectors_by_ids(ids)
        except Exception as e:
            logger.warning(f"Qdrant delete failed for {normalized_path}: {e}")

        # 3) Delete chunk docs from OS
        try:
            delete_chunks_by_path(normalized_path)
        except Exception as e:
            logger.warning(f"OpenSearch chunk delete failed for {normalized_path}: {e}")

        # 4) Delete full-text doc(s) from OS
        try:
            delete_fulltext_by_path(normalized_path)
        except Exception as e:
            logger.warning(
                f"OpenSearch full-text delete failed for {normalized_path}: {e}"
            )
            
    os_acc: Dict[str, Any] = {"indexed": 0, "errors": []}

    def _os_index_batch(group: Sequence[Dict[str, Any]]) -> None:
        n, errs = index_documents(list(group))  # whatever your indexer returns
        os_acc["indexed"] += int(n)
        if errs:
            os_acc["errors"].extend(errs)

    # --- VECTORS FIRST (batched): embed + upsert, then OS per batch ---
    try:
        logger.info(f"Embedding + upserting {len(chunks)} chunks to Qdrant in batches (wait=True).")
        ok = qdrant_utils.index_chunks_in_batches(chunks, os_index_batch=_os_index_batch)

        if not ok:
            raise RuntimeError("Qdrant upsert returned falsy")
    except Exception as e:
        logger.error(f"❌ Vector indexing failed: {e}")
        log.fail(stage="index_vec", error_type=e.__class__.__name__, reason=str(e))
        raise RuntimeError(f"Vector indexing failed for {normalized_path}: {e}") from e

    try:
        index_fulltext_document(full_doc)
    except Exception as e:
        logger.warning(f"OpenSearch full-text indexing failed: {e}")
        log.fail(stage="index_fulltext", error_type=e.__class__.__name__, reason=str(e))
        raise RuntimeError(
            f"OpenSearch full-text indexing failed for {normalized_path}: {e}"
        ) from e

    try:
        set_inventory_number_of_chunks(normalized_path, len(chunks))
    except Exception:
        pass

    # Mark last_indexed so the watch inventory immediately reflects indexing
    try:
        set_inventory_last_indexed(normalized_path, indexed_at)
    except Exception:
        pass

    final_status = "Duplicate & Indexed" if is_dup else "Success"
    log.done(status=final_status)
    return {
        "success": True,
        "num_chunks": len(chunks),
        "path": normalized_path,
        "status": final_status,
    }





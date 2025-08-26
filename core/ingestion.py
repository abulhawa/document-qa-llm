import os
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Iterable, Union
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
from utils.opensearch_utils import (
    is_file_up_to_date,
    is_duplicate_checksum,
    index_documents,
    set_has_embedding_true_by_ids,
    index_fulltext_document,
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
      3) SMALL workloads  -> do ALL locally (OS + embed + Qdrant + flag flip)
         LARGE workloads -> queue ALL to Celery (OS + embed + Qdrant + flag flip)

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
            return {"success": False, "status": "Load failed", "path": normalized_path}

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
                "success": False,
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
            "size_bytes": size_bytes,
            "checksum": checksum,
            "text_full": full_text,
        }
        try:
            index_fulltext_document(full_doc)
        except Exception as e:
            logger.warning(f"Full-text indexing failed: {e}")

        logger.info("✂️ Splitting document into chunks")
        try:
            chunks = split_documents(docs_list)
        except Exception as e:
            logger.error(f"❌ Failed to split document: {e}")
            log.fail(stage="extract", error_type=e.__class__.__name__, reason=str(e))
            return {"success": False, "status": "Split failed", "path": normalized_path}

        if not chunks:
            logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
            log.done(status="No valid content found")
            return {
                "success": False,
                "status": "No valid content found",
                "path": normalized_path,
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
        chunk["has_embedding"] = False  # we flip to True below after Qdrant upsert

    # Optional: on force+replace, purge existing entries
    if force and replace:
        try:
            from utils.opensearch_utils import delete_files_by_path_checksum
            from utils.qdrant_utils import delete_vectors_by_path_checksum

            logger.info(
                f"♻️ Reingest replace: deleting existing docs/vectors for checksum={checksum}"
            )
            try:
                delete_files_by_path_checksum([(normalized_path, checksum)])
            except Exception as e:
                logger.warning(f"OpenSearch pre-delete failed: {e}")
            try:
                delete_vectors_by_path_checksum([(normalized_path, checksum)])
            except Exception as e:
                logger.warning(f"Qdrant pre-delete failed: {e}")
        except Exception as e:
            logger.warning(f"Pre-delete imports failed: {e}")

    try:
        logger.info(f"Indexing {len(chunks)} chunks to OpenSearch.")
        index_documents(chunks)
    except Exception as e:
        logger.error(f"❌ OpenSearch chunk indexing failed: {e}")
        log.fail(
            stage="index_os",
            error_type=e.__class__.__name__,
            reason=str(e),
        )
        return {
            "success": False,
            "status": "Local indexing failed",
            "path": normalized_path,
            "num_chunks": len(chunks),
        }

    try:
        logger.info(f"Embedding + upserting {len(chunks)} chunks to Qdrant.")
        ok = qdrant_utils.index_chunks(chunks)  # embeds + upserts
        if not ok:
            raise RuntimeError("Qdrant upsert returned falsy")

        ids = [c["id"] for c in chunks]
        updated, _errs = set_has_embedding_true_by_ids(ids)

        final_status = "Duplicate & Indexed" if is_dup else "Success"
        log.done(status=final_status)
        return {
            "success": True,
            "num_chunks": len(chunks),
            "path": normalized_path,
            "status": final_status,
        }
    except Exception as e:
        logger.error(f"❌ Local pipeline failed: {e}")
        stage = "index_vec" if "qdrant" in str(e).lower() else "flip_flag"
        log.fail(stage=stage, error_type=e.__class__.__name__, reason=str(e))
        return {
            "success": False,
            "status": "Local indexing failed",
            "path": normalized_path,
            "num_chunks": len(chunks),
        }

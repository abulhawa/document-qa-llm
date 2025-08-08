import os
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    logger,
    EMBEDDING_BATCH_SIZE,
    INGEST_MAX_WORKERS,
    INGEST_IO_CONCURRENCY,
    INGEST_MAX_FAILURES,
)
from utils.file_utils import compute_checksum, get_file_timestamps
from core.file_loader import load_documents
from core.chunking import split_documents
from core.opensearch_store import index_documents, is_file_up_to_date
from core.vector_store import index_chunks
from core.embedding_tasks import embed_and_index_chunks
from tracing import (
    start_span,
    TOOL,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
)

# --- Concurrency from config.py ---
MAX_WORKERS: int = INGEST_MAX_WORKERS
IO_CONCURRENCY: int = INGEST_IO_CONCURRENCY
MAX_FAILURES: int = INGEST_MAX_FAILURES
MAX_FILES_FOR_FULL_EMBEDDING = 15
CHUNK_EMBEDDING_THRESHOLD = 60
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
    force: bool = False,
    replace: bool = True,
    total_files: int = 1,
) -> Dict[str, Any]:
    """Ingest a single file path. Optionally force reingestion and replace existing data.

    Returns a result dict with keys: success, status, path, (optional) num_chunks.
    """
    logger.info(f"📥 Starting ingestion for: {path}")
    normalized_path = os.path.normpath(path).replace("\\", "/")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    checksum = compute_checksum(normalized_path)

    if not force and is_file_up_to_date(checksum):
        logger.info(f"✅ File already indexed and unchanged: {normalized_path}")
        return {
            "success": False,
            "status": "Already indexed",
            "path": normalized_path,
        }

    timestamps = get_file_timestamps(normalized_path)
    created = timestamps.get("created")
    modified = timestamps.get("modified")
    indexed_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"📄 Loading: {normalized_path}")
    try:
        with _io_guard():
            docs = load_documents(normalized_path)
    except Exception as e:
        logger.error(f"❌ Failed to load document: {e}")
        return {"success": False, "status": "Load failed", "path": normalized_path}

    logger.info("✂️ Splitting document into chunks")
    try:
        chunks = split_documents(docs)
    except Exception as e:
        logger.error(f"❌ Failed to split document: {e}")
        return {"success": False, "status": "Split failed", "path": normalized_path}

    if not chunks:
        logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
        return {"success": False, "status": "No valid content found", "path": normalized_path}

    logger.info(f"🧩 Split into {len(chunks)} chunks")

    skip_embedding = len(chunks) > CHUNK_EMBEDDING_THRESHOLD or (
        total_files > MAX_FILES_FOR_FULL_EMBEDDING and len(chunks) > 30
    )

    for i, chunk in enumerate(chunks):
        chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{checksum}-{i}"))
        chunk["chunk_index"] = i
        chunk["path"] = normalized_path
        chunk["checksum"] = checksum
        chunk["filetype"] = ext
        chunk["indexed_at"] = indexed_at
        chunk["created_at"] = created
        chunk["modified_at"] = modified
        chunk["page"] = chunk.get("page", None)
        chunk["location_percent"] = round((i / len(chunks)) * 100)
        chunk["has_embedding"] = not skip_embedding

    # If reingestion is forced and replace=True, clear existing index entries for this checksum first
    if force and replace:
        try:
            from utils.opensearch_utils import delete_files_by_checksum
            from utils.qdrant_utils import delete_vectors_by_checksum
            logger.info(f"♻️ Reingest replace: deleting existing docs/vectors for checksum={checksum}")
            try:
                delete_files_by_checksum([checksum])
            except Exception as e:
                logger.warning(f"OpenSearch pre-delete failed: {e}")
            try:
                delete_vectors_by_checksum(checksum)
            except Exception as e:
                logger.warning(f"Qdrant pre-delete failed: {e}")
        except Exception as e:
            logger.warning(f"Pre-delete imports failed: {e}")

    logger.info(f"Indexing {len(chunks)} chunks to OpenSearch for: {normalized_path}")
    try:
        index_documents(chunks)
    except Exception as e:
        logger.error(f"❌ Failed to index document in OpenSearch: {e}")
        return {
            "success": False,
            "status": "OpenSearch indexing failed",
            "path": normalized_path,
        }

    try:
        if skip_embedding:
            logger.info(
                f"📦 Large batch detected — deferring embedding of {len(chunks)} chunks to Celery (batch size {EMBEDDING_BATCH_SIZE})."
            )
            print(
                f"📣 Sending task to Celery with broker: {embed_and_index_chunks.app.conf.broker_url}"
            )
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
                # Consider adding task retry/backoff/acks_late in the Celery task decorator
                embed_and_index_chunks.delay(chunks=batch)
            status_message = "Partially indexed - embedding will run in the background"
        else:
            logger.info(f"Embedding {len(chunks)} chunks from {normalized_path}.")
            index_chunks(chunks)
            status_message = "Successfully indexed"
    except Exception as e:
        logger.error(f"❌ Failed to embed chunks: {e}")
        return {
            "success": False,
            "status": "Indexed, but embedding failed",
            "path": normalized_path,
            "num_chunks": len(chunks),
        }

    return {
        "success": True,
        "num_chunks": len(chunks),
        "path": normalized_path,
        "status": status_message,
    }


def ingest(
    inputs: List[str],
    *,
    expand_dirs: bool = True,
    force: bool = False,
    replace: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[Dict[str, Any]]:
    """Ingest a list of file paths and/or directories.

    - If expand_dirs=True, directories are traversed for .pdf/.docx/.txt files.
    - Reingestion is controlled by force/replace flags.
    """
    # Expand inputs
    doc_files: List[str] = []
    for p in inputs:
        if os.path.isfile(p) and p.lower().endswith((".pdf", ".docx", ".txt")):
            doc_files.append(p)
        elif expand_dirs and os.path.isdir(p):
            for dirpath, _, filenames in os.walk(p):
                for fname in filenames:
                    if fname.lower().endswith((".pdf", ".docx", ".txt")):
                        full_path = os.path.join(dirpath, fname)
                        doc_files.append(full_path)

    if not doc_files:
        logger.warning("⚠️ No valid document files found in provided inputs.")
        return []

    start_time = time.time()
    results: List[Dict[str, Any]] = []
    total = len(doc_files)
    completed = 0
    failures = 0

    with start_span("ingest.batch", TOOL) as span:
        # Thread pool for parallel ingestion
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_path = {
                executor.submit(ingest_one, f, force=force, replace=replace, total_files=total): f
                for f in doc_files
            }
            for future in as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    result = future.result()
                except Exception as e:
                    failures += 1
                    logger.exception(f"❌ Ingestion failed for {p}: {e}")
                    record_span_error(span, e)
                    result = {"success": False, "status": str(e), "path": p}
                results.append(result)
                completed += 1

                # Circuit breaker: stop early on many failures
                if MAX_FAILURES and failures >= MAX_FAILURES:
                    logger.error(
                        f"⛔ Circuit breaker tripped: {failures} failures (limit {MAX_FAILURES}). Stopping early."
                    )
                    # Try to cancel any remaining futures
                    for f in future_to_path.keys():
                        f.cancel()
                    break

                if progress_callback:
                    elapsed = time.time() - start_time
                    try:
                        progress_callback(completed, total, elapsed)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

    return results

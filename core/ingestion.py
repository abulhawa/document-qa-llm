import os
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable, Optional, Iterable, Union
from core.document_preprocessor import preprocess_to_documents, PreprocessConfig
from core.file_loader import load_documents
from core.chunking import split_documents
from concurrent.futures import ThreadPoolExecutor, as_completed
from worker.celery_worker import app as celery_app
from config import (
    logger,
    EMBEDDING_BATCH_SIZE,
    INGEST_MAX_WORKERS,
    INGEST_IO_CONCURRENCY,
    INGEST_MAX_FAILURES,
)
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
)
from utils.ingest_logging import IngestLogEmitter


# --- Concurrency from config.py ---
MAX_WORKERS: int = INGEST_MAX_WORKERS
IO_CONCURRENCY: int = INGEST_IO_CONCURRENCY
MAX_FAILURES: int = INGEST_MAX_FAILURES

# Size heuristics (unchanged behavior)
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
    op: str = "ingest",
    source: str = "ingest_page",
    run_id: str | None = None,
    retry_of: str | None = None,
) -> Dict[str, Any]:
    """
    Ingest a single file path:
      1) load + split locally
      2) build chunk metadata (UUIDv5 ids)
      3) SMALL files  -> do ALL locally (OS + embed + Qdrant + flag flip)
         LARGE files  -> queue ALL to Celery (OS + embed + Qdrant + flag flip)

    Returns:
      dict with keys: success, status, path, and optionally num_chunks, batches
    """
    logger.info(f"📥 Starting ingestion for: {path}")
    normalized_path = normalize_path(path)
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    log = IngestLogEmitter(path=normalized_path, op=op, source=source, run_id=run_id)
    if retry_of:
        log.set(retry_of=retry_of)
    with log:
        checksum = compute_checksum(normalized_path)
        size_bytes = get_file_size(normalized_path)
        log.set(
            checksum=checksum,
            path_hash=hash_path(normalized_path),
            bytes=size_bytes,
            size=format_file_size(size_bytes),
        )

        # Skip if already indexed and unchanged (based on checksum + path)
        if not force and is_file_up_to_date(checksum, normalized_path):
            logger.info(f"✅ File already indexed and unchanged: {normalized_path}")
            log.done(status="Already indexed")
            return {
                "success": False,
                "status": "Already indexed",
                "path": normalized_path,
            }

        if not force and is_duplicate_checksum(checksum, normalized_path):
            logger.info(f"♻️ Duplicate file detected: {normalized_path}")
            log.done(status="Duplicate")
            return {"success": False, "status": "Duplicate", "path": normalized_path}

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
        chunk["id"] = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{normalized_path}-{i}")
        )
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
        # Celery will flip this after embedding
        chunk["has_embedding"] = False

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

    # Decide small vs large (same thresholds as before)
    skip_embedding = len(chunks) > CHUNK_EMBEDDING_THRESHOLD or (
        total_files > MAX_FILES_FOR_FULL_EMBEDDING and len(chunks) > 30
    )

    if not skip_embedding:
        # SMALL file → do EVERYTHING locally
        try:
            logger.info(f"Indexing {len(chunks)} chunks to OpenSearch (small file).")
            index_documents(chunks)

            logger.info(f"Embedding + upserting {len(chunks)} chunks locally.")
            ok = qdrant_utils.index_chunks(chunks)  # embeds + upserts
            if not ok:
                raise RuntimeError("Qdrant upsert returned falsy")

            ids = [c["id"] for c in chunks]
            updated, _errs = set_has_embedding_true_by_ids(ids)

            log.done(status="Success")
            return {
                "success": True,
                "num_chunks": len(chunks),
                "batches": 0,
                "path": normalized_path,
                "status": "Successfully indexed",
            }
        except Exception as e:
            logger.error(f"❌ Local pipeline failed: {e}")
            stage = "index_vec" if "qdrant" in str(e).lower() else "index_os"
            log.fail(stage=stage, error_type=e.__class__.__name__, reason=str(e))
            return {
                "success": False,
                "status": "Local indexing failed",
                "path": normalized_path,
                "num_chunks": len(chunks),
            }

    else:  # LARGE file → queue EVERYTHING to Celery (OS + embed + Qdrant + flip)
        batches = 0
        try:
            logger.info(
                f"📦 Large file — queuing full pipeline for {len(chunks)} chunks "
                f"in batches of {EMBEDDING_BATCH_SIZE}."
            )
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
                celery_app.send_task(
                    "core.ingestion_tasks.index_and_embed_chunks",
                    args=[batch],
                )
                batches += 1

            log.done(status="Success")
            return {
                "success": True,
                "num_chunks": len(chunks),
                "batches": batches,
                "path": normalized_path,
                "status": "Partially indexed — background worker will finish",
            }
        except Exception as e:
            logger.error(f"❌ Failed to enqueue background tasks: {e}")
            log.fail(stage="index_os", error_type=e.__class__.__name__, reason=str(e))
            return {
                "success": False,
                "status": "Queueing failed",
                "path": normalized_path,
                "num_chunks": len(chunks),
            }


def ingest(
    inputs: Union[str, Iterable[str]],
    *,
    expand_dirs: bool = True,
    force: bool = False,
    replace: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    op: str = "ingest",
    source: str = "ingest_page",
    run_id: str | None = None,
    retry_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Ingest one or more file paths and/or directories.

    `inputs` may be a single string path or any iterable of paths.
    If `expand_dirs` is True, any directories will be walked for supported types.
    """
    # Normalize inputs to a list
    if isinstance(inputs, (str, os.PathLike)):
        iter_inputs = [inputs]
    else:
        iter_inputs = list(inputs)

    # Expand into concrete files
    doc_files: List[str] = []
    for p in iter_inputs:
        if os.path.isfile(p) and p.lower().endswith((".pdf", ".docx", ".txt")):
            doc_files.append(normalize_path(p))
        elif expand_dirs and os.path.isdir(p):
            for dirpath, _, filenames in os.walk(p):
                for fname in filenames:
                    if fname.lower().endswith((".pdf", ".docx", ".txt")):
                        full_path = os.path.join(dirpath, fname)
                        doc_files.append(normalize_path(full_path))

    if not doc_files:
        logger.warning("⚠️ No valid document files found in provided inputs.")
        return []

    if run_id is None:
        run_id = str(uuid.uuid4())

    start_time = time.time()
    results: List[Dict[str, Any]] = []
    total = len(doc_files)
    completed = 0
    failures = 0

    # Thread pool for parallel ingestion
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(
                ingest_one,
                f,
                force=force,
                replace=replace,
                total_files=total,
                op=op,
                source=source,
                run_id=run_id,
                retry_of=(retry_map.get(f) if retry_map else None),
            ): f
            for f in doc_files
        }
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                result = future.result()
            except Exception as e:
                failures += 1
                logger.exception(f"❌ Ingestion failed for {p}: {e}")
                result = {"success": False, "status": str(e), "path": p}
            results.append(result)
            completed += 1

            # Circuit breaker on too many failures
            if MAX_FAILURES and failures >= MAX_FAILURES:
                logger.error(
                    f"⛔ Circuit breaker tripped: {failures} failures (limit {MAX_FAILURES}). Stopping early."
                )
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

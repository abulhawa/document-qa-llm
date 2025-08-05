import os
import uuid
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import logger
from utils import compute_checksum, get_file_timestamps
from core.file_loader import load_documents
from core.chunking import split_documents
from core.opensearch_store import index_documents, is_file_up_to_date
from core.vector_store import index_chunks
from tracing import (
    start_span,
    TOOL,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
)

MAX_WORKERS = 8
CHUNK_EMBEDDING_THRESHOLD = 60
MAX_FILES_FOR_FULL_EMBEDDING = 15


def ingest_file(path: str, total_files: int = 1) -> Dict[str, Any]:
    logger.info(f"\U0001f4e5 Starting ingestion for: {path}")
    normalized_path = os.path.normpath(path).replace("\\", "/")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    checksum = compute_checksum(normalized_path)

    if is_file_up_to_date(checksum):
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

    docs = load_documents(normalized_path)
    if not docs:
        logger.warning(f"⚠️ No valid content found in: {normalized_path}")
        return {"success": False, "status": "No content found", "path": normalized_path}

    chunks = split_documents(docs)
    if not chunks:
        logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
        return {"success": False, "status": "Chunking failed", "path": normalized_path}

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

    logger.info(f"Indexing {len(chunks)} chunks to OpenSearch for: {normalized_path}")
    try:
        index_documents(chunks)
    except Exception as e:
        logger.error(f"❌ Failed to index document in OpenSearch: {e}")
        return {
            "success": False,
            "status": "OpenSearch indexing failed",
            "path": normalized_path,
            "num_chunks": len(chunks),
        }

    logger.info(f"✅ Indexed {len(chunks)} chunks to OpenSearch for: {normalized_path}")

    try:
        if skip_embedding:
            logger.info(
                f"Skipping embedding for {len(chunks)} chunks from {normalized_path} due to threshold."
            )
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


def ingest_files(
    paths: List[str],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    total = len(paths)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(ingest_file, f, total): f for f in paths}
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)
            completed += 1
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(completed, total, elapsed)

    return results


def ingest_paths(
    paths: List[str],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[Dict[str, Any]]:
    doc_files: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.lower().endswith((".pdf", ".docx", ".txt")):
            doc_files.append(path)
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for fname in filenames:
                    if fname.lower().endswith((".pdf", ".docx", ".txt")):
                        full_path = os.path.join(dirpath, fname)
                        doc_files.append(full_path)

    if not doc_files:
        logger.warning("⚠️ No valid document files found in provided paths.")
        return []

    return ingest_files(doc_files, progress_callback)

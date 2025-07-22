import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import logger
from utils import compute_checksum, get_file_timestamps
from core.file_loader import load_documents
from core.chunking import split_documents
from core.vector_store import is_file_up_to_date, index_chunks
from tracing import get_tracer

tracer = get_tracer(__name__)
MAX_WORKERS = 4  # Tune here or load from config/env


@tracer.start_as_current_span("ingest")
def ingest(path: str) -> Dict[str, Any]:
    """
    Entry point: handles both single file and folder ingestion.
    Uses multithreading for folders.
    """
    if not os.path.exists(path):
        logger.error(f"❌ Path does not exist: {path}")
        return {"success": False, "reason": "Path does not exist"}

    if os.path.isfile(path):
        return ingest_file(path)

    # Handle folder ingestion with parallelism
    logger.info(f"📁 Ingesting folder: {path}")
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith((".pdf", ".docx", ".txt"))
    ]

    results = ingest_files(files)

    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"✅ Folder ingestion complete: {success_count}/{len(files)} files succeeded")
    return {"success": True, "num_files": len(files), "results": results}


@tracer.start_as_current_span("ingest_file")
def ingest_file(path: str) -> Dict[str, Any]:
    """
    Ingest a single file: load, chunk, index.
    Embedding is handled inside index_chunks().
    """
    logger.info(f"📥 Starting ingestion for: {path}")

    normalized_path = os.path.normpath(path).replace("\\", "/")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    checksum = compute_checksum(normalized_path)

    if is_file_up_to_date(checksum):
        logger.info(f"✅ File already indexed and unchanged: {normalized_path}")
        return {"success": False, "reason": "Already indexed", "path": normalized_path}

    timestamps = get_file_timestamps(normalized_path)
    created = timestamps.get("created")
    modified = timestamps.get("modified")
    indexed_at = datetime.now(timezone.utc).isoformat()

    docs = load_documents(normalized_path)
    if not docs:
        logger.warning(f"⚠️ No valid documents found in: {normalized_path}")
        return {"success": False, "reason": "No content found", "path": normalized_path}

    chunks = split_documents(docs)
    if not chunks:
        logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
        return {"success": False, "reason": "Chunking failed", "path": normalized_path}

    logger.info(f"🧩 Split into {len(chunks)} chunks")

    texts: List[str] = []
    metadata_list: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk["text"])
        meta = {
            "path": normalized_path,
            "checksum": checksum,
            "filename": os.path.basename(normalized_path),
            "filetype": ext,
            "indexed_at": indexed_at,
            "created": created,
            "modified": modified,
            "chunk_index": i,
        }
        if "page" in chunk:
            meta["page"] = chunk["page"]
        else:
            pct = round((i / len(chunks)) * 100)
            meta["location_percent"] = min(pct, 100)
        metadata_list.append(meta)

    success = index_chunks(texts, metadata_list)
    if not success:
        logger.warning(f"❌ Failed to index chunks for: {normalized_path}")
        return {"success": False, "reason": "Embedding or upsert failed", "path": normalized_path}

    logger.info(f"✅ Indexed {len(texts)} chunks for: {normalized_path}")
    return {
        "success": True,
        "num_chunks": len(texts),
        "path": normalized_path,
    }


def ingest_files(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest a list of files in parallel using threads.
    Returns a list of per-file result dicts.
    """
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(ingest_file, f): f for f in paths}
        for future in as_completed(future_to_path):
            results.append(future.result())
    return results


def ingest_paths(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Accepts a list of file and/or folder paths.
    Recursively collects all .pdf, .docx, and .txt files and ingests them in parallel.
    Returns a list of per-file result dicts.
    """
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

    return ingest_files(doc_files)

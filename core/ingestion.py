import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import logger
from utils import compute_checksum, get_file_timestamps
from core.file_loader import load_documents
from core.chunking import split_documents
from core.vector_store import is_file_up_to_date, index_chunks
from core.opensearch_store import index_documents

from tracing import (
    start_span,
    TOOL,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
)

MAX_WORKERS = 8  # Tune here or load from config/env


def ingest_file(path: str) -> Dict[str, Any]:
    """Ingest a single file: load, chunk, index. Embedding is inside index_chunks()."""
    logger.info(f"📥 Starting ingestion for: {path}")
    normalized_path = os.path.normpath(path).replace("\\", "/")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    checksum = compute_checksum(normalized_path)

    try:
        if is_file_up_to_date(checksum):
            logger.info(f"✅ File already indexed and unchanged: {normalized_path}")
            return {
                "success": False,
                "reason": "Already indexed",
                "path": normalized_path,
            }

        timestamps = get_file_timestamps(normalized_path)
        created = timestamps.get("created")
        modified = timestamps.get("modified")
        indexed_at = datetime.now(timezone.utc).isoformat()

        docs = load_documents(normalized_path)
        if not docs:
            logger.warning(f"⚠️ No valid content found in: {normalized_path}")
            return {
                "success": False,
                "reason": "No content found",
                "path": normalized_path,
            }

        chunks = split_documents(docs)
        if not chunks:
            logger.warning(f"⚠️ No chunks generated from: {normalized_path}")
            return {
                "success": False,
                "reason": "Chunking failed",
                "path": normalized_path,
            }

        logger.info(f"🧩 Split into {len(chunks)} chunks")

        texts: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            texts.append(chunk["text"])
            meta = {
                "path": normalized_path,
                "checksum": checksum,
                "filetype": ext,
                "indexed_at": indexed_at,
                "created": created,
                "modified": modified,
                "chunk_index": i,
                "page": chunk.get("page", None),
            }
            pct = round((i / len(chunks)) * 100)
            meta["location_percent"] = min(pct, 100)
            metadata_list.append(meta)

        success = index_chunks(texts, metadata_list)

        if not success:
            logger.warning(f"❌ Failed to index chunks for: {normalized_path}")
            return {
                "success": False,
                "reason": "Embedding or upsert failed",
                "path": normalized_path,
            }

        # Index full document in OpenSearch
        try:
            index_documents(
                [
                    {
                        "path": normalized_path,
                        "content": " ".join(doc.page_content for doc in docs),
                        "checksum": checksum,
                        "created_at": created,
                        "modified_at": modified,
                    }
                ]
            )
        except Exception as e:
            logger.error(f"❌ Failed to index document in OpenSearch: {e}")

        logger.info(f"✅ Indexed {len(texts)} chunks for: {normalized_path}")
        return {
            "success": True,
            "num_chunks": len(texts),
            "path": normalized_path,
        }

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return {"success": False, "reason": str(e), "path": normalized_path}


def ingest_files(paths: List[str]) -> List[Dict[str, Any]]:
    """Ingest a list of files in parallel using threads. Returns a list of result dicts."""
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(ingest_file, f): f for f in paths}
        for future in as_completed(future_to_path):
            results.append(future.result())

    return results


def ingest_paths(paths: List[str]) -> List[Dict[str, Any]]:
    """Accepts file and/or folder paths. Collects matching files and ingests them in parallel."""
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

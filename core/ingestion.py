# ingestion.py

import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from utils import compute_checksum, get_file_timestamps
from file_loader import load_documents  # to be implemented per format (PDF, DOCX, etc.)
from embedding import embed
from vector_store import index_chunks  # wraps Qdrant client
from config import CHUNK_BATCH_SIZE


def process_file(path: str) -> Dict[str, Any]:
    """Load, split, and prepare chunks + metadata from a single file."""
    if not os.path.exists(path):
        return {"success": False, "reason": "File not found", "path": path}

    checksum = compute_checksum(path)

    # Timestamp info
    try:
        created, modified = get_file_timestamps(path)
    except Exception as e:
        return {"success": False, "reason": f"Stat error: {e}", "path": path}

    # Load and chunk
    try:
        chunks = load_and_split_file(path)
    except Exception as e:
        return {"success": False, "reason": f"Loading error: {e}", "path": path}

    if not chunks:
        return {"success": False, "reason": "No chunks extracted", "path": path}

    timestamp = datetime.now(timezone.utc).isoformat()
    texts, metadata = [], []
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    for i, chunk in enumerate(chunks):
        texts.append(chunk.page_content)
        meta = {
            "path": os.path.normpath(path).replace("\\", "/"),
            "filename": os.path.basename(path),
            "filetype": ext,
            "checksum": checksum,
            "chunk_index": i,
            "indexed_at": timestamp,
            "created": created,
            "modified": modified,
        }
        if "page" in chunk.metadata:
            meta["page"] = chunk.metadata["page"]
        else:
            meta["location_percent"] = round((i / len(chunks)) * 100)
        metadata.append(meta)

    return {
        "success": True,
        "path": path,
        "texts": texts,
        "metadata": metadata,
        "num_chunks": len(texts),
    }


def embed_and_index_chunks(texts: List[str], metadata: List[Dict[str, Any]]) -> bool:
    """Embed texts and store vectors + metadata in Qdrant."""
    if not texts:
        return False

    try:
        embeddings = embed(texts)
        index_chunks(embeddings, metadata)
        return True
    except Exception as e:
        # log inside upsert/embed modules
        return False


def run_batch_ingestion(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Processes and indexes a list of files in chunked batches."""
    results = []
    batch_texts, batch_metadata = [], []

    for path in file_paths:
        result = process_file(path)
        results.append(result)

        if result.get("success"):
            batch_texts.extend(result["texts"])
            batch_metadata.extend(result["metadata"])

        if len(batch_texts) >= CHUNK_BATCH_SIZE:
            embed_and_index_chunks(batch_texts, batch_metadata)
            batch_texts.clear()
            batch_metadata.clear()

    # Final flush
    if batch_texts:
        embed_and_index_chunks(batch_texts, batch_metadata)

    return results

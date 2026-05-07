"""Document chunk construction for the ingestion pipeline.

This module owns the first chunk records produced from preprocessed
LangChain `Document` objects. It does not assign final storage IDs or
file-level ingestion metadata. `ingestion.orchestrator.ingest_one` adds
those fields after all chunks for a file have been produced.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import logger, CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split LangChain `Document` objects into character-based chunk records.

    The splitter uses `RecursiveCharacterTextSplitter` with paragraph,
    newline, whitespace, and finally character-level separators. Defaults
    come from `config.CHUNK_SIZE` and `config.CHUNK_OVERLAP`.

    The returned `chunk_index` is local to each input `Document`; the
    orchestrator rewrites it to a file-level index before indexing. The
    returned `location_percent` is also a local estimate and is overwritten
    by the orchestrator with a file-level position.

    Args:
        docs: List of LangChain `Document` objects. Expected metadata keys:
              - "source": full file path (preferred), or "path" fallback
              - "page": page number (optional)
        chunk_size: Max number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A list of dictionaries. Each dict contains:
            - text:              str, the chunk text
            - page:              int | None, page number if available
            - path:              str, full source path if available
            - chunk_index:       int, index of the chunk within its document
            - location_percent:  float, approx position of the chunk in the doc (0..100)
    """
    logger.info(f"Splitting {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Dict[str, Any]] = []

    for doc in docs:
        # Extract text + metadata
        raw_text = getattr(doc, "page_content", "") or ""
        # Be defensive: ensure string type
        raw_text = str(raw_text)

        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("path") or "unknown"
        page = metadata.get("page", None)

        # Split text
        splits = splitter.split_text(raw_text)
        total = max(len(splits), 1)

        # Build per-chunk records with richer metadata
        for i, chunk in enumerate(splits):
            # location_percent maps first chunk to 0 and last to ~100
            denom = max(total - 1, 1)
            location_percent = round((i / denom) * 100.0, 2)

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page,
                    "path": source,
                    "chunk_index": i,
                    "location_percent": location_percent,
                }
            )

        logger.debug(f"Split document {source} into {len(splits)} chunks.")

    logger.info(f"Generated {len(all_chunks)} total chunks.")
    return all_chunks

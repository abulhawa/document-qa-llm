from typing import List, Dict, Any, TYPE_CHECKING

from config import logger, CHUNK_SIZE, CHUNK_OVERLAP

if TYPE_CHECKING:
    from langchain_core.documents import Document


def split_documents(
    docs: List["Document"],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split a list of LangChain Document objects into chunks with metadata.

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter

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

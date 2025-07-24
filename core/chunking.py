from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import logger, CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split a list of LangChain Document objects into chunks with metadata.

    Args:
        docs: List of LangChain Document objects.
        chunk_size: Max number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A list of dictionaries, each containing a chunk and its metadata.
    """
    logger.info(f"Splitting {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Dict[str, Any]] = []
    for doc_index, doc in enumerate(docs):
        raw_text = doc.page_content
        metadata = doc.metadata or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", None)

        splits = splitter.split_text(raw_text)

        for i, chunk in enumerate(splits):
            chunk_data = {
                "text": chunk,
                "chunk_index": i,
                "source": source,
                "page": page,
                "doc_index": doc_index,
            }
            all_chunks.append(chunk_data)

        logger.debug(
            f"Split document {doc_index} ({source}) into {len(splits)} chunks."
        )

    logger.info(f"Generated {len(all_chunks)} total chunks.")
    return all_chunks

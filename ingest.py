import os
from datetime import datetime, timezone
from typing import List
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import logger, CHUNK_SIZE, CHUNK_OVERLAP
from utils import compute_checksum
from vector_store import is_file_already_indexed, upsert_embeddings


def load_documents(path: str) -> List[Document]:
    """Load documents from file or folder."""
    docs: List[Document] = []

    if os.path.isfile(path):
        paths = [path]
    else:
        paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith((".pdf", ".docx", ".txt"))
        ]

    for file_path in paths:
        if file_path.endswith(".pdf"):
            docs += PyPDFLoader(file_path).load()
        elif file_path.endswith(".docx"):
            docs += Docx2txtLoader(file_path).load()
        elif file_path.endswith(".txt"):
            docs += TextLoader(file_path).load()

    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split documents into %d chunks", len(chunks))
    return chunks


def ingest(path: str) -> None:
    """Main ingestion pipeline: load, split, embed, store in Qdrant."""
    logger.info("📥 Starting ingestion for: %s", path)

    if not os.path.exists(path):
        logger.error("Path does not exist: %s", path)
        return

    checksum: str = compute_checksum(path)

    if is_file_already_indexed(checksum):
        logger.info("✅ File already indexed and unchanged: %s", path)
        return

    documents = load_documents(path)
    if not documents:
        logger.warning("⚠️ No valid documents found in: %s", path)
        return

    chunks = split_documents(documents)
    texts: List[str] = [chunk.page_content for chunk in chunks]
    timestamp: str = datetime.now(timezone.utc).isoformat()

    upsert_embeddings(texts, path, checksum, timestamp)
    logger.info("📦 Ingestion complete: %d chunks stored for %s", len(texts), path)

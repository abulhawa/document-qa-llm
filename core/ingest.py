import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import logger, CHUNK_SIZE, CHUNK_OVERLAP
from utils import compute_checksum
from vector_store import is_file_already_indexed, index_chunks


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


def ingest(path: str) -> Dict[str, Any]:
    """Main ingestion pipeline: load, split, embed, store in Qdrant.
    Returns a result dict with success status and reason or chunk count.
    """
    logger.info("📥 Starting ingestion for: %s", path)

    if not os.path.exists(path):
        logger.error("Path does not exist: %s", path)
        return {"success": False, "reason": "File does not exist"}

    normalized_path = os.path.normpath(path).replace("\\", "/")
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")  # e.g., 'pdf'

    checksum: str = compute_checksum(normalized_path)

    if is_file_already_indexed(checksum):
        logger.info("✅ File already indexed and unchanged: %s", normalized_path)
        return {"success": False, "reason": "Already indexed"}

    # Get file timestamps
    try:
        stat = os.stat(normalized_path)
        created = datetime.fromtimestamp(stat.st_ctime).isoformat(" ", "seconds")
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(" ", "seconds")
    except Exception as e:
        logger.exception("Failed to get file timestamps: %s", e)
        return {"success": False, "reason": "Failed to read file timestamps"}
    
    doc_segments  = load_documents(normalized_path)
    if not doc_segments :
        logger.warning("⚠️ No valid documents found in: %s", normalized_path)
        return {"success": False, "reason": "No content found"}

    chunks = split_documents(doc_segments)
    if not chunks:
        logger.warning("⚠️ No chunks generated from: %s", normalized_path)
        return {"success": False, "reason": "Chunking failed"}

    # Build per-chunk metadata
    timestamp: str = datetime.now(timezone.utc).isoformat()
    texts: List[str] = []
    metadata_list: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk.page_content)
        meta = {
            "path": normalized_path,
            "checksum": checksum,
            "filename": os.path.basename(normalized_path),
            "filetype": ext,
            "checksum": checksum,
            "indexed_at": timestamp,
            "created": created,
            "modified": modified,
            "chunk_index": i,
        }
        if "page" in chunk.metadata:
            meta["page"] = chunk.metadata["page"]
        else:
            pct = round((i / len(chunks)) * 100)
            meta["location_percent"] = min(pct, 100)
        metadata_list.append(meta)

    success = index_chunks(texts, metadata_list)
    if not success:
        logger.warning("❌ Failed to index chunks for %s", normalized_path)
        return {"success": False, "reason": "Embedding or upsert failed"}
    
    logger.info("✅ Indexed %d chunks for %s", len(texts), normalized_path)
    return {
        "success": True,
        "num_chunks": len(texts),
        "path": normalized_path,
    }

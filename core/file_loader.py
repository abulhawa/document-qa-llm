from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os
from config import logger


def load_documents(path: str) -> List[Document]:
    """Load a document from a file path."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        elif ext == ".docx":
            return Docx2txtLoader(path).load()
        elif ext == ".txt":
            return TextLoader(path).load()
        else:
            logger.warning("Unsupported file type: %s", path)
            return []
    except Exception as e:
        logger.exception("Failed to load file: %s", path)
        return []

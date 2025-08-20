import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from config import logger


_TXT_FALLBACK_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")

def _load_txt_with_fallbacks(path: str) -> List[Document]:
    # Try autodetect first (if supported by your langchain version)
    try:
        docs = TextLoader(path, autodetect_encoding=True).load()
        for d in docs:
            d.metadata.setdefault("encoding", "autodetect")
        logger.info("Loaded %s with autodetect_encoding=True", path)
        return docs
    except TypeError:
        pass
    except Exception as e:
        logger.debug("Autodetect failed for %s: %r", path, e)

    last_err: Optional[Exception] = None
    for enc in _TXT_FALLBACK_ENCODINGS:
        try:
            docs = TextLoader(path, encoding=enc).load()
            for d in docs:
                d.metadata["encoding"] = enc
            logger.info("Loaded %s with encoding=%s", path, enc)
            return docs
        except Exception as e:
            last_err = e

    # Last resort: don't crash - salvage text
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        logger.warning("Loaded %s with utf-8(errors=replace) due to %r", path, last_err)
        return [Document(page_content=text, metadata={"source": path, "encoding": "utf-8(errors=replace)"})]
    except Exception as e:
        logger.exception("Failed to salvage text from %s: %r", path, e)
        return []

def load_documents(path: str) -> List[Document]:
    """Load a document from a file path."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        elif ext == ".docx":
            return Docx2txtLoader(path).load()
        elif ext == ".txt":
            return _load_txt_with_fallbacks(path)
        else:
            logger.warning("Unsupported file type: %s", path)
            return []
    except Exception:
        logger.exception("Failed to load file: %s", path)
        return []

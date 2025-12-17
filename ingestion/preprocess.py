import os
from typing import Iterable, List

from config import logger
from core.chunking import split_documents
from core.document_preprocessor import PreprocessConfig, preprocess_to_documents


def preprocess_documents(docs_like: Iterable, normalized_path: str, ext: str):
    """Run document preprocessing with graceful fallback to the original docs."""

    try:
        return preprocess_to_documents(
            docs_like=docs_like,
            source_path=normalized_path,
            cfg=PreprocessConfig(),
            doc_type=ext,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Preprocess step skipped due to error: %s", e)
        return docs_like


def chunk_documents(docs_list: List):
    """Split preprocessed documents into chunks."""

    return split_documents(docs_list)


def build_full_text(docs_list: List) -> str:
    """Combine document contents into a single full-text payload."""

    return "\n\n".join(getattr(d, "page_content", "") for d in docs_list).strip()


__all__ = ["preprocess_documents", "chunk_documents", "build_full_text"]

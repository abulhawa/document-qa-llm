from __future__ import annotations

from typing import List, Dict, Any, Tuple

# Reuse existing implementations and keep a stable API
from utils.opensearch_utils import (
    index_documents,
    list_files_from_opensearch,
    get_chunk_ids_by_path,
    delete_chunks_by_path,
    get_duplicate_checksums,
    get_files_by_checksum,
    is_file_up_to_date,
    is_duplicate_checksum,
)

__all__ = [
    "index_documents",
    "list_files_from_opensearch",
    "get_chunk_ids_by_path",
    "delete_chunks_by_path",
    "get_duplicate_checksums",
    "get_files_by_checksum",
    "is_file_up_to_date",
    "is_duplicate_checksum",
]


import threading
from contextlib import contextmanager
from typing import Optional, Tuple

from config import INGEST_IO_CONCURRENCY
from core.file_loader import load_documents
from utils.file_utils import (
    compute_checksum,
    format_file_size,
    get_file_size,
    get_file_timestamps,
    hash_path,
    normalize_path,
)

# IO semaphore to limit simultaneous file reads (helps avoid too many open files)
IO_CONCURRENCY: int = INGEST_IO_CONCURRENCY
_io_semaphore = threading.Semaphore(IO_CONCURRENCY)


@contextmanager
def _io_guard():
    _io_semaphore.acquire()
    try:
        yield
    finally:
        _io_semaphore.release()


def normalize_paths(path: str, fs_path: Optional[str] = None) -> Tuple[str, str]:
    """Return normalized logical and filesystem paths for ingestion."""

    normalized_path = normalize_path(path)
    io_path = normalize_path(fs_path) if fs_path else normalized_path
    return normalized_path, io_path


def load_file_documents(io_path: str):
    """Load file contents using the shared IO guard."""

    with _io_guard():
        return load_documents(io_path)


def file_fingerprint(io_path: str):
    """Compute checksum, size, and timestamps for the ingestion source file."""

    checksum = compute_checksum(io_path)
    size_bytes = get_file_size(io_path)
    timestamps = get_file_timestamps(io_path)
    return checksum, size_bytes, timestamps


__all__ = [
    "IO_CONCURRENCY",
    "normalize_paths",
    "load_file_documents",
    "file_fingerprint",
    "format_file_size",
    "hash_path",
]

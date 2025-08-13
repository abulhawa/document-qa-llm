import os
import hashlib
from datetime import datetime


def compute_checksum(path: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_path(path: str) -> str:
    """Normalize path to use forward slashes."""
    return os.path.normpath(path).replace("\\", "/")


def hash_path(path: str) -> str:
    """Return a stable hash for a given path string."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def get_file_size(path: str) -> int:
    """Return file size in bytes, or 0 if unavailable."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def get_file_timestamps(path: str) -> dict:
    """
    Returns creation and modification timestamps in ISO format.
    """
    try:
        stat = os.stat(path)
        created = datetime.fromtimestamp(stat.st_ctime)
        modified = datetime.fromtimestamp(stat.st_mtime)
        return {"created": created, "modified": modified}
    except Exception as e:
        # Still return keys with fallback values
        return {"created": "", "modified": ""}

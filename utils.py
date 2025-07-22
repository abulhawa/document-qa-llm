import os
import hashlib
from datetime import datetime
from typing import Dict


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


def get_file_timestamps(path: str) -> Dict[str, str]:
    """
    Returns creation and modification timestamps in ISO format.
    """
    try:
        stat = os.stat(path)
        created = datetime.fromtimestamp(stat.st_ctime).isoformat(" ", "seconds")
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(" ", "seconds")
        return {"created": created, "modified": modified}
    except Exception as e:
        # Still return keys with fallback values
        return {"created": "", "modified": ""}

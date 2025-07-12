import hashlib
from config import logger

def compute_checksum(path: str) -> str:
    """Compute SHA-256 checksum for a given file path."""
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        checksum = sha256.hexdigest()
        logger.debug("Computed checksum for %s: %s", path, checksum)
        return checksum
    except FileNotFoundError:
        logger.error("File not found for checksum computation: %s", path)
        return ""

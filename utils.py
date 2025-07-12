import hashlib

def compute_checksum(path):
    """Return SHA-256 checksum for a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
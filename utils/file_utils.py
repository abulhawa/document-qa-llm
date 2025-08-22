import os, sys, subprocess
import hashlib
from typing import Any
from datetime import datetime, timezone
from config import logger

__all__ = [
    "compute_checksum",
    "normalize_path",
    "hash_path",
    "get_file_size",
    "get_file_timestamps",
    "format_file_size",
]


def compute_checksum(path: str) -> str:
    """Compute SHA256 checksum of a file."""
    path = normalize_path(path)
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
    path = normalize_path(path)
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def format_file_size(num_bytes: Any) -> str:
    """Return a human-friendly string for a byte size.

    Uses binary multiples (KB, MB, GB, ...) and keeps one decimal place for
    non-integer values. Falls back to ``0 B`` for invalid inputs.
    """
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if units[idx] == "B":
        return f"{int(size)} {units[idx]}"
    else:
        return f"{size:.1f} {units[idx]}"


def get_file_timestamps(path: str) -> dict:
    """
    Returns creation and modification timestamps in ISO format.
    """
    path = normalize_path(path)
    try:
        st = os.stat(path)
    except Exception as e:
        # Still return keys with fallback values
        return {"created": "", "modified": ""}
    # Created: macOS has st_birthtime. Windows uses st_ctime as creation time.
    created_ts = getattr(st, "st_birthtime", None)
    if created_ts is None and os.name == "nt":
        created_ts = st.st_ctime  # Windows creation time
    # Linux usually has no birth time. Keep None.

    # Modified: always available. Prefer nanosecond precision if present.
    if hasattr(st, "st_mtime_ns"):
        modified_dt = datetime.fromtimestamp(st.st_mtime_ns / 1e9, tz=timezone.utc)
    else:
        modified_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)

    created_dt = (
        datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts is not None else None
    )

    return {"created": created_dt, "modified": modified_dt}


def open_file_local(path: str) -> None:
    """Open a file on the machine running Streamlit."""
    if not path:
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        (f"Could not open file: {e}")


def show_in_folder(path: str) -> None:
    """Reveal file in its folder (selects the file on Windows/macOS)."""
    if not path:
        return
    try:
        if sys.platform.startswith("win"):
            # /select, must be a single token; pass via shell to support commas
            win_path = path.replace("/", "\\")
            subprocess.run(["explorer", "/select,", win_path], shell=True, check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        else:
            subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
    except Exception as e:
        logger.warning(f"Could not open folder: {e}")

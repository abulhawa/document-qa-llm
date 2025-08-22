from datetime import datetime
from typing import Any


def _parse_any(ts: Any) -> datetime | None:
    """Best-effort conversion of ``ts`` to ``datetime``.

    Accepts ``datetime`` objects, ISO 8601 strings (with optional ``Z``), and
    Unix epoch values (int/float or numeric strings). Returns ``None`` if the
    value cannot be interpreted as a timestamp.
    """

    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        if s.isdigit():
            try:
                return datetime.fromtimestamp(float(s))
            except Exception:
                return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def format_timestamp(ts: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert a timestamp to a formatted string or ``'N/A'`` if invalid."""

    dt = _parse_any(ts)
    return dt.strftime(fmt) if dt else "N/A"


def format_date(ts: Any, fmt: str = "%d %B %Y") -> str:
    """Return a human-friendly date from common timestamp forms."""

    if not ts:
        return ""

    dt = _parse_any(ts)
    if dt:
        return dt.strftime(fmt)
    return str(ts)[:10]  # last resort

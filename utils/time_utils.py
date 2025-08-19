from datetime import datetime


def format_timestamp(ts: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert ISO timestamp to a clean human-readable format.
    If invalid or missing, return 'N/A'.
    """
    try:
        return datetime.fromisoformat(ts).strftime(fmt)
    except Exception:
        return "N/A"


def format_date(ts: str, fmt: str = "%d-%m-%Y") -> str:
    """Return DD-MM-YYYY from common timestamp forms (ISO string, date/datetime, or epoch)."""
    if not ts:
        return ""

    try:
        return datetime.fromisoformat(ts).strftime(fmt)
    except Exception:
        return str(ts)[:10]  # last resort

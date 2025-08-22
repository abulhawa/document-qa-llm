from datetime import datetime


def format_timestamp(ts: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert ISO timestamp to a human-readable format.

    If invalid or missing, return 'N/A'.
    """
    try:
        return datetime.fromisoformat(ts).strftime(fmt)
    except Exception:
        return "N/A"


def format_timestamp_ampm(ts: str) -> str:
    """Return timestamp as 'YYYY-MM-DD H:MM AM/PM'.

    The hour omits any leading zero and seconds are discarded. Returns 'N/A'
    if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts)
        time_part = dt.strftime("%I:%M %p").lstrip("0")
        return f"{dt.strftime('%Y-%m-%d')} {time_part}"
    except Exception:
        return "N/A"


def format_date(ts: str, fmt: str = "%d %B %Y") -> str:
    """Return DD-MM-YYYY from common timestamp forms (ISO string, date/datetime, or epoch)."""
    if not ts:
        return ""

    try:
        return datetime.fromisoformat(ts).strftime(fmt)
    except Exception:
        return str(ts)[:10]  # last resort


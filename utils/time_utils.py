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
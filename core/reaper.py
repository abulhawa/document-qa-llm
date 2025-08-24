from time import time
from core.job_queue import (
    r,
    k,
    add_retry,
    rem_active,
    _active_started_fallback,
    _lock,
)
from core.job_control import get_state


def reap_stale_active(job_id: str, ttl_seconds: int = 900) -> int:
    """Move actives older than ttl to needs_retry (if job not running)."""
    if get_state(job_id) == "running":
        return 0
    now = int(time())
    if r:
        started = r.hgetall(k(job_id, "active_started")) or {}
    else:
        with _lock:
            started = (_active_started_fallback.get(job_id) or {}).copy()
    moved = 0
    for path, ts in started.items():
        if now - int(ts) >= ttl_seconds:
            add_retry(job_id, path)
            rem_active(job_id, path)
            moved += 1
    return moved

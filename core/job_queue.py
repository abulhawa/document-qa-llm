import os
from typing import List
import redis
import threading
from time import time as _now

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/2")
_redis: redis.Redis | None | bool = None
_lock = threading.Lock()
_pending_fallback: dict[str, List[str]] = {}
_active_fallback: dict[str, set] = {}
_active_started_fallback: dict[str, dict[str, int]] = {}
_retry_fallback: dict[str, List[str]] = {}


def _client() -> redis.Redis | None:
    global _redis
    if _redis is None:
        try:
            conn = redis.from_url(
                _redis_url, decode_responses=True, socket_connect_timeout=0.1
            )
            conn.ping()
            _redis = conn
        except Exception:
            _redis = False
    return _redis if _redis is not False else None


def k(job_id: str, suffix: str) -> str:
    return f"job:{job_id}:{suffix}"


r = _client()


def _pending_key(job_id: str) -> str:
    return f"job:{job_id}:pending"


def _active_key(job_id: str) -> str:
    return k(job_id, "active")


def _active_started_key(job_id: str) -> str:
    return k(job_id, "active_started")


def _retry_key(job_id: str) -> str:
    return k(job_id, "needs_retry")


def push_pending(job_id: str, path: str) -> None:
    r = _client()
    if r:
        r.rpush(_pending_key(job_id), path)
    else:
        with _lock:
            _pending_fallback.setdefault(job_id, []).append(path)


def pop_pending(job_id: str) -> str | None:
    r = _client()
    if r:
        return r.lpop(_pending_key(job_id))
    with _lock:
        lst = _pending_fallback.get(job_id, [])
        return lst.pop(0) if lst else None


def pending_count(job_id: str) -> int:
    r = _client()
    if r:
        return r.llen(_pending_key(job_id))
    with _lock:
        return len(_pending_fallback.get(job_id, []))


def add_active(job_id: str, path: str) -> None:
    r = _client()
    ts = int(_now())
    if r:
        r.sadd(_active_key(job_id), path)
        r.hset(_active_started_key(job_id), path, ts)
    else:
        with _lock:
            _active_fallback.setdefault(job_id, set()).add(path)
            _active_started_fallback.setdefault(job_id, {})[path] = ts


def rem_active(job_id: str, path: str) -> None:
    r = _client()
    if r:
        r.srem(_active_key(job_id), path)
        r.hdel(_active_started_key(job_id), path)
    else:
        with _lock:
            _active_fallback.setdefault(job_id, set()).discard(path)
            started = _active_started_fallback.get(job_id, {})
            started.pop(path, None)


def pop_all_active(job_id: str) -> List[str]:
    r = _client()
    key = _active_key(job_id)
    if r:
        paths = list(r.smembers(key))
        if paths:
            r.delete(key)
            r.delete(_active_started_key(job_id))
        return paths
    with _lock:
        paths = list(_active_fallback.get(job_id, set()))
        _active_fallback.pop(job_id, None)
        _active_started_fallback.pop(job_id, None)
        return paths


def active_count(job_id: str) -> int:
    r = _client()
    if r:
        return r.scard(_active_key(job_id))
    with _lock:
        return len(_active_fallback.get(job_id, set()))


def add_retry(job_id: str, path: str) -> None:
    r = _client()
    if r:
        r.rpush(_retry_key(job_id), path)
    else:
        with _lock:
            _retry_fallback.setdefault(job_id, []).append(path)


def pop_all_retry(job_id: str) -> List[str]:
    r = _client()
    key = _retry_key(job_id)
    if r:
        items = r.lrange(key, 0, -1)
        if items:
            r.delete(key)
        return items
    with _lock:
        items = list(_retry_fallback.get(job_id, []))
        _retry_fallback.pop(job_id, None)
        return items


def retry_count(job_id: str) -> int:
    r = _client()
    if r:
        return r.llen(_retry_key(job_id))
    with _lock:
        return len(_retry_fallback.get(job_id, []))


def inflight(job_id: str) -> int:
    return active_count(job_id)

def pending_len(job_id: str) -> int:
    client = _client()
    return int(client.llen(k(job_id, "pending")) or 0) if client else 0

def retry_len(job_id: str) -> int:
    client = _client()
    return int(client.llen(k(job_id, "needs_retry")) or 0) if client else 0


def celery_queue_len(queue: str | None = None) -> int:
    """Return length of the Celery task queue for quick diagnostics."""
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    queue = queue or os.getenv("CELERY_TASK_QUEUE", "celery")
    try:
        client = redis.from_url(
            broker_url, decode_responses=True, socket_connect_timeout=0.1
        )
        return int(client.llen(queue) or 0)
    except Exception:
        return 0

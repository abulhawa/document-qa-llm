import os
from typing import Dict, List, Set
import redis
import threading

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/2")
_redis: redis.Redis | None | bool = None
_lock = threading.Lock()
_state_fallback: Dict[str, str] = {}
_stats_fallback: Dict[str, Dict[str, int]] = {}
_tasks_fallback: Dict[str, set] = {}


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


def _state_key(job_id: str) -> str:
    return f"job:{job_id}:state"


def _stats_key(job_id: str) -> str:
    return f"job:{job_id}:stats"


def _tasks_key(job_id: str) -> str:
    return f"job:{job_id}:tasks"


def set_state(job_id: str, state: str) -> None:
    r = _client()
    if r:
        r.set(_state_key(job_id), state)
    else:
        with _lock:
            _state_fallback[job_id] = state


def get_state(job_id: str) -> str | None:
    r = _client()
    if r:
        return r.get(_state_key(job_id))
    with _lock:
        return _state_fallback.get(job_id)


def incr_stat(job_id: str, field: str, by: int = 1) -> int:
    r = _client()
    if r:
        return r.hincrby(_stats_key(job_id), field, by)
    with _lock:
        stats = _stats_fallback.setdefault(job_id, {})
        stats[field] = stats.get(field, 0) + by
        return stats[field]


def get_stats(job_id: str) -> Dict[str, int]:
    r = _client()
    if r:
        raw = r.hgetall(_stats_key(job_id))
        return {k: int(v) for k, v in raw.items()}
    with _lock:
        return dict(_stats_fallback.get(job_id, {}))


def add_task(job_id: str, task_id: str) -> None:
    r = _client()
    if r:
        r.sadd(_tasks_key(job_id), task_id)
    else:
        with _lock:
            _tasks_fallback.setdefault(job_id, set()).add(task_id)


def pop_all_tasks(job_id: str) -> List[str]:
    r = _client()
    key = _tasks_key(job_id)
    if r:
        tasks = list(r.smembers(key))
        if tasks:
            r.delete(key)
        return tasks
    with _lock:
        tasks = list(_tasks_fallback.get(job_id, set()))
        _tasks_fallback.pop(job_id, None)
        return tasks


def all_jobs() -> List[str]:
    """Return a sorted list of all known job IDs.

    Job identifiers may be present in state, stats or task tracking. This
    helper aggregates IDs from all sources, supporting both Redis-backed and
    in-memory fallback storage.
    """
    r = _client()
    if r:
        job_ids: Set[str] = set()
        for pattern in ("job:*:state", "job:*:stats", "job:*:tasks"):
            for key in r.scan_iter(pattern):
                parts = key.split(":")
                if len(parts) >= 3:
                    job_ids.add(parts[1])
        return sorted(job_ids)
    with _lock:
        job_ids = set(_state_fallback.keys()) | set(_stats_fallback.keys()) | set(
            _tasks_fallback.keys()
        )
        return sorted(job_ids)

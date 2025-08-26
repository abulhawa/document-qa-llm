from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Any
import time
from ui.celery_client import get_ui_celery

@dataclass
class TaskRecord:
    path: str
    task_id: str
    enqueued_at: float

def add_records(existing: List[Dict[str, Any]] | None, paths: List[str], task_ids: List[str]) -> List[Dict[str, Any]]:
    """Append new (path, task_id) records; returns the updated list (as plain dicts for session storage)."""
    now = time.time()
    new = [TaskRecord(p, t, now) for p, t in zip(paths, task_ids)]
    base = list(existing or [])
    base.extend(asdict(n) for n in new)
    return base

def fetch_states(task_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Return {task_id: {'state': 'PENDING'|'STARTED'|..., 'result': <obj or None>}}."""
    app = get_ui_celery()
    out: Dict[str, Dict[str, Any]] = {}
    for tid in task_ids:
        res = app.AsyncResult(tid)
        out[tid] = {"state": res.state, "result": (res.result if res.ready() else None)}
    return out

def clear_finished(records: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    finished = {"SUCCESS", "FAILURE", "REVOKED"}
    keep = []
    for r in records:
        st = states.get(r["task_id"], {}).get("state", "UNKNOWN")
        if st not in finished:
            keep.append(r)
    return keep
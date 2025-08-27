from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Any
import time
from ui.celery_client import get_ui_celery


@dataclass
class TaskRecord:
    path: str
    task_id: str
    enqueued_at: float
    action: str = "ingest"  # "ingest" | "reembed" | "reingest" | "delete"


def add_records(
    existing: List[Dict[str, Any]] | None,
    paths: List[str],
    task_ids: List[str],
    *,
    action: str = "ingest",
) -> List[Dict[str, Any]]:
    """
    Append new (path, task_id) records for the task panel.
    - `action`: "ingest" | "reembed" | "reingest" | "delete"
    Dedupe by task_id to avoid duplicates on reruns.
    """
    if len(paths) != len(task_ids):
        # Be strict; mismatched inputs usually indicate a bug upstream.
        n = min(len(paths), len(task_ids))
        paths, task_ids = paths[:n], task_ids[:n]

    now = time.time()
    base = list(existing or [])
    seen_ids = {r.get("task_id") for r in base}

    for p, t in zip(paths, task_ids):
        if t in seen_ids:
            continue
        rec = TaskRecord(path=p, task_id=t, enqueued_at=now, action=action)
        base.append(asdict(rec))

    return base


def fetch_states(task_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Return {task_id: {'state': 'PENDING'|'STARTED'|..., 'result': <obj or None>}}."""
    app = get_ui_celery()
    out: Dict[str, Dict[str, Any]] = {}
    for tid in task_ids:
        res = app.AsyncResult(tid)
        out[tid] = {"state": res.state, "result": (res.result if res.ready() else None)}
    return out


def clear_finished(
    records: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    finished = {"SUCCESS", "FAILURE", "REVOKED"}
    keep = []
    for r in records:
        st = states.get(r["task_id"], {}).get("state", "UNKNOWN")
        if st not in finished:
            keep.append(r)
    return keep

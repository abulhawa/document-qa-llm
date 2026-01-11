"""Use case for collecting running task metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from ui.celery_admin import (
    fetch_overview,
    failed_count_lookback,
    list_failed_tasks,
    redis_queue_depth,
)


def _short(val: Any, limit: int = 120) -> str:
    if val is None:
        s = ""
    elif isinstance(val, (dict, list, tuple)):
        try:
            s = json.dumps(val, ensure_ascii=False)
        except Exception:
            s = str(val)
    else:
        s = str(val)
    return s[:limit]


def _fmt_time(val: Any) -> str:
    """Render inspector's eta/time_start in local time, handling floats and strings."""
    if val in (None, "", 0):
        return ""
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 10**12:  # ms epoch
            ts = ts / 1000.0
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(val)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return val
    return str(val)


def _normalize_tasks(
    kind: str, by_worker: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for worker, items in (by_worker or {}).items():
        for task in items or []:
            req = task.get("request") or {}
            args_raw = task.get("args") or req.get("argsrepr") or req.get("args")
            kwargs_raw = task.get("kwargs") or req.get("kwargsrepr") or req.get("kwargs")
            eta_raw = task.get("eta") or req.get("eta") or task.get("time_start")
            rows.append(
                {
                    "Type": kind,
                    "Worker": worker,
                    "Task": task.get("name") or task.get("type"),
                    "ID": task.get("id") or req.get("id"),
                    "Args": _short(args_raw),
                    "Kwargs": _short(kwargs_raw),
                    "ETA": _fmt_time(eta_raw),
                }
            )
    return rows[:200]


def fetch_running_tasks_snapshot(
    *,
    failed_window_hours: int,
    failed_page: int,
    failed_page_size: int,
    timeout: float = 0.5,
    cache_ttl: float = 2.0,
) -> Dict[str, Any]:
    """Collect and normalize background task info for the UI."""
    overview = fetch_overview(timeout=timeout, cache_ttl=cache_ttl)
    failed_count = failed_count_lookback(failed_window_hours)
    failed_rows, failed_total = list_failed_tasks(
        failed_window_hours, failed_page, failed_page_size
    )
    return {
        "overview": overview,
        "failed_count": failed_count,
        "failed_rows": failed_rows,
        "failed_total": failed_total,
        "queue_depth": redis_queue_depth("ingest"),
        "tables": {
            "active": _normalize_tasks("Active", overview.get("active", {})),
            "reserved": _normalize_tasks("Reserved", overview.get("reserved", {})),
            "scheduled": _normalize_tasks("Scheduled", overview.get("scheduled", {})),
        },
    }

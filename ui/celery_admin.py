from typing import Dict, Any, List, Tuple, cast
import time, os, json, redis
from opensearchpy import OpenSearch
from ui.celery_client import get_ui_celery

# --- fast inspector with tiny cache ---
_LAST = {"t": 0.0, "payload": None}


def fetch_overview(timeout: float = 0.5, cache_ttl: float = 2.0) -> Dict[str, Any]:
    now = time.time()
    if _LAST["payload"] and (now - _LAST["t"] < cache_ttl):
        return _LAST["payload"]

    insp = get_ui_celery().control.inspect(timeout=timeout)
    active = insp.active() or {}
    reserved = insp.reserved() or {}
    scheduled = insp.scheduled() or {}

    payload = {
        "active": active,
        "reserved": reserved,
        "scheduled": scheduled,
        "counts": {
            "active": sum(len(v or []) for v in active.values()),
            "reserved": sum(len(v or []) for v in reserved.values()),
            "scheduled": sum(len(v or []) for v in scheduled.values()),
        },
    }
    _LAST.update(t=now, payload=payload)
    return payload


def redis_queue_depth(queue: str = "ingest") -> int:
    url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    r = redis.from_url(url)
    try:
        return cast(int, r.llen(queue))
    except Exception:
        return -1


def revoke_task(
    task_id: str, *, terminate: bool = False, signal: str = "SIGTERM"
) -> None:
    get_ui_celery().control.revoke(task_id, terminate=terminate, signal=signal)


# --- failed (needs audit index) ---
_OS_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
_AUDIT = os.getenv("CELERY_AUDIT_INDEX", "celery_task_runs")


def failed_count_lookback(hours: int = 24) -> int | None:
    try:
        os_client = OpenSearch(hosts=[_OS_URL])
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"event.keyword": "failure"}},
                        {"range": {"@timestamp": {"gte": f"now-{hours}h"}}},
                    ]
                }
            },
            "track_total_hits": True,
            "size": 0,
        }
        resp = os_client.search(index=_AUDIT, body=body)
        tot = resp["hits"]["total"]
        return tot["value"] if isinstance(tot, dict) else int(tot)
    except Exception:
        return None


def list_failed_tasks(hours: int, page: int, size: int) -> Tuple[list[dict], int]:
    """
    Returns (rows, total) of failed tasks in the last `hours`.
    Each row has: Time, Task, Task ID, State, Error, Args, Kwargs.
    """
    try:
        os_client = OpenSearch(hosts=[_OS_URL])
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"event.keyword": "failure"}},
                        {"range": {"@timestamp": {"gte": f"now-{hours}h"}}},
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "from": page * size,
            "size": size,
            "track_total_hits": True,
        }
        resp = os_client.search(index=_AUDIT, body=body)
        total = (
            resp["hits"]["total"]["value"]
            if isinstance(resp["hits"]["total"], dict)
            else int(resp["hits"]["total"])
        )

        rows = []
        for hit in resp["hits"]["hits"]:
            s = hit.get("_source", {})
            err = s.get("error")
            if not err:
                # some tasks store error under result.error
                r = s.get("result")
                if isinstance(r, dict) and "error" in r:
                    err = r["error"]
            args = s.get("args")
            kwargs = s.get("kwargs")
            rows.append(
                {
                    "Time": s.get("@timestamp", ""),
                    "Task": s.get("task", ""),
                    "Task ID": s.get("task_id", ""),
                    "State": s.get("state", ""),
                    "Error": (
                        json.dumps(err, ensure_ascii=False)
                        if isinstance(err, (dict, list))
                        else (err or "")
                    )[:180],
                    "Args": (
                        json.dumps(args, ensure_ascii=False)
                        if isinstance(args, (dict, list))
                        else (str(args) if args is not None else "")
                    )[:120],
                    "Kwargs": (
                        json.dumps(kwargs, ensure_ascii=False)
                        if isinstance(kwargs, (dict, list))
                        else (str(kwargs) if kwargs is not None else "")
                    )[:120],
                }
            )
        return rows, total
    except Exception:
        return [], 0

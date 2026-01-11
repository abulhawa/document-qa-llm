"""Use case for the worker emergency controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, cast

from celery import Celery
import redis


@dataclass(frozen=True)
class WorkerEmergencyStatus:
    queue_lengths: Dict[str, int]
    active: int
    reserved: int
    scheduled: int
    broker_ok: bool
    result_ok: bool
    broker_error: Optional[str] = None
    result_error: Optional[str] = None
    celery_error: Optional[str] = None


def run_compose(args: List[str], *, compose_dir: Path, compose_project: str) -> subprocess.CompletedProcess[str]:
    """Run `docker compose` with a controlled working directory and project."""
    cmd = ["docker", "compose", "-p", compose_project] + args
    return subprocess.run(
        cmd,
        cwd=str(compose_dir),
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )


def queue_len(client: redis.Redis, qname: str) -> int:
    try:
        return int(cast(int, client.llen(qname)))
    except Exception:
        return -1


def inspect_counts(app: Celery) -> Dict[str, int]:
    i = app.control.inspect(timeout=0.8)
    active = sum(len(v) for v in (i.active() or {}).values())
    reserved = sum(len(v) for v in (i.reserved() or {}).values())
    scheduled = sum(len(v) for v in (i.scheduled() or {}).values())
    return {"active": active, "reserved": reserved, "scheduled": scheduled}


def _check_redis(client: redis.Redis) -> tuple[bool, Optional[str]]:
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def load_status(
    app: Celery,
    broker_client: redis.Redis,
    result_client: redis.Redis,
    queue_names: List[str],
) -> WorkerEmergencyStatus:
    broker_ok, broker_error = _check_redis(broker_client)
    result_ok, result_error = _check_redis(result_client)

    queue_lengths = {q: queue_len(broker_client, q) for q in queue_names}

    try:
        counts = inspect_counts(app)
        celery_error = None
    except Exception as exc:  # noqa: BLE001
        counts = {"active": 0, "reserved": 0, "scheduled": 0}
        celery_error = str(exc)

    return WorkerEmergencyStatus(
        queue_lengths=queue_lengths,
        active=counts["active"],
        reserved=counts["reserved"],
        scheduled=counts["scheduled"],
        broker_ok=broker_ok,
        result_ok=result_ok,
        broker_error=broker_error,
        result_error=result_error,
        celery_error=celery_error,
    )


def revoke_all_active(app: Celery, signal: str = "SIGTERM") -> int:
    i = app.control.inspect(timeout=0.8)
    active_map = i.active() or {}
    count = 0
    for tasks in active_map.values():
        for t in tasks:
            tid = t.get("id")
            if tid:
                try:
                    app.control.revoke(tid, terminate=True, signal=signal)
                    count += 1
                except Exception:
                    pass
    return count


def rate_limit_zero(app: Celery, task_name: str) -> None:
    try:
        app.control.rate_limit(task_name, "0/m")
    except Exception:
        pass


def purge_queues(client: redis.Redis, qnames: List[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for q in qnames:
        try:
            deleted = int(cast(int, client.delete(q)))  # 1 if deleted, 0 if not present
            result[q] = deleted
        except Exception:
            result[q] = -1
    return result

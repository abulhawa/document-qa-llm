from typing import List, Literal
from ui.celery_client import get_ui_celery
from ui.infra_warmup import force_warmup
import os, time
from redis import Redis


IngestMode = Literal["ingest", "reingest", "reembed"]

MAX_OUTSTANDING = int(os.getenv("MAX_OUTSTANDING", "200"))
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("INGEST_QUEUE", "ingest")


def _outstanding(app) -> int:
    # queued in Redis + active + reserved across workers
    r = Redis.from_url(BROKER_URL)
    queued = r.llen(QUEUE_NAME)
    insp = app.control.inspect(timeout=0.5)
    active = sum(len(v) for v in (insp.active() or {}).values())
    reserved = sum(len(v) for v in (insp.reserved() or {}).values())
    return queued + active + reserved # type: ignore


def enqueue_paths(paths: List[str], *, mode: IngestMode = "ingest") -> List[str]:
    """
    Enqueue ingestion for host paths selected in the UI.
    mode:
      - "ingest": skip if already indexed (no force, no replace)
      - "reembed": recompute embeddings & update vectors (no replace of chunks)
      - "reingest": full re-index & re-embed (replace)
    """
    force_warmup()
    app = get_ui_celery()
    sig = app.signature("tasks.ingest_document")
    out = []
    for p in paths:
        # keep backlog bounded
        while _outstanding(app) >= MAX_OUTSTANDING:
            time.sleep(0.5)
        out.append(
            sig.clone(kwargs={"host_path": p, "mode": mode})
            .apply_async(queue=QUEUE_NAME)
            .id
        )
    return out


def enqueue_delete_by_path(paths: List[str]) -> List[str]:
    """
    Enqueue deletion by the stored index path (exact value in OpenSearch/Qdrant payloads).
    Use this for rows selected in Index Viewer where 'Path' comes from the index.
    """
    force_warmup()
    app = get_ui_celery()
    sig = app.signature("tasks.delete_document")
    return [sig.clone(kwargs={"path": p}).apply_async().id for p in paths]

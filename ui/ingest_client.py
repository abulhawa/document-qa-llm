from typing import List, Literal
from ui.celery_client import get_ui_celery
from ui.infra_warmup import force_warmup


IngestMode = Literal["ingest", "reingest", "reembed"]


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
    return [
        sig.clone(kwargs={"host_path": p, "mode": mode}).apply_async().id for p in paths
    ]


def enqueue_delete_by_path(paths: List[str]) -> List[str]:
    """
    Enqueue deletion by the stored index path (exact value in OpenSearch/Qdrant payloads).
    Use this for rows selected in Index Viewer where 'Path' comes from the index.
    """
    force_warmup()
    app = get_ui_celery()
    sig = app.signature("tasks.delete_document")
    return [sig.clone(kwargs={"path": p}).apply_async().id for p in paths]

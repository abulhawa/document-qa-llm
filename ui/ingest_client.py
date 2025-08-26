from typing import List
from ui.celery_client import get_ui_celery

def enqueue_paths(paths: List[str], *, mode: str = "reingest") -> List[str]:
    """
    mode: "reingest" (full re-index + re-embed) or "reembed" (re-embed-only-ish).
    """
    app = get_ui_celery()
    sig = app.signature("tasks.ingest_document")
    return [sig.clone(kwargs={"host_path": p, "mode": mode}).apply_async().id for p in paths]
from typing import List
from ui.celery_client import get_ui_celery

def enqueue_paths(paths: List[str]) -> List[str]:
    app = get_ui_celery()
    return [
        app.signature("tasks.ingest_document", kwargs={"host_path": p}).apply_async().id
        for p in paths
    ]

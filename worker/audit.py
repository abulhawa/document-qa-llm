from datetime import datetime, timezone
from typing import Any, Dict
from opensearchpy import OpenSearch
import os

OS_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
INDEX  = os.getenv("CELERY_AUDIT_INDEX", "celery_task_runs")
_client = OpenSearch(hosts=[OS_URL])

def log_task(event: str, task_id: str, task_name: str, **fields: Any) -> None:
    doc = {"@timestamp": datetime.now().astimezone().isoformat(),
           "event": event, "task_id": task_id, "task": task_name}
    doc.update({k: v for k, v in fields.items() if v is not None})
    try:
        if not _client.indices.exists(index=INDEX):
            _client.indices.create(index=INDEX)
    except Exception:
        pass
    _client.index(index=INDEX, body=doc)
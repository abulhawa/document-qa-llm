from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.opensearch_client import get_client
from config import INGEST_PLAN_INDEX
from utils.opensearch_utils import ensure_ingest_plan_index_exists
from utils.file_utils import hash_path


def add_planned_ingestions(paths: List[str]) -> None:
    """Add file paths to the planned ingestion index with Pending status."""
    ensure_ingest_plan_index_exists()
    client = get_client()
    now = datetime.now().astimezone().isoformat()
    for p in paths:
        doc = {"path": p, "status": "Pending", "added_at": now}
        client.index(index=INGEST_PLAN_INDEX, id=hash_path(p), body=doc)


def update_plan_status(path: str, status: str) -> None:
    """Update status of a planned ingestion entry."""
    client = get_client()
    body = {
        "doc": {
            "status": status,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
    }
    client.update(index=INGEST_PLAN_INDEX, id=hash_path(path), body=body)


def get_planned_ingestions(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return planned ingestions, optionally filtered by status."""
    ensure_ingest_plan_index_exists()
    client = get_client()
    query: Dict[str, Any]
    if status:
        query = {"term": {"status": status}}
    else:
        query = {"match_all": {}}
    body = {"query": query, "size": 1000}
    res = client.search(index=INGEST_PLAN_INDEX, body=body)
    hits = res.get("hits", {}).get("hits", [])
    return [h.get("_source", {}) for h in hits]


def clear_planned_ingestions() -> None:
    """Remove all planned ingestion entries."""
    client = get_client()
    client.delete_by_query(index=INGEST_PLAN_INDEX, body={"query": {"match_all": {}}})

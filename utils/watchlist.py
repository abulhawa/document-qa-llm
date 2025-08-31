from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime, timezone

from core.opensearch_client import get_client
from config import WATCHLIST_INDEX
from utils.file_utils import normalize_path
from utils.opensearch.indexes import ensure_index_exists


WATCHLIST_INDEX_SETTINGS: Dict[str, Any] = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "prefix": {"type": "keyword"},
            "added_at": {"type": "date"},
            "active": {"type": "boolean"},
            "note": {"type": "text"},
            "last_refreshed": {"type": "date"},
            "last_total": {"type": "integer"},
            "last_indexed": {"type": "integer"},
            "last_unindexed": {"type": "integer"},
        }
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_watchlist_prefixes() -> List[str]:
    """Return the list of tracked prefixes from the watchlist index."""
    ensure_index_exists(WATCHLIST_INDEX)
    client = get_client()
    try:
        resp = client.search(
            index=WATCHLIST_INDEX,
            body={
                "size": 1000,
                "query": {"term": {"active": True}},
                "_source": ["prefix"],
                "sort": [{"prefix": "asc"}],
            },
        )
    except Exception:
        return []
    hits = resp.get("hits", {}).get("hits", [])
    return [h.get("_source", {}).get("prefix", "") for h in hits if h.get("_source", {}).get("prefix")]


def add_watchlist_prefix(prefix: str) -> bool:
    """Add or activate a prefix in the watchlist index."""
    p = (prefix or "").strip()
    if not p:
        return False
    np = normalize_path(p)
    ensure_index_exists(WATCHLIST_INDEX)
    client = get_client()
    doc = {"prefix": np, "active": True, "added_at": _now_iso()}
    client.update(
        index=WATCHLIST_INDEX,
        id=np,
        body={"doc": doc, "doc_as_upsert": True},
        refresh=False,  # type: ignore
    )
    return True


def remove_watchlist_prefix(prefix: str) -> bool:
    """Deactivate a prefix from the watchlist index."""
    p = (prefix or "").strip()
    if not p:
        return False


def get_watchlist_meta(prefix: str) -> Dict[str, Any]:
    """Return watchlist metadata for a prefix (may be empty)."""
    ensure_index_exists(WATCHLIST_INDEX)
    client = get_client()
    np = normalize_path(prefix)
    try:
        resp = client.get(index=WATCHLIST_INDEX, id=np)
        return resp.get("_source", {}) or {}
    except Exception:
        return {}


def update_watchlist_stats(prefix: str, total: int, indexed: int, unindexed: int) -> None:
    """Persist last_refreshed and last_* counters for a prefix."""
    ensure_index_exists(WATCHLIST_INDEX)
    client = get_client()
    np = normalize_path(prefix)
    doc = {
        "last_refreshed": _now_iso(),
        "last_total": int(total),
        "last_indexed": int(indexed),
        "last_unindexed": int(unindexed),
    }
    client.update(
        index=WATCHLIST_INDEX,
        id=np,
        body={"doc": doc, "doc_as_upsert": True},
        refresh=False,  # type: ignore
    )
    np = normalize_path(p)
    ensure_index_exists(WATCHLIST_INDEX)
    client = get_client()
    try:
        client.update(
            index=WATCHLIST_INDEX,
            id=np,
            body={"doc": {"active": False}},
            refresh=False,  # type: ignore
        )
        return True
    except Exception:
        return False

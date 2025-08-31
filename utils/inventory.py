from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from core.opensearch_client import get_client
from opensearchpy import helpers

from config import (
    WATCH_INVENTORY_INDEX,
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    logger,
)
from utils.file_utils import normalize_path
from utils.opensearch.indexes import ensure_index_exists
import os


INVENTORY_INDEX_SETTINGS: Dict[str, Any] = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "path": {"type": "keyword"},
            "size": {"type": "long"},
            "mtime_iso": {"type": "date", "format": "strict_date_time"},
            "checksum": {"type": "keyword"},
            "first_seen": {"type": "date"},
            "last_seen": {"type": "date"},
            "exists_now": {"type": "boolean"},
            "last_indexed": {"type": "date"},
            "number_of_chunks": {"type": "integer"},
            "indexed_chunked_count": {"type": "integer"},
        }
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_watch_inventory_for_paths(paths: List[str]) -> int:
    """Bulk upsert inventory docs for given paths.

    Sets exists_now=True and updates last_seen. If a doc does not exist, initializes first_seen.
    Use last_indexed presence as the source of truth for indexing status.
    """
    if not paths:
        return 0
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    actions: List[Dict[str, Any]] = []
    now = _now_iso()
    for p in paths:
        np = normalize_path(p)
        doc = {"path": np, "exists_now": True, "last_seen": now}
        actions.append(
            {
                "_op_type": "update",
                "_index": WATCH_INVENTORY_INDEX,
                "_id": np,
                "doc": doc,
                "doc_as_upsert": True,
                "upsert": {**doc, "first_seen": now},
            }
        )
    helpers.bulk(client, actions)
    return len(actions)


def seed_watch_inventory_from_fulltext(path_prefix: str, size: int = 10000) -> int:
    """Populate watch inventory from already-indexed full-text docs.

    Uses the full-text index (1 doc per file) to upsert inventory entries setting
    last_indexed from indexed_at. Also sets exists_now=True and first_seen/last_seen.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    n_pref = normalize_path(path_prefix)
    body: Dict[str, Any] = {
        "track_total_hits": True,
        "query": {"prefix": {"path": n_pref}},
        "_source": ["path", "checksum", "indexed_at", "modified_at", "size_bytes"],
        "size": size,
        "sort": [{"path": "asc"}],
    }
    resp = client.search(index=FULLTEXT_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return 0
    actions: List[Dict[str, Any]] = []
    for h in hits:
        s = h.get("_source", {})
        p = normalize_path(s.get("path", ""))
        if not p:
            continue
        indexed_at = s.get("indexed_at") or _now_iso()
        modified_at = s.get("modified_at")
        size_bytes = s.get("size_bytes")
        checksum = s.get("checksum")
        doc = {
            "path": p,
            "size": size_bytes,
            "mtime_iso": modified_at,
            "checksum": checksum,
            "exists_now": True,
            "first_seen": indexed_at,
            "last_seen": indexed_at,
            "last_indexed": indexed_at,
        }
        actions.append(
            {
                "_op_type": "update",
                "_index": WATCH_INVENTORY_INDEX,
                "_id": p,
                "doc": doc,
                "doc_as_upsert": True,
                "upsert": doc,
            }
        )
    if actions:
        helpers.bulk(client, actions)
    return len(actions)


def count_watch_inventory_remaining(path_prefix: Optional[str] = None) -> int:
    """Count files that exist now but are not yet indexed.

    If path_prefix is provided, restrict to entries whose path starts with it.
    """
    client = get_client()
    filters: List[Dict[str, Any]] = [{"term": {"exists_now": True}}]
    if path_prefix:
        filters.append({"prefix": {"path": normalize_path(path_prefix)}})
    body: Dict[str, Any] = {
        "size": 0,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": filters,
                "must_not": [{"exists": {"field": "last_indexed"}}],
            }
        },
    }
    resp = client.search(index=WATCH_INVENTORY_INDEX, body=body)
    return int(resp.get("hits", {}).get("total", {}).get("value", 0))


def count_watch_inventory_total(path_prefix: Optional[str] = None) -> int:
    """Count files that currently exist (exists_now=True), optionally under a prefix."""
    client = get_client()
    filters: List[Dict[str, Any]] = [{"term": {"exists_now": True}}]
    if path_prefix:
        filters.append({"prefix": {"path": normalize_path(path_prefix)}})
    body: Dict[str, Any] = {
        "size": 0,
        "track_total_hits": True,
        "query": {"bool": {"filter": filters}},
    }
    resp = client.search(index=WATCH_INVENTORY_INDEX, body=body)
    return int(resp.get("hits", {}).get("total", {}).get("value", 0))


def list_watch_inventory_unindexed_paths(path_prefix: str, size: int = 10) -> List[str]:
    """Return up to `size` file paths under prefix that are exists_now and missing last_indexed."""
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    filters: List[Dict[str, Any]] = [
        {"term": {"exists_now": True}},
        {"prefix": {"path": normalize_path(path_prefix)}},
    ]
    body: Dict[str, Any] = {
        "size": max(1, int(size)),
        "track_total_hits": False,
        "query": {
            "bool": {
                "filter": filters,
                "must_not": [{"exists": {"field": "last_indexed"}}],
            }
        },
        "_source": ["path"],
        "sort": [{"path": "asc"}],
    }
    resp = client.search(index=WATCH_INVENTORY_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    out: List[str] = []
    for h in hits:
        p = (h.get("_source", {}) or {}).get("path")
        if p:
            out.append(p)
    return out


def list_watch_inventory_unindexed_paths_all(
    path_prefix: str, *, limit: int = 2000, page_size: int = 500
) -> List[str]:
    """Return up to `limit` unindexed file paths under prefix using scroll.

    Uses exists_now=True and must_not last_indexed, filtered by path prefix.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    filters: List[Dict[str, Any]] = [
        {"term": {"exists_now": True}},
        {"prefix": {"path": normalize_path(path_prefix)}},
    ]
    body: Dict[str, Any] = {
        "size": max(1, int(page_size)),
        "track_total_hits": False,
        "query": {
            "bool": {
                "filter": filters,
                "must_not": [{"exists": {"field": "last_indexed"}}],
            }
        },
        "_source": ["path"],
        "sort": [{"path": "asc"}],
    }
    out: List[str] = []
    try:
        resp = client.search(index=WATCH_INVENTORY_INDEX, body=body, scroll="2m")
        scroll_id = resp.get("_scroll_id")
        hits = resp.get("hits", {}).get("hits", [])
        while hits and len(out) < limit:
            for h in hits:
                p = (h.get("_source", {}) or {}).get("path")
                if p:
                    out.append(p)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
            if not scroll_id:
                break
            resp = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp.get("_scroll_id")
            hits = resp.get("hits", {}).get("hits", [])
    except Exception:
        return out
    finally:
        try:
            if 'scroll_id' in locals() and scroll_id:
                client.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass
    return out


def list_inventory_paths_needing_reingest(path_prefix: str, limit: int = 2000, page_size: int = 500) -> List[str]:
    """Heuristically find files that likely changed since last indexing.

    Criteria (any):
    - mtime_iso > last_indexed (both present)
    - indexed_chunked_count != number_of_chunks (both present)
    Only considers exists_now=True under the given path prefix.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    n_pref = normalize_path(path_prefix)
    body: Dict[str, Any] = {
        "size": max(1, int(page_size)),
        "track_total_hits": False,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"exists_now": True}},
                    {"prefix": {"path": n_pref}},
                ],
                "must": [{"exists": {"field": "last_indexed"}}],
                "should": [
                    {
                        "script": {
                            "script": {
                                "source": "doc.containsKey('mtime_iso') && doc.containsKey('last_indexed') && doc['mtime_iso'].size()!=0 && doc['last_indexed'].size()!=0 && doc['mtime_iso'].value.toInstant().isAfter(doc['last_indexed'].value.toInstant())",
                                "lang": "painless",
                            }
                        }
                    },
                    {
                        "script": {
                            "script": {
                                "source": "doc.containsKey('indexed_chunked_count') && doc.containsKey('number_of_chunks') && doc['indexed_chunked_count'].size()!=0 && doc['number_of_chunks'].size()!=0 && doc['indexed_chunked_count'].value != doc['number_of_chunks'].value",
                                "lang": "painless",
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["path"],
        "sort": [{"path": "asc"}],
    }
    out: List[str] = []
    try:
        resp = client.search(index=WATCH_INVENTORY_INDEX, body=body, scroll="2m")
        scroll_id = resp.get("_scroll_id")
        hits = resp.get("hits", {}).get("hits", [])
        while hits and len(out) < limit:
            for h in hits:
                p = (h.get("_source", {}) or {}).get("path")
                if p:
                    out.append(p)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
            if not scroll_id:
                break
            resp = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp.get("_scroll_id")
            hits = resp.get("hits", {}).get("hits", [])
    except Exception:
        return out
    finally:
        try:
            if 'scroll_id' in locals() and scroll_id:
                client.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass
    return out


def scan_watch_inventory_for_prefix(
    path_prefix: str,
    *,
    include_checksum: bool = False,
    allowed_suffixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Scan the local filesystem under a prefix and reconcile inventory.

    - Upserts/updates entries for files found (exists_now=True, size, mtime_iso, checksum optional, last_seen=now).
    - Marks entries as exists_now=False if they were not seen in this scan (based on last_seen < now) under the same prefix.

    Returns a summary dict: {"found": int, "upserts": int, "marked_missing": int}.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    n_pref = normalize_path(path_prefix)
    now = _now_iso()

    found_paths: List[str] = []
    actions: List[Dict[str, Any]] = []

    # Walk filesystem
    try:
        # Normalize allowed suffix set (default: pdf/docx/txt)
        if allowed_suffixes is None:
            allowed_set = {".pdf", ".docx", ".txt"}
        else:
            allowed_set = set(
                [(s if s.startswith(".") else f".{s}").lower() for s in allowed_suffixes]
            )
        for root, _, files in os.walk(n_pref):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if allowed_set and ext not in allowed_set:
                    continue
                p = normalize_path(os.path.join(root, name))
                try:
                    size = os.path.getsize(p)
                except Exception:
                    size = None
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc).isoformat()
                except Exception:
                    mtime = None
                doc: Dict[str, Any] = {
                    "path": p,
                    "exists_now": True,
                    "last_seen": now,
                }
                if size is not None:
                    doc["size"] = int(size)
                if mtime is not None:
                    doc["mtime_iso"] = mtime
                # Skip checksum for performance unless requested
                if include_checksum:
                    try:
                        from utils.file_utils import compute_checksum
                        doc["checksum"] = compute_checksum(p)
                    except Exception:
                        pass

                actions.append(
                    {
                        "_op_type": "update",
                        "_index": WATCH_INVENTORY_INDEX,
                        "_id": p,
                        "doc": doc,
                        "doc_as_upsert": True,
                        "upsert": {**doc, "first_seen": now},
                    }
                )
                found_paths.append(p)
    except Exception:
        # If walk fails entirely, return zeros
        found_paths = []

    upserts = 0
    if actions:
        helpers.bulk(client, actions)
        upserts = len(actions)

    # Mark missing: any exists_now=True under prefix whose last_seen < now (same allowed suffixes)
    marked_missing = 0
    try:
        should_suffix = [
            {"wildcard": {"path.keyword": f"*{s}"}} for s in (allowed_set or [])
        ]
        resp = client.update_by_query(
            index=WATCH_INVENTORY_INDEX,
            body={
                "script": {"source": "ctx._source.exists_now = false", "lang": "painless"},
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"exists_now": True}},
                            {"prefix": {"path": n_pref}},
                        ],
                        "must": [{"range": {"last_seen": {"lt": now}}}],
                        **({"should": should_suffix, "minimum_should_match": 1} if should_suffix else {}),
                    }
                },
            },
            refresh=False,  # type: ignore
        )
        marked_missing = int(resp.get("updated", 0))
    except Exception:
        pass

    return {"found": len(found_paths), "upserts": upserts, "marked_missing": marked_missing}


def seed_inventory_indexed_chunked_count(path_prefix: str, size: int = 10000) -> int:
    """Populate indexed_chunked_count from the documents index by counting chunks per path.

    Filters by path prefix using wildcard query on path.keyword.
    """
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    n_pref = normalize_path(path_prefix)
    body: Dict[str, Any] = {
        "size": 0,
        "query": {"wildcard": {"path.keyword": f"{n_pref}*"}},
        "aggs": {"by_path": {"terms": {"field": "path.keyword", "size": size}}},
    }
    resp = client.search(index=CHUNKS_INDEX, body=body)
    buckets = resp.get("aggregations", {}).get("by_path", {}).get("buckets", [])
    if not buckets:
        return 0
    actions: List[Dict[str, Any]] = []
    now = _now_iso()
    for b in buckets:
        p = b.get("key")
        if not p:
            continue
        cnt = int(b.get("doc_count", 0))
        doc = {"path": p, "exists_now": True, "last_seen": now, "indexed_chunked_count": cnt}
        actions.append(
            {
                "_op_type": "update",
                "_index": WATCH_INVENTORY_INDEX,
                "_id": p,
                "doc": doc,
                "doc_as_upsert": True,
                "upsert": {**doc, "first_seen": now, "last_indexed": now},
            }
        )
    if actions:
        helpers.bulk(client, actions)
    return len(actions)


def set_inventory_number_of_chunks(path: str, number_of_chunks: int) -> None:
    """Update inventory.number_of_chunks for entries matching the exact path."""
    ensure_index_exists(WATCH_INVENTORY_INDEX)
    client = get_client()
    try:
        client.update_by_query(
            index=WATCH_INVENTORY_INDEX,
            body={
                "script": {
                    "source": "ctx._source.number_of_chunks = params.n; ctx._source.last_seen = params.t; ctx._source.exists_now = true",
                    "lang": "painless",
                    "params": {"n": int(number_of_chunks), "t": _now_iso()},
                },
                "query": {"term": {"path": normalize_path(path)}},
            },
            refresh=False, # type: ignore
        )
    except Exception:
        pass

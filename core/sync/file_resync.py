from __future__ import annotations

import hashlib
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from config import (
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    QDRANT_COLLECTION,
    QDRANT_URL,
    logger,
)
from core.opensearch_client import get_client
from qdrant_client import QdrantClient, models
from utils.file_utils import compute_checksum, normalize_path

CHUNK_READ_BYTES = 1024 * 1024  # 1MB slices for quick fingerprinting
DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _quick_fingerprint(path: str, size_bytes: int) -> Tuple[int, float, str, str]:
    """Return a lightweight fingerprint using first/last slices plus metadata."""

    first_hash = ""
    last_hash = ""
    try:
        with open(path, "rb") as f:
            first_part = f.read(CHUNK_READ_BYTES)
            first_hash = hashlib.sha256(first_part).hexdigest()
            if size_bytes > CHUNK_READ_BYTES:
                # Seek near the end for the last slice
                f.seek(max(size_bytes - CHUNK_READ_BYTES, 0))
                last_part = f.read(CHUNK_READ_BYTES)
                last_hash = hashlib.sha256(last_part).hexdigest()
    except Exception as e:
        logger.debug("Fingerprint failed for %s: %s", path, e)
    mtime = 0.0
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        pass
    return (size_bytes, mtime, first_hash, last_hash)


def _stat_metadata(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        modified_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone()
        size_bytes = st.st_size
    except Exception:
        modified_dt = None
        size_bytes = 0
    return {"modified_at": modified_dt, "size_bytes": size_bytes}


def fetch_index_state() -> Dict[str, Dict[str, Any]]:
    """Fetch the current indexed state from OpenSearch keyed by checksum."""

    client = get_client()
    try:
        resp = client.search(
            index=CHUNKS_INDEX,
            body={
                "size": 0,
                "aggs": {
                    "by_checksum": {
                        "terms": {"field": "checksum.keyword", "size": 10000},
                        "aggs": {
                            "top_hit": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": [
                                        "path",
                                        "filename",
                                        "filetype",
                                        "bytes",
                                        "size",
                                        "modified_at",
                                        "indexed_at",
                                    ],
                                }
                            }
                        },
                    }
                },
            },
        )
    except Exception as e:
        logger.error("Failed to fetch index state: %s", e)
        return {}

    buckets = (
        resp.get("aggregations", {}).get("by_checksum", {}).get("buckets", [])
    )
    index_state: Dict[str, Dict[str, Any]] = {}
    for bucket in buckets:
        checksum = bucket.get("key")
        hits = bucket.get("top_hit", {}).get("hits", {}).get("hits", [])
        if not checksum or not hits:
            continue
        source = hits[0].get("_source", {})
        path = source.get("path")
        filename = source.get("filename") or os.path.basename(path or "")
        index_state[checksum] = {
            "checksum": checksum,
            "path": path,
            "filename": filename,
            "filetype": source.get("filetype"),
            "size_bytes": source.get("bytes") or source.get("size"),
            "modified_at": source.get("modified_at"),
            "indexed_at": source.get("indexed_at"),
        }
    return index_state


def scan_disk(roots: List[str], allowed_ext: set[str]) -> Dict[str, Any]:
    """Scan local disk roots and return state keyed by checksum."""

    allowed_ext = {e.lower().strip() for e in (allowed_ext or set()) if e}
    files_by_checksum: Dict[str, Dict[str, Any]] = {}
    conflicts: Dict[str, List[Dict[str, Any]]] = {}
    fingerprint_cache: Dict[Tuple[int, float, str, str], str] = {}

    for root in roots:
        if not root:
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if allowed_ext and ext not in allowed_ext:
                    continue
                path = normalize_path(os.path.join(dirpath, name))
                metadata = _stat_metadata(path)
                size_bytes = metadata.get("size_bytes") or 0
                fp = _quick_fingerprint(path, size_bytes)
                checksum = fingerprint_cache.get(fp)
                if not checksum:
                    try:
                        checksum = compute_checksum(path)
                        fingerprint_cache[fp] = checksum
                    except Exception as e:
                        logger.warning("Skipping file due to checksum error %s: %s", path, e)
                        continue
                entry = {
                    "checksum": checksum,
                    "path": path,
                    "filename": os.path.basename(path),
                    "size_bytes": size_bytes,
                    "modified_at": metadata.get("modified_at"),
                    "filetype": ext.lstrip("."),
                }
                if checksum in files_by_checksum:
                    conflicts.setdefault(checksum, [files_by_checksum[checksum]]).append(
                        entry
                    )
                else:
                    files_by_checksum[checksum] = entry
    return {"files": files_by_checksum, "conflicts": conflicts}


def reconcile(index_state: Dict[str, Dict[str, Any]], disk_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    disk_files = disk_state.get("files", disk_state)
    conflicts = disk_state.get("conflicts", {})

    rows: List[Dict[str, Any]] = []
    checksums = set(index_state.keys()) | set(disk_files.keys())

    for checksum in sorted(checksums):
        idx_entry = index_state.get(checksum)
        disk_entry = disk_files.get(checksum)
        if checksum in conflicts:
            rows.append(
                {
                    "status": "conflict",
                    "checksum": checksum,
                    "old_path": idx_entry.get("path") if idx_entry else None,
                    "new_path": ", ".join([c.get("path", "") for c in conflicts[checksum]]),
                    "confidence": 1.0,
                    "actions": {"update_opensearch": False, "update_qdrant": False},
                }
            )
            continue

        if idx_entry and disk_entry:
            old_path = normalize_path(idx_entry.get("path")) if idx_entry.get("path") else None
            new_path = normalize_path(disk_entry.get("path")) if disk_entry.get("path") else None
            if old_path != new_path:
                status = "moved"
                actions = {"update_opensearch": True, "update_qdrant": True}
            else:
                status = "unchanged"
                actions = {"update_opensearch": False, "update_qdrant": False}
            rows.append(
                {
                    "status": status,
                    "checksum": checksum,
                    "old_path": idx_entry.get("path"),
                    "new_path": disk_entry.get("path"),
                    "confidence": 1.0,
                    "actions": actions,
                }
            )
        elif idx_entry and not disk_entry:
            rows.append(
                {
                    "status": "missing_on_disk",
                    "checksum": checksum,
                    "old_path": idx_entry.get("path"),
                    "new_path": None,
                    "confidence": 1.0,
                    "actions": {"update_opensearch": False, "update_qdrant": False},
                }
            )
        elif disk_entry and not idx_entry:
            rows.append(
                {
                    "status": "new_untracked",
                    "checksum": checksum,
                    "old_path": None,
                    "new_path": disk_entry.get("path"),
                    "confidence": 1.0,
                    "actions": {"update_opensearch": False, "update_qdrant": False},
                }
            )
    return rows


def _update_opensearch_paths(checksum: str, new_path: str) -> int:
    client = get_client()
    filename = os.path.basename(new_path)
    now = datetime.now(timezone.utc).isoformat()
    script = {
        "source": "ctx._source.path=params.path; ctx._source.filename=params.filename; ctx._source.path_updated_at=params.path_updated_at;",
        "lang": "painless",
        "params": {"path": new_path, "filename": filename, "path_updated_at": now},
    }
    total_updated = 0
    for index in (FULLTEXT_INDEX, CHUNKS_INDEX):
        resp = client.update_by_query(
            index=index,
            body={"script": script, "query": {"term": {"checksum": checksum}}},
            refresh=True,
            conflicts="proceed",
        )
        total_updated += resp.get("updated", 0)
    return total_updated


def _update_qdrant_payload(checksum: str, new_path: str) -> int:
    client = QdrantClient(url=QDRANT_URL)
    filename = os.path.basename(new_path)
    result = client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={"path": new_path, "filename": filename},
        wait=True,
        filter=models.Filter(
            must=[models.FieldCondition(key="checksum", match=models.MatchValue(value=checksum))]
        ),
    )
    if isinstance(result, dict):
        return int(result.get("result", {}).get("count", 0))
    if hasattr(result, "result") and isinstance(result.result, dict):
        return int(result.result.get("count", 0))
    return 0


def apply_updates(rows: List[Dict[str, Any]], dry_run: bool = True) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "counts_by_status": Counter(),
        "updated_opensearch": 0,
        "updated_qdrant": 0,
        "errors": [],
    }
    for row in rows:
        summary["counts_by_status"].update([row.get("status", "")])

    if dry_run:
        summary["counts_by_status"] = dict(summary["counts_by_status"])
        return summary

    for row in rows:
        if row.get("status") != "moved":
            continue
        if row.get("confidence", 1.0) < 1.0:
            continue
        actions = row.get("actions", {})
        if not actions.get("update_opensearch") and not actions.get("update_qdrant"):
            continue
        checksum = row.get("checksum")
        new_path = row.get("new_path")
        if not checksum or not new_path:
            continue
        try:
            if actions.get("update_opensearch"):
                summary["updated_opensearch"] += _update_opensearch_paths(checksum, new_path)
        except Exception as e:
            logger.error("OpenSearch update failed for %s: %s", checksum, e)
            summary["errors"].append(str(e))
        try:
            if actions.get("update_qdrant"):
                summary["updated_qdrant"] += _update_qdrant_payload(checksum, new_path)
        except Exception as e:
            logger.error("Qdrant update failed for %s: %s", checksum, e)
            summary["errors"].append(str(e))

    summary["counts_by_status"] = dict(summary["counts_by_status"])
    return summary

"""Use case for the index viewer UI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List
import queue

from ui.ingest_client import enqueue_delete_by_path, enqueue_paths
from utils.opensearch_utils import list_files_from_opensearch
from utils.qdrant_utils import count_qdrant_chunks_by_checksum


def fetch_indexed_files() -> List[Dict[str, Any]]:
    """Load indexed file metadata from OpenSearch."""
    return list_files_from_opensearch()


def prefetch_indexed_files(out_q: "queue.Queue") -> None:
    """Background fetch that doesn't touch Streamlit APIs."""
    try:
        files = fetch_indexed_files()
        out_q.put({"ok": True, "files": files})
    except Exception as exc:  # noqa: BLE001
        out_q.put({"ok": False, "error": str(exc)})


def compute_qdrant_counts(checksums: Iterable[str]) -> Dict[str, int]:
    """Resolve Qdrant chunk counts for the given checksums."""
    checksum_list = [cs for cs in checksums if cs]
    if not checksum_list:
        return {}

    results: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(count_qdrant_chunks_by_checksum, checksum): checksum
            for checksum in checksum_list
        }
        for future in as_completed(futures):
            checksum = futures[future]
            try:
                results[checksum] = future.result() or 0
            except Exception:  # noqa: BLE001
                results[checksum] = 0

    return results


def enqueue_reembed(paths: List[str]) -> List[str]:
    """Queue files for re-embedding."""
    return enqueue_paths(paths, mode="reembed")


def enqueue_reingest(paths: List[str]) -> List[str]:
    """Queue files for re-ingestion."""
    return enqueue_paths(paths, mode="reingest")


def enqueue_delete(paths: List[str]) -> List[str]:
    """Queue files for deletion."""
    return enqueue_delete_by_path(paths)

"""Use case for watchlist operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ui.ingest_client import enqueue_paths
from utils.inventory import (
    count_watch_inventory_remaining,
    count_watch_inventory_total,
    count_watch_inventory_unindexed_missing_size,
    count_watch_inventory_unindexed_quick_wins,
    list_inventory_paths_needing_reingest,
    list_watch_inventory_unindexed_paths,
    list_watch_inventory_unindexed_paths_all,
    list_watch_inventory_unindexed_paths_simple,
    list_watch_inventory_unindexed_quick_wins,
    scan_watch_inventory_for_prefix,
    seed_inventory_indexed_chunked_count,
    seed_watch_inventory_from_fulltext,
)
from utils.watchlist import (
    add_watchlist_prefix,
    get_watchlist_meta,
    get_watchlist_prefixes,
    remove_watchlist_prefix,
    update_watchlist_scan_stats,
    update_watchlist_stats,
)


@dataclass
class WatchlistStatus:
    prefix: str
    remaining: int = 0
    total: int = 0
    indexed: int = 0
    percent_indexed: float = 0.0
    quick_wins: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    preview: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class WatchlistRefreshAllResult:
    total_fulltext: int = 0
    total_chunks: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class WatchlistRefreshResult:
    imported: int = 0
    chunk_counts: int = 0
    total: int = 0
    remaining: int = 0
    indexed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class WatchlistScanResult:
    found: int = 0
    marked_missing: int = 0
    total: int = 0
    remaining: int = 0
    indexed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class WatchlistQuickWinResult:
    count: int = 0
    missing_size: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class WatchlistQueueResult:
    task_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def load_watchlist_prefixes() -> List[str]:
    return get_watchlist_prefixes()


def add_prefix(prefix: str) -> bool:
    return add_watchlist_prefix(prefix)


def remove_prefix(prefix: str) -> bool:
    return remove_watchlist_prefix(prefix)


def refresh_all_status(prefixes: List[str]) -> WatchlistRefreshAllResult:
    result = WatchlistRefreshAllResult()
    for pref in prefixes:
        try:
            result.total_fulltext += seed_watch_inventory_from_fulltext(pref)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{pref}: {exc}")
        try:
            result.total_chunks += seed_inventory_indexed_chunked_count(pref)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{pref}: {exc}")
    return result


def get_status(
    prefix: str, *, preview_size: int = 10, quick_win_size: int = 102_400
) -> WatchlistStatus:
    status = WatchlistStatus(prefix=prefix)
    try:
        status.remaining = count_watch_inventory_remaining(prefix)
    except Exception as exc:  # noqa: BLE001
        status.errors.append(str(exc))
    try:
        status.quick_wins = count_watch_inventory_unindexed_quick_wins(
            prefix, quick_win_size
        )
    except Exception as exc:  # noqa: BLE001
        status.errors.append(str(exc))
    try:
        status.total = count_watch_inventory_total(prefix)
        status.indexed = max(0, status.total - status.remaining)
        status.percent_indexed = (status.indexed / status.total) if status.total else 0.0
    except Exception as exc:  # noqa: BLE001
        status.errors.append(str(exc))
    try:
        status.meta = get_watchlist_meta(prefix)
    except Exception as exc:  # noqa: BLE001
        status.errors.append(str(exc))
    try:
        status.preview = list_watch_inventory_unindexed_paths(prefix, size=preview_size)
    except Exception as exc:  # noqa: BLE001
        status.errors.append(str(exc))
    return status


def refresh_status(prefix: str) -> WatchlistRefreshResult:
    result = WatchlistRefreshResult()
    try:
        result.imported = seed_watch_inventory_from_fulltext(prefix)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    try:
        result.chunk_counts = seed_inventory_indexed_chunked_count(prefix)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    try:
        result.total = count_watch_inventory_total(prefix)
        result.remaining = count_watch_inventory_remaining(prefix)
        result.indexed = max(0, result.total - result.remaining)
        update_watchlist_stats(prefix, result.total, result.indexed, result.remaining)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    return result


def scan_folder(prefix: str) -> WatchlistScanResult:
    result = WatchlistScanResult()
    summary: Dict[str, Any] = {}
    try:
        summary = scan_watch_inventory_for_prefix(prefix)
        result.found = int(summary.get("found", 0))
        result.marked_missing = int(summary.get("marked_missing", 0))
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    try:
        result.total = count_watch_inventory_total(prefix)
        result.remaining = count_watch_inventory_remaining(prefix)
        result.indexed = max(0, result.total - result.remaining)
        update_watchlist_stats(prefix, result.total, result.indexed, result.remaining)
        update_watchlist_scan_stats(
            prefix,
            found=result.found,
            marked_missing=result.marked_missing,
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    return result


def sync_from_indices(prefix: str) -> List[str]:
    errors: List[str] = []
    try:
        seed_watch_inventory_from_fulltext(prefix)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    try:
        seed_inventory_indexed_chunked_count(prefix)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return errors


def get_remaining(prefix: str) -> tuple[int, List[str]]:
    errors: List[str] = []
    remaining = 0
    try:
        remaining = count_watch_inventory_remaining(prefix)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return remaining, errors


def list_unindexed_paths(prefix: str, *, limit: int = 2000) -> List[str]:
    paths = list_watch_inventory_unindexed_paths_simple(prefix, limit=limit)
    if not paths:
        paths = list_watch_inventory_unindexed_paths_all(prefix, limit=limit)
    return paths


def list_quick_win_paths(
    prefix: str, *, limit: int = 2000, max_size_bytes: int = 102_400
) -> List[str]:
    return list_watch_inventory_unindexed_quick_wins(
        prefix, limit=limit, max_size_bytes=max_size_bytes
    )


def get_quick_win_counts(
    prefix: str, *, max_size_bytes: int = 102_400
) -> WatchlistQuickWinResult:
    result = WatchlistQuickWinResult()
    try:
        result.count = count_watch_inventory_unindexed_quick_wins(
            prefix, max_size_bytes
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    try:
        result.missing_size = count_watch_inventory_unindexed_missing_size(prefix)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    return result


def list_reingest_paths(prefix: str, *, limit: int = 2000) -> List[str]:
    return list_inventory_paths_needing_reingest(prefix, limit=limit)


def import_known_files(prefix: str) -> tuple[int, List[str]]:
    errors: List[str] = []
    count = 0
    try:
        count = seed_watch_inventory_from_fulltext(prefix)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return count, errors


def import_chunk_counts(prefix: str) -> tuple[int, List[str]]:
    errors: List[str] = []
    count = 0
    try:
        count = seed_inventory_indexed_chunked_count(prefix)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return count, errors


def queue_ingest(paths: List[str], *, mode: str = "ingest") -> WatchlistQueueResult:
    result = WatchlistQueueResult()
    if not paths:
        return result
    try:
        result.task_ids = enqueue_paths(paths, mode=mode)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
    return result

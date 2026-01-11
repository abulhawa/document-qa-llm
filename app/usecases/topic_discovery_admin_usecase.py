"""Use case for topic discovery admin controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.qdrant_file_vectors import (
    build_missing_file_vectors,
    ensure_file_vectors_collection,
    get_file_vectors_count,
    get_unique_checksums_in_chunks,
    sample_file_vectors,
)


def get_file_vector_status() -> tuple[int, int, float]:
    unique_checksums = get_unique_checksums_in_chunks()
    file_vectors_count = get_file_vectors_count()
    total_files = len(unique_checksums)
    coverage = (file_vectors_count / total_files * 100) if total_files else 0.0
    return total_files, file_vectors_count, coverage


def ensure_collection(*, recreate: bool = False) -> None:
    ensure_file_vectors_collection(recreate=recreate)


def build_file_vectors(
    *,
    k: int,
    batch: int,
    limit: int | None,
    force: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return build_missing_file_vectors(
        k=k,
        batch=batch,
        limit=limit,
        force=force,
        progress_callback=progress_callback,
    )


def sample_vectors(*, limit: int = 5) -> list[dict[str, Any]]:
    return sample_file_vectors(limit=limit)

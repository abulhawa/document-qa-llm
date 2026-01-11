"""Use case for topic discovery overview orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from config import logger
from services.topic_discovery_clusters import (
    clear_cluster_cache,
    cluster_cache_exists,
    ensure_macro_grouping,
    load_last_cluster_cache,
    run_topic_discovery_clustering,
)
from utils.timing import set_run_id, timed_block


def run_clustering(settings: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, bool, str]:
    """Run topic discovery clustering and return results."""
    run_id = uuid.uuid4().hex[:8]
    set_run_id(run_id)
    with timed_block(
        "action.topic_discovery.run_clustering",
        extra={
            "run_id": run_id,
            "min_cluster_size": settings["min_cluster_size"],
            "min_samples": settings["min_samples"],
            "use_umap": settings["use_umap"],
        },
        logger=logger,
    ):
        result, used_cache = run_topic_discovery_clustering(
            min_cluster_size=settings["min_cluster_size"],
            min_samples=settings["min_samples"],
            metric="cosine",
            use_umap=settings["use_umap"],
            umap_config=settings["umap_config"],
            macro_k_range=settings["macro_k_range"],
            allow_cache=True,
        )
    return result, used_cache, run_id


def load_cached(settings: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Load cached clustering and ensure macro grouping."""
    cached = load_last_cluster_cache()
    if cached is None:
        return None
    return ensure_macro_grouping(
        cached,
        macro_k_range=settings["macro_k_range"],
    )


def clear_cache() -> bool:
    """Clear cached clustering artifacts."""
    return clear_cluster_cache()


def has_cached_run() -> bool:
    """Check whether a cached clustering run exists."""
    return cluster_cache_exists()

"""Use case for admin hub page rendering."""

from __future__ import annotations

import runpy
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List

import streamlit as st
import redis

from app.usecases import topic_discovery_overview_usecase, topic_naming_usecase
from app.usecases import worker_emergency_usecase

PAGES_DIR = Path(__file__).resolve().parents[2] / "pages"

ADMIN_TABS: tuple[tuple[str, str], ...] = (
    ("Running Tasks", "30_running_tasks.py"),
    ("Worker Emergency", "worker_emergency.py"),
)


def render_admin_page(filename: str) -> None:
    """Render an admin subpage within the hub context."""
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)


def clear_cache() -> Dict[str, Any]:
    """Clear application caches used by admin tooling."""
    results: Dict[str, Any] = {
        "topic_discovery_cache": topic_discovery_overview_usecase.clear_cache()
    }
    naming_cache_dir = topic_naming_usecase.TOPIC_NAMING_CACHE_DIR
    if naming_cache_dir.exists():
        shutil.rmtree(naming_cache_dir)
        results["topic_naming_cache"] = True
    else:
        results["topic_naming_cache"] = False
    return results


def purge_queues(queue_names: List[str]) -> Dict[str, int]:
    """Purge broker queues by name."""
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(broker_url, decode_responses=True)
    return worker_emergency_usecase.purge_queues(client, queue_names)

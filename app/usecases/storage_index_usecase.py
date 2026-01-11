"""Use case for storage and index hub page rendering."""

from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parents[2] / "pages"

STORAGE_INDEX_TABS: tuple[tuple[str, str], ...] = (
    ("Search", "5_search.py"),
    ("File Index Viewer", "2_index_viewer.py"),
    ("Ingestion Logs", "4_ingest_logs.py"),
    ("File Path Re-Sync", "7_file_resync.py"),
    ("Duplicate Files", "3_duplicates_viewer.py"),
    ("Watchlist", "6_watchlist.py"),
)


def render_storage_index_page(filename: str) -> None:
    """Render a storage/index subpage within the hub context."""
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)

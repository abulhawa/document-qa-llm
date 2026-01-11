"""Use case for admin hub page rendering."""

from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

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

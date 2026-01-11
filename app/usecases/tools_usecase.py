"""Use case for tools hub page rendering."""

from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parents[2] / "pages"

TOOLS_TABS: tuple[tuple[str, str], ...] = (
    ("Smart File Sorter", "tools_file_sorter.py"),
)


def render_tools_page(filename: str) -> None:
    """Render a tools subpage within the hub context."""
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)

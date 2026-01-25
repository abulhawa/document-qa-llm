"""Use case for tools hub page rendering."""

from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parents[2] / "pages"

SMART_FILE_SORTER_PRESETS: dict[str, dict[str, object]] = {
    "Balanced (default)": {
        "move_threshold": 0.7,
        "weight_meta": 0.55,
        "weight_content": 0.30,
        "weight_keyword": 0.15,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.65,
        "llm_max_items": 200,
    },
    "Path-heavy (folder structure)": {
        "move_threshold": 0.75,
        "weight_meta": 0.7,
        "weight_content": 0.2,
        "weight_keyword": 0.1,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.65,
        "llm_max_items": 200,
    },
    "Content-heavy (document text)": {
        "move_threshold": 0.65,
        "weight_meta": 0.35,
        "weight_content": 0.5,
        "weight_keyword": 0.15,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.6,
        "llm_max_items": 200,
    },
    "LLM assist (low-confidence only)": {
        "move_threshold": 0.7,
        "weight_meta": 0.5,
        "weight_content": 0.3,
        "weight_keyword": 0.2,
        "use_llm_fallback": True,
        "llm_confidence_floor": 0.75,
        "llm_max_items": 300,
    },
}

DEFAULT_SMART_FILE_SORTER_PRESET = "Balanced (default)"

TOOLS_TABS: tuple[tuple[str, str], ...] = (
    ("Smart File Sorter", "tools_file_sorter.py"),
)


def get_smart_file_sorter_presets() -> dict[str, dict[str, object]]:
    return SMART_FILE_SORTER_PRESETS


def render_tools_page(filename: str) -> None:
    """Render a tools subpage within the hub context."""
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)

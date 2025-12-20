import runpy
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Tools", layout="wide")

PAGES_DIR = Path(__file__).resolve().parent


def render_page(filename: str) -> None:
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)


tabs = st.tabs(
    [
        "Smart File Sorter",
    ]
)

with tabs[0]:
    render_page("tools_file_sorter.py")

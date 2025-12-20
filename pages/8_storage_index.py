import runpy
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Storage & Index", layout="wide")

PAGES_DIR = Path(__file__).resolve().parent


def render_page(filename):
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)


tabs = st.tabs(
    [
        "Search",
        "File Index Viewer",
        "Ingestion Logs",
        "File Path Re-Sync",
        "Duplicate Files",
        "Watchlist",
    ]
)

with tabs[0]:
    render_page("5_search.py")

with tabs[1]:
    render_page("2_index_viewer.py")

with tabs[2]:
    render_page("4_ingest_logs.py")

with tabs[3]:
    render_page("7_file_resync.py")

with tabs[4]:
    render_page("3_duplicates_viewer.py")

with tabs[5]:
    render_page("6_watchlist.py")

import runpy
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Admin", layout="wide")

PAGES_DIR = Path(__file__).resolve().parent


def render_page(filename):
    st.session_state["_nav_context"] = "hub"
    try:
        runpy.run_path(str(PAGES_DIR / filename))
    finally:
        st.session_state.pop("_nav_context", None)


tabs = st.tabs(
    [
        "Running Tasks",
        "Worker Emergency",
    ]
)

with tabs[0]:
    render_page("30_running_tasks.py")

with tabs[1]:
    render_page("worker_emergency.py")

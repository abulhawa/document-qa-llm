import streamlit as st

from ui.topic_discovery import (
    render_admin_tab,
    render_naming_tab,
    render_overview_tab,
    render_review_tab,
)

st.set_page_config(page_title="Topic Discovery", layout="wide")

st.title("Topic Discovery")

tabs = st.tabs(["Overview", "Naming", "Naming Review", "Admin"])

with tabs[0]:
    render_overview_tab()

with tabs[1]:
    render_naming_tab()

with tabs[2]:
    render_review_tab()

with tabs[3]:
    render_admin_tab()

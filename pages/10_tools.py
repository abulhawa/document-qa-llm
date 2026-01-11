import streamlit as st

from app.usecases.tools_usecase import TOOLS_TABS, render_tools_page

st.set_page_config(page_title="Tools", layout="wide")


tabs = st.tabs([label for label, _ in TOOLS_TABS])

for tab, (_, filename) in zip(tabs, TOOLS_TABS):
    with tab:
        render_tools_page(filename)

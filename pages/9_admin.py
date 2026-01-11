import streamlit as st

from app.usecases.admin_usecase import ADMIN_TABS, render_admin_page

st.set_page_config(page_title="Admin", layout="wide")


tabs = st.tabs([label for label, _ in ADMIN_TABS])

for tab, (_, filename) in zip(tabs, ADMIN_TABS):
    with tab:
        render_admin_page(filename)

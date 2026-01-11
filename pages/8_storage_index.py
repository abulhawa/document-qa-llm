import streamlit as st

from app.usecases.storage_index_usecase import (
    STORAGE_INDEX_TABS,
    render_storage_index_page,
)

st.set_page_config(page_title="Storage & Index", layout="wide")


tabs = st.tabs([label for label, _ in STORAGE_INDEX_TABS])

for tab, (_, filename) in zip(tabs, STORAGE_INDEX_TABS):
    with tab:
        render_storage_index_page(filename)

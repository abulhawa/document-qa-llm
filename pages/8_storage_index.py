import streamlit as st

from app.usecases.storage_index_usecase import (
    STORAGE_INDEX_TABS,
    render_storage_index_page,
)

st.set_page_config(page_title="Storage & Index", layout="wide")
st.title("Storage & Index")

labels = [label for label, _ in STORAGE_INDEX_TABS]
selected_label = st.segmented_control(
    "Section",
    labels,
    default=labels[0],
    label_visibility="collapsed",
)

selected_filename = dict(STORAGE_INDEX_TABS)[selected_label or labels[0]]
render_storage_index_page(selected_filename)

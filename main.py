import streamlit as st

st.set_page_config(
    page_title="Document QA", layout="wide", initial_sidebar_state="expanded"
)

pages = [
    st.Page("pages/0_chat.py", title="Ask Your Documents"),
    st.Page("pages/1_ingest.py", title="Ingest Documents"),
    st.Page("pages/8_storage_index.py", title="Storage & Index"),
    st.Page("pages/9_admin.py", title="Admin"),
]

pg = st.navigation(pages, position="top")
pg.run()

import streamlit as st

st.set_page_config(
    page_title="Document QA", layout="wide", initial_sidebar_state="expanded"
)

pages = [
    st.Page("pages/0_chat.py", title="Ask Your Documents", icon="💬"),
    st.Page("pages/1_ingest.py", title="Ingest Documents", icon="📥"),
    st.Page(
        "pages/2_index_viewer.py",
        title="File Index Viewer",
        icon="📂",
    ),
    st.Page(
        "pages/3_duplicates_viewer.py",
        title="Duplicate Files",
        icon="🗂️",
    ),
]

pg = st.navigation(pages, position="top")
pg.run()

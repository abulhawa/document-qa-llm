import streamlit as st

from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp
from utils.fulltext_search import search_documents

st.set_page_config(page_title="Search", layout="wide")
st.title("🔎 Search")

if "page" not in st.session_state:
    st.session_state.page = 0
if "page_size" not in st.session_state:
    st.session_state.page_size = 10
if "sort" not in st.session_state:
    st.session_state.sort = "relevance"
if "results" not in st.session_state:
    st.session_state.results = None
if "q" not in st.session_state:
    st.session_state.q = ""
if "file_types" not in st.session_state:
    st.session_state.file_types = []
if "path_prefix" not in st.session_state:
    st.session_state.path_prefix = ""
if "modified_range" not in st.session_state:
    st.session_state.modified_range = (None, None)


def run_search() -> None:
    params = {
        "q": st.session_state.q,
        "from_": st.session_state.page * st.session_state.page_size,
        "size": st.session_state.page_size,
        "sort": st.session_state.sort,
    }
    if st.session_state.file_types:
        params["filetypes"] = st.session_state.file_types
    if st.session_state.path_prefix:
        params["path_prefix"] = st.session_state.path_prefix
    if st.session_state.modified_range:
        start, end = st.session_state.modified_range
        if start:
            params["modified_from"] = start.isoformat()
        if end:
            params["modified_to"] = end.isoformat()
    st.session_state.results = search_documents(**params)


search_col, sort_col = st.columns([4, 1], vertical_alignment="bottom")
with search_col:
    st.text_input(
        "Search",
        key="q",
        on_change=run_search,
        placeholder="Type to search",
    )
with sort_col:
    st.selectbox(
        "Sort",
        ["relevance", "modified"],
        key="sort",
        on_change=run_search,
    )

filters_col1, filters_col2 = st.columns(2, vertical_alignment="bottom")
with filters_col1:
    st.text_input("Path prefix", key="path_prefix", on_change=run_search)
with filters_col2:
    st.multiselect("File type", options=[], key="file_types", on_change=run_search)
st.date_input("Modified range", key="modified_range", on_change=run_search)

if st.session_state.results:
    for hit in st.session_state.results.get("hits", []):
        st.markdown(
            f"**{hit.get('filename', '')}** - {format_file_size(hit.get('size_bytes', 0))}"
        )
        st.caption(
            f"{hit.get('path', '')} • {format_timestamp(hit.get('modified_at'))}"
        )
        for frag in hit.get("highlights", [])[:3]:
            st.write(f"…{frag}…", unsafe_allow_html=True)
else:
    st.info("Enter a query to search your documents")

left, right = st.columns(2)
with left:
    st.number_input(
        "Page size", 1, 100, key="page_size", on_change=run_search
    )
with right:
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Prev") and st.session_state.page > 0:
            st.session_state.page -= 1
            run_search()
    with col_next:
        if st.button("Next"):
            st.session_state.page += 1
            run_search()

import streamlit as st
from datetime import date, timedelta
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp
from utils.fulltext_search import search_documents


@st.cache_data(ttl=180, show_spinner=False)
def cached_search_documents(**params):
    # cache keys must be hashable; you already pass filetypes as tuple in current_params()
    p = dict(params)
    if isinstance(p.get("filetypes"), tuple):
        p["filetypes"] = list(p["filetypes"])  # your utils expect list/None
    return search_documents(**p)


st.set_page_config(page_title="Search", layout="wide")
st.title("🔎 Search")

_defaults = {
    "page": 0,
    "page_size": 10,
    "sort": "relevance",
    "q": "",
    "filetypes": [],
    "path_prefix": "",
}

for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


def run_search() -> None:
    q = (st.session_state.get("q") or "").strip()
    if not q:
        st.session_state.results = None
        return

    params = {
        "q": q,
        "from_": st.session_state.page * st.session_state.page_size,
        "size": st.session_state.page_size,
        "sort": st.session_state.sort,
        "path_prefix": st.session_state.path_prefix or None,
        "filetypes": st.session_state.filetypes or None,  # keep None for “no filter”
    }

    # Only add date filters when the corresponding checkbox is enabled
    if st.session_state.get("enable_modified"):
        start, end = st.session_state.modified_range
        params["modified_from"] = start.isoformat()
        params["modified_to"] = end.isoformat()

    if st.session_state.get("enable_created"):
        start, end = st.session_state.created_range
        params["created_from"] = start.isoformat()
        params["created_to"] = end.isoformat()

    try:
        st.session_state.results = search_documents(**params)
    except Exception as e:
        st.session_state.results = None
        st.error(f"Search failed: {e}")


def _reset_and_search() -> None:
    st.session_state.page = 0


params = current_params()
res = cached_search_documents(**params) if params else None


search_col, sort_col = st.columns([4, 1], vertical_alignment="bottom")
with search_col:
    st.text_input(
        "Search",
        key="q",
        value="",
        on_change=_reset_and_search,
        placeholder="Type to search",
    )
with sort_col:
    st.selectbox(
        "Sort",
        ["relevance", "modified"],
        key="sort",
        on_change=_reset_and_search,
    )

filters_col1, filters_col2 = st.columns(2, vertical_alignment="bottom")
with filters_col1:
    st.text_input("Path prefix", key="path_prefix", on_change=_reset_and_search)
with filters_col2:
    # populate options from last search aggs (if any)
    options = []
    if res and res.get("aggs"):
        options = [
            b["key"] for b in res["aggs"].get("filetypes", {}).get("buckets", [])
        ]
    st.multiselect(
        "File type", options=options, key="filetypes", on_change=_reset_and_search
    )
# Modified range (optional)
st.checkbox(
    "Filter by modified date",
    key="enable_modified",
    value=False,
)
st.date_input(
    "Modified range",
    key="modified_range",
    value=(date.today() - timedelta(days=90), date.today()),
    disabled=not st.session_state.enable_modified,
)

# Created range (optional)
st.checkbox(
    "Filter by created date",
    key="enable_created",
    value=False,
if res:
    st.markdown(f"Found {res.get('total', 0)} results • {res.get('took', 0)} ms")
    for hit in res.get("hits", []):
        st.markdown(
            f"**{hit.get('filename', '')}** - {format_file_size(hit.get('size_bytes', 0))}"
        )
        st.caption(
            f"{hit.get('path', '')} • {format_timestamp(hit.get('modified_at'))}"
        )
        for frag in hit.get("highlights", [])[:3]:
            st.markdown(f"…{frag}…", unsafe_allow_html=True)
else:
    st.info("Enter a query to search your documents")

left, right = st.columns(2, vertical_alignment="bottom")
with left:
    st.number_input("Page size", 1, 100, key="page_size", on_change=_reset_and_search)
with right:
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("◀ Prev", disabled=st.session_state.page <= 0):
            st.session_state.page -= 1
    with col_next:
        # disable "Next" when no more hits
        can_next = bool(res) and (
            (st.session_state.page + 1) * st.session_state.page_size
            < res.get("total", 0)
        )
        if st.button("Next", disabled=not can_next):
            st.session_state.page += 1

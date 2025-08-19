import streamlit as st
from datetime import date, time, datetime, timezone
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp, format_date
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
    "path_contains": "",
}

for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


def _iso_start(d: date | None) -> str | None:
    return datetime.combine(d, time.min, tzinfo=timezone.utc).isoformat() if d else None


def _iso_end(d: date | None) -> str | None:
    return datetime.combine(d, time.max, tzinfo=timezone.utc).isoformat() if d else None


def current_params() -> dict | None:
    q = (st.session_state.get("q") or "").strip()
    if not q:
        return None
    params = {
        "q": q,
        "from_": st.session_state.page * st.session_state.page_size,
        "size": st.session_state.page_size,
        "sort": st.session_state.sort,
        "path_contains": (st.session_state.path_contains or None),
        "filetypes": tuple(st.session_state.filetypes) or None,
    }

    # Only add date filters when the corresponding checkbox is enabled
    if st.session_state.get("enable_modified"):
        params["modified_from"] = _iso_start(st.session_state.get("modified_from"))
        params["modified_to"] = _iso_end(st.session_state.get("modified_to"))

    if st.session_state.get("enable_created"):
        params["created_from"] = _iso_start(st.session_state.get("created_from"))
        params["created_to"] = _iso_end(st.session_state.get("created_to"))

    return params


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
    st.text_input("Path contains", key="path_contains", on_change=_reset_and_search)
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
# Modified range filter
st.checkbox(
    "Filter by modified date",
    key="enable_modified",
    value=False,
)
mod_from, mod_to = st.columns(2)
with mod_from:
    st.date_input(
        "From:",
        key="modified_from",
        value=None,
        format="DD/MM/YYYY",
        disabled=not st.session_state.enable_modified,
    )
with mod_to:
    st.date_input(
        "To:",
        key="modified_to",
        value=None,
        format="DD/MM/YYYY",
        min_value=st.session_state.modified_from,
        disabled=not st.session_state.enable_modified,
    )
# Created range filter
st.checkbox(
    "Filter by created date",
    key="enable_created",
    value=False,
)
created_from, created_to = st.columns(2)
with created_from:
    st.date_input(
        "From:",
        key="created_from",
        value=None,
        format="DD/MM/YYYY",
        disabled=not st.session_state.enable_created,
    )
with created_to:
    st.date_input(
        "To:",
        key="created_to",
        value=None,
        format="DD/MM/YYYY",
        min_value=st.session_state.created_from,
        disabled=not st.session_state.enable_created,
    )

if res:
            date_str = format_date(hit.get("modified_at"))
            st.markdown(f"**{filename}** • {date_str}")

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

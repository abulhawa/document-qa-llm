import streamlit as st
import math
from datetime import date, time, datetime, tzinfo
from utils.file_utils import format_file_size, open_file_local
from utils.time_utils import format_timestamp, format_date
from utils.fulltext_search import search_documents
from utils.opensearch_utils import (
    list_files_missing_fulltext,
)
from ui.ingest_client import enqueue_paths
from ui.task_status import add_records

@st.cache_data(ttl=180, show_spinner=False)
def cached_search_documents(**params):
    # cache keys must be hashable; you already pass filetypes as tuple in current_params()
    p = dict(params)
    if isinstance(p.get("filetypes"), tuple):
        p["filetypes"] = list(p["filetypes"])  # your utils expect list/None
    return search_documents(**p)


st.set_page_config(page_title="Search", layout="wide")
st.title("🔎 Search")

PAGE_SIZE_OPTIONS = [5, 25, 50, 100]
DEFAULT_PAGE_SIZE = 25

_defaults = {
    "page": 0,
    "page_size": DEFAULT_PAGE_SIZE,
    "sort": "relevance",
    "q": "",
    "filetypes": [],
    "path_contains": "",
}

for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo  # type: ignore[return-value]


def _iso_start(d: date | None) -> str | None:
    return (
        datetime.combine(d, time.min, tzinfo=_local_tz()).isoformat() if d else None
    )


def _iso_end(d: date | None) -> str | None:
    return (
        datetime.combine(d, time.max, tzinfo=_local_tz()).isoformat() if d else None
    )


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
        "modified_from": _iso_start(st.session_state.get("modified_from")),
        "modified_to": _iso_end(st.session_state.get("modified_to")),
        "created_from": _iso_start(st.session_state.get("created_from")),
        "created_to": _iso_end(st.session_state.get("created_to")),
    }
    return params


def _reset_and_search() -> None:
    st.session_state.page = 0


params = current_params()
res = cached_search_documents(**params) if params else None


if st.checkbox("Show files missing from full-text index"):
    missing_files = list_files_missing_fulltext()
    if not missing_files:
        st.success("All indexed files are present in the full-text index.")
    else:
        st.warning(f"{len(missing_files)} file(s) missing from full-text index:")
        paths = [f.get("path", "") for f in missing_files if f.get("path")]
        for p in paths:
            st.code(p, language="")

        if st.button("Rebuild full-text (reingest)", key="reindex_missing"):
            with st.spinner(f"Queuing reingest for {len(paths)} file(s)…"):
                task_ids = enqueue_paths(paths, mode="reingest")
                st.session_state["ingest_tasks"] = add_records(
                    st.session_state.get("ingest_tasks"),
                    paths,
                    task_ids,
                    action="reingest",
                )
            st.success(f"Queued reingest for {len(paths)} file(s).")
            # st.rerun()


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

d1, d2 = st.columns(2)
with d1:
    mod_from, mod_to = st.columns(2)
    with mod_from:
        st.date_input(
            "From:",
            key="modified_from",
            value=None,
            format="DD/MM/YYYY",
        )
    with mod_to:
        st.date_input(
            "To:",
            key="modified_to",
            value=None,
            format="DD/MM/YYYY",
            min_value=st.session_state.modified_from,
        )
with d2:
    created_from, created_to = st.columns(2)
    with created_from:
        st.date_input(
            "From:",
            key="created_from",
            value=None,
            format="DD/MM/YYYY",
        )
    with created_to:
        st.date_input(
            "To:",
            key="created_to",
            value=None,
            format="DD/MM/YYYY",
            min_value=st.session_state.created_from,
        )

total_results = res.get("total", 0) if res else 0
if res:
    st.markdown(f"Found {total_results} results • {res.get('took', 0)} ms")
    with st.container(height=500):
        if total_results == 0:
            st.markdown(f'No results were found for "{st.session_state.q}"!')
        for i, hit in enumerate(res.get("hits", [])):
            path = hit.get("path", "") or ""
            filename = hit.get("filename")
            date_str = format_date(hit.get("modified_at"))
            st.markdown(f"**{filename}** • {date_str}")

            # first highlight inline, rest in expander
            frags = hit.get("highlights", [])
            if frags:
                frags = [f.replace("\n\n", "  \n") for f in frags]
                st.markdown(f"…{frags[0]}…", unsafe_allow_html=True)
                if len(frags) > 1:
                    with st.expander(f"Show {min(len(frags)-1, 4)} more highlight(s)…"):
                        for frag in frags[1:]:
                            st.markdown(f"…{frag}…", unsafe_allow_html=True)

            # actions row (buttons are optional)
            c1, c2, _ = st.columns(3, vertical_alignment="bottom")
            with c1:
                if st.button("Open", key=f"open_{i}"):
                    open_file_local(path)
            with c2:
                # easy "copy path"
                st.code(path, language="")

            st.divider()
else:
    st.info("Enter a query to search your documents")

page_size = st.session_state.page_size
# Clamp page into valid bounds [0, total_pages-1]
total_pages = max(1, math.ceil(total_results / page_size))  # at least 1 virtual page
st.session_state.page = min(max(st.session_state.page, 0), total_pages - 1)

can_prev = st.session_state.page > 0
can_next = (st.session_state.page + 1) < total_pages

left, right = st.columns(2, vertical_alignment="bottom")
with left:
    st.selectbox(
        "Results per page",
        PAGE_SIZE_OPTIONS,
        key="page_size",
        on_change=_reset_and_search,
    )

with right:
    col_prev, info_col, col_next = st.columns(
        [1, 1, 1], gap="small", vertical_alignment="center"
    )
    with col_prev:
        st.button(
            "◀ Prev",
            key="pager_prev",
            disabled=not can_prev,
            on_click=lambda: st.session_state.__setitem__(
                "page", max(0, st.session_state.page - 1)
            ),
        )
    with info_col:
        st.markdown(f"**Page {st.session_state.page + 1} of {total_pages}**")
    with col_next:
        st.button(
            "Next ▶",
            key="pager_next",
            disabled=not can_next,
            on_click=lambda: st.session_state.__setitem__(
                "page", min(total_pages - 1, st.session_state.page + 1)
            ),
        )

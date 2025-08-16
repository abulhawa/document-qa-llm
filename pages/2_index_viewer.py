import streamlit as st
import pandas as pd
import threading
import queue
import time
import os
import sys
import subprocess
from typing import List, Dict, Any, Tuple

from utils.time_utils import format_timestamp
from utils.opensearch_utils import (
    list_files_from_opensearch,
    delete_files_by_path_checksum,
    get_duplicate_checksums,
    get_chunks_by_path,
)
from utils.qdrant_utils import (
    count_qdrant_chunks_by_path,
    delete_vectors_by_path_checksum,
    index_chunks,
)
from utils.file_utils import format_file_size
from core.ingestion import ingest
from config import logger

def _open_file(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        st.error(f"Failed to open file: {e}")


def _show_in_folder(path: str) -> None:
    folder = os.path.dirname(path)
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", path], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        else:
            subprocess.run(["xdg-open", folder], check=False)
    except Exception as e:
        st.error(f"Failed to open folder: {e}")


def _sync_os_to_qdrant(path: str, checksum: str) -> None:
    chunks = get_chunks_by_path(path)
    if not chunks:
        st.warning("No OpenSearch chunks found for path")
        return
    delete_vectors_by_path_checksum([(path, checksum)])
    ok = index_chunks(chunks)
    if ok:
        st.toast("Synced embeddings for file.", icon="✅")
    else:
        st.error("Failed to sync embeddings.")



st.set_page_config(page_title="File Index Viewer", layout="wide")
st.title("📂 File Index Viewer")


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_indexed_files() -> List[Dict[str, Any]]:
    # Fast path: do NOT compute per-file Qdrant counts here to keep UI responsive
    return list_files_from_opensearch()


def _prefetch_index_in_bg(out_q: "queue.Queue") -> None:
    """Background fetch that doesn't touch Streamlit APIs."""
    try:
        files = list_files_from_opensearch()
        out_q.put({"ok": True, "files": files})
    except Exception as e:
        out_q.put({"ok": False, "error": str(e)})


def trigger_refresh() -> None:
    """Kick off a background refresh if one isn't already running."""
    if not st.session_state.get("_prefetch_running"):
        st.session_state["_prefetch_running"] = True
        st.session_state["_prefetch_started_at"] = time.time()
        st.session_state.pop("_prefetched_files", None)
        st.session_state.pop("_prefetch_error", None)
        q: "queue.Queue" = st.session_state.setdefault(
            "_prefetch_queue", queue.Queue(maxsize=1)
        )
        # clear any stale item
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t = threading.Thread(target=_prefetch_index_in_bg, args=(q,), daemon=True)
        t.start()


def _get_files_fast() -> List[Dict[str, Any]]:
    override = st.session_state.pop("_files_override", None)
    if override is not None:
        return override
    return load_indexed_files()


def build_table_data(files: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for f in files:
        rows.append(
            {
                "Filename": f.get("filename", ""),
                "Path": f.get("path", ""),
                "Filetype": f.get("filetype", ""),
                "Modified": format_timestamp(f.get("modified_at") or ""),
                "Created": format_timestamp(f.get("created_at") or ""),
                "Indexed": format_timestamp(f.get("indexed_at") or ""),
                "Size": f.get("bytes", 0),
                "OpenSearch Chunks": f.get("num_chunks", 0),
                "Qdrant Chunks": f.get("qdrant_count", 0),
                "Checksum": f.get("checksum", ""),
            }
        )
    df = pd.DataFrame(rows)
    if "Modified" in df.columns:
        df = df.sort_values("Modified", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def render_filtered_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # Filters
    colf1, colf2, colf3, colf4 = st.columns([3, 2, 2, 1])
    with colf1:
        path_filter = st.text_input(
            "Filter by path substring", value=st.session_state.get("path_filter", "")
        )
        st.session_state["path_filter"] = path_filter
    with colf2:
        only_missing = st.checkbox(
            "Only missing embeddings (Qdrant=0)",
            value=st.session_state.get("embed_filter", False),
        )
        st.session_state["embed_filter"] = only_missing
    with colf3:
        selection_mode = st.toggle(
            "Selection mode",
            value=st.session_state.get("selection_mode", False),
            help="Enable to select specific rows for batch actions",
        )
        st.session_state["selection_mode"] = selection_mode
    with colf4:
        if st.button("↻ Refresh", use_container_width=True):
            trigger_refresh()

    # Apply filters
    fdf = df.copy()
    if path_filter:
        fdf = fdf[fdf["Path"].str.contains(path_filter, case=False, na=False)]

    # Decide if we need Qdrant counts for this view
    need_counts = only_missing
    show_qdrant_counts = st.checkbox(
        "Show Qdrant counts (slower)",
        value=False,
        help="Compute counts only for visible rows",
    )
    need_counts = need_counts or show_qdrant_counts

    if need_counts and not fdf.empty:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        visible_paths = fdf["Path"].dropna().astype(str).unique().tolist()
        memo = st.session_state.setdefault("_qdrant_count_memo", {})
        missing = [cs for cs in visible_paths if cs not in memo]
        if missing:
            with st.spinner(f"Counting Qdrant chunks for {len(missing)} file(s)…"):
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = {
                        ex.submit(count_qdrant_chunks_by_path, cs): cs for cs in missing
                    }
                    for fut in as_completed(futs):
                        cs = futs[fut]
                        try:
                            memo[cs] = fut.result() or 0
                        except Exception:
                            memo[cs] = 0
        # update counts in the DataFrame for filtering and display
        fdf["Qdrant Chunks"] = (
            fdf["Path"].astype(str).map(memo).fillna(fdf.get("Qdrant Chunks", 0))
        )

    # Now that counts exist (if needed), apply the 'missing embeddings' filter
    if only_missing:
        fdf = fdf[(fdf["Qdrant Chunks"].fillna(0) == 0)]

    st.caption(f"{len(fdf)} file(s) match current filters.")

    # Poll background queue (non-blocking) to capture results
    _q = st.session_state.get("_prefetch_queue")
    if _q is not None:
        try:
            msg = _q.get_nowait()
            st.session_state["_prefetch_running"] = False
            if msg.get("ok"):
                st.session_state["_prefetched_files"] = msg["files"]
                st.session_state.pop("_prefetch_error", None)
            else:
                st.session_state["_prefetch_error"] = msg.get("error", "Unknown error")
        except queue.Empty:
            pass

    # Non-blocking refresh status/apply
    status_col1, status_col2 = st.columns([3, 2])
    with status_col1:
        now = time.time()
        if st.session_state.get("_prefetch_running"):
            # timeout safeguard
            started = st.session_state.get("_prefetch_started_at", now)
            if now - started > 20:
                st.session_state["_prefetch_running"] = False
                st.session_state["_prefetch_error"] = "Refresh timed out after 20s"
            st.caption("Updating… (fetching latest from OpenSearch)")
            # gentle auto-rerun throttle while waiting
            last = st.session_state.get("_prefetch_last_rerun", 0.0)
            if now - last > 1.2:
                st.session_state["_prefetch_last_rerun"] = now
                st.rerun()
        elif st.session_state.get("_prefetch_error"):
            st.error(f"Refresh failed: {st.session_state['_prefetch_error']}")
        elif st.session_state.get("_prefetched_files") is not None:
            st.success("New data is ready to apply.", icon="✅")
    with status_col2:
        if st.session_state.get("_prefetched_files") is not None:
            if not selection_mode:
                new_files = st.session_state.pop("_prefetched_files")
                load_indexed_files.clear()
                st.session_state["_files_override"] = new_files
                st.toast("Table updated.", icon="✅")
                st.rerun()
            else:
                if st.button("Apply updated data", use_container_width=True):
                    new_files = st.session_state.pop("_prefetched_files")
                    load_indexed_files.clear()
                    st.session_state["_files_override"] = new_files
                    st.rerun()

    table_key = "file_index_editor" if selection_mode else "file_index_viewer"
    render_action_buttons(fdf, table_key)

    if selection_mode:
        # Selection helpers
        h1, h2, h3 = st.columns([2, 2, 6])
        if h1.button("Select duplicates"):
            dups = set(get_duplicate_checksums())
            st.session_state["select_checksums"] = list(dups)
            st.toast(
                f"Found {len(dups)} duplicate checksum(s). Preselecting those in the table."
            )
            st.rerun()
        if h2.button("Clear selection"):
            st.session_state.pop("select_checksums", None)
            st.rerun()

        # ensure Select column exists before editor
        if "Select" not in fdf.columns:
            fdf.insert(0, "Select", False)

        # Preselect rows if we have stored selections
        if "select_checksums" in st.session_state:
            selected_set = set(st.session_state["select_checksums"] or [])
            fdf["Select"] = fdf["Checksum"].astype(str).isin(selected_set)
        display_df = fdf.copy()
        display_df["Size"] = display_df["Size"].apply(format_file_size)
        edited = st.data_editor(
            display_df,
            hide_index=True,
            use_container_width=True,
            disabled=[
                "Filename",
                "Path",
                "Checksum",
                "Filetype",
                "Modified",
                "Created",
                "Indexed",
                "Size",
                "OpenSearch Chunks",
                "Qdrant Chunks",
            ],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
            },
            num_rows="fixed",
            key="file_index_editor",
        )
        # Sync current selection into session so it survives auto-apply updates
        try:
            st.session_state["select_checksums"] = (
                edited.loc[edited["Select"] == True, "Checksum"]
                .astype(str)
                .unique()
                .tolist()  # noqa: E712
            )
        except Exception:
            pass
        return edited, table_key
    else:
        display_df = fdf.copy()
        display_df["Size"] = display_df["Size"].apply(format_file_size)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            st.dataframe(
                fdf,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.data_editor(
                display_df,
                hide_index=True,
                use_container_width=True,
                disabled=[
                    "Filename",
                    "Path",
                    "Checksum",
                    "Filetype",
                    "Modified",
                    "Created",
                    "Indexed",
                    "Size",
                    "OpenSearch Chunks",
                    "Qdrant Chunks",
                ],
                num_rows="fixed",
                key="file_index_viewer",
            )
        return fdf, table_key


def render_action_buttons(fdf: pd.DataFrame, key: str) -> None:
    sel = st.session_state.get(key, {}).get("selection", {})
    rows = sel.get("rows", []) if isinstance(sel, dict) else []
    selected = fdf.iloc[rows] if rows else pd.DataFrame()
    disabled = selected.empty

    b1, b2, b3, b4, b5 = st.columns(5)

    def _pairs() -> List[Tuple[str, str]]:
        return list(
            selected[["Path", "Checksum"]]
            .dropna()
            .astype(str)
            .itertuples(index=False, name=None)
        )

    if b1.button("📄 Open", use_container_width=True, disabled=disabled):
        _open_file(selected.iloc[0]["Path"])

    if b2.button("📂 Show", use_container_width=True, disabled=disabled):
        _show_in_folder(selected.iloc[0]["Path"])

    if b3.button("🔁 Sync", use_container_width=True, disabled=disabled):
        try:
            for r in selected.itertuples(index=False):
                _sync_os_to_qdrant(r.Path, r.Checksum)
            load_indexed_files.clear()
        except Exception as e:
            logger.exception(f"Sync failed: {e}")
            st.error(f"Sync failed: {e}")

    if b4.button("🔄 Reingest", use_container_width=True, disabled=disabled):
        try:
            ingest(selected["Path"].astype(str).tolist(), force=True, op="reingest", source="viewer")
            st.success(f"Queued reingestion for {len(selected)} file(s).")
            load_indexed_files.clear()
        except Exception as e:
            logger.exception(f"Reingest failed: {e}")
            st.error(f"Reingest failed: {e}")

    if b5.button("🗑️ Delete", use_container_width=True, disabled=disabled):
        try:
            pairs = _pairs()
            delete_files_by_path_checksum(pairs)
            delete_vectors_by_path_checksum(pairs)
            st.success(f"Deleted {len(pairs)} file(s).")
            load_indexed_files.clear()
        except Exception as e:
            logger.exception(f"Delete failed: {e}")
            st.error(f"Delete failed: {e}")


def run_batch_actions(fdf: pd.DataFrame) -> None:
    st.subheader("Batch actions")
    with st.form("batch_actions_form", clear_on_submit=False):
        a1, a2 = st.columns([2, 2])
        with a1:
            action = st.selectbox("Action", ["Reingest", "Delete"], index=0)
        with a2:
            scope_options = ["All filtered"]
            if "Select" in fdf.columns and fdf["Select"].any():
                scope_options.append("Only selected")
            scope = st.selectbox("Scope", scope_options, index=0)

        confirm = ""
        if action == "Delete":
            confirm = st.text_input(
                "Type to confirm (e.g., DELETE 3 paths)", value=""
            )

        submitted = st.form_submit_button("Run")

    if not submitted:
        return

    # Compute target rows
    if scope == "Only selected" and "Select" in fdf.columns:
        targets = fdf[fdf["Select"] == True]  # noqa: E712
    else:
        targets = fdf

    if targets.empty:
        st.warning("No rows to act on.")
        return

    if action == "Delete":
        expected = len(targets)
        if confirm.strip() != f"DELETE {expected}":
            st.error(f"Confirmation mismatch. Please type exactly: DELETE {expected}")
            return

    pairs = (
        targets[["Path", "Checksum"]]
        .dropna()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    pairs = list(pairs)
    paths = sorted({p for p, _ in pairs})

    try:
        if action == "Reingest":
            with st.spinner(f"Queuing reingestion for {len(paths)} file(s)…"):
                ingest(paths, force=True, op="reingest", source="viewer")
            st.success(f"Queued reingestion for {len(paths)} file(s).")
        elif action == "Delete":
            with st.spinner(f"Deleting {len(pairs)} file(s) from OpenSearch…"):
                deleted = delete_files_by_path_checksum(pairs)
            st.info(
                f"OpenSearch deleted {deleted} chunks across {len(pairs)} file(s).",
            )
            with st.spinner(
                f"Deleting vectors in Qdrant for {len(pairs)} path(s)…",

            ):
                delete_vectors_by_path_checksum(pairs)
            st.success("Qdrant deletion requested.")
        # Refresh cache after action
        load_indexed_files.clear()
    except Exception as e:
        logger.exception(f"Batch action failed: {e}")
        st.error(f"Batch action failed: {e}")


files = _get_files_fast()
df = build_table_data(files)
fdf, _ = render_filtered_table(df)
run_batch_actions(fdf)

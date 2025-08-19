import streamlit as st
import pandas as pd
import threading
import queue
import time
import os, sys, subprocess
from typing import List, Dict, Any

from utils.time_utils import format_timestamp
from utils.opensearch_utils import (
    list_files_from_opensearch,
    delete_files_by_path_checksum,
)
from utils.qdrant_utils import (
    count_qdrant_chunks_by_path,
    delete_vectors_by_path_checksum,
)
from utils.file_utils import format_file_size
from core.ingestion import ingest
from config import logger

st.set_page_config(page_title="File Index Viewer", layout="wide")
st.title("📂 File Index Viewer")

# Confirm when opening/showing > N files
MAX_BULK_OPEN = 10


# ---------- OS helpers for row actions (Open / Show in folder) ----------
def open_file_local(path: str) -> None:
    """Open a file on the machine running Streamlit."""
    if not path:
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        st.warning(f"Could not open file: {e}")


def show_in_folder(path: str) -> None:
    """Reveal file in its folder (selects the file on Windows/macOS)."""
    if not path:
        return
    try:
        if sys.platform.startswith("win"):
            # /select, must be a single token; pass via shell to support commas
            win_path = path.replace("/", "\\")
            subprocess.run(["explorer", "/select,", win_path], shell=True, check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        else:
            subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
    except Exception as e:
        st.warning(f"Could not open folder: {e}")


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


def render_filtered_table(df: pd.DataFrame) -> pd.DataFrame:
    # Filters
    colf1, colf2, colf3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    with colf1:
        c_l, c_r = st.columns([5, 1], vertical_alignment="bottom")

        # A nonce forces a brand-new text_input instance on reset
        pf_nonce = st.session_state.setdefault("path_filter_nonce", 0)

        with c_r:
            if st.button(
                "Reset", use_container_width=True, help="Clear filter and reset table"
            ):
                # Clear app-level filter value and recreate the widget next run
                st.session_state["path_filter"] = ""
                st.session_state["path_filter_nonce"] = pf_nonce + 1  # new widget key
                # Preserve selection on this rerun
                st.session_state["_suppress_next_selection_overwrite"] = True
                st.rerun()

        with c_l:
            # Use a widget key that changes with the nonce so its visual value resets
            path_filter_input = st.text_input(
                "Filter by path substring",
                value=st.session_state.get("path_filter", ""),
                key=f"path_filter_widget_{st.session_state['path_filter_nonce']}",
            )

        # Store the current value under a stable app-level key (NOT the widget key)
        st.session_state["path_filter"] = path_filter_input

    # Use the current filter value everywhere below
    path_filter = st.session_state.get("path_filter", "")
    with colf2:
        only_missing = st.checkbox(
            "Only missing embeddings (Qdrant=0)",
            value=st.session_state.get("embed_filter", False),
        )
        st.session_state["embed_filter"] = only_missing
    with colf3:
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
    # Track if any non-table control changed this run (used to preserve selection)
    controls_changed = False

    def _changed(k, v):
        prev = st.session_state.get(f"_prev_{k}", "__MISSING__")
        st.session_state[f"_prev_{k}"] = v
        return prev != "__MISSING__" and prev != v

    if _changed("path_filter", path_filter):
        controls_changed = True
    if _changed("embed_filter", only_missing):
        controls_changed = True
    if _changed("show_qdrant_counts", show_qdrant_counts):
        controls_changed = True

    # one-shot suppression flag (if True, we won't overwrite saved selection with an empty one)
    st.session_state["_suppress_next_selection_overwrite"] = controls_changed

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
            new_files = st.session_state.pop("_prefetched_files")
            load_indexed_files.clear()
            st.session_state["_files_override"] = new_files
            st.toast("Table updated.", icon="✅")
            st.rerun()
    # ---------- Single table (st.dataframe) with persistent selection ----------

    # Default sort (best-effort) before showing
    display_df = fdf.copy()

    # Keep 'Size' numeric for tests; only format for display via Styler
    use_styler = "Size" in display_df.columns
    if use_styler:
        styled_df = display_df.style.format({"Size": format_file_size})
    else:
        styled_df = display_df

    # A nonce lets us hard-reset the widget when you click "Clear selection"
    nonce = st.session_state.setdefault("file_index_table_nonce", 0)

    # Placeholder for the action bar (will appear *above* the table)
    action_bar = st.container()

    # ---- Render the table and read selection (UI is the source of truth) ----
    event = st.dataframe(
        display_df if not use_styler else styled_df,
        hide_index=True,
        use_container_width=True,
        key=f"file_index_table_{nonce}",
        on_select="rerun",  # rerun on checkbox change
        selection_mode="multi-row",  # checkbox UI
    )

    # Current selection = rows checked in the widget (relative to display_df)
    sel_idx = (event or {}).get("selection", {}).get("rows", [])
    try:
        selected_paths = (
            (display_df.iloc[sel_idx]["Path"].dropna().astype(str).unique().tolist())
            if sel_idx
            else []
        )
    except Exception:
        selected_paths = []

    # -------------------- Action bar (above the table) --------------------
    with action_bar:
        st.caption(f"Selected {len(selected_paths)} file(s)")

        c1, c2, c3 = st.columns([1.4, 1.6, 1.2])

        with c1:
            if st.button("📂 Open selected", use_container_width=True):
                if not selected_paths:
                    st.info("Select one or more rows first.")
                else:
                    for p in sorted(selected_paths):
                        open_file_local(p)
                    st.success(f"Opened {len(selected_paths)} file(s).")

        with c2:
            if st.button("📁 Show folders (selected)", use_container_width=True):
                if not selected_paths:
                    st.info("Select one or more rows first.")
                else:
                    for p in sorted(selected_paths):
                        show_in_folder(p)
                    st.success("Done.")

        with c3:
            if st.button("❌ Clear selection", use_container_width=True):
                # Bump the table key to visually uncheck all boxes
                st.session_state["file_index_table_nonce"] = nonce + 1
                st.rerun()
    # Helpful hint
    st.caption("Tip: sort/filter first, then use the checkboxes to select rows.")
    # ---- Bulk apply to ALL rows currently shown (no checkboxes needed) ----
    with st.expander(
        f"Bulk apply to ALL rows currently shown ({len(display_df)})", expanded=False
    ):
        # All visible rows in the table right now
        all_paths = (
            display_df.get("Path", pd.Series([], dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        n_all = len(all_paths)

        # Safety confirm: only required when over threshold
        need_confirm = n_all > MAX_BULK_OPEN
        if need_confirm:
            bulk_confirm = st.checkbox(
                f"Confirm bulk action on {n_all} files (> {MAX_BULK_OPEN})",
                value=False,
                key=f"bulk_confirm_{nonce}",
                help="Prevents accidental mass actions",
            )
        else:
            bulk_confirm = True  # not required for small batches

        bc1, bc2 = st.columns(2)

        if bc1.button(
            f"📂 Open ALL shown ({n_all})",
            use_container_width=True,
            key=f"open_all_{nonce}",
        ):
            if not all_paths:
                st.info("No rows shown.")
            elif not bulk_confirm:
                st.warning("Tick the confirm checkbox to proceed.")
            else:
                for p in all_paths:
                    open_file_local(p)
                st.success(f"Opened {n_all} file(s).")

        if bc2.button(
            f"📁 Show folders for ALL shown ({n_all})",
            use_container_width=True,
            key=f"show_all_{nonce}",
        ):
            if not all_paths:
                st.info("No rows shown.")
            elif not bulk_confirm:
                st.warning("Tick the confirm checkbox to proceed.")
            else:
                for p in all_paths:
                    show_in_folder(p)
                st.success("Done.")
    return fdf


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
            confirm = st.text_input("Type to confirm (e.g., DELETE 3 paths)", value="")

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


def render_row_actions(fdf: pd.DataFrame) -> None:
    st.subheader("Row actions")
    name_col = (
        "Filename"
        if "Filename" in fdf.columns
        else ("Path" if "Path" in fdf.columns else None)
    )
    if not name_col or fdf.empty:
        st.info("No rows to act on.")
        return

    options = fdf[name_col].astype(str).tolist()
    idx = st.selectbox(
        "Pick a file",
        options=list(range(len(options))),
        format_func=lambda i: options[i],
        key="row_pick",
    )
    row = fdf.iloc[int(idx)]
    st.markdown(f"**Selected File:** `{row[name_col]}`")
    c1, c2 = st.columns(2)

    if c1.button("🔄 Reingest File", use_container_width=True):
        try:
            ingest([row["Path"]], force=True, op="reingest", source="viewer")
            st.success(f"Reingestion triggered for: {row[name_col]}")
            load_indexed_files.clear()
        except Exception as e:
            logger.exception(f"Row reingest failed: {e}")
            st.error(f"Row reingest failed: {e}")

    if c2.button("🗑️ Delete from Index", use_container_width=True):
        try:
            delete_files_by_path_checksum([(row["Path"], row["Checksum"])])
            delete_vectors_by_path_checksum([(row["Path"], row["Checksum"])])
            st.success(f"Deleted: {row[name_col]}")
            load_indexed_files.clear()
        except Exception as e:
            logger.exception(f"Row delete failed: {e}")
            st.error(f"Row delete failed: {e}")


files = _get_files_fast()
df = build_table_data(files)
fdf = render_filtered_table(df)
run_batch_actions(fdf)
st.divider()
render_row_actions(fdf)

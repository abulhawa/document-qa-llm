import streamlit as st
from config import WATCH_INVENTORY_INDEX, WATCHLIST_INDEX
from app.usecases import watchlist_usecase
from ui.task_status import add_records
from components.task_panel import render_task_panel
from utils.opensearch.indexes import ensure_index_exists

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Watchlist", layout="wide")

ensure_index_exists(index=WATCH_INVENTORY_INDEX)

st.title("Watchlist")
st.caption("Track folders and see how many files still need indexing. Use the actions to import known files and update counts.")

# Load persisted watchlist prefixes
ensure_index_exists(index=WATCHLIST_INDEX)
watched = st.session_state.setdefault(
    "watched_prefixes", watchlist_usecase.load_watchlist_prefixes()
)
st.session_state.setdefault("ingest_tasks", [])

with st.container(border=True):
    st.subheader("Add Tracked Folder")
    with st.form("add_prefix"):
        new_prefix = st.text_input("Path prefix (folder)", value="")
        submitted = st.form_submit_button("Add")
        if submitted:
            p = (new_prefix or "").strip()
            if not p:
                st.warning("Enter a valid path prefix")
            elif p in watched:
                st.info("Already tracked")
            else:
                ok = watchlist_usecase.add_prefix(p)
                if ok:
                    watched.append(p)
                    st.success("Added")
                else:
                    st.warning("Could not add prefix")

st.divider()

# Batch actions for all watched folders
if watched:
    with st.container(border=True):
        st.subheader("All Watched Folders")
        if st.button(
            "Refresh all",
            help="Import known files and chunk counts for every watched folder, then update counts.",
            key="seed-all",
        ):
            result = watchlist_usecase.refresh_all_status(list(watched))
            if result.errors:
                st.warning("Some folders could not be refreshed.")
            st.success(
                "Refreshed all. Imported known files: "
                f"{result.total_fulltext}, updated chunk counts: {result.total_chunks}."
            )

if not watched:
    st.info("No tracked folders yet. Add one above.")
else:
    for pref in list(watched):
        with st.container(border=True):
            st.subheader(pref)
            status = watchlist_usecase.get_status(pref)
            # Show current remaining for context
            remaining_now = status.remaining
            st.metric("Unindexed files", remaining_now)
            # Inline quick-win count for clarity
            st.caption(f"Quick wins (<=100KB): {status.quick_wins}")
            # Bold summaries and progress
            st.progress(
                status.percent_indexed,
                text=f"Indexed {status.indexed} / {status.total}",
            )
            meta = status.meta
            last_idx = int(meta.get("last_indexed", 0) or 0)
            last_tot = int(meta.get("last_total", 0) or 0)
            last_ref = meta.get("last_refreshed") or ""
            delta = status.indexed - last_idx
            if last_ref:
                st.caption(f"Last refreshed: {last_ref}  •  Indexed Δ: {delta:+}")
            last_scan = meta.get("last_scanned") or ""
            if last_scan:
                found = int(meta.get("last_scan_found", 0) or 0)
                missing = int(meta.get("last_scan_marked_missing", 0) or 0)
                st.caption(f"Last scan: {last_scan}  •  Found: {found}  •  Marked missing: {missing}")
            if status.errors:
                st.warning("Some status metrics failed to load.")
            # Preview a few unindexed file paths
            if status.preview:
                st.caption("Sample of unindexed files")
                st.table({"Path": status.preview})
            c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1, 1])
            with c0:
                if st.button(
                    "Refresh folder status",
                    key=f"seed-{pref}",
                    help="Import known files and chunk counts from existing indexes, then update the unindexed count.",
                ):
                    with st.spinner("Refreshing status ..."):
                        result = watchlist_usecase.refresh_status(pref)
                    if result.errors:
                        st.warning("Some refresh steps failed.")
                    st.success(
                        "Imported known files: "
                        f"{result.imported}, updated chunk counts: {result.chunk_counts}. "
                        f"Unindexed now: {result.remaining}"
                    )
            with c1:
                if st.button(
                    "Scan disk for changes",
                    key=f"scan-{pref}",
                    help="Walk the folder on disk to detect new/removed PDF, DOCX and TXT files and update inventory.",
                ):
                    with st.spinner("Scanning disk ..."):
                        result = watchlist_usecase.scan_folder(pref)
                    if result.errors:
                        st.warning("Some scan steps failed.")
                    st.success(
                        "Scan complete. Found: "
                        f"{result.found}, marked missing: {result.marked_missing}."
                    )
            with c1:
                if st.button(
                    "Check unindexed",
                    key=f"rem-{pref}",
                    help="Count files in this folder that are not yet indexed.",
                ):
                    with st.spinner("Computing ..."):
                        r, errors = watchlist_usecase.get_remaining(pref)
                    if errors:
                        st.warning("Unable to compute remaining files.")
                    st.success(f"Remaining: {r}")
            with c2:
                if st.button(
                    "Import known files",
                    key=f"sft-{pref}",
                    help="Add files already present in the File Index into the watch inventory, marking them as indexed.",
                ):
                    with st.spinner("Importing known files ..."):
                        n, errors = watchlist_usecase.import_known_files(pref)
                    if errors:
                        st.warning("Some files could not be imported.")
                    st.success(f"Imported {n} files from index")
            with c3:
                if st.button(
                    "Import chunk counts",
                    key=f"scc-{pref}",
                    help="Record how many chunks each file has in the index (advanced).",
                ):
                    with st.spinner("Importing chunk counts ..."):
                        n, errors = watchlist_usecase.import_chunk_counts(pref)
                    if errors:
                        st.warning("Some chunk counts could not be updated.")
                    st.success(f"Updated chunk counts for {n} files")
            with c4:
                if st.button(
                    "Stop tracking",
                    key=f"rm-{pref}",
                    help="Remove this folder from your watchlist.",
                ):
                    if watchlist_usecase.remove_prefix(pref):
                        watched.remove(pref)
                        st.warning("Folder removed from watchlist.")
                    else:
                        st.warning("Failed to remove. Try again.")
            with c5:
                if st.button(
                    "Ingest remaining",
                    key=f"ingest-{pref}",
                    help="Queue ingestion jobs for unindexed files under this folder (up to 2000).",
                ):
                    # Auto-sync from indices so we don't queue already-indexed files
                    with st.spinner("Syncing status from indices ..."):
                        sync_errors = watchlist_usecase.sync_from_indices(pref)
                    with st.spinner("Collecting unindexed files ..."):
                        # Prefer simple fetch; fallback to scroll if needed
                        paths = watchlist_usecase.list_unindexed_paths(pref, limit=2000)
                    if not paths:
                        st.info("No unindexed files found under this folder.")
                    else:
                        with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                            result = watchlist_usecase.queue_ingest(paths, mode="ingest")
                            st.session_state["ingest_tasks"] = add_records(
                                st.session_state.get("ingest_tasks"),
                                paths,
                                result.task_ids,
                                action="ingest",
                            )
                        if sync_errors or result.errors:
                            st.warning("Some ingestion steps failed.")
                        st.success(
                            f"Queued {len(result.task_ids)} file(s) for ingestion."
                        )

            # Smart re-ingest (changed files)
            with st.expander("Smart re-ingest (changed files)", expanded=False):
                if st.button(
                    "Find and re-ingest changed",
                    key=f"reingest-{pref}",
                    help="Re-ingest files whose modified time is newer than last indexed, or whose chunk counts drifted.",
                ):
                    with st.spinner("Finding changed files ..."):
                        re_files = watchlist_usecase.list_reingest_paths(
                            pref, limit=2000
                        )
                    if not re_files:
                        st.info("No changed files found under this folder.")
                    else:
                        with st.spinner(f"Queuing {len(re_files)} files for re-ingest ..."):
                            result = watchlist_usecase.queue_ingest(
                                re_files, mode="reingest"
                            )
                            st.session_state["ingest_tasks"] = add_records(
                                st.session_state.get("ingest_tasks"),
                                re_files,
                                result.task_ids,
                                action="reingest",
                            )
                        if result.errors:
                            st.warning("Some re-ingest steps failed.")
                        st.success(
                            f"Queued {len(result.task_ids)} file(s) for re-ingestion."
                        )

            # Ingest controls
            with st.expander("Ingest controls", expanded=False):
                cap = st.slider(
                    "Max files to queue",
                    min_value=500,
                    max_value=5000,
                    value=2000,
                    step=100,
                    help="Upper bound for how many files to enqueue in one go.",
                    key=f"cap-{pref}",
                )
                cA, cB, cC, cD = st.columns(4)
                with cA:
                    if st.button(
                        "Preview unindexed",
                        key=f"preview-unindexed-{pref}",
                        help="Show how many unindexed files exist; respects the cap in the next step.",
                    ):
                        total_unidx, errors = watchlist_usecase.get_remaining(pref)
                        if errors:
                            st.warning("Unable to compute unindexed preview.")
                        st.info(f"Unindexed total: {total_unidx}. Cap will queue up to {min(cap, total_unidx)}.")
                with cB:
                    if st.button(
                        "Queue unindexed",
                        key=f"queue-unindexed-{pref}",
                        help="Enqueue up to the cap of unindexed files under this folder.",
                    ):
                        with st.spinner("Collecting unindexed files ..."):
                            paths = watchlist_usecase.list_unindexed_paths(
                                pref, limit=cap
                            )
                        if not paths:
                            st.info("No unindexed files found under this folder.")
                        else:
                            with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                                result = watchlist_usecase.queue_ingest(
                                    paths, mode="ingest"
                                )
                                st.session_state["ingest_tasks"] = add_records(
                                    st.session_state.get("ingest_tasks"),
                                    paths,
                                    result.task_ids,
                                    action="ingest",
                                )
                            if result.errors:
                                st.warning("Some ingestion steps failed.")
                            st.success(
                                f"Queued {len(result.task_ids)} file(s) for ingestion."
                            )
                with cC:
                    if st.button(
                        "Preview quick wins (<=100KB)",
                        key=f"preview-quick-{pref}",
                        help="Count unindexed files up to 100 KB; respects the cap in the next step.",
                    ):
                        quick_win_result = watchlist_usecase.get_quick_win_counts(
                            pref, max_size_bytes=102_400
                        )
                        if quick_win_result.errors:
                            st.warning("Unable to compute quick-win counts.")
                        if quick_win_result.missing_size:
                            st.info(
                                "Quick wins total (<=100KB): "
                                f"{quick_win_result.count}. Cap will queue up to {min(cap, quick_win_result.count)}. "
                                f"Note: {quick_win_result.missing_size} unindexed file(s) missing size; "
                                "run 'Scan disk for changes' to populate sizes."
                            )
                        else:
                            st.info(
                                "Quick wins total (<=100KB): "
                                f"{quick_win_result.count}. Cap will queue up to {min(cap, quick_win_result.count)}."
                            )
                with cD:
                    if st.button(
                        "Queue quick wins (<=100KB)",
                        key=f"queue-quick-{pref}",
                        help="Enqueue up to the cap of small unindexed files (<= 100 KB).",
                    ):
                        with st.spinner("Collecting small unindexed files ..."):
                            paths = watchlist_usecase.list_quick_win_paths(
                                pref, limit=cap, max_size_bytes=102_400
                            )
                        if not paths:
                            st.info("No small unindexed files found.")
                        else:
                            with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                                result = watchlist_usecase.queue_ingest(
                                    paths, mode="ingest"
                                )
                                st.session_state["ingest_tasks"] = add_records(
                                    st.session_state.get("ingest_tasks"),
                                    paths,
                                    result.task_ids,
                                    action="ingest",
                                )
                            if result.errors:
                                st.warning("Some ingestion steps failed.")
                            st.success(
                                f"Queued {len(result.task_ids)} quick-win file(s) for ingestion."
                            )

# Small task panel to track queued work from this page
should_rerun, updated = render_task_panel(st.session_state.get("ingest_tasks", []))
if should_rerun:
    st.session_state["ingest_tasks"] = updated
    st.rerun()

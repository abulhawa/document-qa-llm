import streamlit as st
from config import WATCH_INVENTORY_INDEX, WATCHLIST_INDEX
from utils.inventory import (
    seed_watch_inventory_from_fulltext,
    seed_inventory_indexed_chunked_count,
    count_watch_inventory_remaining,
    count_watch_inventory_total,
    list_watch_inventory_unindexed_paths,
    list_watch_inventory_unindexed_paths_all,
    list_watch_inventory_unindexed_paths_simple,
    scan_watch_inventory_for_prefix,
    count_watch_inventory_unindexed_quick_wins,
    count_watch_inventory_unindexed_missing_size,
    list_watch_inventory_unindexed_paths_filtered,
    list_watch_inventory_unindexed_quick_wins,
)
from ui.ingest_client import enqueue_paths
from ui.task_status import add_records
from components.task_panel import render_task_panel
from utils.opensearch.indexes import ensure_index_exists
from utils.watchlist import (
    get_watchlist_prefixes,
    add_watchlist_prefix,
    remove_watchlist_prefix,
    get_watchlist_meta,
    update_watchlist_stats,
    update_watchlist_scan_stats,
)

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Watchlist", layout="wide")

ensure_index_exists(index=WATCH_INVENTORY_INDEX)

st.title("Watchlist")
st.caption("Track folders and see how many files still need indexing. Use the actions to import known files and update counts.")

# Load persisted watchlist prefixes
ensure_index_exists(index=WATCHLIST_INDEX)
watched = st.session_state.setdefault("watched_prefixes", get_watchlist_prefixes())
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
                ok = add_watchlist_prefix(p)
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
            total_fulltext = 0
            total_chunks = 0
            for pref in list(watched):
                try:
                    total_fulltext += seed_watch_inventory_from_fulltext(pref)
                    total_chunks += seed_inventory_indexed_chunked_count(pref)
                except Exception:
                    pass
            st.success(
                f"Refreshed all. Imported known files: {total_fulltext}, updated chunk counts: {total_chunks}."
            )

if not watched:
    st.info("No tracked folders yet. Add one above.")
else:
    for pref in list(watched):
        with st.container(border=True):
            st.subheader(pref)
            # Show current remaining for context
            remaining_now = 0
            try:
                remaining_now = count_watch_inventory_remaining(pref)
                st.metric("Unindexed files", remaining_now)
            except Exception:
                pass
            # Inline quick-win count for clarity
            try:
                qw = count_watch_inventory_unindexed_quick_wins(pref, 102_400)
                st.caption(f"Quick wins (<=100KB): {qw}")
            except Exception:
                pass
            # Bold summaries and progress
            try:
                total_now = count_watch_inventory_total(pref)
                indexed_now = max(0, total_now - remaining_now)
                pct = (indexed_now / total_now) if total_now else 0
                st.progress(pct, text=f"Indexed {indexed_now} / {total_now}")
                meta = get_watchlist_meta(pref)
                last_idx = int(meta.get("last_indexed", 0) or 0)
                last_tot = int(meta.get("last_total", 0) or 0)
                last_ref = meta.get("last_refreshed") or ""
                delta = indexed_now - last_idx
                if last_ref:
                    st.caption(f"Last refreshed: {last_ref}  •  Indexed Δ: {delta:+}")
                last_scan = meta.get("last_scanned") or ""
                if last_scan:
                    found = int(meta.get("last_scan_found", 0) or 0)
                    missing = int(meta.get("last_scan_marked_missing", 0) or 0)
                    st.caption(f"Last scan: {last_scan}  •  Found: {found}  •  Marked missing: {missing}")
            except Exception:
                pass
            # Preview a few unindexed file paths
            try:
                preview = list_watch_inventory_unindexed_paths(pref, size=10)
                if preview:
                    st.caption("Sample of unindexed files")
                    st.table({"Path": preview})
            except Exception:
                pass
            c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1, 1])
            with c0:
                if st.button(
                    "Refresh folder status",
                    key=f"seed-{pref}",
                    help="Import known files and chunk counts from existing indexes, then update the unindexed count.",
                ):
                    with st.spinner("Refreshing status ..."):
                        n1 = seed_watch_inventory_from_fulltext(pref)
                        n2 = seed_inventory_indexed_chunked_count(pref)
                        total_now = count_watch_inventory_total(pref)
                        r = count_watch_inventory_remaining(pref)
                        indexed_now = max(0, total_now - r)
                        update_watchlist_stats(pref, total_now, indexed_now, r)
                    st.success(f"Imported known files: {n1}, updated chunk counts: {n2}. Unindexed now: {r}")
            with c1:
                if st.button(
                    "Scan disk for changes",
                    key=f"scan-{pref}",
                    help="Walk the folder on disk to detect new/removed PDF, DOCX and TXT files and update inventory.",
                ):
                    with st.spinner("Scanning disk ..."):
                        summary = scan_watch_inventory_for_prefix(pref)
                        # Update stats after scan
                        total_now = count_watch_inventory_total(pref)
                        r = count_watch_inventory_remaining(pref)
                        indexed_now = max(0, total_now - r)
                        update_watchlist_stats(pref, total_now, indexed_now, r)
                        update_watchlist_scan_stats(
                            pref,
                            found=int(summary.get("found", 0)),
                            marked_missing=int(summary.get("marked_missing", 0)),
                        )
                    st.success(
                        f"Scan complete. Found: {summary.get('found',0)}, marked missing: {summary.get('marked_missing',0)}."
                    )
            with c1:
                if st.button(
                    "Check unindexed",
                    key=f"rem-{pref}",
                    help="Count files in this folder that are not yet indexed.",
                ):
                    with st.spinner("Computing ..."):
                        r = count_watch_inventory_remaining(pref)
                    st.success(f"Remaining: {r}")
            with c2:
                if st.button(
                    "Import known files",
                    key=f"sft-{pref}",
                    help="Add files already present in the File Index into the watch inventory, marking them as indexed.",
                ):
                    with st.spinner("Importing known files ..."):
                        n = seed_watch_inventory_from_fulltext(pref)
                    st.success(f"Imported {n} files from index")
            with c3:
                if st.button(
                    "Import chunk counts",
                    key=f"scc-{pref}",
                    help="Record how many chunks each file has in the index (advanced).",
                ):
                    with st.spinner("Importing chunk counts ..."):
                        n = seed_inventory_indexed_chunked_count(pref)
                    st.success(f"Updated chunk counts for {n} files")
            with c4:
                if st.button(
                    "Stop tracking",
                    key=f"rm-{pref}",
                    help="Remove this folder from your watchlist.",
                ):
                    if remove_watchlist_prefix(pref):
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
                        try:
                            seed_watch_inventory_from_fulltext(pref)
                            seed_inventory_indexed_chunked_count(pref)
                        except Exception:
                            pass
                    with st.spinner("Collecting unindexed files ..."):
                        # Prefer simple fetch; fallback to scroll if needed
                        paths = list_watch_inventory_unindexed_paths_simple(
                            pref, limit=2000
                        )
                        if not paths:
                            paths = list_watch_inventory_unindexed_paths_all(
                                pref, limit=2000
                            )
                    if not paths:
                        st.info("No unindexed files found under this folder.")
                    else:
                        with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                            task_ids = enqueue_paths(paths, mode="ingest")
                            st.session_state["ingest_tasks"] = add_records(
                                st.session_state.get("ingest_tasks"),
                                paths,
                                task_ids,
                                action="ingest",
                            )
                        st.success(f"Queued {len(task_ids)} file(s) for ingestion.")

            # Smart re-ingest (changed files)
            with st.expander("Smart re-ingest (changed files)", expanded=False):
                if st.button(
                    "Find and re-ingest changed",
                    key=f"reingest-{pref}",
                    help="Re-ingest files whose modified time is newer than last indexed, or whose chunk counts drifted.",
                ):
                    from utils.inventory import list_inventory_paths_needing_reingest
                    with st.spinner("Finding changed files ..."):
                        re_files = list_inventory_paths_needing_reingest(pref, limit=2000)
                    if not re_files:
                        st.info("No changed files found under this folder.")
                    else:
                        with st.spinner(f"Queuing {len(re_files)} files for re-ingest ..."):
                            task_ids = enqueue_paths(re_files, mode="reingest")
                            st.session_state["ingest_tasks"] = add_records(
                                st.session_state.get("ingest_tasks"),
                                re_files,
                                task_ids,
                                action="reingest",
                            )
                        st.success(f"Queued {len(task_ids)} file(s) for re-ingestion.")

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
                        total_unidx = count_watch_inventory_remaining(pref)
                        st.info(f"Unindexed total: {total_unidx}. Cap will queue up to {min(cap, total_unidx)}.")
                with cB:
                    if st.button(
                        "Queue unindexed",
                        key=f"queue-unindexed-{pref}",
                        help="Enqueue up to the cap of unindexed files under this folder.",
                    ):
                        with st.spinner("Collecting unindexed files ..."):
                            paths = list_watch_inventory_unindexed_paths_simple(
                                pref, limit=cap
                            )
                            if not paths:
                                paths = list_watch_inventory_unindexed_paths_all(
                                    pref, limit=cap
                                )
                        if not paths:
                            st.info("No unindexed files found under this folder.")
                        else:
                            with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                                task_ids = enqueue_paths(paths, mode="ingest")
                                st.session_state["ingest_tasks"] = add_records(
                                    st.session_state.get("ingest_tasks"),
                                    paths,
                                    task_ids,
                                    action="ingest",
                                )
                            st.success(f"Queued {len(task_ids)} file(s) for ingestion.")
                with cC:
                    if st.button(
                        "Preview quick wins (<=100KB)",
                        key=f"preview-quick-{pref}",
                        help="Count unindexed files up to 100 KB; respects the cap in the next step.",
                    ):
                        cnt = count_watch_inventory_unindexed_quick_wins(pref, 102_400)
                        missing_sz = count_watch_inventory_unindexed_missing_size(pref)
                        if missing_sz:
                            st.info(
                                f"Quick wins total (<=100KB): {cnt}. Cap will queue up to {min(cap, cnt)}. "
                                f"Note: {missing_sz} unindexed file(s) missing size; run 'Scan disk for changes' to populate sizes."
                            )
                        else:
                            st.info(
                                f"Quick wins total (<=100KB): {cnt}. Cap will queue up to {min(cap, cnt)}."
                            )
                with cD:
                    if st.button(
                        "Queue quick wins (<=100KB)",
                        key=f"queue-quick-{pref}",
                        help="Enqueue up to the cap of small unindexed files (<= 100 KB).",
                    ):
                        with st.spinner("Collecting small unindexed files ..."):
                            paths = list_watch_inventory_unindexed_quick_wins(
                                pref, limit=cap, max_size_bytes=102_400
                            )
                        if not paths:
                            st.info("No small unindexed files found.")
                        else:
                            with st.spinner(f"Queuing {len(paths)} files for ingestion ..."):
                                task_ids = enqueue_paths(paths, mode="ingest")
                                st.session_state["ingest_tasks"] = add_records(
                                    st.session_state.get("ingest_tasks"),
                                    paths,
                                    task_ids,
                                    action="ingest",
                                )
                            st.success(f"Queued {len(task_ids)} quick-win file(s) for ingestion.")

# Small task panel to track queued work from this page
should_rerun, updated = render_task_panel(st.session_state.get("ingest_tasks", []))
if should_rerun:
    st.session_state["ingest_tasks"] = updated
    st.rerun()

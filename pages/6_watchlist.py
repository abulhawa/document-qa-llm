import streamlit as st
from config import WATCH_INVENTORY_INDEX, WATCHLIST_INDEX
from utils.inventory import (
    seed_watch_inventory_from_fulltext,
    seed_inventory_indexed_chunked_count,
    count_watch_inventory_remaining,
)
from utils.opensearch.indexes import ensure_index_exists
from utils.watchlist import (
    get_watchlist_prefixes,
    add_watchlist_prefix,
    remove_watchlist_prefix,
)

ensure_index_exists(index=WATCH_INVENTORY_INDEX)

st.title("Watch Inventory")
st.caption("Track folders and see how many files still need indexing. Use the actions to import known files and update counts.")

# Load persisted watchlist prefixes
ensure_index_exists(index=WATCHLIST_INDEX)
watched = st.session_state.setdefault("watched_prefixes", get_watchlist_prefixes())

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

if not watched:
    st.info("No tracked folders yet. Add one above.")
else:
    for pref in list(watched):
        with st.container(border=True):
            st.subheader(pref)
            # Show current remaining for context
            try:
                remaining_now = count_watch_inventory_remaining(pref)
                st.metric("Unindexed files", remaining_now)
            except Exception:
                pass
            c0, c1, c2, c3, c4 = st.columns([1, 1, 1, 1, 1])
            with c0:
                if st.button(
                    "Refresh folder status",
                    key=f"seed-{pref}",
                    help="Import known files and chunk counts from existing indexes, then update the unindexed count.",
                ):
                    with st.spinner("Refreshing status ..."):
                        n1 = seed_watch_inventory_from_fulltext(pref)
                        n2 = seed_inventory_indexed_chunked_count(pref)
                        r = count_watch_inventory_remaining(pref)
                    st.success(f"Imported known files: {n1}, updated chunk counts: {n2}. Unindexed now: {r}")
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


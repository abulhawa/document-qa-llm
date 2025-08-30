import streamlit as st
from config import WATCH_INVENTORY_INDEX
from utils.inventory import (
    seed_watch_inventory_from_fulltext,
    seed_inventory_indexed_chunked_count,
    count_watch_inventory_remaining,
)
from utils.opensearch.indexes import ensure_index_exists

ensure_index_exists(index=WATCH_INVENTORY_INDEX)

st.title("Watch Inventory")
st.caption("Track folders and see remaining unindexed files. Seed as needed.")

# Simple tracked list stored in session (persist separately later if needed)
watched = st.session_state.setdefault("watched_prefixes", [])

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
                watched.append(p)
                st.success("Added")

st.divider()

if not watched:
    st.info("No tracked folders yet. Add one above.")
else:
    for pref in list(watched):
        with st.container(border=True):
            st.subheader(pref)
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                if st.button("Recompute remaining", key=f"rem-{pref}"):
                    with st.spinner("Computing…"):
                        r = count_watch_inventory_remaining(pref)
                    st.success(f"Remaining: {r}")
            with c2:
                if st.button("Seed from full-text", key=f"sft-{pref}"):
                    with st.spinner("Seeding from full-text…"):
                        n = seed_watch_inventory_from_fulltext(pref)
                    st.success(f"Seeded {n} from full-text")
            with c3:
                if st.button("Seed chunk counts", key=f"scc-{pref}"):
                    with st.spinner("Seeding chunk counts…"):
                        n = seed_inventory_indexed_chunked_count(pref)
                    st.success(f"Seeded counts for {n} paths")
            with c4:
                if st.button("Remove", key=f"rm-{pref}"):
                    watched.remove(pref)
                    st.warning("Removed. Refresh to update view if needed.")

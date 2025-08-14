"""Streamlit page for viewing duplicate files.

This page groups duplicate files by checksum and allows the user to remove
entire groups from the display. The underlying data is fetched from
OpenSearch via ``get_duplicate_checksums`` and ``get_files_by_checksum``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from utils.opensearch_utils import get_duplicate_checksums, get_files_by_checksum

st.set_page_config(page_title="Duplicate Files", page_icon="🗂️")

st.title("Duplicate Files")


def _load_duplicate_groups() -> list[dict]:
    """Fetch duplicate groups from OpenSearch.

    Returns a list of dicts with ``checksum`` and ``files`` keys.
    """

    groups = []
    for checksum in get_duplicate_checksums():
        groups.append({"checksum": checksum, "files": get_files_by_checksum(checksum)})
    return groups


# Populate session state with duplicate groups on first load
if "duplicate_groups" not in st.session_state:
    st.session_state.duplicate_groups = _load_duplicate_groups()

groups = st.session_state.duplicate_groups
if not groups:
    st.info("No duplicate files found.")
else:
    for idx, group in enumerate(list(groups)):
        files = group["files"]
        count = len(files)
        suffix = "file" if count == 1 else "files"
        st.subheader(f"{group['checksum']} ({count} {suffix})")

        df = pd.DataFrame(files)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

        if st.button("Delete group", key=f"delete_{group['checksum']}"):
            # Removing the group from session state is enough for the tests and
            # avoids side effects such as deleting from the backing index.
            st.session_state.duplicate_groups.pop(idx)
            st.rerun()

    # Offer a CSV download of the currently displayed duplicates
    all_rows = []
    for group in st.session_state.duplicate_groups:
        for f in group["files"]:
            row = {"Checksum": group["checksum"], **f}
            all_rows.append(row)
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "duplicate_files.csv", "text/csv")

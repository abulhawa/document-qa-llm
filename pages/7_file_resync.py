import os
from collections import Counter
from typing import List

import pandas as pd
import streamlit as st

from core.sync.file_resync import (
    DEFAULT_ALLOWED_EXTENSIONS,
    apply_updates,
    fetch_index_state,
    reconcile,
    scan_disk,
)

st.set_page_config(page_title="File Path Re-Sync", layout="wide")
st.title("🔁 File Path Re-Sync")

DEFAULT_ROOT = os.getenv("LOCAL_SYNC_ROOT", "")


def _parse_roots(raw: str) -> List[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _parse_exts(raw: str) -> set[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    exts = {p if p.startswith(".") else f".{p}" for p in parts}
    return exts or set(DEFAULT_ALLOWED_EXTENSIONS)


def _render_summary(rows: List[dict]) -> None:
    counts = Counter([r.get("status", "") for r in rows])
    cols = st.columns(len(counts) or 1)
    for col, (status, cnt) in zip(cols, counts.items()):
        col.metric(status.replace("_", " ").title(), cnt)


def _render_table(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No results to show yet. Run a scan to populate this table.")
        return df

    status_options = [
        "moved",
        "unchanged",
        "missing_on_disk",
        "new_untracked",
        "conflict",
    ]
    default_selection = [s for s in status_options if s in df["status"].unique()]
    selected = st.multiselect(
        "Filter by status", options=status_options, default=default_selection
    )
    if selected:
        df = df[df["status"].isin(selected)]

    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv_data, file_name="file_resync.csv")
    return df


with st.expander("Instructions", expanded=False):
    st.markdown(
        """
        * Scan configured roots for files and match them against indexed checksums.
        * Conflicts (same checksum on multiple paths) are never updated automatically.
        * Dry-run is enabled by default; uncheck it only after reviewing the table.
        """
    )

roots_input = st.text_area(
    "Sync roots (one per line)",
    value=DEFAULT_ROOT,
    placeholder="/path/to/drive/root",
)
ext_input = st.text_input(
    "Allowed extensions (comma-separated)",
    value=", ".join(sorted(DEFAULT_ALLOWED_EXTENSIONS)),
)
dry_run = st.checkbox("Dry-run", value=True)
confirm_apply = st.checkbox(
    "I have reviewed the report and want to apply metadata updates.", value=False
)

col1, col2 = st.columns([1, 1])
with col1:
    scan_clicked = st.button("Scan Files", use_container_width=True)
with col2:
    apply_clicked = st.button(
        "Apply Updates", use_container_width=True, disabled="file_resync_rows" not in st.session_state
    )

if scan_clicked:
    roots = _parse_roots(roots_input)
    exts = _parse_exts(ext_input)
    if not roots:
        st.error("Please provide at least one root to scan.")
    else:
        with st.spinner("Scanning disk and indexes…"):
            index_state = fetch_index_state()
            disk_state = scan_disk(roots, exts)
            rows = reconcile(index_state, disk_state)
        st.session_state["file_resync_rows"] = rows
        st.success(f"Scan complete. {len(rows)} row(s) ready.")

rows = st.session_state.get("file_resync_rows", [])
if rows:
    st.subheader("Reconciliation Report")
    _render_summary(rows)
    filtered_df = _render_table(rows)

if apply_clicked and rows:
    st.warning("Updates will touch metadata only; conflicts and non-moved rows are skipped.")
    if confirm_apply:
        with st.spinner("Applying path updates…"):
            summary = apply_updates(rows, dry_run=dry_run)
        st.success("Done." if not summary.get("errors") else "Completed with errors.")
        st.json(summary)
    else:
        st.info("Enable the confirmation checkbox to apply updates.")

import os
from typing import List

import pandas as pd
import streamlit as st

from core.sync.file_resync import (
    DEFAULT_ALLOWED_EXTENSIONS,
    ApplyOptions,
    apply_plan,
    build_reconciliation_plan,
    scan_files,
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


def _render_summary(counts: dict) -> None:
    counts = {k: v for k, v in counts.items() if v}
    if not counts:
        st.info("No plan items yet. Run a scan to populate this view.")
        return
    cols = st.columns(len(counts))
    for col, (bucket, cnt) in zip(cols, counts.items()):
        col.metric(bucket.title(), cnt)


def _render_table(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No results to show yet. Run a scan to populate this table.")
        return df

    bucket_options = ["SAFE", "REVIEW", "BLOCKED", "INFO"]
    default_selection = [s for s in bucket_options if s in df["bucket"].unique()]
    selected = st.multiselect(
        "Filter by bucket", options=bucket_options, default=default_selection
    )
    if selected:
        df = df[df["bucket"].isin(selected)]

    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv_data, file_name="file_resync.csv")
    return df


with st.expander("Workflow & Safety", expanded=False):
    st.markdown(
        """
        **Phase A: Scan & Plan** (dry run only)
        - Builds a plan without changing data.

        **Phase B: Apply SAFE actions**
        - Canonical/alias updates that are unambiguous.
        - Optional ingestion of clearly missing files.
        - No deletions.

        **Phase C: Apply destructive actions**
        - REVIEW + SAFE buckets.
        - Optional orphan cleanup and retire-on-replace.
        - Use with care; vectors/chunks/full-text may be deleted.

        *Orphaned content* = indexed content whose canonical path **and** all aliases are missing on disk within scanned roots.
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

st.subheader("Apply Options")
opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
with opt_col1:
    ingest_missing_safe = st.checkbox("Ingest missing (SAFE phase)", value=False)
with opt_col2:
    ingest_missing_destructive = st.checkbox("Ingest missing (Destructive phase)", value=False)
with opt_col3:
    delete_orphaned = st.checkbox("Delete orphaned content (Destructive)", value=False)
with opt_col4:
    retire_replaced = st.checkbox("Retire replaced content (Destructive)", value=False)

col1, col2 = st.columns([1, 1])
with col1:
    scan_clicked = st.button("Scan & Plan", use_container_width=True)
with col2:
    apply_safe_clicked = st.button(
        "Apply SAFE actions",
        use_container_width=True,
        disabled="file_resync_plan" not in st.session_state,
    )

apply_destructive_clicked = st.button(
    "Apply destructive actions",
    use_container_width=True,
    disabled="file_resync_plan" not in st.session_state,
)

if scan_clicked:
    roots = _parse_roots(roots_input)
    exts = _parse_exts(ext_input)
    if not roots:
        st.error("Please provide at least one root to scan.")
    else:
        with st.spinner("Scanning disk and building plan…"):
            scan_result = scan_files(roots, exts)
            plan = build_reconciliation_plan(scan_result, roots, retire_replaced_content=retire_replaced)
        st.session_state["file_resync_plan"] = plan
        st.session_state["file_resync_scan_meta"] = {
            "ignored": scan_result.ignored_files,
            "scanned_roots": scan_result.scanned_roots_successful,
            "failed_roots": scan_result.scanned_roots_failed,
        }
        st.success(
            f"Scan complete. {len(plan.items)} plan item(s) found across {len(plan.counts)} buckets."
        )

plan = st.session_state.get("file_resync_plan")
scan_meta = st.session_state.get("file_resync_scan_meta", {})
if plan:
    st.subheader("Reconciliation Report")
    if scan_meta:
        st.info(
            f"Roots scanned: {', '.join(scan_meta.get('scanned_roots', [])) or 'None'}; "
            f"failed roots: {', '.join(scan_meta.get('failed_roots', [])) or 'None'}; "
            f"ignored temp/small files: {scan_meta.get('ignored', 0)}"
        )
    _render_summary(plan.counts)
    rows = plan.as_rows()
    filtered_df = _render_table(rows)

if apply_safe_clicked and plan:
    with st.spinner("Applying SAFE actions…"):
        result = apply_plan(
            plan,
            ApplyOptions(
                ingest_missing=ingest_missing_safe,
                apply_safe_only=True,
                delete_orphaned=False,
                retire_replaced_content=False,
            ),
        )
    st.success("SAFE actions completed." if not result.errors else "SAFE actions completed with warnings.")
    st.json(
        {
            "ingested": result.ingested,
            "updated_fulltext": result.updated_fulltext,
            "updated_chunks": result.updated_chunks,
            "updated_qdrant": result.updated_qdrant,
            "deleted_checksums": result.deleted_checksums,
            "errors": result.errors,
        }
    )

if apply_destructive_clicked and plan:
    with st.spinner("Applying destructive actions…"):
        result = apply_plan(
            plan,
            ApplyOptions(
                ingest_missing=ingest_missing_destructive,
                apply_safe_only=False,
                delete_orphaned=delete_orphaned,
                retire_replaced_content=retire_replaced,
            ),
        )
    st.success("Destructive actions completed." if not result.errors else "Destructive actions completed with warnings.")
    st.json(
        {
            "ingested": result.ingested,
            "updated_fulltext": result.updated_fulltext,
            "updated_chunks": result.updated_chunks,
            "updated_qdrant": result.updated_qdrant,
            "deleted_checksums": result.deleted_checksums,
            "errors": result.errors,
        }
    )

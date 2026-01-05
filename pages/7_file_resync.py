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
from ui.ingestion_ui import run_root_picker

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="File Path Re-Sync", layout="wide")
st.title("🔁 File Path Re-Sync")

DEFAULT_ROOT = os.getenv("LOCAL_SYNC_ROOT", "")
REASON_ORDER = [
    "DUPLICATE_INDEX_DOCS",
    "NOT_INDEXED",
    "ADD_ALIAS",
    "REMOVE_ALIAS",
    "SET_CANONICAL",
    "CANONICAL_AMBIGUOUS",
    "ORPHANED_INDEX_CONTENT",
    "PATH_REPLACED",
    "MIXED",
]
REASON_DETAIL_MAP = {
    "DUPLICATE_INDEX_DOCS": "Multiple full-text docs share the same checksum.",
    "NOT_INDEXED": "File exists on disk but is missing from the index.",
    "ADD_ALIAS": "Disk path exists that is not in aliases yet.",
    "REMOVE_ALIAS": "Alias path is missing on disk within scanned roots.",
    "SET_CANONICAL": (
        "Canonical path missing; auto-selected via shortest path, then newest mtime, then first."
    ),
    "CANONICAL_AMBIGUOUS": "Legacy state; canonical selection is now automatic.",
    "ORPHANED_INDEX_CONTENT": "Indexed content has no disk paths under scanned roots.",
    "PATH_REPLACED": "Canonical path now points to a different checksum.",
    "MIXED": "Multiple reasons apply; review actions and details.",
}
REASON_ACTION_MAP = {
    "DUPLICATE_INDEX_DOCS": "Blocked: dedupe full-text docs before applying.",
    "NOT_INDEXED": "Optional ingest (checkbox in Apply phase).",
    "ADD_ALIAS": "Apply: add alias path.",
    "REMOVE_ALIAS": "Apply: remove alias path.",
    "SET_CANONICAL": "Apply: set canonical using auto-selection rules.",
    "CANONICAL_AMBIGUOUS": "No longer used; auto-selection applies.",
    "ORPHANED_INDEX_CONTENT": "Optional delete (Destructive checkbox).",
    "PATH_REPLACED": "Manual review; optional retire replaced content.",
    "MIXED": "Review actions list for applyable steps.",
}


def _friendly_service_error(exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    name = exc.__class__.__name__.lower()
    if "connection" in name or "connection" in lowered or "remotedisconnected" in lowered:
        return "Index services are unavailable. Start OpenSearch and Qdrant, then retry."
    if "opensearch" in lowered:
        return "OpenSearch is unavailable. Start OpenSearch, then retry."
    if "qdrant" in lowered:
        return "Qdrant is unavailable. Start Qdrant, then retry."
    return f"Failed to contact indexing services: {msg}"


def _render_service_error(exc: Exception, action: str) -> None:
    st.error(f"{action} failed. {_friendly_service_error(exc)}")
    st.caption(f"Error details: {exc.__class__.__name__}: {exc}")


def _parse_exts(raw: str) -> set[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    exts = {p if p.startswith(".") else f".{p}" for p in parts}
    return exts or set(DEFAULT_ALLOWED_EXTENSIONS)


def _split_reasons(reason: str) -> List[str]:
    if not reason:
        return []
    parts = [p.strip() for p in str(reason).split(";") if p.strip()]
    return parts or []


def _map_reason_details(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_DETAIL_MAP["MIXED"]
    mapped = [REASON_DETAIL_MAP.get(p, f"Unknown reason: {p}") for p in parts]
    return "; ".join(dict.fromkeys(mapped))


def _map_reason_actions(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_ACTION_MAP["MIXED"]
    mapped = [REASON_ACTION_MAP.get(p, f"Manual review for {p}") for p in parts]
    return "; ".join(dict.fromkeys(mapped))


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

    df = df.copy()
    df["reason_detail"] = df["reason"].apply(_map_reason_details)
    df["apply_action"] = df["reason"].apply(_map_reason_actions)
    preferred_cols = [
        "bucket",
        "reason",
        "reason_detail",
        "apply_action",
        "checksum",
        "content_id",
        "indexed_paths",
        "disk_paths",
        "actions",
        "new_checksum",
        "explanation",
    ]
    df = df[[c for c in preferred_cols if c in df.columns]]

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

with st.expander("Reason Map (Mismatch -> Apply Action)", expanded=False):
    mapping_rows = [
        {
            "Reason": reason,
            "Meaning": REASON_DETAIL_MAP.get(reason, ""),
            "Apply action": REASON_ACTION_MAP.get(reason, ""),
        }
        for reason in REASON_ORDER
    ]
    st.table(pd.DataFrame(mapping_rows))

if "file_resync_roots" not in st.session_state:
    st.session_state["file_resync_roots"] = [DEFAULT_ROOT] if DEFAULT_ROOT else []

roots_col1, roots_col2 = st.columns([1, 1], gap="small")
with roots_col1:
    if st.button("Select Folder Root"):
        picked = run_root_picker()
        if picked:
            current = st.session_state.get("file_resync_roots", [])
            merged = list(dict.fromkeys(current + picked))
            st.session_state["file_resync_roots"] = merged
with roots_col2:
    if st.button("Clear Roots"):
        st.session_state["file_resync_roots"] = []

roots = st.session_state.get("file_resync_roots", [])
if roots:
    st.dataframe(
        pd.DataFrame({"Sync roots": roots}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No roots selected yet. Use the folder picker to add one.")
ext_input = st.text_input(
    "Allowed extensions (comma-separated)",
    value=", ".join(sorted(DEFAULT_ALLOWED_EXTENSIONS)),
)

st.subheader("Plan & Apply Actions")
phase_col1, phase_col2, phase_col3 = st.columns(3)

with phase_col1:
    st.markdown("**Step 1: Scan & Plan**")
    st.caption("Dry run that builds the reconciliation plan based on the options below.")
    scan_clicked = st.button("Scan & Plan", use_container_width=True)

with phase_col2:
    st.markdown("**Step 2: Apply SAFE actions**")
    st.caption("Unambiguous updates only; no deletions.")
    ingest_missing_safe = st.checkbox("Ingest missing (SAFE phase)", value=False)
    apply_safe_clicked = st.button(
        "Apply SAFE actions",
        use_container_width=True,
        disabled="file_resync_plan" not in st.session_state,
    )

with phase_col3:
    st.markdown("**Step 3: Apply destructive actions**")
    st.caption("Includes REVIEW bucket; may delete content.")
    ingest_missing_destructive = st.checkbox("Ingest missing (Destructive phase)", value=False)
    delete_orphaned = st.checkbox("Delete orphaned content (Destructive)", value=False)
    retire_replaced = st.checkbox("Retire replaced content (Destructive)", value=False)
    apply_destructive_clicked = st.button(
        "Apply destructive actions",
        use_container_width=True,
        disabled="file_resync_plan" not in st.session_state,
    )

if scan_clicked:
    exts = _parse_exts(ext_input)
    if not roots:
        st.error("Please provide at least one root to scan.")
    else:
        try:
            with st.spinner("Scanning disk and building plan…"):
                scan_result = scan_files(roots, exts)
                plan = build_reconciliation_plan(
                    scan_result, roots, retire_replaced_content=retire_replaced
                )
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("file_resync_plan", None)
            st.session_state.pop("file_resync_scan_meta", None)
            _render_service_error(e, "Scan & plan")
            st.stop()
        else:
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
    try:
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
    except Exception as e:  # noqa: BLE001
        _render_service_error(e, "SAFE apply")
        st.stop()
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
    try:
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
    except Exception as e:  # noqa: BLE001
        _render_service_error(e, "Destructive apply")
        st.stop()
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

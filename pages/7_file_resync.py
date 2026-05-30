"""Streamlit UI for the File Path Re-Sync reconciliation workflow.

This page is a thin UI over ``app.usecases.file_resync_usecase``. It keeps the
workflow explicit: configure a scan, review the dry-run plan, then apply
automatic or reviewed changes with separate controls.
"""

import os
import uuid
from typing import List, cast

import pandas as pd
import streamlit as st

from app.schemas import FileResyncApplyRequest, FileResyncPlanItem, FileResyncScanRequest
from config import logger
from ui.ingestion_ui import run_root_picker
from utils.timing import set_run_id, timed_block

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="File Path Re-Sync", layout="wide")
st.title("🔁 File Path Re-Sync")

st.caption(
    "Find files that moved, disappeared, were copied, or were replaced after indexing. "
    "Scan first; nothing changes until you apply a plan."
)

DEFAULT_ROOT = os.getenv("LOCAL_SYNC_ROOT", "")
DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MIN_INGEST_BYTES = 1024
TEMP_PREFIXES = ("~$",)
TEMP_SUFFIXES = (".tmp", ".temp", ".swp", ".swx", ".part")
IGNORE_DIR_NAMES = {".git", ".obsidian", ".cache", "node_modules", "__pycache__"}
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
        "Canonical path missing; single disk replacement can be applied automatically."
    ),
    "CANONICAL_AMBIGUOUS": "Canonical path missing but multiple disk paths exist.",
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
    "CANONICAL_AMBIGUOUS": "Review deterministic canonical candidate before applying.",
    "ORPHANED_INDEX_CONTENT": "Optional delete (Destructive checkbox).",
    "PATH_REPLACED": "Manual review; optional retire replaced content.",
    "MIXED": "Review actions list for applyable steps.",
}
BUCKET_LABEL_MAP = {
    "SAFE": "Automatic fix",
    "INFO": "New file",
    "REVIEW": "Needs review",
    "BLOCKED": "Blocked",
}
BUCKET_RISK_MAP = {
    "SAFE": "Low",
    "INFO": "Low",
    "REVIEW": "Medium",
    "BLOCKED": "High",
}
SUMMARY_LABEL_MAP = {
    "SAFE": "Automatic fixes",
    "INFO": "New files",
    "REVIEW": "Need review",
    "BLOCKED": "Blocked",
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


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


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


def _first_path(paths: List[str]) -> str:
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    return f"{paths[0]} (+{len(paths) - 1} more)"


def _friendly_change(item: FileResyncPlanItem) -> str:
    reasons = set(_split_reasons(item.reason))
    if "DUPLICATE_INDEX_DOCS" in reasons:
        return "The index has duplicate records for the same file content."
    if "NOT_INDEXED" in reasons:
        return "This file exists on disk but is not indexed yet."
    if "PATH_REPLACED" in reasons:
        return "An indexed path now points to different file content."
    if "ORPHANED_INDEX_CONTENT" in reasons:
        return "Indexed content no longer appears under the scanned roots."
    if "CANONICAL_AMBIGUOUS" in reasons:
        return "The main indexed path is missing and multiple replacements exist."
    if "SET_CANONICAL" in reasons:
        return "The main indexed path is missing and one replacement was found."
    if "ADD_ALIAS" in reasons and "REMOVE_ALIAS" in reasons:
        return "The file has moved or has changed duplicate locations."
    if "ADD_ALIAS" in reasons:
        return "The same file content exists at an additional disk location."
    if "REMOVE_ALIAS" in reasons:
        return "A previously known duplicate location is missing from disk."
    return item.explanation or "This item needs reconciliation."


def _friendly_action(item: FileResyncPlanItem) -> str:
    actions = {action.type for action in item.actions}
    if item.bucket == "BLOCKED":
        return "Fix duplicate index records before applying changes."
    if item.bucket == "INFO" and "INGEST_NEW" in actions:
        return "Enable ingest missing files, then apply automatic fixes."
    if "DELETE_CONTENT" in actions:
        return "Review carefully; deletion only runs when its checkbox is enabled."
    if "SET_CANONICAL" in actions and item.bucket == "REVIEW":
        return "Choose whether the suggested main path is correct before applying."
    if "SET_CANONICAL" in actions:
        return "Apply automatic fixes to update the main path."
    if "ADD_ALIAS" in actions or "REMOVE_ALIAS" in actions:
        return "Apply automatic fixes to update known locations."
    return "No automatic action is available."


def _render_summary(counts: dict) -> None:
    visible_counts = {k: int(counts.get(k, 0)) for k in ["SAFE", "INFO", "REVIEW", "BLOCKED"]}
    if not any(visible_counts.values()):
        st.info("No plan items yet. Run a scan to populate this view.")
        return
    cols = st.columns(4)
    for col, bucket in zip(cols, ["SAFE", "INFO", "REVIEW", "BLOCKED"]):
        col.metric(SUMMARY_LABEL_MAP[bucket], visible_counts[bucket])


def _next_step_message(counts: dict) -> tuple[str, str]:
    blocked = int(counts.get("BLOCKED", 0))
    review = int(counts.get("REVIEW", 0))
    safe = int(counts.get("SAFE", 0))
    info = int(counts.get("INFO", 0))
    if blocked:
        return (
            "warning",
            "Some items are blocked. Open technical details and fix duplicate index records before applying changes.",
        )
    if safe or info:
        return (
            "success",
            "Start with automatic fixes. They do not delete content; new files are ingested only if you enable that option.",
        )
    if review:
        return (
            "warning",
            "Only reviewed changes were found. Read the rows carefully before enabling any delete options.",
        )
    return ("info", "No changes are needed for the scanned roots.")


def _plan_items_to_rows(items: List[FileResyncPlanItem]) -> List[dict]:
    return [
        {
            "status": BUCKET_LABEL_MAP.get(item.bucket, item.bucket),
            "risk": BUCKET_RISK_MAP.get(item.bucket, "Unknown"),
            "what_changed": _friendly_change(item),
            "recommended_action": _friendly_action(item),
            "indexed_location": _first_path(item.indexed_paths),
            "disk_location": _first_path(item.disk_paths),
            "bucket": item.bucket,
            "reason": item.reason,
            "checksum": item.checksum,
            "content_id": item.content_id,
            "indexed_paths": "; ".join(item.indexed_paths),
            "disk_paths": "; ".join(item.disk_paths),
            "actions": ", ".join(sorted({action.type for action in item.actions})),
            "explanation": item.explanation,
            "new_checksum": item.new_checksum,
        }
        for item in items
    ]


def _render_table(rows: List[dict], controls_disabled: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No results to show yet. Run a scan to populate this table.")
        return df

    status_options = [
        BUCKET_LABEL_MAP[bucket]
        for bucket in ["SAFE", "INFO", "REVIEW", "BLOCKED"]
        if BUCKET_LABEL_MAP[bucket] in set(df["status"])
    ]
    selected = st.multiselect(
        "Filter by status",
        options=status_options,
        default=status_options,
        disabled=controls_disabled,
    )
    if selected:
        df = df.loc[df["status"].isin(selected)]

    df = df.copy()
    reason_series = cast(pd.Series, df["reason"])
    df["reason_detail"] = reason_series.apply(_map_reason_details)
    df["apply_action"] = reason_series.apply(_map_reason_actions)
    friendly_cols = [
        "status",
        "risk",
        "what_changed",
        "recommended_action",
        "indexed_location",
        "disk_location",
    ]
    technical_cols = [
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

    friendly_df = df[[c for c in friendly_cols if c in df.columns]]
    st.dataframe(friendly_df, width="stretch", hide_index=True)
    csv_data = friendly_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export current view",
        data=csv_data,
        file_name="file_resync.csv",
        disabled=controls_disabled,
    )

    show_technical = st.checkbox(
        "Show technical details",
        value=False,
        disabled=controls_disabled,
        help="Show checksums, raw reason codes, content IDs, and low-level actions.",
    )
    if show_technical:
        technical_df = df[[c for c in technical_cols if c in df.columns]]
        st.dataframe(technical_df, width="stretch", hide_index=True)
        st.download_button(
            "Export technical details",
            data=technical_df.to_csv(index=False).encode("utf-8"),
            file_name="file_resync_technical.csv",
            disabled=controls_disabled,
        )
    return cast(pd.DataFrame, friendly_df)


if "file_resync_roots" not in st.session_state:
    st.session_state["file_resync_roots"] = [DEFAULT_ROOT] if DEFAULT_ROOT else []

scan_clicked = False
apply_safe_clicked = False
apply_destructive_clicked = False
ingest_missing_safe = False
delete_orphaned = False
retire_replaced = False

step_cols = st.columns(3)
step_cols[0].markdown("**1. Choose folders**")
step_cols[0].caption("Pick the disk roots that should be checked.")
step_cols[1].markdown("**2. Review plan**")
step_cols[1].caption("The scan is a dry run and does not change indexes.")
step_cols[2].markdown("**3. Apply changes**")
step_cols[2].caption("Automatic fixes and reviewed changes are separate.")

st.subheader("Configure Scan")
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
    st.markdown("**Selected folders**")
    for root in roots:
        st.code(root, language=None)
else:
    st.info("No roots selected yet. Use the folder picker to add one.")
ext_input = st.text_input(
    "Allowed extensions (comma-separated)",
    value=", ".join(sorted(DEFAULT_ALLOWED_EXTENSIONS)),
)
with st.expander("Scan filters", expanded=False):
    min_ingest_bytes = st.number_input(
        "Minimum file size in bytes",
        min_value=0,
        value=MIN_INGEST_BYTES,
        step=256,
        help="Files smaller than this are ignored before checksum calculation.",
    )
    temp_prefixes_input = st.text_input(
        "Temporary filename prefixes",
        value=", ".join(TEMP_PREFIXES),
    )
    temp_suffixes_input = st.text_input(
        "Temporary filename suffixes",
        value=", ".join(TEMP_SUFFIXES),
    )
    ignore_dirs_input = st.text_input(
        "Ignored directory names",
        value=", ".join(sorted(IGNORE_DIR_NAMES)),
    )

scan_clicked = st.button("Scan & build plan", width="stretch")

if scan_clicked:
    exts = _parse_exts(ext_input)
    if not roots:
        st.error("Please provide at least one root to scan.")
    else:
        try:
            run_id = uuid.uuid4().hex[:8]
            st.session_state["_run_id"] = run_id
            set_run_id(run_id)
            with st.spinner("Scanning disk and building plan…"):
                with timed_block(
                    "action.file_resync.scan_plan",
                    extra={"run_id": run_id, "roots": len(roots), "extensions": sorted(exts)},
                    logger=logger,
                ):
                    from app.usecases.file_resync_usecase import scan_and_plan

                    plan, scan_meta = scan_and_plan(
                        FileResyncScanRequest(
                            roots=roots,
                            allowed_extensions=sorted(exts),
                            min_ingest_bytes=int(min_ingest_bytes),
                            temp_prefixes=_parse_csv(temp_prefixes_input),
                            temp_suffixes=_parse_csv(temp_suffixes_input),
                            ignore_dir_names=_parse_csv(ignore_dirs_input),
                        ),
                        retire_replaced_content=retire_replaced,
                    )
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("file_resync_plan", None)
            st.session_state.pop("file_resync_scan_meta", None)
            _render_service_error(e, "Scan & plan")
            st.stop()
        else:
            st.session_state["file_resync_plan"] = plan
            st.session_state["file_resync_scan_meta"] = scan_meta
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
    next_step_kind, next_step_text = _next_step_message(plan.counts)
    if next_step_kind == "success":
        st.success(next_step_text)
    elif next_step_kind == "warning":
        st.warning(next_step_text)
    else:
        st.info(next_step_text)
    st.caption(
        "Review the plain-language report first. Use technical details only when you need "
        "checksums, raw action codes, or index identifiers."
    )
    rows = _plan_items_to_rows(plan.items)
    filtered_df = _render_table(rows)

    st.subheader("Apply Changes")
    apply_col1, apply_col2 = st.columns(2, gap="large")
    with apply_col1:
        st.markdown("**Automatic fixes**")
        st.caption("Applies low-risk path and alias updates. Optionally ingests new files. Never deletes content.")
        ingest_missing_safe = st.checkbox("Also ingest new files", value=False)
        can_apply_automatic = bool(plan.counts.get("SAFE")) or (
            bool(plan.counts.get("INFO")) and ingest_missing_safe
        )
        apply_safe_clicked = st.button(
            "Apply automatic fixes",
            width="stretch",
            disabled=not can_apply_automatic,
        )

    with apply_col2:
        st.markdown("**Reviewed changes**")
        st.caption("Applies reviewed items. Deletions run only when explicitly enabled below.")
        delete_orphaned = st.checkbox("Delete orphaned index content")
        retire_replaced = st.checkbox("Retire replaced content")
        apply_destructive_clicked = st.button(
            "Apply reviewed changes",
            width="stretch",
            disabled=not bool(plan.counts.get("REVIEW")),
        )

    with st.expander("Technical reference", expanded=False):
        st.markdown(
            """
            - Scan & build plan is a dry run.
            - Automatic fixes update paths and aliases only; they never delete indexed content.
            - Reviewed changes may include deletions only when the matching checkbox is enabled.
            """
        )
        mapping_rows = [
            {
                "Reason": reason,
                "Meaning": REASON_DETAIL_MAP.get(reason, ""),
                "Apply action": REASON_ACTION_MAP.get(reason, ""),
            }
            for reason in REASON_ORDER
        ]
        st.table(pd.DataFrame(mapping_rows))

if apply_safe_clicked and plan:
    result = None
    try:
        run_id = uuid.uuid4().hex[:8]
        st.session_state["_run_id"] = run_id
        set_run_id(run_id)
        with st.spinner("Applying SAFE actions…"):
            with timed_block(
                "action.file_resync.apply_safe_actions",
                extra={"run_id": run_id, "ingest_missing": ingest_missing_safe},
                logger=logger,
            ):
                from app.usecases.file_resync_usecase import apply_plan as apply_resync_plan

                result = apply_resync_plan(
                    FileResyncApplyRequest(
                        items=plan.items,
                        ingest_missing=ingest_missing_safe,
                        apply_safe_only=True,
                        delete_orphaned=False,
                        retire_replaced_content=False,
                    )
                )
    except Exception as e:  # noqa: BLE001
        _render_service_error(e, "SAFE apply")
        st.stop()
    if result is None:
        st.stop()
    assert result is not None
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
    result = None
    try:
        run_id = uuid.uuid4().hex[:8]
        st.session_state["_run_id"] = run_id
        set_run_id(run_id)
        with st.spinner("Applying destructive actions…"):
            with timed_block(
                "action.file_resync.apply_destructive_actions",
                extra={
                    "run_id": run_id,
                    "ingest_missing": False,
                    "delete_orphaned": delete_orphaned,
                    "retire_replaced": retire_replaced,
                },
                logger=logger,
            ):
                from app.usecases.file_resync_usecase import apply_plan as apply_resync_plan

                result = apply_resync_plan(
                    FileResyncApplyRequest(
                        items=plan.items,
                        ingest_missing=False,
                        apply_safe_only=False,
                        delete_orphaned=delete_orphaned,
                        retire_replaced_content=retire_replaced,
                    )
                )
    except Exception as e:  # noqa: BLE001
        _render_service_error(e, "Destructive apply")
        st.stop()
    if result is None:
        st.stop()
    assert result is not None
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

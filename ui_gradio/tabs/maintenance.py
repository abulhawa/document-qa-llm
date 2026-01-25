"""Knowledge base maintenance tab."""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List

import gradio as gr
import pandas as pd

from app.schemas import FileResyncApplyRequest, FileResyncPlanItem, FileResyncScanRequest
from app.usecases import (
    admin_usecase,
    duplicates_usecase,
    file_resync_usecase,
    index_viewer_usecase,
    watchlist_usecase,
)
from core.sync.file_resync import DEFAULT_ALLOWED_EXTENSIONS

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


def _parse_exts(raw: str) -> set[str]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    exts = {part if part.startswith(".") else f".{part}" for part in parts}
    return exts or set(DEFAULT_ALLOWED_EXTENSIONS)


def _split_reasons(reason: str) -> list[str]:
    if not reason:
        return []
    parts = [part.strip() for part in str(reason).split(";") if part.strip()]
    return parts or []


def _map_reason_details(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_DETAIL_MAP["MIXED"]
    mapped = [REASON_DETAIL_MAP.get(part, f"Unknown reason: {part}") for part in parts]
    return "; ".join(dict.fromkeys(mapped))


def _map_reason_actions(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_ACTION_MAP["MIXED"]
    mapped = [REASON_ACTION_MAP.get(part, f"Manual review for {part}") for part in parts]
    return "; ".join(dict.fromkeys(mapped))


def _plan_items_to_rows(items: List[FileResyncPlanItem]) -> List[dict]:
    return [
        {
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


def _build_filtered_df(rows: Iterable[dict], buckets: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if buckets:
        df = df.loc[df["bucket"].isin(buckets)]
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
    return df[[col for col in preferred_cols if col in df.columns]]


def _format_scan_summary(plan, scan_meta: dict) -> str:
    if plan is None:
        return "No plan available yet. Run a scan to populate results."
    scanned = ", ".join(scan_meta.get("scanned_roots", [])) or "None"
    failed = ", ".join(scan_meta.get("failed_roots", [])) or "None"
    ignored = scan_meta.get("ignored", 0)
    return (
        f"Plan generated with {len(plan.items)} item(s). Buckets: {plan.counts}. "
        f"Roots scanned: {scanned}; failed roots: {failed}; ignored files: {ignored}."
    )


def _write_csv(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _roots_to_df(roots: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Sync roots": roots}) if roots else pd.DataFrame({"Sync roots": []})


def build_maintenance_tab() -> None:
    duplicate_selection = gr.State(None)
    resync_roots = gr.State([DEFAULT_ROOT] if DEFAULT_ROOT else [])
    resync_plan = gr.State(None)
    resync_scan_meta = gr.State({})
    resync_rows = gr.State([])
    resync_filtered = gr.State(pd.DataFrame())

    with gr.Accordion("Index Viewer", open=True):
        index_button = gr.Button("Load indexed files", variant="primary")
        index_table = gr.Dataframe(
            headers=["Path", "Filetype", "Modified", "Created"],
            datatype=["str", "str", "str", "str"],
            row_count=0,
            column_count=(4, "fixed"),
            interactive=False,
        )

    def load_indexed():
        files = index_viewer_usecase.fetch_indexed_files()
        rows = [
            {
                "Path": file_meta.get("path"),
                "Filetype": file_meta.get("filetype"),
                "Modified": file_meta.get("modified_at"),
                "Created": file_meta.get("created_at"),
            }
            for file_meta in files
        ]
        return pd.DataFrame(rows)

    index_button.click(load_indexed, outputs=[index_table])

    with gr.Accordion("Duplicate Files", open=False):
        dup_button = gr.Button("Load duplicates", variant="primary")
        dup_table = gr.Dataframe(
            headers=["Checksum", "Location", "Filetype", "Created", "Modified"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(5, "fixed"),
            interactive=False,
        )
        delete_button = gr.Button("Delete Selected")
        dup_status = gr.Markdown()

    def load_duplicates():
        response = duplicates_usecase.lookup_duplicates()
        rows = duplicates_usecase.format_duplicate_rows(response)
        trimmed = [
            {
                "Checksum": row.get("Checksum"),
                "Location": row.get("Location"),
                "Filetype": row.get("Filetype"),
                "Created": row.get("Created"),
                "Modified": row.get("Modified"),
            }
            for row in rows
        ]
        return pd.DataFrame(trimmed)

    def select_duplicate(evt: gr.SelectData):
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return row_index

    def delete_selected(selection, df):
        if selection is None or df is None:
            return "Select a duplicate row first."
        try:
            row = df.iloc[int(selection)]
        except Exception:
            return "Unable to read selection."
        path = row.get("Location") if hasattr(row, "get") else row["Location"]
        if not path:
            return "Selected row has no path."
        task_ids = index_viewer_usecase.enqueue_delete([path])
        return f"Queued deletion for {path}. Task IDs: {', '.join(task_ids)}"

    dup_button.click(load_duplicates, outputs=[dup_table])
    dup_table.select(select_duplicate, outputs=[duplicate_selection])
    delete_button.click(delete_selected, inputs=[duplicate_selection, dup_table], outputs=[dup_status])

    with gr.Accordion("Watchlist", open=False):
        watchlist_prefix = gr.Textbox(label="Watchlist prefix")
        add_watchlist = gr.Button("Add to Watchlist")
        watchlist_status = gr.Markdown()

    def add_prefix(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to add."
        added = watchlist_usecase.add_prefix(prefix)
        return "Prefix added." if added else "Prefix already exists or failed."

    add_watchlist.click(add_prefix, inputs=[watchlist_prefix], outputs=[watchlist_status])

    with gr.Accordion("File Resync", open=False):
        gr.Markdown(
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
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                root_input = gr.Textbox(label="Add root", placeholder="/path/to/root")
            with gr.Column(scale=1):
                add_root_button = gr.Button("Add root", variant="secondary")
            with gr.Column(scale=1):
                clear_roots_button = gr.Button("Clear roots", variant="secondary")

        root_status = gr.Markdown()
        roots_table = gr.Dataframe(
            value=_roots_to_df([DEFAULT_ROOT] if DEFAULT_ROOT else []),
            headers=["Sync roots"],
            datatype=["str"],
            row_count=0,
            column_count=(1, "fixed"),
            interactive=False,
        )
        ext_input = gr.Textbox(
            label="Allowed extensions (comma-separated)",
            value=", ".join(sorted(DEFAULT_ALLOWED_EXTENSIONS)),
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Step 1: Scan & Plan**")
                scan_button = gr.Button("Scan & Plan", variant="primary")
            with gr.Column():
                gr.Markdown("**Step 2: Apply SAFE actions**")
                ingest_missing_safe = gr.Checkbox(
                    label="Ingest missing (SAFE phase)", value=False
                )
                apply_safe_button = gr.Button(
                    "Apply SAFE actions",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("**Step 3: Apply destructive actions**")
                ingest_missing_destructive = gr.Checkbox(
                    label="Ingest missing (Destructive phase)", value=False
                )
                delete_orphaned = gr.Checkbox(
                    label="Delete orphaned content (Destructive)", value=False
                )
                retire_replaced = gr.Checkbox(
                    label="Retire replaced content (Destructive)", value=False
                )
                apply_destructive_button = gr.Button(
                    "Apply destructive actions",
                    interactive=False,
                )

        scan_summary = gr.Markdown()
        apply_summary = gr.Markdown()

        bucket_filters = gr.CheckboxGroup(
            label="Filter by bucket",
            choices=["SAFE", "REVIEW", "BLOCKED", "INFO"],
            value=["SAFE", "REVIEW", "BLOCKED", "INFO"],
        )
        plan_table = gr.Dataframe(
            headers=[
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
            ],
            row_count=0,
            column_count=(11, "fixed"),
            interactive=False,
        )
        export_button = gr.DownloadButton("Export CSV", visible=False)
        apply_result = gr.JSON()

        reason_map_rows = [
            {
                "Reason": reason,
                "Meaning": REASON_DETAIL_MAP.get(reason, ""),
                "Apply action": REASON_ACTION_MAP.get(reason, ""),
            }
            for reason in REASON_ORDER
        ]
        gr.Dataframe(
            value=pd.DataFrame(reason_map_rows),
            headers=["Reason", "Meaning", "Apply action"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Reason Map (Mismatch -> Apply Action)",
        )

    def _add_root(roots: list[str], new_root: str):
        if not new_root:
            return roots, _roots_to_df(roots), "Enter a root path to add."
        entries = [part.strip() for part in new_root.split(",") if part.strip()]
        merged = list(dict.fromkeys(roots + entries))
        message = f"Added {len(merged) - len(roots)} root(s)."
        return merged, _roots_to_df(merged), message

    def _clear_roots():
        return [], _roots_to_df([]), "Cleared all roots."

    def _scan_plan(roots: list[str], raw_exts: str, retire_replaced_flag: bool, buckets: list[str]):
        if not roots:
            empty_df = pd.DataFrame()
            return (
                None,
                {},
                [],
                empty_df,
                "Please provide at least one root to scan.",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )
        try:
            exts = sorted(_parse_exts(raw_exts))
            plan, scan_meta = file_resync_usecase.scan_and_plan(
                FileResyncScanRequest(roots=roots, allowed_extensions=exts),
                retire_replaced_content=retire_replaced_flag,
            )
        except Exception as exc:  # noqa: BLE001
            empty_df = pd.DataFrame()
            return (
                None,
                {},
                [],
                empty_df,
                f"Scan failed: {exc}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

        rows = _plan_items_to_rows(plan.items)
        filtered_df = _build_filtered_df(rows, buckets)
        csv_path = _write_csv(filtered_df)
        return (
            plan,
            scan_meta,
            rows,
            filtered_df,
            _format_scan_summary(plan, scan_meta),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=csv_path, visible=bool(csv_path)),
        )

    def _filter_table(rows: list[dict], buckets: list[str]):
        filtered_df = _build_filtered_df(rows, buckets)
        csv_path = _write_csv(filtered_df)
        return filtered_df, filtered_df, gr.update(value=csv_path, visible=bool(csv_path))

    def _apply_actions(
        plan,
        ingest_missing: bool,
        apply_safe_only: bool,
        delete_orphaned_flag: bool,
        retire_replaced_flag: bool,
    ):
        if plan is None:
            return "No plan available. Run a scan first.", None
        try:
            result = file_resync_usecase.apply_plan(
                FileResyncApplyRequest(
                    items=plan.items,
                    ingest_missing=ingest_missing,
                    apply_safe_only=apply_safe_only,
                    delete_orphaned=delete_orphaned_flag,
                    retire_replaced_content=retire_replaced_flag,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return f"Apply failed: {exc}", None
        summary = "Actions completed." if not result.errors else "Actions completed with warnings."
        return summary, {
            "ingested": result.ingested,
            "updated_fulltext": result.updated_fulltext,
            "updated_chunks": result.updated_chunks,
            "updated_qdrant": result.updated_qdrant,
            "deleted_checksums": result.deleted_checksums,
            "errors": result.errors,
        }

    add_root_button.click(
        _add_root,
        inputs=[resync_roots, root_input],
        outputs=[resync_roots, roots_table, root_status],
    )
    clear_roots_button.click(
        _clear_roots,
        outputs=[resync_roots, roots_table, root_status],
    )

    scan_button.click(
        _scan_plan,
        inputs=[resync_roots, ext_input, retire_replaced, bucket_filters],
        outputs=[
            resync_plan,
            resync_scan_meta,
            resync_rows,
            plan_table,
            scan_summary,
            apply_safe_button,
            apply_destructive_button,
            export_button,
        ],
    )

    bucket_filters.change(
        _filter_table,
        inputs=[resync_rows, bucket_filters],
        outputs=[plan_table, resync_filtered, export_button],
    )

    apply_safe_button.click(
        _apply_actions,
        inputs=[resync_plan, ingest_missing_safe, gr.State(True), gr.State(False), gr.State(False)],
        outputs=[apply_summary, apply_result],
    )
    apply_destructive_button.click(
        _apply_actions,
        inputs=[
            resync_plan,
            ingest_missing_destructive,
            gr.State(False),
            delete_orphaned,
            retire_replaced,
        ],
        outputs=[apply_summary, apply_result],
    )

    with gr.Accordion("Admin & Worker", open=False):
        queue_names = gr.Textbox(
            label="Queues to purge",
            value="ingest,celery",
        )
        clear_cache = gr.Button("Clear Cache")
        purge_queues = gr.Button("Purge Queues")
        admin_status = gr.Markdown()

    def clear_all():
        result = admin_usecase.clear_cache()
        return f"Cache clear result: {result}"

    def purge(queue_text: str) -> str:
        queues = [name.strip() for name in queue_text.split(",") if name.strip()]
        result = admin_usecase.purge_queues(queues)
        return f"Purge result: {result}"

    clear_cache.click(clear_all, outputs=[admin_status])
    purge_queues.click(purge, inputs=[queue_names], outputs=[admin_status])

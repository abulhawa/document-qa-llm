"""Running task admin tab."""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from app.usecases.running_tasks_usecase import fetch_running_tasks_snapshot
from ui.celery_admin import revoke_task
from ui.task_status import clear_finished, fetch_states

FAILED_WINDOWS = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7}
SESSION_HEADERS = ["Path", "Action", "Task ID", "State", "Result"]


def _format_metrics(overview: dict, fails: int | None, window_label: str, queue_depth: int):
    counts = overview.get("counts", {})
    return (
        f"**Active**\n\n{counts.get('active', 0)}",
        f"**Reserved**\n\n{counts.get('reserved', 0)}",
        f"**Scheduled**\n\n{counts.get('scheduled', 0)}",
        f"**Failed ({window_label})**\n\n{'—' if fails is None else fails}",
        f"**Queue depth**\n\n{queue_depth}",
    )


def _format_failed_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Task", "Task ID", "Time", "State", "Error"])
    trimmed = [
        {
            "Task": r.get("Task"),
            "Task ID": r.get("Task ID"),
            "Time": r.get("Time"),
            "State": r.get("State"),
            "Error": r.get("Error"),
        }
        for r in rows
    ]
    return pd.DataFrame(trimmed)


def _format_session_rows(records: list[dict], states: dict) -> pd.DataFrame:
    rows = []
    for record in records:
        state_info = states.get(record["task_id"], {})
        result = state_info.get("result")
        if isinstance(result, dict):
            result = {
                key: result[key]
                for key in ("status", "checksum", "n_chunks", "path")
                if key in result
            }
        rows.append(
            {
                "Path": record.get("path", ""),
                "Action": record.get("action"),
                "Task ID": record["task_id"],
                "State": state_info.get("state", "UNKNOWN"),
                "Result": json.dumps(result, ensure_ascii=False)[:160] if result else "",
            }
        )
    return pd.DataFrame(rows, columns=SESSION_HEADERS)


def build_running_tasks_tab(session_tasks_state: gr.State) -> None:
    snapshot_state = gr.State({})

    with gr.Row():
        failed_window = gr.Dropdown(
            choices=list(FAILED_WINDOWS.keys()),
            value="24h",
            label="Failed lookback",
        )
        refresh_snapshot = gr.Button("Refresh snapshot", variant="primary")

    with gr.Row():
        active_metric = gr.Markdown("**Active**\n\n—")
        reserved_metric = gr.Markdown("**Reserved**\n\n—")
        scheduled_metric = gr.Markdown("**Scheduled**\n\n—")
        failed_metric = gr.Markdown("**Failed**\n\n—")
        queue_metric = gr.Markdown("**Queue depth**\n\n—")

    with gr.Accordion("Failed tasks (paged)", open=False):
        with gr.Row():
            failed_page_size = gr.Dropdown(
                choices=[25, 50, 100, 200],
                value=25,
                label="Page size",
            )
            failed_page = gr.Number(
                value=0,
                precision=0,
                label="Page",
            )
        failed_summary = gr.Markdown()
        failed_table = gr.Dataframe(
            headers=["Task", "Task ID", "Time", "State", "Error"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(5, "fixed"),
            interactive=False,
        )

    with gr.Tabs():
        with gr.Tab("Active"):
            active_table = gr.Dataframe(
                headers=["Type", "Worker", "Task", "ID", "Args", "Kwargs", "ETA"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                row_count=0,
                column_count=(7, "fixed"),
                interactive=False,
            )
        with gr.Tab("Reserved"):
            reserved_table = gr.Dataframe(
                headers=["Type", "Worker", "Task", "ID", "Args", "Kwargs", "ETA"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                row_count=0,
                column_count=(7, "fixed"),
                interactive=False,
            )
        with gr.Tab("Scheduled"):
            scheduled_table = gr.Dataframe(
                headers=["Type", "Worker", "Task", "ID", "Args", "Kwargs", "ETA"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                row_count=0,
                column_count=(7, "fixed"),
                interactive=False,
            )

    gr.Markdown("---")
    gr.Markdown("### My session tasks")
    session_note = gr.Markdown("No tasks enqueued in this session yet.")
    session_table = gr.Dataframe(
        headers=SESSION_HEADERS,
        datatype=["str", "str", "str", "str", "str"],
        row_count=0,
        column_count=(5, "fixed"),
        interactive=False,
    )
    with gr.Row():
        refresh_session = gr.Button("🔄 Refresh")
        clear_session = gr.Button("🧹 Clear finished")
    with gr.Row():
        revoke_task_id = gr.Textbox(
            label="Revoke Task ID",
            placeholder="Paste a Task ID…",
        )
        terminate_task = gr.Checkbox(
            label="Terminate (SIGTERM)",
            value=False,
            info="Only applies to STARTED tasks; ignored for queued ones.",
        )
        revoke_button = gr.Button("🚫 Revoke")
    revoke_status = gr.Markdown()

    def load_snapshot(window_label: str, page: float, page_size: int):
        hours = FAILED_WINDOWS.get(window_label, 24)
        snapshot = fetch_running_tasks_snapshot(
            failed_window_hours=hours,
            failed_page=int(page or 0),
            failed_page_size=int(page_size or 25),
        )
        overview = snapshot["overview"]
        fails = snapshot["failed_count"]
        metrics = _format_metrics(overview, fails, window_label, snapshot["queue_depth"])
        rows = snapshot["failed_rows"]
        total = snapshot["failed_total"]
        summary = (
            "No failures in this window."
            if not rows
            else f"{total} failed task(s) in {window_label}. Showing {len(rows)}."
        )
        return (
            snapshot,
            *metrics,
            summary,
            _format_failed_rows(rows),
            pd.DataFrame(snapshot["tables"]["active"]),
            pd.DataFrame(snapshot["tables"]["reserved"]),
            pd.DataFrame(snapshot["tables"]["scheduled"]),
        )

    def refresh_session_rows(records: list[dict]):
        if not records:
            return records, pd.DataFrame(columns=SESSION_HEADERS), "No tasks enqueued in this session yet."
        states = fetch_states([record["task_id"] for record in records])
        return records, _format_session_rows(records, states), ""

    def clear_session_rows(records: list[dict]):
        if not records:
            return records, pd.DataFrame(columns=SESSION_HEADERS), "No tasks enqueued in this session yet."
        states = fetch_states([record["task_id"] for record in records])
        updated = clear_finished(records, states)
        note = "Cleared finished tasks." if len(updated) < len(records) else "No finished tasks to clear."
        return updated, _format_session_rows(updated, states), note

    def revoke_from_session(task_id: str, terminate: bool, records: list[dict]):
        revoke_id = (task_id or "").strip()
        if not revoke_id:
            return "Enter a Task ID.", _format_session_rows(records, fetch_states([r["task_id"] for r in records])) if records else pd.DataFrame(columns=SESSION_HEADERS)
        state_info = fetch_states([revoke_id]).get(revoke_id, {})
        state = state_info.get("state", "UNKNOWN")
        if state in {"SUCCESS", "FAILURE", "REVOKED"}:
            message = f"Task already in terminal state ({state}); revoke skipped."
        else:
            try:
                revoke_task(revoke_id, terminate=terminate)
                message = f"Revoke sent for {revoke_id} (state was {state})."
            except Exception as exc:
                message = f"Revoke failed: {exc}"
        updated_table = (
            _format_session_rows(records, fetch_states([r["task_id"] for r in records]))
            if records
            else pd.DataFrame(columns=SESSION_HEADERS)
        )
        return message, updated_table

    refresh_snapshot.click(
        load_snapshot,
        inputs=[failed_window, failed_page, failed_page_size],
        outputs=[
            snapshot_state,
            active_metric,
            reserved_metric,
            scheduled_metric,
            failed_metric,
            queue_metric,
            failed_summary,
            failed_table,
            active_table,
            reserved_table,
            scheduled_table,
        ],
    )
    failed_window.change(
        load_snapshot,
        inputs=[failed_window, failed_page, failed_page_size],
        outputs=[
            snapshot_state,
            active_metric,
            reserved_metric,
            scheduled_metric,
            failed_metric,
            queue_metric,
            failed_summary,
            failed_table,
            active_table,
            reserved_table,
            scheduled_table,
        ],
    )
    failed_page_size.change(
        load_snapshot,
        inputs=[failed_window, failed_page, failed_page_size],
        outputs=[
            snapshot_state,
            active_metric,
            reserved_metric,
            scheduled_metric,
            failed_metric,
            queue_metric,
            failed_summary,
            failed_table,
            active_table,
            reserved_table,
            scheduled_table,
        ],
    )
    failed_page.change(
        load_snapshot,
        inputs=[failed_window, failed_page, failed_page_size],
        outputs=[
            snapshot_state,
            active_metric,
            reserved_metric,
            scheduled_metric,
            failed_metric,
            queue_metric,
            failed_summary,
            failed_table,
            active_table,
            reserved_table,
            scheduled_table,
        ],
    )

    refresh_session.click(
        refresh_session_rows,
        inputs=[session_tasks_state],
        outputs=[session_tasks_state, session_table, session_note],
    )
    clear_session.click(
        clear_session_rows,
        inputs=[session_tasks_state],
        outputs=[session_tasks_state, session_table, session_note],
    )
    revoke_button.click(
        revoke_from_session,
        inputs=[revoke_task_id, terminate_task, session_tasks_state],
        outputs=[revoke_status, session_table],
    )

"""Ingestion pipeline tab."""

from __future__ import annotations

from typing import Any, cast
import json
import subprocess

import gradio as gr
import pandas as pd

from app.gradio_utils import format_ingest_logs
from app.schemas import IngestLogRequest, IngestRequest, IngestMode
from app.usecases import ingest_logs_usecase, ingest_usecase
from ui.task_status import add_records, clear_finished, fetch_states


def build_ingest_tab() -> None:
    ingest_state = gr.State([])
    folder_state = gr.State([])
    selected_state = gr.State([])

    task_columns = ["Path", "Task ID", "Action", "State", "Result"]

    with gr.Row():
        with gr.Column(scale=1):
            uploads = gr.File(
                label="Upload files",
                file_count="multiple",
                type="filepath",
            )
            folder_button = gr.Button("Select Folder")
            selected_status = gr.Markdown("No paths selected.")
            mode = gr.Dropdown(
                choices=["ingest", "reingest"],
                value="ingest",
                label="Mode",
            )
            start_button = gr.Button("Start Ingestion", variant="primary")
            refresh_logs = gr.Button("Refresh Logs")
            status = gr.Markdown()
        with gr.Column(scale=2):
            logs = gr.Code(label="Ingestion Logs", language="markdown")

    with gr.Accordion("Selected Paths", open=False):
        selected_table = gr.Dataframe(
            headers=["Selected Path"],
            datatype=["str"],
            row_count=0,
            column_count=(1, "fixed"),
            interactive=False,
        )

    with gr.Accordion("Task Panel", open=True):
        task_status = gr.Markdown("No tasks enqueued in this session yet.")
        task_table = gr.Dataframe(
            headers=task_columns,
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(5, "fixed"),
            interactive=False,
        )
        with gr.Row():
            refresh_tasks = gr.Button("Refresh task status")
            clear_tasks = gr.Button("Clear finished")

    def run_folder_picker() -> list[str]:
        try:
            result = subprocess.run(
                ["python", "file_picker.py", "folder"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return []
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    def normalize_paths(paths: list[str]) -> list[str]:
        return [path.replace("\\", "/") for path in paths]

    def build_selected_table(paths: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"Selected Path": normalize_paths(paths)})

    def combine_paths(
        uploads_list: list[str] | None,
        folder_paths: list[str] | None,
    ) -> list[str]:
        combined: list[str] = []
        if folder_paths:
            combined.extend(folder_paths)
        if uploads_list:
            combined.extend(uploads_list)
        return combined

    def update_selected_paths(
        uploads_list: list[str] | None,
        folder_paths: list[str] | None,
    ):
        combined = combine_paths(uploads_list, folder_paths)
        if combined:
            message = f"Selected {len(combined)} path(s)."
        else:
            message = "No paths selected."
        return combined, build_selected_table(combined), message

    def pick_folder(uploads_list: list[str] | None):
        folder_paths = run_folder_picker()
        combined, table, message = update_selected_paths(uploads_list, folder_paths)
        return folder_paths, combined, table, message

    def summarize_result(result: Any) -> str:
        if not isinstance(result, dict) or not result:
            return ""
        fields = {k: result.get(k) for k in ("status", "checksum", "n_chunks") if k in result}
        if not fields:
            return ""
        return ", ".join(f"{key}={value}" for key, value in fields.items())

    def build_task_panel(records: list[dict[str, Any]]):
        if not records:
            empty = pd.DataFrame(columns=task_columns)
            return "No tasks enqueued in this session yet.", empty
        ids = [record["task_id"] for record in records]
        states = fetch_states(ids)
        rows = []
        for record in records:
            task_id = record["task_id"]
            state = states.get(task_id, {}).get("state", "UNKNOWN")
            result = states.get(task_id, {}).get("result")
            rows.append(
                {
                    "Path": record.get("path"),
                    "Task ID": task_id,
                    "Action": str(record.get("action", "")),
                    "State": state,
                    "Result": summarize_result(result),
                }
            )
        return f"{len(records)} task(s) queued in this session.", pd.DataFrame(rows)

    def start_ingestion(
        selected_paths: list[str],
        mode: str,
        state: list[dict[str, Any]],
        progress: gr.Progress = gr.Progress(),
    ):
        if not selected_paths:
            return "No files selected.", state, "", *build_task_panel(state)
        progress(0, desc="Queueing files")
        request = IngestRequest(paths=selected_paths, mode=cast(IngestMode, mode))
        response = ingest_usecase.ingest(request)
        progress(1, desc="Queued")
        message = f"Queued {response.queued_count} files."
        if response.errors:
            message += "\n" + "\n".join(response.errors)
        log_response = ingest_logs_usecase.fetch_ingest_logs(IngestLogRequest())
        updated_records = add_records(
            state,
            selected_paths,
            response.task_ids,
            action=mode,
        )
        task_message, task_table = build_task_panel(updated_records)
        return (
            message,
            updated_records,
            format_ingest_logs(log_response.logs),
            task_message,
            task_table,
        )

    def load_logs() -> str:
        log_response = ingest_logs_usecase.fetch_ingest_logs(IngestLogRequest())
        return format_ingest_logs(log_response.logs)

    def refresh_task_panel(records: list[dict[str, Any]]):
        task_message, task_table = build_task_panel(records)
        return task_message, task_table, records

    def clear_finished_tasks(records: list[dict[str, Any]]):
        if not records:
            task_message, task_table = build_task_panel(records)
            return task_message, task_table, records
        states = fetch_states([record["task_id"] for record in records])
        updated = clear_finished(records, states)
        task_message, task_table = build_task_panel(updated)
        return task_message, task_table, updated

    uploads.change(
        update_selected_paths,
        inputs=[uploads, folder_state],
        outputs=[selected_state, selected_table, selected_status],
    )
    folder_button.click(
        pick_folder,
        inputs=[uploads],
        outputs=[folder_state, selected_state, selected_table, selected_status],
    )
    start_button.click(
        start_ingestion,
        inputs=[selected_state, mode, ingest_state],
        outputs=[status, ingest_state, logs, task_status, task_table],
    )
    refresh_logs.click(load_logs, outputs=[logs])
    refresh_tasks.click(
        refresh_task_panel,
        inputs=[ingest_state],
        outputs=[task_status, task_table, ingest_state],
    )
    clear_tasks.click(
        clear_finished_tasks,
        inputs=[ingest_state],
        outputs=[task_status, task_table, ingest_state],
    )

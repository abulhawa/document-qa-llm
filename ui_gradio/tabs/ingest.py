"""Ingestion pipeline tab."""

from __future__ import annotations

from typing import Any, cast
import json
import subprocess

import gradio as gr
import pandas as pd

from app.gradio_utils import normalize_date_input
from app.schemas import IngestLogRequest, IngestMode, IngestRequest
from app.usecases import ingest_logs_usecase, ingest_usecase
from ui.task_status import add_records, clear_finished, fetch_states
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp

LOG_HEADERS = ["Path", "Size", "Status", "Error", "Reason", "Stage", "Attempt"]
TASK_HEADERS = ["Path", "Task ID", "Action", "State", "Result"]


def build_ingest_tab(session_tasks_state: gr.State) -> None:
    ingest_state = gr.State([])
    folder_state = gr.State([])
    selected_state = gr.State([])

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
            gr.Markdown("### Log filters")
            path_filter = gr.Textbox(label="Path contains", value="")
            status_filter = gr.Dropdown(
                choices=[
                    "All",
                    "Failed",
                    "Success",
                    "Already indexed",
                    "Duplicate & Indexed",
                    "No valid content found",
                ],
                value="All",
                label="Status",
            )
            start_date = gr.DateTime(label="Start date", include_time=False)
            end_date = gr.DateTime(label="End date", include_time=False)
        with gr.Column(scale=2):
            logs = gr.Dataframe(
                headers=LOG_HEADERS,
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                row_count=0,
                column_count=(len(LOG_HEADERS), "fixed"),
                interactive=False,
                label="Ingestion Logs",
            )

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
            headers=TASK_HEADERS,
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(len(TASK_HEADERS), "fixed"),
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
        # Dedupe while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for path in combined:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
        return ordered

    def update_selected_paths(
        uploads_list: list[str] | None,
        folder_paths: list[str] | None,
    ):
        combined = combine_paths(uploads_list, folder_paths)
        message = f"Selected {len(combined)} path(s)." if combined else "No paths selected."
        return combined, build_selected_table(combined), message

    def pick_folder(uploads_list: list[str] | None):
        folder_paths = run_folder_picker()
        combined, table, message = update_selected_paths(uploads_list, folder_paths)
        return folder_paths, combined, table, message

    def build_log_rows(request: IngestLogRequest) -> pd.DataFrame:
        log_response = ingest_logs_usecase.fetch_ingest_logs(request)
        rows = [
            {
                "Path": log.path,
                "Size": format_file_size(log.bytes or 0),
                "Status": log.status,
                "Error": log.error_type,
                "Reason": (log.reason or "")[:100],
                "Stage": log.stage,
                "Attempt": format_timestamp(log.attempt_at) if log.attempt_at else "",
            }
            for log in log_response.logs
        ]
        return pd.DataFrame(rows, columns=LOG_HEADERS)

    def build_log_request(
        status_value: str,
        path_value: str,
        start_value: object,
        end_value: object,
    ) -> IngestLogRequest:
        status_param = None if status_value == "All" else status_value
        return IngestLogRequest(
            status=status_param,
            path_query=path_value or None,
            start_date=normalize_date_input(start_value),
            end_date=normalize_date_input(end_value),
            size=200,
        )

    def summarize_result(result: Any) -> str:
        if not isinstance(result, dict) or not result:
            return ""
        fields = {k: result.get(k) for k in ("status", "checksum", "n_chunks") if k in result}
        if not fields:
            return ""
        return ", ".join(f"{key}={value}" for key, value in fields.items())

    def build_task_panel(records: list[dict[str, Any]]):
        if not records:
            empty = pd.DataFrame(columns=TASK_HEADERS)
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
        return f"{len(records)} task(s) queued in this session.", pd.DataFrame(rows, columns=TASK_HEADERS)

    def start_ingestion(
        selected_paths: list[str],
        mode_value: str,
        records: list[dict[str, Any]],
        session_records: list[dict[str, Any]],
        status_value: str,
        path_value: str,
        start_value: object,
        end_value: object,
        progress: gr.Progress = gr.Progress(),
    ):
        log_request = build_log_request(status_value, path_value, start_value, end_value)

        if not selected_paths:
            task_message, task_df = build_task_panel(records)
            return "No files selected.", records, session_records, build_log_rows(log_request), task_message, task_df

        progress(0, desc="Queueing files")
        request = IngestRequest(paths=selected_paths, mode=cast(IngestMode, mode_value))
        response = ingest_usecase.ingest(request)
        progress(1, desc="Queued")

        message = f"Queued {response.queued_count} files."
        if response.errors:
            message += "\n" + "\n".join(response.errors)

        updated_records = add_records(records, selected_paths, response.task_ids, action=mode_value)
        updated_session = add_records(session_records, selected_paths, response.task_ids, action=mode_value)
        task_message, task_df = build_task_panel(updated_records)

        return (
            message,
            updated_records,
            updated_session,
            build_log_rows(log_request),
            task_message,
            task_df,
        )

    def load_logs(
        status_value: str,
        path_value: str,
        start_value: object,
        end_value: object,
    ) -> pd.DataFrame:
        log_request = build_log_request(status_value, path_value, start_value, end_value)
        return build_log_rows(log_request)

    def refresh_task_panel(records: list[dict[str, Any]], session_records: list[dict[str, Any]]):
        task_message, task_df = build_task_panel(records)
        return task_message, task_df, records, session_records

    def clear_finished_tasks(records: list[dict[str, Any]], session_records: list[dict[str, Any]]):
        task_ids = {record["task_id"] for record in records}
        task_ids.update(record["task_id"] for record in session_records)
        states = fetch_states(task_ids)
        updated_records = clear_finished(records, states)
        updated_session = clear_finished(session_records, states)
        task_message, task_df = build_task_panel(updated_records)
        return task_message, task_df, updated_records, updated_session

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
        inputs=[
            selected_state,
            mode,
            ingest_state,
            session_tasks_state,
            status_filter,
            path_filter,
            start_date,
            end_date,
        ],
        outputs=[status, ingest_state, session_tasks_state, logs, task_status, task_table],
    )
    refresh_logs.click(
        load_logs,
        inputs=[status_filter, path_filter, start_date, end_date],
        outputs=[logs],
    )
    refresh_tasks.click(
        refresh_task_panel,
        inputs=[ingest_state, session_tasks_state],
        outputs=[task_status, task_table, ingest_state, session_tasks_state],
    )
    clear_tasks.click(
        clear_finished_tasks,
        inputs=[ingest_state, session_tasks_state],
        outputs=[task_status, task_table, ingest_state, session_tasks_state],
    )

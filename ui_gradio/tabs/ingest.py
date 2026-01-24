"""Ingestion pipeline tab."""

from __future__ import annotations

from typing import cast

import pandas as pd
import gradio as gr

from app.gradio_utils import normalize_date_input
from app.schemas import IngestLogRequest, IngestRequest, IngestMode
from app.usecases import ingest_logs_usecase, ingest_usecase
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp


LOG_HEADERS = ["Path", "Size", "Status", "Error", "Reason", "Stage", "Attempt"]


def build_ingest_tab() -> None:
    ingest_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            uploads = gr.File(
                label="Upload files",
                file_count="multiple",
                type="filepath",
            )
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

    def start_ingestion(
        files: list[str] | None,
        mode: str,
        state: list[str],
        status_value: str,
        path_value: str,
        start_value: object,
        end_value: object,
        progress: gr.Progress = gr.Progress(),
    ):
        if not files:
            return "No files selected.", state, pd.DataFrame(columns=LOG_HEADERS)
        progress(0, desc="Queueing files")
        request = IngestRequest(paths=files, mode=cast(IngestMode, mode))
        response = ingest_usecase.ingest(request)
        progress(1, desc="Queued")
        message = f"Queued {response.queued_count} files."
        if response.errors:
            message += "\n" + "\n".join(response.errors)
        log_request = build_log_request(status_value, path_value, start_value, end_value)
        return message, response.task_ids, build_log_rows(log_request)

    def load_logs(
        status_value: str,
        path_value: str,
        start_value: object,
        end_value: object,
    ) -> pd.DataFrame:
        log_request = build_log_request(status_value, path_value, start_value, end_value)
        return build_log_rows(log_request)

    start_button.click(
        start_ingestion,
        inputs=[
            uploads,
            mode,
            ingest_state,
            status_filter,
            path_filter,
            start_date,
            end_date,
        ],
        outputs=[status, ingest_state, logs],
    )
    refresh_logs.click(
        load_logs,
        inputs=[status_filter, path_filter, start_date, end_date],
        outputs=[logs],
    )

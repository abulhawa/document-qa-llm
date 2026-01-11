"""Ingestion pipeline tab."""

from __future__ import annotations

import gradio as gr

from app.gradio_utils import format_ingest_logs
from app.schemas import IngestLogRequest, IngestRequest
from app.usecases import ingest_logs_usecase, ingest_usecase


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
        with gr.Column(scale=2):
            logs = gr.Code(label="Ingestion Logs", language="text")

    def start_ingestion(
        files: list[str] | None,
        mode_value: str,
        state: list[str],
        progress: gr.Progress = gr.Progress(),
    ):
        if not files:
            return "No files selected.", state, ""
        progress(0, desc="Queueing files")
        request = IngestRequest(paths=files, mode=mode_value)
        response = ingest_usecase.ingest(request)
        progress(1, desc="Queued")
        message = f"Queued {response.queued_count} files."
        if response.errors:
            message += "\n" + "\n".join(response.errors)
        log_response = ingest_logs_usecase.fetch_ingest_logs(IngestLogRequest())
        return message, response.task_ids, format_ingest_logs(log_response.logs)

    def load_logs() -> str:
        log_response = ingest_logs_usecase.fetch_ingest_logs(IngestLogRequest())
        return format_ingest_logs(log_response.logs)

    start_button.click(
        start_ingestion,
        inputs=[uploads, mode, ingest_state],
        outputs=[status, ingest_state, logs],
    )
    refresh_logs.click(load_logs, outputs=[logs])

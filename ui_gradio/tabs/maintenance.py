"""Knowledge base maintenance tab."""

from __future__ import annotations

import os
from pathlib import Path

from celery import Celery
import gradio as gr
import pandas as pd
import redis

from app.schemas import FileResyncScanRequest
from app.usecases import (
    admin_usecase,
    duplicates_usecase,
    file_resync_usecase,
    index_viewer_usecase,
    worker_emergency_usecase,
    watchlist_usecase,
)

# ----------------------------
# Worker emergency env config
# ----------------------------
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
QUEUE_NAMES = os.getenv("CELERY_QUEUES", "ingest,celery").split(",")
DEFAULT_TASK = os.getenv("CELERY_DEFAULT_TASK", "tasks.ingest_document")
COMPOSE_DIR = Path(os.getenv("COMPOSE_DIR", Path.cwd()))
COMPOSE_PROJECT = os.getenv("COMPOSE_PROJECT", "document_qa")
CELERY_SERVICE = os.getenv("CELERY_SERVICE", "celery")


def build_maintenance_tab() -> None:
    duplicate_selection = gr.State(None)
    celery_app = Celery(broker=BROKER_URL, backend=RESULT_BACKEND)
    broker_client = redis.Redis.from_url(BROKER_URL, decode_responses=True)
    result_client = redis.Redis.from_url(RESULT_BACKEND, decode_responses=True)

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

    with gr.Accordion("Watchlist & File Resync", open=False):
        watchlist_prefix = gr.Textbox(label="Watchlist prefix")
        add_watchlist = gr.Button("Add to Watchlist")
        resync_button = gr.Button("Scan & Plan Resync")
        watchlist_status = gr.Markdown()

    def add_prefix(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to add."
        added = watchlist_usecase.add_prefix(prefix)
        return "Prefix added." if added else "Prefix already exists or failed."

    def scan_resync(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to scan."
        response, meta = file_resync_usecase.scan_and_plan(
            FileResyncScanRequest(roots=[prefix])
        )
        return (
            f"Plan generated with {len(response.items)} items. "
            f"Buckets: {response.counts}. "
            f"Scanned: {meta.get('scanned_roots', [])}."
        )

    add_watchlist.click(add_prefix, inputs=[watchlist_prefix], outputs=[watchlist_status])
    resync_button.click(scan_resync, inputs=[watchlist_prefix], outputs=[watchlist_status])

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

    with gr.Accordion("Worker Emergency Operations", open=False):
        status_button = gr.Button("Refresh worker status", variant="primary")
        queue_status = gr.Dataframe(
            headers=["Queue", "Length"],
            datatype=["str", "number"],
            row_count=0,
            column_count=(2, "fixed"),
            interactive=False,
        )
        counts_status = gr.Markdown()
        alerts_status = gr.Markdown()

        def load_worker_status():
            status = worker_emergency_usecase.load_status(
                celery_app,
                broker_client,
                result_client,
                [name.strip() for name in QUEUE_NAMES if name.strip()],
            )
            queue_rows = [
                {"Queue": name, "Length": status.queue_lengths.get(name, -1)}
                for name in status.queue_lengths.keys()
            ]
            counts = (
                f"**Active:** {status.active} | "
                f"**Reserved:** {status.reserved} | "
                f"**Scheduled:** {status.scheduled}"
            )
            alerts = []
            if not status.broker_ok:
                alerts.append(f"Broker connectivity failed: {status.broker_error}")
            if not status.result_ok:
                alerts.append(f"Result backend connectivity failed: {status.result_error}")
            if status.celery_error:
                alerts.append(f"Celery inspection failed: {status.celery_error}")
            alert_text = "\n".join(f"- {entry}" for entry in alerts) if alerts else "All checks OK."
            return pd.DataFrame(queue_rows), counts, alert_text

        status_button.click(load_worker_status, outputs=[queue_status, counts_status, alerts_status])

        with gr.Accordion("Pause/Resume Queue Consumption", open=False):
            queue_name = gr.Textbox(label="Queue name", value=QUEUE_NAMES[0] if QUEUE_NAMES else "ingest")
            pause_button = gr.Button("Cancel consumer (pause)")
            resume_button = gr.Button("Add consumer (resume)")
            pause_status = gr.Markdown()

            def pause_queue(name: str) -> str:
                if not name:
                    return "Enter a queue name."
                result = celery_app.control.cancel_consumer(name)
                return f"Cancelled consumer for queue {name}. Result: {result}"

            def resume_queue(name: str) -> str:
                if not name:
                    return "Enter a queue name."
                result = celery_app.control.add_consumer(name)
                return f"Added consumer for queue {name}. Result: {result}"

            pause_button.click(pause_queue, inputs=[queue_name], outputs=[pause_status])
            resume_button.click(resume_queue, inputs=[queue_name], outputs=[pause_status])

        with gr.Accordion("Autoscale Worker Pool", open=False):
            autoscale_max = gr.Number(label="Max processes", value=8, precision=0)
            autoscale_min = gr.Number(label="Min processes", value=0, precision=0)
            autoscale_button = gr.Button("Apply autoscale")
            autoscale_status = gr.Markdown()

            def apply_autoscale(max_val: float, min_val: float) -> str:
                if max_val is None or min_val is None:
                    return "Enter max and min values."
                max_int = int(max_val)
                min_int = int(min_val)
                result = celery_app.control.autoscale(max_int, min_int)
                return f"Autoscale applied: max={max_int}, min={min_int}. Result: {result}"

            autoscale_button.click(
                apply_autoscale,
                inputs=[autoscale_max, autoscale_min],
                outputs=[autoscale_status],
            )

        with gr.Accordion("Stop/Start Worker via Docker Compose", open=False):
            stop_button = gr.Button("Stop worker container", variant="primary")
            start_button = gr.Button("Start worker container")
            compose_status = gr.Markdown()

            def stop_worker() -> str:
                result = worker_emergency_usecase.run_compose(
                    ["stop", CELERY_SERVICE],
                    compose_dir=COMPOSE_DIR,
                    compose_project=COMPOSE_PROJECT,
                )
                return result.stdout or result.stderr or "(no output)"

            def start_worker() -> str:
                result = worker_emergency_usecase.run_compose(
                    ["up", "-d", CELERY_SERVICE],
                    compose_dir=COMPOSE_DIR,
                    compose_project=COMPOSE_PROJECT,
                )
                return result.stdout or result.stderr or "(no output)"

            stop_button.click(stop_worker, outputs=[compose_status])
            start_button.click(start_worker, outputs=[compose_status])

        with gr.Accordion("Rate Limit & Revoke Active Tasks", open=False):
            rate_button = gr.Button("Apply 0/m rate limit")
            revoke_term = gr.Button("Revoke active (SIGTERM)")
            revoke_kill = gr.Button("Revoke active (SIGKILL)")
            revoke_status = gr.Markdown()

            def rate_limit() -> str:
                worker_emergency_usecase.rate_limit_zero(celery_app, DEFAULT_TASK)
                return f"Rate limit applied: {DEFAULT_TASK} set to 0/m."

            def revoke_active(signal: str) -> str:
                count = worker_emergency_usecase.revoke_all_active(celery_app, signal=signal)
                return f"Revoked {count} active task(s) with {signal}."

            rate_button.click(rate_limit, outputs=[revoke_status])
            revoke_term.click(lambda: revoke_active("SIGTERM"), outputs=[revoke_status])
            revoke_kill.click(lambda: revoke_active("SIGKILL"), outputs=[revoke_status])

        with gr.Accordion("Purge Queues", open=False):
            purge_text = gr.Textbox(
                label="Queues to purge (comma-separated)",
                value=",".join(name.strip() for name in QUEUE_NAMES if name.strip()),
            )
            purge_button = gr.Button("Purge queues")
            purge_status = gr.Markdown()

            def purge_selected(queue_text: str) -> str:
                queues = [name.strip() for name in queue_text.split(",") if name.strip()]
                result = worker_emergency_usecase.purge_queues(broker_client, queues)
                return f"Purge result: {result}"

            purge_button.click(purge_selected, inputs=[purge_text], outputs=[purge_status])

        with gr.Accordion("Flush Redis DBs", open=False):
            flush_broker = gr.Button("Flush broker DB (DB 0)")
            flush_results = gr.Button("Flush result DB (DB 1)")
            flush_status = gr.Markdown()

            def flush_broker_db() -> str:
                broker_client.flushdb()
                return "Broker FLUSHDB completed (DB 0)."

            def flush_results_db() -> str:
                result_client.flushdb()
                return "Result FLUSHDB completed (DB 1)."

            flush_broker.click(flush_broker_db, outputs=[flush_status])
            flush_results.click(flush_results_db, outputs=[flush_status])

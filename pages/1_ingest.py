# pages/1_Ingest_Documents.py
import streamlit as st
import pandas as pd

from ui.ingest_client import enqueue_paths
from ui.ingestion_ui import run_file_picker, run_folder_picker
from ui.task_status import add_records
from components.task_panel import render_task_panel
from tracing import start_span, CHAIN, INPUT_VALUE, OUTPUT_VALUE, STATUS_OK

st.set_page_config(page_title="Ingest Documents", layout="wide")
st.title("📥 Ingest Documents")
st.session_state.setdefault("ingest_tasks", [])

col1, col2 = st.columns([1, 1], gap="small")

selected_files = []
with col1:
    if st.button("📄 Select File(s)"):
        selected_files = run_file_picker()
with col2:
    if st.button("📂 Select Folder"):
        selected_files = run_folder_picker()

if selected_files:
    st.success(f"Found {len(selected_files)} path(s).")
    status_table = st.empty()
    status_line = st.empty()
    progress_bar = st.progress(0)
    eta_display = st.empty()

    # Show the list being processed (no persistence across refresh)
    df = pd.DataFrame({"Selected Path": [p.replace("\\", "/") for p in selected_files]})
    status_table.dataframe(df, height=300)

    def update_progress(done: int, total: int, elapsed: float):
        # Foreground progress: files loaded/split/enqueued (no Celery polling)
        progress_bar.progress(done / max(total, 1))
        if done:
            eta = (elapsed / done) * (total - done)
            eta_display.text(
                f"{done}/{total} files processed… (elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)"
            )
        else:
            eta_display.text(
                f"{done}/{total} files processed… (elapsed: {elapsed:.1f}s)"
            )

    with start_span("Ingestion chain", CHAIN) as span:
        if len(selected_files) > 5:
            preview = selected_files[:5] + [
                f"... and {len(selected_files) - 5} more not shown here"
            ]
        else:
            preview = selected_files

        span.set_attribute(INPUT_VALUE, preview)

        # Enqueue ingestion to the worker; progress bar shows "queued" state only.
        task_ids = enqueue_paths(selected_files, mode="ingest")
        status_line.info(f"Queued {len(task_ids)} file(s) for ingestion.")

        # Persist tasks in session (path -> task_id); UI polling happens via task panel
        st.session_state["ingest_tasks"] = add_records(
            st.session_state.get("ingest_tasks"),
            selected_files,
            task_ids,
        )

        # Mark foreground progress as "all queued"
        progress_bar.progress(1.0)
        eta_display.text(f"Queued {len(selected_files)} / {len(selected_files)} files.")

        # Tracing: record what we actually did in async mode
        span.set_attribute("files_queued", len(task_ids))
        span.set_attribute("task_ids_preview", str(task_ids[:5]))
        span.set_attribute(OUTPUT_VALUE, f"queued {len(task_ids)}")
        span.set_status(STATUS_OK)

# Render a small task panel so users can refresh/clear task states
should_rerun, updated = render_task_panel(st.session_state.get("ingest_tasks", []))
if should_rerun:
    st.session_state["ingest_tasks"] = updated
    st.rerun()

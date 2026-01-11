# pages/1_Ingest_Documents.py
import streamlit as st
import pandas as pd

from app.schemas import IngestRequest
from app.usecases.ingest_usecase import ingest
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

    with start_span("Ingestion chain", CHAIN) as span:
        if len(selected_files) > 5:
            preview = selected_files[:5] + [
                f"... and {len(selected_files) - 5} more not shown here"
            ]
        else:
            preview = selected_files

        span.set_attribute(INPUT_VALUE, preview)

        # Enqueue ingestion to the worker; progress bar shows "queued" state only.
        response = ingest(IngestRequest(paths=selected_files, mode="ingest"))
        if response.task_ids:
            st.session_state["ingest_tasks"] = add_records(
                st.session_state.get("ingest_tasks"),
                selected_files,
                response.task_ids,
                action="ingest",
            )
            status_line.info(
                f"Queued {response.queued_count} file(s) for ingestion."
            )

        # Mark foreground progress as "all queued"
        progress_bar.progress(1.0)
        eta_display.text(
            f"Queued {response.queued_count} / {len(selected_files)} files."
        )
        if response.errors:
            st.error("\n".join(response.errors))

        # Tracing: record what we actually did in async mode
        span.set_attribute("files_queued", response.queued_count)
        span.set_attribute("task_ids_preview", str(response.task_ids[:5]))
        span.set_attribute(OUTPUT_VALUE, f"queued {response.queued_count}")
        span.set_status(STATUS_OK)

# Render a small task panel so users can refresh/clear task states
should_rerun, updated = render_task_panel(st.session_state.get("ingest_tasks", []))
if should_rerun:
    st.session_state["ingest_tasks"] = updated
    st.rerun()

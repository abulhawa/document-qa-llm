# pages/1_Ingest_Documents.py
import streamlit as st
import pandas as pd
from ui.ingest_client import enqueue_ingest, job_stats
from ui.ingestion_ui import run_file_picker, run_folder_picker
from tracing import start_span, CHAIN, INPUT_VALUE, STATUS_OK
from utils.opensearch_utils import (
    ensure_index_exists,
    ensure_fulltext_index_exists,
    ensure_ingest_log_index_exists,
    missing_indices,
)
from core.job_queue import push_pending, pending_count, active_count, retry_count
from core.job_control import set_state, get_state, incr_stat, get_stats
from core.job_commands import pause_job, resume_job, cancel_job, stop_job

st.set_page_config(page_title="Ingest Documents", layout="wide")
st.title("📥 Ingest Documents")

JOB_ID = "default"

import os as _os
if not _os.getenv("PYTEST_CURRENT_TEST"):
    with st.expander("Jobs", expanded=False):
        if st.button("Register Folder(s)"):
            paths = run_folder_picker()
            for p in paths:
                push_pending(JOB_ID, p)
                incr_stat(JOB_ID, "registered", 1)
            set_state(JOB_ID, "running")

        state = get_state(JOB_ID)
        if state is None:
            set_state(JOB_ID, "idle")
            state = "idle"
        stats = get_stats(JOB_ID)
        st.write(
            f"state={state} pending={pending_count(JOB_ID)} active={active_count(JOB_ID)} "
            f"needs_retry={retry_count(JOB_ID)} done={stats.get('done',0)} failed={stats.get('failed',0)}"
        )

        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Pause"):
            pause_job(JOB_ID)
        if c2.button("Resume"):
            resume_job(JOB_ID)
        if c3.button("Cancel"):
            cancel_job(JOB_ID)
        if c4.button("Stop"):
            stop_job(JOB_ID)

missing = missing_indices()
if missing:
    st.warning("Missing OpenSearch indices: " + ", ".join(missing))
    if st.button("🛠️ Create Indices"):
        ensure_index_exists()
        ensure_fulltext_index_exists()
        ensure_ingest_log_index_exists()
        st.success("Created required indices. Please try again.")

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
    df = pd.DataFrame({"Selected Path": [p.replace("\\", "/") for p in selected_files]})
    st.dataframe(df, height=300)

    with start_span("Ingestion chain", CHAIN) as span:
        preview = (
            selected_files[:5] + [f"... and {len(selected_files) - 5} more not shown here"]
            if len(selected_files) > 5
            else selected_files
        )
        span.set_attribute(INPUT_VALUE, preview)
        enqueue_ingest(selected_files, op="ingest", source="ingest_page")
        span.set_attribute("enqueued_files", len(selected_files))
        span.set_status(STATUS_OK)

    st.success(f"Queued {len(selected_files)} file(s) for ingestion.")

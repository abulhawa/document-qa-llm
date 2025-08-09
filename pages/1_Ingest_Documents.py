import time
import streamlit as st
import pandas as pd
from celery.result import AsyncResult
from worker.celery_worker import app
from config import EMBEDDING_BATCH_SIZE
from ui.ingestion_ui import run_file_picker, run_folder_picker
from tracing import start_span, CHAIN, INPUT_VALUE, OUTPUT_VALUE, STATUS_OK

st.set_page_config(page_title="Ingest Documents", layout="wide")
st.title("📥 Ingest Documents")

col1, col2 = st.columns([1, 1], gap="small")

selected_files: list[str] = []
with col1:
    if st.button("📄 Select File(s)"):
        selected_files = run_file_picker()
with col2:
    if st.button("📂 Select Folder"):
        selected_files = run_folder_picker()

if selected_files:
    st.session_state["selected_files"] = selected_files

files = st.session_state.get("selected_files", [])

if files and "task_id" not in st.session_state:
    st.success(f"Found {len(files)} path(s).")
    df = pd.DataFrame({"Selected Path": [p.replace('\\', '/') for p in files]})
    st.dataframe(df, height=300)

    with start_span("Ingestion chain", CHAIN) as span:
        preview = files[:5] + [f"... and {len(files) - 5} more not shown here"] if len(files) > 5 else files
        span.set_attribute(INPUT_VALUE, preview)
        async_result = app.send_task(
            "core.ingestion_tasks.ingest_paths",
            args=[files],
            kwargs={
                "force": False,
                "replace": True,
                "batch_size": EMBEDDING_BATCH_SIZE,
            },
        )
        st.session_state["task_id"] = async_result.id
        span.set_attribute(OUTPUT_VALUE, f"task_id={async_result.id}")
        span.set_status(STATUS_OK)

if "task_id" in st.session_state:
    task_id = st.session_state["task_id"]
    res = AsyncResult(task_id, app=app)
    st.write(f"Task state: {res.state}")
    progress_bar = st.progress(0)
    info = res.info or {}
    total = info.get("total_files") or 1
    done = info.get("files_done", 0)
    progress_bar.progress(done / total)
    st.write(f"{done}/{total} files processed")
    if res.state == "SUCCESS":
        st.success("✅ Ingestion complete. Embedding will continue in the background.")
        del st.session_state["task_id"]
        st.session_state.pop("selected_files", None)
    else:
        if st.button("Cancel Task"):
            res.revoke(terminate=True, signal="SIGTERM")
            del st.session_state["task_id"]
            st.session_state.pop("selected_files", None)
        else:
            time.sleep(1)
            st.experimental_rerun()

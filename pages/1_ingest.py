# pages/1_Ingest_Documents.py
import streamlit as st
import pandas as pd
import threading

from core.ingestion import ingest
from ui.ingestion_ui import run_file_picker, run_folder_picker
from tracing import start_span, CHAIN, INPUT_VALUE, OUTPUT_VALUE, STATUS_OK
from utils.opensearch_utils import (
    ensure_index_exists,
    ensure_fulltext_index_exists,
    ensure_ingest_log_index_exists,
    ensure_ingest_plan_index_exists,
    missing_indices,
)
from utils.ingest_plans import (
    add_planned_ingestions,
    get_planned_ingestions,
    update_plan_status,
    clear_planned_ingestions,
)

st.set_page_config(page_title="Ingest Documents", layout="wide")
st.title("📥 Ingest Documents")

missing = missing_indices()
if missing:
    st.warning("Missing OpenSearch indices: " + ", ".join(missing))
    if st.button("🛠️ Create Indices"):
        ensure_index_exists()
        ensure_fulltext_index_exists()
        ensure_ingest_log_index_exists()
        ensure_ingest_plan_index_exists()
        st.success("Created required indices. Please try again.")

existing_plans = get_planned_ingestions()
if existing_plans:
    st.subheader("Planned Ingestions")
    st.dataframe(pd.DataFrame(existing_plans), height=200)
    if st.button("🧹 Clear Planned Ingestions"):
        clear_planned_ingestions()
        st.experimental_rerun()

col1, col2 = st.columns([1, 1], gap="small")

selected_files = []
with col1:
    if st.button("📄 Select File(s)"):
        selected_files = run_file_picker()
with col2:
    if st.button("📂 Select Folder"):
        selected_files = run_folder_picker()

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

if selected_files:
    add_planned_ingestions(selected_files)
    for p in selected_files:
        update_plan_status(p, "Processing")
    st.success(f"Found {len(selected_files)} path(s).")
    status_table = st.empty()
    status_line = st.empty()
    progress_bar = st.progress(0)
    eta_display = st.empty()

    stop_event = st.session_state.stop_event
    if st.button("⏹️ Interrupt"):
        stop_event.set()

    # Show the list being processed (no persistence across refresh)
    df = pd.DataFrame({"Selected Path": [p.replace("\\", "/") for p in selected_files]})
    status_table.dataframe(df, height=300)

    def update_progress(done: int, total: int, elapsed: float):
        if stop_event.is_set():
            raise RuntimeError("Interrupted")
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

        # Ingest now does: load → split → enqueue Celery batches (OS + Qdrant in background)
        results = ingest(
            selected_files,
            progress_callback=update_progress,
            stop_event=stop_event,
        )

        for r in results:
            update_plan_status(
                r.get("path", ""),
                "Completed" if r.get("success") else "Failed",
            )

        successes = [r for r in results if r.get("success")]
        failures = [
            (r.get("path"), r.get("status")) for r in results if not r.get("success")
        ]
        # Categorise successful ingests by whether they were indexed immediately or
        # queued for background processing.
        direct_successes = [
            r
            for r in successes
            if "background" not in r.get("status", "").lower()
            and "partially" not in r.get("status", "").lower()
        ]
        queued_successes = [
            r
            for r in successes
            if "background" in r.get("status", "").lower()
            or "partially" in r.get("status", "").lower()
        ]

        span.set_attribute("indexed_files", len(successes))
        span.set_attribute("indexed_files_direct", len(direct_successes))
        span.set_attribute("indexed_files_queued", len(queued_successes))
        span.set_attribute("failed_files", len(failures))
        span.set_attribute("failed_files_details", str(failures))
        span.set_attribute(
            OUTPUT_VALUE,
            f"{len(direct_successes)} direct, {len(queued_successes)} queued, {len(failures)} failed",
        )
        span.set_status(STATUS_OK)

    # Foreground complete message (handles direct vs background indexing)
    if stop_event.is_set():
        status_line.warning(
            f"⛔ Ingestion interrupted after processing {len(results)} file(s)."
        )
    else:
        total = len(results)
        direct_count = len(direct_successes)
        queued_count = len(queued_successes)
        if direct_count and queued_count:
            status_line.success(
                f"✅ Indexed {direct_count} file(s) immediately and queued {queued_count} / {total} for background indexing."
            )
        elif direct_count:
            status_line.success(f"✅ Indexed {direct_count} / {total} file(s) immediately.")
        elif queued_count:
            status_line.success(
                f"✅ Queued {queued_count} / {total} file(s) for background indexing."
            )
        else:
            status_line.warning("⚠️ No files were indexed.")

    stop_event.clear()

    # Summary table for this run only (no persistence)
    # Shows status message from ingest: e.g., "Queued for background indexing (N batches)"
    summary_df = pd.DataFrame(
        {
            "File": [r.get("path", "") for r in results],
            "Status": [
                ("✅" if r.get("success") else "❌") + " " + r.get("status", "")
                for r in results
            ],
            "Num. Chunks": [r.get("num_chunks", 0) for r in results],
        }
    )
    status_table.dataframe(summary_df, height=300)

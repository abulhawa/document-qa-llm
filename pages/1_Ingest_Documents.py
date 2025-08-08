import streamlit as st
import pandas as pd
from core.ingestion import ingest
from ui.ingestion_ui import run_file_picker, run_folder_picker
from tracing import start_span, CHAIN, INPUT_VALUE, OUTPUT_VALUE, STATUS_OK

st.set_page_config(page_title="Ingest Documents", layout="wide")
st.title("📥 Ingest Documents")

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
    df = pd.DataFrame({"Selected Path": [p.replace("\\", "/") for p in selected_files]})
    status_table.dataframe(df, height=300)

    def update_progress(done: int, total: int, elapsed: float):
        progress_bar.progress(done / total)
        if done:
            eta = (elapsed / done) * (total - done)
            eta_display.text(
                f"{done}/{total} files ingested... (elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)"
            )
        else:
            eta_display.text(
                f"{done}/{total} files ingested... (elapsed: {elapsed:.1f}s)"
            )

    with start_span("Ingestion chain", CHAIN) as span:
        if len(selected_files) > 5:
            preview = selected_files[:5] + [
                f"... and {len(selected_files) - 5} more not shown here"
            ]
        else:
            preview = selected_files
            
        span.set_attribute(INPUT_VALUE, preview)
        results = ingest(selected_files, progress_callback=update_progress)

        successes = [r for r in results if r["success"]]
        failures = [(r["path"], r["status"]) for r in results if not r["success"]]
        span.set_attribute("indexed_files", len(successes))
        span.set_attribute("failed_files", len(failures))
        span.set_attribute("failed_files_details", str(failures))
        span.set_attribute(OUTPUT_VALUE, f"{len(successes)} indexed, {len(failures)} failed")
        span.set_status(STATUS_OK)

    status_line.success(f"✅ Indexed {len(successes)} / {len(results)} file(s).")

    summary_df = pd.DataFrame(
        {
            "File": [r["path"] for r in results],
            "Status": [("✅" if r["success"] else "❌") + " " + r["status"] for r in results],
            "Num. Chunks": [r.get("num_chunks", 0) for r in results],
        }
    )
    status_table.dataframe(summary_df, height=300)

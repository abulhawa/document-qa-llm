import streamlit as st
import pandas as pd
from datetime import datetime

from core.ingestion import ingest
from utils.opensearch_utils import search_ingest_logs

st.set_page_config(page_title="Ingestion Logs", layout="wide")
st.title("📝 Ingestion Logs")

# Filters
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    path_filter = st.text_input("Path contains", value="")
with col2:
    start_date = st.date_input("Start date", value=None)
with col3:
    end_date = st.date_input("End date", value=None)

start_str = start_date.isoformat() if start_date else None
end_str = end_date.isoformat() if end_date else None

logs = search_ingest_logs(
    status="failed", path_query=path_filter or None, start=start_str, end=end_str, size=200
)

if logs:
    df = pd.DataFrame(
        [
            {
                "Path": l.get("path"),
                "Error": l.get("error_type"),
                "Reason": (l.get("reason") or "")[:100],
                "Stage": l.get("stage"),
                "Attempt": l.get("attempt_at"),
                "log_id": l.get("log_id"),
            }
            for l in logs
        ]
    )
    df_display = df.drop(columns=["log_id"])
    st.dataframe(df_display, height=400)

    paths = df["Path"].tolist()
    retry_map = {row.Path: row.log_id for row in df.itertuples()}

    if st.button("Reingest all failed"):
        ingest(paths, force=True, op="reingest", source="viewer", retry_map=retry_map)
        st.success(f"Reingested {len(paths)} file(s)")

    for row in df.itertuples():
        if st.button("Reingest", key=row.log_id):
            ingest([row.Path], force=True, op="reingest", source="viewer", retry_map={row.Path: row.log_id})
            st.success(f"Reingested {row.Path}")
else:
    st.info("No failed ingestions found")

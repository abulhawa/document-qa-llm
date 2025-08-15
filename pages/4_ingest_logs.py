import streamlit as st
import pandas as pd
from datetime import datetime

from core.ingestion import ingest
from utils.opensearch_utils import search_ingest_logs
from utils.file_utils import format_file_size

st.set_page_config(page_title="Ingestion Logs", layout="wide")
st.title("📝 Ingestion Logs")

# Filters
col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
with col1:
    path_filter = st.text_input("Path contains", value="")
with col2:
    status_filter = st.selectbox(
        "Status",
        [
            "All",
            "Failed",
            "Success",
            "Already indexed",
            "Duplicate",
            "No valid content found",
        ],
        index=0,
    )
with col3:
    start_date = st.date_input("Start date", value=None)
with col4:
    end_date = st.date_input("End date", value=None)

start_str = start_date.isoformat() if start_date else None
end_str = end_date.isoformat() if end_date else None
status_param = None if status_filter == "All" else status_filter

logs = search_ingest_logs(
    status=status_param,
    path_query=path_filter or None,
    start=start_str,
    end=end_str,
    size=200,
)
if logs:
    df = pd.DataFrame(
        [
            {
                "Path": l.get("path"),
                "Size": l.get("bytes", 0),
                "Status": l.get("status"),
                "Error": l.get("error_type"),
                "Reason": (l.get("reason") or "")[:100],
                "Stage": l.get("stage"),
                "Attempt": l.get("attempt_at"),
            }
            for l in logs
        ]
    )
    st.dataframe(df.style.format({"Size": format_file_size}), height=400)
else:
    st.info("No ingestion logs found")

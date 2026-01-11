import pandas as pd
import streamlit as st

from app.schemas import IngestLogRequest
from app.usecases.ingest_logs_usecase import fetch_ingest_logs
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp

if st.session_state.get("_nav_context") != "hub":
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
            "Duplicate & Indexed",
            "No valid content found",
        ],
        index=0,
    )
with col3:
    start_date = st.date_input("Start date", value=None, format="DD/MM/YYYY")
with col4:
    end_date = st.date_input("End date", value=None, format="DD/MM/YYYY")

start_str = start_date.isoformat() if start_date else None
end_str = end_date.isoformat() if end_date else None
status_param = None if status_filter == "All" else status_filter

response = fetch_ingest_logs(
    IngestLogRequest(
        status=status_param,
        path_query=path_filter or None,
        start_date=start_str,
        end_date=end_str,
        size=200,
    )
)
if response.logs:
    df = pd.DataFrame(
        [
            {
                "Path": log.path,
                "Size": log.bytes or 0,
                "Status": log.status,
                "Error": log.error_type,
                "Reason": (log.reason or "")[:100],
                "Stage": log.stage,
                "Attempt": log.attempt_at,
            }
            for log in response.logs
        ]
    )
    df["Attempt"] = df["Attempt"].apply(lambda x: format_timestamp(x) if x else "")
    st.dataframe(df.style.format({"Size": format_file_size}), height=400)
else:
    st.info("No ingestion logs found")

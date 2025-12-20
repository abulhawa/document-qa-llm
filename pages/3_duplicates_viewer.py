import pandas as pd
import streamlit as st
from utils.opensearch_utils import get_duplicate_checksums, get_files_by_checksum
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp, format_timestamp_ampm

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Duplicate Files", page_icon="🗂️")

st.title("Duplicate Files")

checksums = get_duplicate_checksums()
if not checksums:
    st.info("No duplicate files found.")
else:
    rows = []
    for checksum in checksums:
        files = get_files_by_checksum(checksum)
        for f in files:
            rows.append(
                {
                    "Checksum": checksum,
                    "Location": f.get("path"),
                    "Canonical Path": f.get("canonical_path") or f.get("path"),
                    "Location Type": "Alias" if f.get("location_type") == "alias" else "Canonical",
                    "Filetype": f.get("filetype"),
                    "Created": format_timestamp_ampm(f.get("created_at") or ""),
                    "Modified": format_timestamp_ampm(f.get("modified_at") or ""),
                    "Indexed": format_timestamp(f.get("indexed_at") or ""),
                    "Chunks": f.get("num_chunks"),
                    "Size": f.get("bytes", 0),
                }
            )
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Size": format_file_size}), use_container_width=True)

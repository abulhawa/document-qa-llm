import streamlit as st
from opensearchpy import OpenSearch
from utils.opensearch_utils import list_files_from_opensearch


def render_file_index_viewer(os_client: OpenSearch) -> None:
    st.subheader("📂 Indexed Files (OpenSearch)")

    with st.spinner("Fetching files from OpenSearch..."):
        files = list_files_from_opensearch()

    if not files:
        st.info("No files found in OpenSearch.")
        return

    # Display as DataFrame-style table
    table_data = [
        {
            "Filename": f["filename"],
            "Path": f["path"],
            "Modified": f["modified_at"],
            "Chunks": f["num_chunks"],
            "Checksum": f["checksum"][:8],  # shortened
        }
        for f in files
    ]
    st.dataframe(table_data, use_container_width=True)

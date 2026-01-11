import pandas as pd
import streamlit as st

from app.usecases.duplicates_usecase import (
    format_duplicate_rows,
    format_duplicate_size,
    lookup_duplicates,
)

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Duplicate Files", page_icon="🗂️")

st.title("Duplicate Files")

response = lookup_duplicates()
if not response.groups:
    st.info("No duplicate files found.")
else:
    rows = format_duplicate_rows(response)
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Size": format_duplicate_size}),
        use_container_width=True,
    )

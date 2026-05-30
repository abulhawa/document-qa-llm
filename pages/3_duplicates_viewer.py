import pandas as pd
import streamlit as st

from app.usecases.duplicates_usecase import (
    format_duplicate_rows,
    format_duplicate_size,
    lookup_duplicates,
)
from utils.file_utils import open_file_local, show_in_folder

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Duplicate Files", page_icon="folder")


def load_duplicate_response():
    if "duplicate_response" not in st.session_state:
        with st.spinner("Loading duplicate files..."):
            st.session_state["duplicate_response"] = lookup_duplicates()
    return st.session_state["duplicate_response"]


st.title("Duplicate Files")
st.caption("Select duplicate indexed locations, then open the file or containing folder.")

st.session_state.setdefault("duplicate_table_nonce", 0)

refresh_col, _ = st.columns([1, 4])
with refresh_col:
    if st.button("Refresh duplicates", width="stretch"):
        st.session_state.pop("duplicate_response", None)
        st.rerun()

response = load_duplicate_response()
if not response.groups:
    st.info("No duplicate files found.")
else:
    rows = format_duplicate_rows(response)
    df = pd.DataFrame(rows)
    display_df = df.drop(
        columns=["Checksum", "Canonical Path", "Filetype", "Chunks"],
        errors="ignore",
    )

    action_bar = st.container()
    try:
        event = st.dataframe(
            display_df.style.format({"Size": format_duplicate_size}),
            width="stretch",
            hide_index=True,
            key=f"duplicate_table_{st.session_state['duplicate_table_nonce']}",
            on_select="rerun",
            selection_mode="multi-row",
        )
    except TypeError:
        event = st.dataframe(
            display_df.style.format({"Size": format_duplicate_size}),
            use_container_width=True,
            hide_index=True,
            key=f"duplicate_table_{st.session_state['duplicate_table_nonce']}",
            on_select="rerun",
            selection_mode="multi-row",
        )

    selected_rows = (event or {}).get("selection", {}).get("rows", [])
    selected_df = (
        display_df.iloc[selected_rows].copy()
        if selected_rows
        else pd.DataFrame(columns=display_df.columns)
    )
    selected_paths = (
        selected_df["Location"].dropna().astype(str).unique().tolist()
        if not selected_df.empty
        else []
    )

    with action_bar:
        st.caption(f"Selected {len(selected_paths)} file(s)")
        c1, c2, c3 = st.columns([1.2, 1.6, 1.1])
        with c1:
            if st.button("Open file", width="stretch"):
                if not selected_paths:
                    st.info("Select one or more rows first.")
                else:
                    for path in selected_paths:
                        open_file_local(path)
                    st.success(f"Opened {len(selected_paths)} file(s).")
        with c2:
            if st.button("Open containing folder", width="stretch"):
                if not selected_paths:
                    st.info("Select one or more rows first.")
                else:
                    for path in selected_paths:
                        show_in_folder(path)
                    st.success("Opened containing folder(s).")
        with c3:
            if st.button("Clear selection", width="stretch"):
                st.session_state["duplicate_table_nonce"] += 1
                st.rerun()

    with st.expander("Technical details", expanded=False):
        st.dataframe(
            df.style.format({"Size": format_duplicate_size}),
            width="stretch",
            hide_index=True,
        )

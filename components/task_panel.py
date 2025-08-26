from typing import List, Dict, Any, Tuple
import streamlit as st
from ui.task_status import fetch_states, clear_finished

def render_task_panel(records: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Renders a compact task panel.
    Returns (should_rerun, possibly_updated_records).
    """
    st.subheader("Background ingestion tasks")

    if not records:
        st.caption("No tasks enqueued in this session yet.")
        return False, records

    ids = [r["task_id"] for r in records]
    states = fetch_states(ids)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh task status"):
            return True, records
    with col2:
        if st.button("🧹 Clear finished"):
            return True, clear_finished(records, states)

    for r in records:
        tid = r["task_id"]
        state = states.get(tid, {}).get("state", "UNKNOWN")
        st.write(f"- `{r['path']}`  \n  • Task: `{tid}`  \n  • State: **{state}**")
        res = states.get(tid, {}).get("result")
        if isinstance(res, dict) and res:
            # Show a few common fields if present
            fields = {k: res[k] for k in ("status", "checksum", "n_chunks") if k in res}
            if fields:
                st.caption("  ↳ " + ", ".join(f"{k}={v}" for k, v in fields.items()))

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Refresh task status", key="bot1"):
            return True, records
    with col4:
        if st.button("🧹 Clear finished", key="bot2"):
            return True, clear_finished(records, states)

    return False, records
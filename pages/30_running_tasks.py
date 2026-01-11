import json
import streamlit as st
import pandas as pd
from app.usecases.running_tasks_usecase import fetch_running_tasks_snapshot
from ui.celery_admin import revoke_task
from ui.task_status import fetch_states, clear_finished

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Running Tasks", layout="wide")
st.title("🧵 Running Tasks")

st.session_state.setdefault("ingest_tasks", [])
st.session_state.setdefault("revoke_task_id", "")

# Failed lookback selector
col_win, _, _ = st.columns([1, 1, 3])
with col_win:
    win_label = st.selectbox("Failed lookback", ["1h", "6h", "24h", "7d"], index=2)
win_hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7}[win_label]

c1, c2, c3, c4, c5 = st.columns(5)
snapshot = fetch_running_tasks_snapshot(
    failed_window_hours=win_hours,
    failed_page=st.session_state.get("failed_page_num", 0),
    failed_page_size=st.session_state.get("failed_page_size", 25),
)
overview = snapshot["overview"]
fails = snapshot["failed_count"]
c1.metric("Active", overview["counts"]["active"])
c2.metric("Reserved", overview["counts"]["reserved"])
c3.metric("Scheduled", overview["counts"]["scheduled"])
c4.metric(f"Failed ({win_label})", "—" if fails is None else fails)
c5.metric("Queue depth", snapshot["queue_depth"])

# View failed list (paged) for the selected window
with st.expander(f"View failed tasks ({win_label})", expanded=False):
    col_fs, col_fp = st.columns([1, 1])
    with col_fs:
        failed_page_size = st.selectbox(
            "Page size", [25, 50, 100, 200], index=0, key="failed_page_size"
        )
    with col_fp:
        failed_page = st.number_input(
            "Page", min_value=0, step=1, value=0, key="failed_page_num"
        )

    rows = snapshot["failed_rows"]
    total = snapshot["failed_total"]
    st.caption(f"{total} failed task(s) in {win_label}. Showing {len(rows)}.")

    if not rows:
        st.info("No failures in this window.")
    else:
        # Read-only list (no revoke here; revoke makes sense only for non-terminal states)
        for r in rows:
            st.markdown(
                f"**{r['Task']}**  \n"
                f"ID: `{r['Task ID']}`  \n"
                f"Time: {r['Time']}  \n"
                f"State: {r['State'] or '—'}  \n"
                f"Error: {r['Error'] or '—'}",
                help=f"Args: {r['Args'] or '—'}\n\nKwargs: {r['Kwargs'] or '—'}",
            )


tab1, tab2, tab3 = st.tabs(
    [
        f"Active ({overview['counts']['active']})",
        f"Reserved ({overview['counts']['reserved']})",
        f"Scheduled ({overview['counts']['scheduled']})",
    ]
)
with tab1:
    st.dataframe(
        pd.DataFrame(snapshot["tables"]["active"]),
        use_container_width=True,
        hide_index=True,
    )
with tab2:
    st.dataframe(
        pd.DataFrame(snapshot["tables"]["reserved"]),
        use_container_width=True,
        hide_index=True,
    )
with tab3:
    st.dataframe(
        pd.DataFrame(snapshot["tables"]["scheduled"]),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---- My session tasks (from enqueues) ----
st.subheader("My session tasks")
records = st.session_state.get("ingest_tasks", [])
if not records:
    st.caption("No tasks enqueued in this session yet.")
else:
    ids = [r["task_id"] for r in records]
    states = fetch_states(ids)
    rows = []
    for r in records:
        s = states.get(r["task_id"], {})
        res = s.get("result")
        if isinstance(res, dict):
            res = {
                k: res[k]
                for k in ("status", "checksum", "n_chunks", "path")
                if k in res
            }
        rows.append(
            {
                "Path": r.get("path", ""),
                "Action": r.get("action"),
                "Task ID": r["task_id"],
                "State": s.get("state", "UNKNOWN"),
                "Result": json.dumps(res, ensure_ascii=False)[:160] if res else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    with a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with b:
        if st.button("🧹 Clear finished"):
            st.session_state["ingest_tasks"] = clear_finished(records, states)
            st.rerun()
    with c:
        st.text_input(
            "Revoke Task ID", key="revoke_task_id", placeholder="Paste a Task ID…"
        )
        term = st.checkbox(
            "Terminate (SIGTERM)",
            value=False,
            help="Only applies to STARTED tasks; ignored for queued ones.",
        )
        if st.button("🚫 Revoke"):
            revoke_id = (st.session_state.get("revoke_task_id") or "").strip()
            if not revoke_id:
                st.warning("Enter a Task ID.")
            else:
                # Check state; if terminal, skip revoke
                state_info = fetch_states([revoke_id]).get(revoke_id, {})
                state = state_info.get("state", "UNKNOWN")
                if state in {"SUCCESS", "FAILURE", "REVOKED"}:
                    st.info(
                        f"Task already in terminal state ({state}); revoke skipped."
                    )
                else:
                    try:
                        revoke_task(revoke_id, terminate=term)
                        st.success(f"Revoke sent for {revoke_id} (state was {state}).")
                    except Exception as e:
                        st.error(f"Revoke failed: {e}")

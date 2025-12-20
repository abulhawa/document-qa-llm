import os, json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
from typing import Any, Dict, List
from ui.celery_admin import (
    fetch_overview,
    failed_count_lookback,
    redis_queue_depth,
    revoke_task,
    list_failed_tasks,
)
from ui.task_status import fetch_states, clear_finished

if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Running Tasks", layout="wide")
st.title("🧵 Running Tasks")

st.session_state.setdefault("ingest_tasks", [])
st.session_state.setdefault("revoke_task_id", "")

# ---- Stats (fast) ----
ov = fetch_overview(timeout=0.5, cache_ttl=2.0)
# Failed lookback selector
col_win, _, _ = st.columns([1, 1, 3])
with col_win:
    win_label = st.selectbox("Failed lookback", ["1h", "6h", "24h", "7d"], index=2)
win_hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7}[win_label]
fails = failed_count_lookback(win_hours)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active", ov["counts"]["active"])
c2.metric("Reserved", ov["counts"]["reserved"])
c3.metric("Scheduled", ov["counts"]["scheduled"])
c4.metric(f"Failed ({win_label})", "—" if fails is None else fails)
c5.metric("Queue depth", redis_queue_depth("ingest"))

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

    rows, total = list_failed_tasks(win_hours, failed_page, failed_page_size)
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


def _short(val, limit=120):
    if val is None:
        s = ""
    elif isinstance(val, (dict, list, tuple)):
        try:
            s = json.dumps(val, ensure_ascii=False)
        except Exception:
            s = str(val)
    else:
        s = str(val)
    return s[:limit]

def _fmt_time(val):
    """Render inspector's eta/time_start in local time, handling floats and strings."""
    if val in (None, "", 0):
        return ""
    # numeric epoch (seconds or ms)
    if isinstance(val, (int, float)):
        ts = float(val)
        # Heuristic: if it's ridiculously large, treat as milliseconds
        if ts > 10**12:  # e.g. 1693050000000
            ts = ts / 1000.0
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(val)
    # string: try ISO parse with stdlib
    if isinstance(val, str):
        try:
            # Accept both "YYYY-MM-DD HH:MM:SS" and ISO "YYYY-MM-DDTHH:MM:SS[+TZ]"
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))  # handle trailing Z
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return val  # fallback raw
    return str(val)

# ---- Tables (first 200 rows to keep snappy) ----
def _flatten(kind: str, by_worker: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for worker, items in (by_worker or {}).items():
        for t in items or []:
            req = t.get("request") or {}
            args_raw = t.get("args") or req.get("argsrepr") or req.get("args")
            kwargs_raw = t.get("kwargs") or req.get("kwargsrepr") or req.get("kwargs")
            # Prefer ETA for scheduled entries; otherwise show time_start for active
            eta_raw = t.get("eta") or req.get("eta") or t.get("time_start")
            rows.append({
                "Type": kind,
                "Worker": worker,
                "Task": t.get("name") or t.get("type"),
                "ID": t.get("id") or req.get("id"),
                "Args": _short(args_raw),
                "Kwargs": _short(kwargs_raw),
                "ETA": _fmt_time(eta_raw),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.iloc[:200].reset_index(drop=True)  # keep UI snappy
    return df



tab1, tab2, tab3 = st.tabs(
    [
        f"Active ({ov['counts']['active']})",
        f"Reserved ({ov['counts']['reserved']})",
        f"Scheduled ({ov['counts']['scheduled']})",
    ]
)
with tab1:
    st.dataframe(
        _flatten("Active", ov["active"]), use_container_width=True, hide_index=True
    )
with tab2:
    st.dataframe(
        _flatten("Reserved", ov["reserved"]), use_container_width=True, hide_index=True
    )
with tab3:
    st.dataframe(
        _flatten("Scheduled", ov["scheduled"]),
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

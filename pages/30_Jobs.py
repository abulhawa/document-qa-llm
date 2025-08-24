import time, uuid
from pathlib import Path
import streamlit as st
from core.job_control import set_state, get_state, get_stats, incr_stat
from core.job_queue import push_pending, inflight, pending_len, retry_len
from core.discovery_filters import should_skip
from core.feeder import feed_once
from core.job_commands import pause_job, resume_job, cancel_job, stop_job
from core.reaper import reap_stale_active

st.set_page_config(page_title="Ingestion Jobs", layout="wide")
st.title("🧩 Ingestion Jobs")

# --- Pick / create a job ---
if "job_id" not in st.session_state:
    st.session_state.job_id = uuid.uuid4().hex[:8]

c1, c2 = st.columns([3, 1])
with c1:
    st.text_input("Job ID", key="job_id")
with c2:
    if st.button("New Job ID"):
        st.session_state.job_id = uuid.uuid4().hex[:8]

job_id = st.session_state.job_id

# --- Register paths under C:\ ---
st.subheader("Register files")
root = st.text_input(
    "Folder under C:\\ to scan",
    value=r"C:\Users",
    help="Will recurse; system dirs are skipped.",
)
limit = st.number_input(
    "Max files to register this run", min_value=1, max_value=2_000_000, value=5000, step=1000
)
if st.button("Scan & Register"):
    set_state(job_id, "running")
    added = 0
    for p in Path(root).rglob("*"):
        if p.is_file() and not should_skip(str(p)):
            push_pending(job_id, str(p))
            incr_stat(job_id, "registered", 1)
            added += 1
            if added >= limit:
                break
    st.success(f"Registered {added} files to job {job_id}")

# --- Stats ---

state = get_state(job_id)
stats = get_stats(job_id)
pending = pending_len(job_id)
active = inflight(job_id)
retry = retry_len(job_id)

if state in {"paused", "stopping", "canceled"}:
    moved = reap_stale_active(job_id)
    if moved:
        st.toast(f"Reaped {moved} stale active file(s)")

st.subheader("Status")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("State", state)
m2.metric("Pending", f"{pending:,}")
m3.metric("Active", f"{active:,}")
m4.metric("Needs retry", f"{retry:,}")
m5.metric("Done", f"{stats.get('done',0):,}")
m6.metric("Failed", f"{stats.get('failed',0):,}")

# --- Controls ---
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("▶️ Resume"):
        resume_job(job_id)
        st.rerun()
with c2:
    if st.button("⏸ Pause"):
        pause_job(job_id)
        st.rerun()
with c3:
    if st.button("🛑 Cancel"):
        cancel_job(job_id)
        st.rerun()
with c4:
    if st.button("⏹ Stop (drain)"):
        stop_job(job_id)
        st.rerun()
with c5:
    if st.button("➡️ Feed now"):
        started = feed_once(job_id)
        st.toast(f"Enqueued {started} file(s)")

# --- Auto-feed while running ---
if state == "running":
    started = feed_once(job_id)
    if started:
        st.toast(f"Enqueued {started} file(s)")
    st.autorefresh = st.experimental_rerun  # for older Streamlit, emulate
    time.sleep(1.5)
    st.experimental_rerun()

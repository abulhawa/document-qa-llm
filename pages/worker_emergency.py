"""
Streamlit – Worker Emergency Controls

Single‑worker, single‑queue control panel to safely pause/stop Celery and purge broker queues.
- Designed for your current setup: Redis broker (DB 0), Redis result backend (DB 1), default queue "ingest".
- Includes Docker Compose fallbacks to stop/start the celery container from the UI (optional).

⚠️ Destructive actions are double‑confirmed. Use with care.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from celery import Celery
import redis

from app.usecases.worker_emergency_usecase import (
    WorkerEmergencyStatus,
    load_status,
    purge_queues,
    rate_limit_zero,
    revoke_all_active,
    run_compose,
)

# ----------------------------
# Configuration (env‑driven)
# ----------------------------
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
QUEUE_NAMES = os.getenv("CELERY_QUEUES", "ingest,celery").split(",")  # purge targets
DEFAULT_TASK = os.getenv("CELERY_DEFAULT_TASK", "tasks.ingest_document")

# Docker Compose controls (optional; used for stop/start)
COMPOSE_DIR = Path(os.getenv("COMPOSE_DIR", Path.cwd()))
COMPOSE_PROJECT = os.getenv("COMPOSE_PROJECT", "document_qa")
CELERY_SERVICE = os.getenv("CELERY_SERVICE", "celery")

# ----------------------------
# Clients (lazy)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_celery() -> Celery:
    app = Celery(broker=BROKER_URL, backend=RESULT_BACKEND)
    return app

@st.cache_resource(show_spinner=False)
def get_redis(db_url: str) -> redis.Redis:
    return redis.Redis.from_url(db_url, decode_responses=True)

# UI
# ----------------------------
if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Worker Emergency", page_icon="🧯", layout="wide")
st.title("Worker Emergency")
st.caption("Kill or pause Celery safely and purge pending tasks. Use with care – destructive operations ahead.")

c1, c2, c3 = st.columns(3)
with c1:
    st.write("**Broker**:", BROKER_URL)
with c2:
    st.write("**Result**:", RESULT_BACKEND)
with c3:
    st.write("**Compose dir**:", str(COMPOSE_DIR))

app = get_celery()
rb = get_redis(BROKER_URL)
rr = get_redis(RESULT_BACKEND)

# Status block
st.subheader("Status")
status = load_status(app, rb, rr, QUEUE_NAMES)
qcols = st.columns(len(QUEUE_NAMES) + 3)
for idx, q in enumerate(QUEUE_NAMES):
    qcols[idx].metric(f"Queue: {q}", status.queue_lengths.get(q, -1))
qcols[-3].metric("Active", status.active)
qcols[-2].metric("Reserved", status.reserved)
qcols[-1].metric("Scheduled", status.scheduled)

def render_status_alerts(status_data: WorkerEmergencyStatus) -> None:
    if not status_data.broker_ok:
        st.error(f"Broker connectivity check failed: {status_data.broker_error}")
    if not status_data.result_ok:
        st.error(f"Result backend connectivity check failed: {status_data.result_error}")
    if status_data.celery_error:
        st.warning(f"Celery inspection failed: {status_data.celery_error}")


render_status_alerts(status)

st.divider()

# Controls
st.subheader("Controls")

with st.expander("⛔ Pause/resume queue consumption (cancel/add consumer)", expanded=True):
    st.write("Stops/starts the worker from **fetching** new messages from a queue. Already active/reserved tasks continue unless revoked.")
    qname = st.text_input("Queue name", value=QUEUE_NAMES[0] if QUEUE_NAMES else "ingest")
    cc1, cc2 = st.columns(2)
    if cc1.button("Cancel consumer (pause queue)"):
        try:
            res = get_celery().control.cancel_consumer(qname)
            st.success(f"Cancelled consumer for queue: {qname}{res}")
        except Exception as e:
            st.error(f"cancel_consumer failed: {e}")
    if cc2.button("Add consumer (resume queue)"):
        try:
            res = get_celery().control.add_consumer(qname)
            st.success(f"Added consumer for queue: {qname}{res}")
        except Exception as e:
            st.error(f"add_consumer failed: {e}")

with st.expander("🧮 Autoscale pool (instant pause/resume)", expanded=False):
    st.write("Dynamically set worker pool size. Use **0/0** to fully pause execution; resume by setting a nonzero max.")
    max_val = st.number_input("Max processes", min_value=0, value=8)
    min_val = st.number_input("Min processes", min_value=0, value=0)
    if st.button("Apply autoscale"):
        try:
            res = get_celery().control.autoscale(max_val, min_val)
            st.success(f"Autoscale applied: max={max_val}, min={min_val}{res}")
        except Exception as e:
            st.error(f"autoscale failed: {e}")


with st.expander("🛑 Stop worker (Docker Compose)", expanded=False):
    st.write("Stops the Celery container via Docker Compose (recommended big switch).")
    if st.button("Stop worker container", type="primary"):
        res = run_compose(
            ["stop", CELERY_SERVICE],
            compose_dir=COMPOSE_DIR,
            compose_project=COMPOSE_PROJECT,
        )
        st.code(res.stdout or res.stderr or "(no output)")

with st.expander("⏸️ Pause consumption (rate limit 0/m)", expanded=False):
    st.write("Prevents the worker from pulling new tasks while it stays up.")
    if st.button("Apply 0/m to ingest task"):
        rate_limit_zero(app, DEFAULT_TASK)
        st.success("Rate limit applied: 0/m")

with st.expander("🧨 Revoke running tasks", expanded=False):
    st.write("Force‑terminate any active tasks (SIGTERM). Use SIGKILL only if TERM fails.")
    col_a, col_b = st.columns(2)
    if col_a.button("Revoke active (SIGTERM)"):
        n = revoke_all_active(app, signal="SIGTERM")
        st.success(f"Revoked {n} active task(s) with SIGTERM.")
    if col_b.button("Revoke active (SIGKILL – last resort)"):
        n = revoke_all_active(app, signal="SIGKILL")
        st.warning(f"Revoked {n} active task(s) with SIGKILL.")

with st.expander("🧹 Purge queued messages (Redis broker)", expanded=True):
    st.write("Deletes the listed Redis lists (queues).\n**This is destructive.**")
    purge_names = st.text_input("Queues to purge (comma‑separated)", value=",".join(QUEUE_NAMES))
    confirm = st.text_input("Type PURGE to confirm", value="")
    if st.button("Purge queues"):
        if confirm.strip().upper() != "PURGE":
            st.error("Confirmation text mismatch. Type PURGE to proceed.")
        else:
            names = [n.strip() for n in purge_names.split(",") if n.strip()]
            res = purge_queues(rb, names)
            st.json(res)

with st.expander("🧼 Flush Redis DBs (nuclear)", expanded=False):
    st.write("Flush broker DB 0 (messages) and result DB 1. **Irreversible**.")
    c1, c2 = st.columns(2)
    if c1.button("FLUSHDB broker (DB 0)"):
        try:
            rb.flushdb()
            st.success("Broker FLUSHDB completed (DB 0).")
        except Exception as e:
            st.error(f"Broker FLUSHDB failed: {e}")
    if c2.button("FLUSHDB results (DB 1)"):
        try:
            rr.flushdb()
            st.success("Result FLUSHDB completed (DB 1).")
        except Exception as e:
            st.error(f"Result FLUSHDB failed: {e}")

with st.expander("▶️ Start worker (Docker Compose)", expanded=False):
    st.write("Starts the Celery container via Docker Compose.")
    if st.button("Start worker container"):
        res = run_compose(
            ["up", "-d", CELERY_SERVICE],
            compose_dir=COMPOSE_DIR,
            compose_project=COMPOSE_PROJECT,
        )
        st.code(res.stdout or res.stderr or "(no output)")

st.divider()
st.caption("Tip: keep your ingestion vectors-first with Qdrant upsert(wait=True) so OpenSearch never runs ahead, even under load.")

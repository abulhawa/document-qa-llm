import streamlit as st

from services.qdrant_file_vectors import (
    build_missing_file_vectors,
    ensure_file_vectors_collection,
    get_file_vectors_count,
    get_unique_checksums_in_chunks,
    sample_file_vectors,
)

st.set_page_config(page_title="Topic Discovery", layout="wide")

st.title("Topic Discovery")


tabs = st.tabs(["Overview", "Admin"])

with tabs[0]:
    st.caption("Discover document topics and prepare file-level vectors for clustering.")
    st.info("Step 1 builds file-level vectors from existing chunk embeddings in Qdrant.")

with tabs[1]:
    st.subheader("File vectors")

    status_placeholder = st.empty()

    @st.cache_data(ttl=30)
    def _get_status() -> tuple[int, int, float]:
        unique_checksums = get_unique_checksums_in_chunks()
        file_vectors_count = get_file_vectors_count()
        total_files = len(unique_checksums)
        coverage = (file_vectors_count / total_files * 100) if total_files else 0.0
        return total_files, file_vectors_count, coverage

    def _render_status() -> None:
        try:
            total_files, file_vectors_count, coverage = _get_status()
            with status_placeholder.container():
                cols = st.columns(3)
                cols[0].metric("Unique files in chunks", total_files)
                cols[1].metric("File vectors", file_vectors_count)
                cols[2].metric("Coverage", f"{coverage:.1f}%")
        except Exception as exc:  # noqa: BLE001
            status_placeholder.error(f"Status check failed: {exc}")

    _render_status()

    status_cols = st.columns(2)
    with status_cols[0]:
        if st.button("Refresh status"):
            _get_status.clear()
            _render_status()

    st.divider()

    controls = st.columns(3)
    with controls[0]:
        k_value = st.number_input("Top-k chunks", min_value=1, max_value=64, value=8)
    with controls[1]:
        batch_value = st.number_input("Upsert batch size", min_value=1, max_value=512, value=64)
    with controls[2]:
        limit_value = st.number_input("Limit (0 = all)", min_value=0, max_value=50000, value=0)

    limit_arg = None if limit_value == 0 else int(limit_value)

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Ensure collection"):
            with st.spinner("Ensuring collection..."):
                ensure_file_vectors_collection()
            st.success("Collection is ready.")
            _get_status.clear()
            _render_status()

    def _run_build(force: bool, recreate: bool = False) -> None:
        progress_bar = st.progress(0.0)
        log_placeholder = st.empty()
        logs: list[str] = []

        def _on_progress(payload: dict) -> None:
            total = payload.get("total") or 1
            processed = payload.get("processed", 0)
            progress_bar.progress(min(processed / total, 1.0))
            message = f"{processed}/{total} {payload.get('checksum')} ({payload.get('status')})"
            logs.append(message)
            log_placeholder.text("\n".join(logs[-50:]))

        if recreate:
            ensure_file_vectors_collection(recreate=True)

        with st.spinner("Building file vectors..."):
            stats = build_missing_file_vectors(
                k=int(k_value),
                batch=int(batch_value),
                limit=limit_arg,
                force=force,
                progress_callback=_on_progress,
            )

        progress_bar.progress(1.0)
        st.success(
            f"Processed {stats['processed']} file(s) | Created {stats['created']} | "
            f"Skipped {stats['skipped']} | Errors {stats['errors']}"
        )
        if stats.get("error_list"):
            st.warning("Errors:\n" + "\n".join(stats["error_list"]))
        _get_status.clear()
        _render_status()

    with action_cols[1]:
        if st.button("Build missing file vectors", type="primary"):
            _run_build(force=False)

    with action_cols[2]:
        confirm = st.checkbox("Confirm rebuild all", value=False)
        if st.button("Rebuild all", disabled=not confirm):
            _run_build(force=True, recreate=True)

    st.divider()

    st.subheader("Sanity check")
    st.caption("Sample vectors should have dim=768 and norm close to 1.0.")
    if st.button("Sample 5 vectors"):
        samples = sample_file_vectors(limit=5)
        if samples:
            st.dataframe(samples, use_container_width=True)
        else:
            st.info("No file vectors available to sample.")

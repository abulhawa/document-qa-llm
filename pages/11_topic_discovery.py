import streamlit as st

from services.qdrant_file_vectors import (
    build_missing_file_vectors,
    ensure_file_vectors_collection,
    get_file_vectors_count,
    get_unique_checksums_in_chunks,
    sample_file_vectors,
)
from services.topic_discovery_clusters import (
    attach_representative_checksums,
    build_cluster_cache_result,
    cluster_cache_exists,
    load_all_file_vectors,
    load_last_cluster_cache,
    run_hdbscan,
    save_cluster_cache,
)

st.set_page_config(page_title="Topic Discovery", layout="wide")

st.title("Topic Discovery")


tabs = st.tabs(["Overview", "Admin"])

with tabs[0]:
    st.caption("Discover document topics and prepare file-level vectors for clustering.")
    st.info("Step 1 builds file-level vectors from existing chunk embeddings in Qdrant.")

    st.divider()
    st.subheader("Step 2: Discover clusters")
    st.caption("Run HDBSCAN over file-level vectors to surface topical clusters and outliers.")

    control_cols = st.columns(2)
    with control_cols[0]:
        min_cluster_size = st.slider("min_cluster_size", min_value=5, max_value=50, value=15)
    with control_cols[1]:
        min_samples = st.slider("min_samples", min_value=1, max_value=30, value=10)

    action_cols = st.columns(2)
    run_clicked = action_cols[0].button("Run clustering", type="primary")
    load_clicked = action_cols[1].button("Load last run", disabled=not cluster_cache_exists())

    if run_clicked:
        with st.spinner("Loading file vectors from Qdrant..."):
            checksums, vectors, payloads = load_all_file_vectors()
        if not checksums:
            st.warning("No file vectors found. Run Step 1 first.")
        else:
            with st.spinner("Clustering with HDBSCAN..."):
                labels, probs, clusters = run_hdbscan(
                    vectors=vectors,
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    metric="cosine",
                )
                clusters = attach_representative_checksums(clusters, checksums)
                result = build_cluster_cache_result(
                    checksums=checksums,
                    payloads=payloads,
                    labels=labels,
                    probs=probs,
                    clusters=clusters,
                    params={
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "metric": "cosine",
                    },
                )
                save_cluster_cache(result)
                st.session_state["topic_discovery_clusters"] = result
                st.success("Clustering complete and cached.")

    if load_clicked:
        cached = load_last_cluster_cache()
        if cached is None:
            st.warning("No cached clustering run found.")
        else:
            st.session_state["topic_discovery_clusters"] = cached
            st.success("Loaded cached clustering results.")

    result = st.session_state.get("topic_discovery_clusters")

    if result:
        checksums = result.get("checksums", [])
        payloads = result.get("payloads", [])
        labels = result.get("labels", [])
        probs = result.get("probs", [])
        clusters = result.get("clusters", [])

        total_files = len(checksums)
        outlier_count = sum(1 for label in labels if label == -1)
        cluster_count = len([cluster for cluster in clusters if cluster.get("cluster_id", -1) >= 0])
        outlier_pct = (outlier_count / total_files * 100) if total_files else 0.0

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Total files", total_files)
        metrics_cols[1].metric("Clusters", cluster_count)
        metrics_cols[2].metric("Outliers", outlier_count)
        metrics_cols[3].metric("Outlier %", f"{outlier_pct:.1f}%")

        st.subheader("Cluster summary")
        cluster_rows = sorted(
            [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "size": cluster.get("size"),
                    "avg_prob": round(float(cluster.get("avg_prob", 0.0)), 3),
                }
                for cluster in clusters
            ],
            key=lambda item: item["size"],
            reverse=True,
        )
        if cluster_rows:
            st.dataframe(cluster_rows, use_container_width=True)
        else:
            st.info("No clusters found. Adjust parameters and rerun.")

        payload_lookup = {
            checksum: payloads[idx] if idx < len(payloads) else {}
            for idx, checksum in enumerate(checksums)
        }

        def _format_file_label(checksum: str) -> str:
            payload = payload_lookup.get(checksum, {})
            path = payload.get("path") or payload.get("file_path")
            filename = payload.get("filename") or payload.get("file_name")
            ext = payload.get("ext")
            if path:
                return f"{checksum} • {path}"
            if filename and ext:
                return f"{checksum} • {filename}.{ext}"
            if filename:
                return f"{checksum} • {filename}"
            return checksum

        if cluster_rows:
            cluster_ids = [row["cluster_id"] for row in cluster_rows]
            selected_cluster_id = st.selectbox("Select a cluster", options=cluster_ids)
            selected_cluster = next(
                (cluster for cluster in clusters if cluster.get("cluster_id") == selected_cluster_id),
                None,
            )
            if selected_cluster:
                st.markdown("**Representative files**")
                rep_checksums = selected_cluster.get("representative_checksums", [])
                if rep_checksums:
                    st.table([{"file": _format_file_label(checksum)} for checksum in rep_checksums])
                else:
                    st.info("No representatives found for this cluster.")

                st.markdown("**Lowest-confidence members**")
                member_indices = [
                    idx for idx, label in enumerate(labels) if label == selected_cluster_id
                ]
                if member_indices:
                    member_rows = sorted(
                        [
                            {
                                "file": _format_file_label(checksums[idx]),
                                "prob": round(float(probs[idx]), 3),
                            }
                            for idx in member_indices
                        ],
                        key=lambda item: item["prob"],
                    )
                    st.table(member_rows[:10])
                else:
                    st.info("No members found for this cluster.")

        st.subheader("Outliers")
        outlier_rows = sorted(
            [
                {
                    "file": _format_file_label(checksums[idx]),
                    "prob": round(float(probs[idx]), 3),
                }
                for idx, label in enumerate(labels)
                if label == -1
            ],
            key=lambda item: item["prob"],
        )
        if outlier_rows:
            st.table(outlier_rows[:15])
        else:
            st.info("No outliers detected.")

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

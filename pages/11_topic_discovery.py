import json

import streamlit as st

from services.qdrant_file_vectors import (
    build_missing_file_vectors,
    ensure_file_vectors_collection,
    get_file_vectors_count,
    get_unique_checksums_in_chunks,
    sample_file_vectors,
)
from services.topic_discovery_clusters import (
    clear_cluster_cache,
    cluster_cache_exists,
    ensure_macro_grouping,
    load_last_cluster_cache,
    run_topic_discovery_clustering,
)

st.set_page_config(page_title="Topic Discovery", layout="wide")

st.title("Topic Discovery")


tabs = st.tabs(["Overview", "Admin"])

with tabs[0]:
    st.caption("Discover document topics and prepare file-level vectors for clustering.")
    st.info("Step 1 builds file-level vectors from existing chunk embeddings in Qdrant.")

    st.divider()
    st.subheader("Clustering workflow")
    st.caption("Run a single clustering workflow that builds topics and macro themes.")

    with st.expander("Advanced options", expanded=False):
        control_cols = st.columns(2)
        with control_cols[0]:
            min_cluster_size = st.slider("min_cluster_size", min_value=5, max_value=50, value=10)
        with control_cols[1]:
            min_samples = st.slider("min_samples", min_value=1, max_value=30, value=3)
        use_umap = st.checkbox("Use UMAP before clustering", value=False)
        umap_config = {
            "n_components": 10,
            "n_neighbors": 30,
            "min_dist": 0.1,
            "metric": "cosine",
        }
        if use_umap:
            umap_cols = st.columns(3)
            with umap_cols[0]:
                umap_config["n_components"] = st.number_input(
                    "UMAP n_components", min_value=2, max_value=64, value=10
                )
            with umap_cols[1]:
                umap_config["n_neighbors"] = st.number_input(
                    "UMAP n_neighbors", min_value=2, max_value=200, value=30
                )
            with umap_cols[2]:
                umap_config["min_dist"] = st.number_input(
                    "UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1
                )

        macro_cols = st.columns(2)
        with macro_cols[0]:
            macro_min_k = st.number_input("Macro grouping min k", min_value=2, max_value=20, value=5)
        with macro_cols[1]:
            macro_max_k = st.number_input("Macro grouping max k", min_value=2, max_value=30, value=10)

    action_cols = st.columns(3)
    run_clicked = action_cols[0].button("Run clustering", type="primary")
    load_clicked = action_cols[1].button("Load last run", disabled=not cluster_cache_exists())
    clear_clicked = action_cols[2].button("Clear cache")

    if run_clicked:
        with st.spinner("Running clustering workflow..."):
            result, used_cache = run_topic_discovery_clustering(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
                metric="cosine",
                use_umap=use_umap,
                umap_config=umap_config if use_umap else None,
                macro_k_range=(int(macro_min_k), int(macro_max_k)),
                allow_cache=True,
            )
        if result is None:
            st.warning("No file vectors found. Run Step 1 first.")
        else:
            st.session_state["topic_discovery_clusters"] = result
            if used_cache:
                st.success("Loaded cached clustering run.")
            else:
                st.success("Clustering complete and cached.")

    if load_clicked:
        cached = load_last_cluster_cache()
        if cached is None:
            st.warning("No cached clustering run found.")
        else:
            cached = ensure_macro_grouping(
                cached,
                macro_k_range=(int(macro_min_k), int(macro_max_k)),
            )
            st.session_state["topic_discovery_clusters"] = cached
            st.success("Loaded cached clustering results.")
    if clear_clicked:
        removed = clear_cluster_cache()
        st.session_state.pop("topic_discovery_clusters", None)
        if removed:
            st.success("Cluster cache cleared.")
        else:
            st.info("No cache files to clear.")

    result = st.session_state.get("topic_discovery_clusters")

    if result:
        checksums = result.get("checksums", [])
        payloads = result.get("payloads", [])
        labels = result.get("labels", [])
        probs = result.get("probs", [])
        clusters = result.get("clusters", [])
        parent_summaries = result.get("parent_summaries", [])
        topic_parent_map = {
            int(key): int(value) for key, value in result.get("topic_parent_map", {}).items()
        }
        macro_metrics = result.get("macro_metrics", {})
        file_assignments = result.get("file_assignments", {})

        total_files = len(checksums)
        outlier_count = sum(1 for label in labels if label == -1)
        cluster_count = len([cluster for cluster in clusters if cluster.get("cluster_id", -1) >= 0])
        parent_count = len(parent_summaries)
        outlier_pct = (outlier_count / total_files * 100) if total_files else 0.0
        largest_parent_share = (
            max((parent.get("total_files", 0) for parent in parent_summaries), default=0)
            / total_files
            if total_files
            else 0.0
        )

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Parents", parent_count)
        metrics_cols[1].metric("Topics", cluster_count)
        metrics_cols[2].metric("Outlier %", f"{outlier_pct:.1f}%")
        metrics_cols[3].metric("Largest parent share", f"{largest_parent_share:.1%}")

        st.subheader("Parent summary")
        parent_rows = [
            {
                "parent_id": parent.get("parent_id"),
                "total_files": parent.get("total_files"),
                "n_topics": parent.get("n_topics"),
            }
            for parent in parent_summaries
        ]
        if parent_rows:
            st.dataframe(parent_rows, use_container_width=True)
        else:
            st.info("No parent groups available. Adjust parameters and rerun.")

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

        if parent_rows:
            parent_ids = [row["parent_id"] for row in parent_rows]
            selected_parent_id = st.selectbox("Select a parent group", options=parent_ids)
            selected_topics = [
                cluster
                for cluster in clusters
                if topic_parent_map.get(int(cluster.get("cluster_id", -1))) == selected_parent_id
            ]
            topic_rows = sorted(
                [
                    {
                        "topic_id": cluster.get("cluster_id"),
                        "size": cluster.get("size"),
                        "avg_prob": round(float(cluster.get("avg_prob", 0.0)), 3),
                    }
                    for cluster in selected_topics
                ],
                key=lambda item: item["size"],
                reverse=True,
            )
            if topic_rows:
                st.dataframe(topic_rows, use_container_width=True)
            else:
                st.info("No topics found for this parent group.")

        export_cols = st.columns(2)
        with export_cols[0]:
            if file_assignments:
                st.download_button(
                    "Export destination map",
                    data=json.dumps(file_assignments, indent=2),
                    file_name="topic_destination_map.json",
                    mime="application/json",
                )
            else:
                st.button("Export destination map", disabled=True)

        with st.expander("Diagnostics: topic-level table", expanded=False):
            cluster_rows = sorted(
                [
                    {
                        "topic_id": cluster.get("cluster_id"),
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
                st.info("No topics found. Adjust parameters and rerun.")

        with st.expander("Diagnostics: topic representatives", expanded=False):
            cluster_ids = [cluster.get("cluster_id") for cluster in clusters if cluster.get("cluster_id") is not None]
            if cluster_ids:
                selected_cluster_id = st.selectbox(
                    "Select a topic",
                    options=cluster_ids,
                    key="diagnostic_topic_select",
                )
                selected_cluster = next(
                    (cluster for cluster in clusters if cluster.get("cluster_id") == selected_cluster_id),
                    None,
                )
                if selected_cluster:
                    rep_checksums = selected_cluster.get("representative_checksums", [])
                    if rep_checksums:
                        st.table([{"file": _format_file_label(checksum)} for checksum in rep_checksums])
                    else:
                        st.info("No representatives found for this topic.")
            else:
                st.info("No topics available.")

        with st.expander("Diagnostics: outliers", expanded=False):
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

        with st.expander("Diagnostics: macro clustering metrics", expanded=False):
            selected_k = macro_metrics.get("selected_k")
            silhouette_value = macro_metrics.get("silhouette")
            largest_parent_share = macro_metrics.get("largest_parent_share")
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Selected k", selected_k if selected_k is not None else "n/a")
            metrics_cols[1].metric(
                "Silhouette (cosine)",
                f"{silhouette_value:.3f}" if silhouette_value is not None else "n/a",
            )
            metrics_cols[2].metric(
                "Largest parent share",
                f"{largest_parent_share:.1%}" if largest_parent_share is not None else "n/a",
            )
            auto_split = macro_metrics.get("auto_split", {})
            if auto_split.get("applied"):
                new_parent_ids = auto_split.get("new_parent_ids", [])
                new_shares = auto_split.get("new_shares", [])
                st.caption(
                    "Auto-split applied to parent "
                    f"{auto_split.get('original_parent_id')} "
                    f"(share {auto_split.get('original_share', 0.0):.1%}) → parents "
                    f"{new_parent_ids[0] if len(new_parent_ids) > 0 else 'n/a'} / "
                    f"{new_parent_ids[1] if len(new_parent_ids) > 1 else 'n/a'} "
                    f"(shares {new_shares[0] if len(new_shares) > 0 else 0.0:.1%} / "
                    f"{new_shares[1] if len(new_shares) > 1 else 0.0:.1%})."
                )
            candidate_rows = macro_metrics.get("candidates", [])
            if candidate_rows:
                st.dataframe(candidate_rows, use_container_width=True)

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

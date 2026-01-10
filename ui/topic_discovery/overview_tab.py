import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from config import logger
from services.topic_discovery_clusters import (
    clear_cluster_cache,
    cluster_cache_exists,
    ensure_macro_grouping,
    load_last_cluster_cache,
    run_topic_discovery_clustering,
)
from utils.timing import set_run_id, timed_block

from .shared import format_file_label


def render_overview_tab() -> None:
    _render_intro()
    settings = _render_clustering_settings()
    _handle_clustering_actions(settings)
    result = st.session_state.get("topic_discovery_clusters")
    if result:
        _render_cluster_results(result)


def _render_intro() -> None:
    st.caption("Discover document topics and prepare file-level vectors for clustering.")
    st.info("Step 1 builds file-level vectors from existing chunk embeddings in Qdrant.")
    st.divider()
    st.subheader("Clustering workflow")
    st.caption("Run a single clustering workflow that builds topics and macro themes.")


def _render_clustering_settings() -> dict[str, Any]:
    with st.expander("Advanced options", expanded=False):
        control_cols = st.columns(2)
        with control_cols[0]:
            min_cluster_size = st.slider(
                "min_cluster_size", min_value=5, max_value=50, value=10
            )
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
            macro_min_k = st.number_input(
                "Macro grouping min k", min_value=2, max_value=20, value=5
            )
        with macro_cols[1]:
            macro_max_k = st.number_input(
                "Macro grouping max k", min_value=2, max_value=30, value=10
            )

    return {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "use_umap": use_umap,
        "umap_config": umap_config if use_umap else None,
        "macro_k_range": (int(macro_min_k), int(macro_max_k)),
    }


def _handle_clustering_actions(settings: Mapping[str, Any]) -> None:
    action_cols = st.columns(3)
    run_clicked = action_cols[0].button(
        "Run clustering", type="primary", key="run_clustering"
    )
    load_clicked = action_cols[1].button(
        "Load last run", disabled=not cluster_cache_exists()
    )
    clear_clicked = action_cols[2].button("Clear cache")

    if run_clicked:
        _run_clustering(settings)
    if load_clicked:
        _load_cached(settings)
    if clear_clicked:
        _clear_cache()


def _run_clustering(settings: Mapping[str, Any]) -> None:
    run_id = uuid.uuid4().hex[:8]
    st.session_state["_run_id"] = run_id
    set_run_id(run_id)
    with st.spinner("Running clustering workflow..."):
        with timed_block(
            "action.topic_discovery.run_clustering",
            extra={
                "run_id": run_id,
                "min_cluster_size": settings["min_cluster_size"],
                "min_samples": settings["min_samples"],
                "use_umap": settings["use_umap"],
            },
            logger=logger,
        ):
            result, used_cache = run_topic_discovery_clustering(
                min_cluster_size=settings["min_cluster_size"],
                min_samples=settings["min_samples"],
                metric="cosine",
                use_umap=settings["use_umap"],
                umap_config=settings["umap_config"],
                macro_k_range=settings["macro_k_range"],
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


def _load_cached(settings: Mapping[str, Any]) -> None:
    cached = load_last_cluster_cache()
    if cached is None:
        st.warning("No cached clustering run found.")
        return
    cached = ensure_macro_grouping(
        cached,
        macro_k_range=settings["macro_k_range"],
    )
    st.session_state["topic_discovery_clusters"] = cached
    st.success("Loaded cached clustering results.")


def _clear_cache() -> None:
    removed = clear_cluster_cache()
    st.session_state.pop("topic_discovery_clusters", None)
    if removed:
        st.success("Cluster cache cleared.")
    else:
        st.info("No cache files to clear.")


def _render_cluster_results(result: Mapping[str, Any]) -> None:
    checksums = result.get("checksums", [])
    payloads = result.get("payloads", [])
    labels = result.get("labels", [])
    probs = result.get("probs", [])
    clusters = result.get("clusters", [])
    parent_summaries = result.get("parent_summaries", [])
    topic_parent_map = {
        int(key): int(value)
        for key, value in result.get("topic_parent_map", {}).items()
    }
    macro_metrics = result.get("macro_metrics", {})
    file_assignments = result.get("file_assignments", {})

    _render_summary_metrics(checksums, labels, clusters, parent_summaries)
    parent_rows = _render_parent_summary(parent_summaries)
    payload_lookup = _build_payload_lookup(checksums, payloads)
    if parent_rows:
        _render_parent_topics(parent_rows, clusters, topic_parent_map)
    _render_export_controls(file_assignments)
    _render_diagnostics(
        checksums,
        labels,
        probs,
        clusters,
        payload_lookup,
        macro_metrics,
    )


def _render_summary_metrics(
    checksums: list[Any],
    labels: list[Any],
    clusters: list[Mapping[str, Any]],
    parent_summaries: list[Mapping[str, Any]],
) -> None:
    total_files = len(checksums)
    outlier_count = sum(1 for label in labels if label == -1)
    cluster_count = len(
        [cluster for cluster in clusters if cluster.get("cluster_id", -1) >= 0]
    )
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


def _render_parent_summary(
    parent_summaries: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
    return parent_rows


def _build_payload_lookup(
    checksums: list[str], payloads: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        checksum: dict(payloads[idx]) if idx < len(payloads) else {}
        for idx, checksum in enumerate(checksums)
    }


def _render_parent_topics(
    parent_rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    topic_parent_map: Mapping[int, int],
) -> None:
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
        key=lambda item: int(item.get("size") or 0),
        reverse=True,
    )
    if topic_rows:
        st.dataframe(topic_rows, use_container_width=True)
    else:
        st.info("No topics found for this parent group.")


def _render_export_controls(file_assignments: Mapping[str, Any]) -> None:
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


def _render_diagnostics(
    checksums: list[str],
    labels: list[Any],
    probs: list[Any],
    clusters: list[Mapping[str, Any]],
    payload_lookup: Mapping[str, Mapping[str, Any]],
    macro_metrics: Mapping[str, Any],
) -> None:
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
            key=lambda item: int(item.get("size") or 0),
            reverse=True,
        )
        if cluster_rows:
            st.dataframe(cluster_rows, use_container_width=True)
        else:
            st.info("No topics found. Adjust parameters and rerun.")

    with st.expander("Diagnostics: topic representatives", expanded=False):
        cluster_ids = [
            cluster.get("cluster_id")
            for cluster in clusters
            if cluster.get("cluster_id") is not None
        ]
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
                    st.table(
                        [
                            {
                                "file": format_file_label(
                                    payload_lookup.get(checksum, {}),
                                    checksum,
                                )
                            }
                            for checksum in rep_checksums
                        ]
                    )
                else:
                    st.info("No representatives found for this topic.")
        else:
            st.info("No topics available.")

    with st.expander("Diagnostics: outliers", expanded=False):
        outlier_rows = sorted(
            [
                {
                    "file": format_file_label(
                        payload_lookup.get(checksums[idx], {}),
                        checksums[idx],
                    ),
                    "prob": round(float(probs[idx]), 3),
                }
                for idx, label in enumerate(labels)
                if label == -1
            ],
            key=lambda item: float(item.get("prob") or 0.0),
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
        metrics_cols[0].metric(
            "Selected k", selected_k if selected_k is not None else "n/a"
        )
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

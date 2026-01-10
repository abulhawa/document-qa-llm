import json
import uuid
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

import streamlit as st

from config import logger
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
from services import topic_naming
from services.topic_naming import (
    ClusterProfile,
    ParentProfile,
    build_cluster_profile,
    build_parent_profile,
    get_significant_keywords_from_os,
    hash_profile,
    suggest_child_name_with_llm,
    suggest_parent_name_with_llm,
)

from core.llm import check_llm_status
from ui.ingestion_ui import run_root_picker
from utils.timing import set_run_id, timed_block

st.set_page_config(page_title="Topic Discovery", layout="wide")

st.title("Topic Discovery")

DEFAULT_PROMPT_VERSION = getattr(topic_naming, "DEFAULT_PROMPT_VERSION", "v1")
DEFAULT_LANGUAGE = getattr(topic_naming, "DEFAULT_LANGUAGE", "en")
DEFAULT_MAX_KEYWORDS = getattr(topic_naming, "DEFAULT_MAX_KEYWORDS", 20)
DEFAULT_MAX_PATH_DEPTH = getattr(topic_naming, "DEFAULT_MAX_PATH_DEPTH", 4)
DEFAULT_ROOT_PATH = getattr(topic_naming, "DEFAULT_ROOT_PATH", "")
DEFAULT_TOP_EXTENSION_COUNT = getattr(topic_naming, "DEFAULT_TOP_EXTENSION_COUNT", 5)
MIXEDNESS_WARNING_THRESHOLD = 0.6
CONFIDENCE_MIXEDNESS_FACTOR = 0.5
TOPIC_NAMING_CACHE_DIR = getattr(
    topic_naming, "CACHE_DIR", Path(".cache") / "topic_naming"
)

def _format_file_label(payload: dict[str, Any], checksum: str) -> str:
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


def _format_label(identifier: int, name: str, hide_ids: bool) -> str:
    return name if hide_ids else f"{identifier} — {name}"


def _cluster_profile(
    cluster: dict[str, Any],
    payload_lookup: dict[str, dict[str, Any]],
    include_snippets: bool,
    *,
    max_keywords: int,
    max_path_depth: int | None,
    root_path: str | None,
    top_extension_count: int,
) -> ClusterProfile:
    base = build_cluster_profile(
        cluster,
        payload_lookup,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
        top_extension_count=top_extension_count,
    )
    if include_snippets:
        return base
    keywords = get_significant_keywords_from_os(
        base.representative_checksums,
        snippets=None,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
    )
    return ClusterProfile(
        cluster_id=base.cluster_id,
        size=base.size,
        avg_prob=base.avg_prob,
        centroid=list(base.centroid),
        keyword_entropy=base.keyword_entropy,
        extension_entropy=base.extension_entropy,
        embedding_spread=base.embedding_spread,
        mixedness=base.mixedness,
        representative_checksums=list(base.representative_checksums),
        representative_files=list(base.representative_files),
        representative_paths=list(base.representative_paths),
        representative_snippets=[],
        keywords=keywords,
        top_extensions=list(base.top_extensions),
    )


def _parent_profile(
    parent_id: int,
    child_profiles: list[ClusterProfile],
    *,
    top_extension_count: int,
) -> ParentProfile:
    return build_parent_profile(
        parent_id,
        child_profiles,
        top_extension_count=top_extension_count,
    )


def _cache_hit(
    profile: ClusterProfile | ParentProfile,
    model_id: str,
    *,
    ignore_cache: bool,
) -> bool:
    if ignore_cache:
        return False
    cache_key = hash_profile(
        profile,
        DEFAULT_PROMPT_VERSION,
        model_id,
        language=DEFAULT_LANGUAGE,
    )
    cache_path = Path(TOPIC_NAMING_CACHE_DIR) / f"{cache_key}.json"
    return cache_path.exists()


def _profile_rationale(profile: ClusterProfile | ParentProfile) -> str:
    payload = asdict(profile)
    keywords = ", ".join(payload.get("keywords", [])[:10])
    lines = []
    if keywords:
        lines.append(f"Keywords: {keywords}")
    mixedness = payload.get("mixedness")
    if mixedness is not None:
        lines.append(f"Mixedness: {mixedness:.3f}")
    component_labels = []
    keyword_entropy = payload.get("keyword_entropy")
    if keyword_entropy is not None:
        component_labels.append(f"keyword {keyword_entropy:.3f}")
    extension_entropy = payload.get("extension_entropy")
    if extension_entropy is not None:
        component_labels.append(f"extension {extension_entropy:.3f}")
    embedding_spread = payload.get("embedding_spread")
    if embedding_spread is not None:
        component_labels.append(f"embedding {embedding_spread:.3f}")
    if component_labels:
        lines.append("Mixedness components: " + ", ".join(component_labels))
    if isinstance(profile, ClusterProfile):
        files = payload.get("representative_files", [])
        if files:
            names = [
                entry.get("filename") or entry.get("path") or entry.get("checksum")
                for entry in files
                if entry
            ]
            if names:
                lines.append(f"Representative files: {', '.join(names[:6])}")
        snippets = payload.get("representative_snippets", [])
        if snippets:
            lines.append("Snippets:\n" + "\n".join(snippets[:3]))
        top_extensions = payload.get("top_extensions", [])
        if top_extensions:
            formatted = ", ".join(
                f"{entry.get('extension')} ({entry.get('count')})"
                for entry in top_extensions
                if entry.get("extension")
            )
            if formatted:
                lines.append(f"Top extensions: {formatted}")
        lines.append(f"Cluster size: {payload.get('size')}")
        lines.append(f"Avg prob: {payload.get('avg_prob'):.3f}")
    else:
        lines.append(f"Parent size: {payload.get('size')}")
        cluster_ids = payload.get("cluster_ids", [])
        if cluster_ids:
            lines.append(f"Child clusters: {', '.join(str(cid) for cid in cluster_ids)}")
        top_extensions = payload.get("top_extensions", [])
        if top_extensions:
            formatted = ", ".join(
                f"{entry.get('extension')} ({entry.get('count')})"
                for entry in top_extensions
                if entry.get("extension")
            )
            if formatted:
                lines.append(f"Top extensions: {formatted}")
        lines.append(f"Avg prob: {payload.get('avg_prob'):.3f}")
    return "\n".join(lines)


def _mixedness_warning(profile: ClusterProfile | ParentProfile) -> str:
    if profile.mixedness > MIXEDNESS_WARNING_THRESHOLD:
        return f"High mixedness ({profile.mixedness:.2f})"
    return ""


def _cap_confidence(confidence: float, mixedness: float) -> float:
    cap = 1.0 - mixedness * CONFIDENCE_MIXEDNESS_FACTOR
    return max(0.0, min(float(confidence), cap))


def _merge_warnings(*messages: str) -> str:
    return "; ".join([message for message in messages if message])


def _profile_differentiator(profile: ClusterProfile | ParentProfile) -> str | None:
    for keyword in profile.keywords:
        if keyword:
            return keyword
    for entry in profile.top_extensions:
        extension = entry.get("extension")
        if extension:
            return str(extension)
    return None


def _build_rows(
    *,
    child_profiles: list[ClusterProfile],
    parent_profiles: list[ParentProfile],
    llm_status: Mapping[str, Any],
    ignore_cache: bool,
) -> list[dict[str, Any]]:
    model_id = llm_status.get("current_model") or "default"
    rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    parent_differentiators: list[str | None] = []

    for profile in parent_profiles:
        cache_hit = _cache_hit(profile, model_id, ignore_cache=ignore_cache)
        suggestion = suggest_parent_name_with_llm(
            profile,
            model_id=model_id,
            allow_cache=True,
            ignore_cache=ignore_cache,
        )
        base_confidence = (
            suggestion.confidence if suggestion.confidence is not None else profile.avg_prob
        )
        parent_rows.append(
            {
                "id": profile.parent_id,
                "level": "parent",
                "proposed_name": suggestion.name,
                "confidence": _cap_confidence(base_confidence, profile.mixedness),
                "warnings": _merge_warnings(
                    (suggestion.metadata or {}).get("warning", ""),
                    _mixedness_warning(profile),
                ),
                "rationale": _profile_rationale(profile),
                "cache_hit": cache_hit,
                "source": suggestion.source,
            }
        )
        parent_differentiators.append(_profile_differentiator(profile))

    if parent_rows:
        unique_names = topic_naming.disambiguate_duplicate_names(
            [row["proposed_name"] for row in parent_rows],
            differentiators=parent_differentiators,
        )
        for row, name in zip(parent_rows, unique_names, strict=False):
            row["proposed_name"] = name

    child_rows: list[dict[str, Any]] = []
    child_differentiators: list[str | None] = []
    for profile in child_profiles:
        cache_hit = _cache_hit(profile, model_id, ignore_cache=ignore_cache)
        suggestion = suggest_child_name_with_llm(
            profile,
            model_id=model_id,
            allow_cache=True,
            ignore_cache=ignore_cache,
        )
        base_confidence = (
            suggestion.confidence if suggestion.confidence is not None else profile.avg_prob
        )
        child_rows.append(
            {
                "id": profile.cluster_id,
                "level": "child",
                "proposed_name": suggestion.name,
                "confidence": _cap_confidence(base_confidence, profile.mixedness),
                "warnings": _merge_warnings(
                    (suggestion.metadata or {}).get("warning", ""),
                    _mixedness_warning(profile),
                ),
                "rationale": _profile_rationale(profile),
                "cache_hit": cache_hit,
                "source": suggestion.source,
            }
        )
        child_differentiators.append(_profile_differentiator(profile))

    if child_rows:
        unique_names = topic_naming.disambiguate_duplicate_names(
            [row["proposed_name"] for row in child_rows],
            differentiators=child_differentiators,
        )
        for row, name in zip(child_rows, unique_names, strict=False):
            row["proposed_name"] = name

    rows.extend(parent_rows)
    rows.extend(child_rows)

    return rows


def _update_session_rows(rows: Sequence[Mapping[Hashable, Any]]) -> None:
    normalized_rows = [{str(key): value for key, value in row.items()} for row in rows]
    st.session_state["topic_naming_rows"] = normalized_rows


def _apply_names(
    *,
    rows: list[dict[str, Any]],
    payload_lookup: dict[str, dict[str, Any]],
    file_assignments: dict[str, dict[str, Any]],
    hide_ids: bool,
) -> None:
    parent_names: dict[int, str] = {}
    child_names: dict[int, str] = {}
    for row in rows:
        identifier = int(row["id"])
        name = str(row["proposed_name"]).strip() or "Untitled"
        if row["level"] == "parent":
            parent_names[identifier] = name
        else:
            child_names[identifier] = name

    move_plan: list[dict[str, Any]] = []
    for checksum, assignment in file_assignments.items():
        payload = payload_lookup.get(checksum, {})
        topic_id = int(assignment.get("topic_id", -1))
        parent_id = int(assignment.get("parent_id", -1))
        topic_name = child_names.get(topic_id, f"Topic {topic_id}")
        parent_name = parent_names.get(parent_id, f"Parent {parent_id}")
        combined = (
            f"{_format_label(parent_id, parent_name, hide_ids)} / "
            f"{_format_label(topic_id, topic_name, hide_ids)}"
        )
        move_plan.append(
            {
                "checksum": checksum,
                "file": _format_file_label(payload, checksum),
                "parent": _format_label(parent_id, parent_name, hide_ids),
                "topic": _format_label(topic_id, topic_name, hide_ids),
                "combined_label": combined,
                "prob": assignment.get("prob"),
            }
        )

    st.session_state["topic_discovery_name_map"] = {
        "parents": parent_names,
        "children": child_names,
    }
    st.session_state["topic_discovery_move_plan"] = move_plan

tabs = st.tabs(["Overview", "Naming", "Admin"])

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
    run_clicked = action_cols[0].button("Run clustering", type="primary", key="run_clustering")
    load_clicked = action_cols[1].button("Load last run", disabled=not cluster_cache_exists())
    clear_clicked = action_cols[2].button("Clear cache")

    if run_clicked:
        run_id = uuid.uuid4().hex[:8]
        st.session_state["_run_id"] = run_id
        set_run_id(run_id)
        with st.spinner("Running clustering workflow..."):
            with timed_block(
                "action.topic_discovery.run_clustering",
                extra={
                    "run_id": run_id,
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                    "use_umap": use_umap,
                },
                logger=logger,
            ):
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
                        st.table(
                            [
                                {
                                    "file": _format_file_label(
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
                        "file": _format_file_label(
                            payload_lookup.get(checksums[idx], {}),
                            checksums[idx],
                        ),
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
    st.subheader("Topic naming")
    st.caption("Generate and edit names for parent and child clusters.")

    cluster_result = st.session_state.get("topic_discovery_clusters")
    if not cluster_result:
        st.info("Run clustering in the Overview tab to generate results before naming topics.")
    else:
        llm_status = check_llm_status()

        with st.expander("Naming settings", expanded=False):
            settings_cols = st.columns(2)
            with settings_cols[0]:
                max_keywords = st.slider(
                    "Max keywords",
                    min_value=15,
                    max_value=30,
                    value=DEFAULT_MAX_KEYWORDS,
                )
                max_path_depth = st.number_input(
                    "Path segment depth (0 = no limit)",
                    min_value=0,
                    max_value=12,
                    value=DEFAULT_MAX_PATH_DEPTH,
                )
            with settings_cols[1]:
                if "topic_discovery_root_path" not in st.session_state:
                    st.session_state["topic_discovery_root_path"] = DEFAULT_ROOT_PATH
                root_action_cols = st.columns([1, 1], gap="small")
                with root_action_cols[0]:
                    if st.button("Select Root Folder", key="topic_discovery_root_pick"):
                        picked = run_root_picker()
                        if picked:
                            st.session_state["topic_discovery_root_path"] = picked[0]
                with root_action_cols[1]:
                    if st.button("Clear Root", key="topic_discovery_root_clear"):
                        st.session_state["topic_discovery_root_path"] = ""
                root_path = st.text_input(
                    "Root path prefix to drop (optional)",
                    key="topic_discovery_root_path",
                )
                top_extension_count = st.number_input(
                    "Top extensions to include",
                    min_value=1,
                    max_value=10,
                    value=DEFAULT_TOP_EXTENSION_COUNT,
                )

        max_path_depth_value = int(max_path_depth)
        max_path_depth_value = None if max_path_depth_value == 0 else max_path_depth_value
        root_path_value = root_path.strip() or None

        controls = st.columns(4)
        with controls[0]:
            include_snippets = st.toggle("Include content snippets", value=True)
        with controls[1]:
            hide_ids = st.toggle("Hide numeric IDs", value=False)
        with controls[2]:
            ignore_cache = st.checkbox(
                "Ignore cache (this run)",
                value=st.session_state.get("topic_naming_ignore_cache", False),
                key="topic_naming_ignore_cache",
            )
        with controls[3]:
            # Allow baseline naming even when the LLM is inactive.
            generate_clicked = st.button(
                "Generate names (LLM)",
                type="primary",
            )
            regenerate_clicked = st.button("Regenerate now")

        if not llm_status.get("active"):
            st.warning(
                "LLM is inactive. Baseline naming will be used until a model is loaded."
            )

        payloads = cluster_result.get("payloads", [])
        checksums = cluster_result.get("checksums", [])
        clusters = cluster_result.get("clusters", [])
        parent_summaries = cluster_result.get("parent_summaries", [])
        file_assignments = cluster_result.get("file_assignments", {})

        payload_lookup = {
            checksum: payloads[idx] if idx < len(payloads) else {}
            for idx, checksum in enumerate(checksums)
        }

        topic_parent_map = {
            int(key): int(value)
            for key, value in cluster_result.get("topic_parent_map", {}).items()
        }

        run_naming = generate_clicked or regenerate_clicked
        ignore_cache_for_run = ignore_cache or regenerate_clicked

        if run_naming:
            run_id = uuid.uuid4().hex[:8]
            st.session_state["_run_id"] = run_id
            set_run_id(run_id)
            child_profiles: list[ClusterProfile] = []
            for cluster in clusters:
                profile = _cluster_profile(
                    cluster,
                    payload_lookup,
                    include_snippets,
                    max_keywords=int(max_keywords),
                    max_path_depth=max_path_depth_value,
                    root_path=root_path_value,
                    top_extension_count=int(top_extension_count),
                )
                child_profiles.append(profile)

            parent_profiles: list[ParentProfile] = []
            child_profiles_by_parent: dict[int, list[ClusterProfile]] = {}
            for profile in child_profiles:
                parent_id = topic_parent_map.get(profile.cluster_id, -1)
                child_profiles_by_parent.setdefault(parent_id, []).append(profile)

            for parent in parent_summaries:
                parent_id = int(parent.get("parent_id", -1))
                child_group = child_profiles_by_parent.get(parent_id, [])
                if not child_group:
                    continue
                parent_profiles.append(
                    _parent_profile(
                        parent_id,
                        child_group,
                        top_extension_count=int(top_extension_count),
                    )
                )

            with timed_block(
                "action.topic_discovery.generate_names_llm",
                extra={"run_id": run_id, "ignore_cache": ignore_cache_for_run},
                logger=logger,
            ):
                with timed_block(
                    "step.topic_naming.run",
                    extra={"run_id": run_id, "child_clusters": len(child_profiles)},
                    logger=logger,
                ):
                    rows = _build_rows(
                        child_profiles=child_profiles,
                        parent_profiles=parent_profiles,
                        llm_status=llm_status,
                        ignore_cache=ignore_cache_for_run,
                    )
            if any(row["source"] != "llm" for row in rows):
                st.warning(
                    "LLM naming unavailable for some rows; using baseline names instead."
                )
            _update_session_rows(cast(Sequence[Mapping[Hashable, Any]], rows))

        rows_state = st.session_state.get("topic_naming_rows", [])

        if not rows_state:
            st.info("Generate names to populate the editable table.")
        else:
            rows_df = pd.DataFrame(rows_state)

            if "source" in rows_df.columns:
                rows_df = rows_df.drop(columns=["source"])

            st.subheader("Topic naming table")

            edited_df = st.data_editor(
                rows_df,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.NumberColumn("id", disabled=True),
                    "level": st.column_config.TextColumn("level", disabled=True),
                    "proposed_name": st.column_config.TextColumn("proposed_name"),
                    "confidence": st.column_config.NumberColumn("confidence", disabled=True),
                    "warnings": st.column_config.TextColumn("warnings", disabled=True),
                    "rationale": st.column_config.TextColumn("rationale", disabled=True),
                    "cache_hit": st.column_config.CheckboxColumn("cache_hit", disabled=True),
                },
                disabled=["id", "level", "confidence", "warnings", "rationale", "cache_hit"],
            )

            _update_session_rows(
                cast(Sequence[Mapping[Hashable, Any]], edited_df.to_dict(orient="records"))
            )

            st.caption("Rationale details")
            for row in edited_df.to_dict(orient="records"):
                label = f"{row['level'].title()} {row['id']}"
                with st.expander(label, expanded=False):
                    st.text(row.get("rationale", ""))

            apply_cols = st.columns(2)
            with apply_cols[0]:
                apply_clicked = st.button("Apply names")

            if apply_clicked:
                run_id = uuid.uuid4().hex[:8]
                st.session_state["_run_id"] = run_id
                set_run_id(run_id)
                with timed_block(
                    "action.topic_discovery.apply_names",
                    extra={"run_id": run_id, "rows": len(rows_state)},
                    logger=logger,
                ):
                    _apply_names(
                        rows=st.session_state.get("topic_naming_rows", []),
                        payload_lookup=payload_lookup,
                        file_assignments=file_assignments,
                        hide_ids=hide_ids,
                    )
                st.success("Applied names to the dry-run move plan.")

            move_plan = st.session_state.get("topic_discovery_move_plan")
            if move_plan:
                st.subheader("Dry-run move plan")
                st.dataframe(pd.DataFrame(move_plan).head(200), use_container_width=True)

with tabs[2]:
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

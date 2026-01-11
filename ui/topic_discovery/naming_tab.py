import math
import time
import uuid
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from config import logger
from core.llm import check_llm_status
from services import topic_naming
from services.topic_naming import (
    ClusterProfile,
    ParentProfile,
    build_cluster_profile,
    build_parent_profile,
    get_significant_keywords_from_os,
    hash_profile,
    suggest_child_name_with_llm,
    suggest_child_names_with_llm_batch,
    suggest_parent_name_with_llm,
    suggest_parent_names_with_llm_batch,
)
from ui.ingestion_ui import run_root_picker
from utils.timing import set_run_id, timed_block

from .shared import format_file_label, format_label

DEFAULT_PROMPT_VERSION = getattr(topic_naming, "DEFAULT_PROMPT_VERSION", "v1")
DEFAULT_LANGUAGE = getattr(topic_naming, "DEFAULT_LANGUAGE", "en")
DEFAULT_MAX_KEYWORDS = getattr(topic_naming, "DEFAULT_MAX_KEYWORDS", 20)
DEFAULT_MAX_PATH_DEPTH = getattr(topic_naming, "DEFAULT_MAX_PATH_DEPTH", 4)
DEFAULT_ROOT_PATH = getattr(topic_naming, "DEFAULT_ROOT_PATH", "")
DEFAULT_TOP_EXTENSION_COUNT = getattr(topic_naming, "DEFAULT_TOP_EXTENSION_COUNT", 5)
DEFAULT_LLM_BATCH_SIZE = getattr(topic_naming, "DEFAULT_LLM_BATCH_SIZE", 10)
DEFAULT_ALLOW_RAW_PARENT_EVIDENCE = getattr(
    topic_naming, "DEFAULT_ALLOW_RAW_PARENT_EVIDENCE", False
)
MIXEDNESS_WARNING_THRESHOLD = getattr(topic_naming, "MIXEDNESS_WARNING_THRESHOLD", 0.6)
CONFIDENCE_MIXEDNESS_FACTOR = 0.5
TOPIC_NAMING_CACHE_DIR = getattr(
    topic_naming, "CACHE_DIR", Path(".cache") / "topic_naming"
)


def render_naming_tab() -> None:
    st.subheader("Topic naming")
    st.caption("Generate and edit names for parent and child clusters.")

    cluster_result = st.session_state.get("topic_discovery_clusters")
    if not cluster_result:
        st.info("Run clustering in the Overview tab to generate results before naming topics.")
        return

    llm_status = check_llm_status()
    settings = _render_naming_settings()
    controls = _render_naming_controls()

    if not llm_status.get("active"):
        st.warning("LLM is inactive. Baseline naming will be used until a model is loaded.")

    payloads, checksums, clusters, parent_summaries, file_assignments = _extract_cluster_data(
        cluster_result
    )
    payload_lookup = _build_payload_lookup(checksums, payloads)
    topic_parent_map = {
        int(key): int(value)
        for key, value in cluster_result.get("topic_parent_map", {}).items()
    }

    _render_llm_estimate(
        clusters,
        parent_summaries,
        fast_mode=controls["fast_mode"],
        llm_batch_size=settings["llm_batch_size"],
    )

    if controls["generate_clicked"]:
        _run_naming(
            clusters=clusters,
            parent_summaries=parent_summaries,
            topic_parent_map=topic_parent_map,
            payload_lookup=payload_lookup,
            include_snippets=controls["include_snippets"],
            max_keywords=settings["max_keywords"],
            max_path_depth=settings["max_path_depth"],
            root_path=settings["root_path"],
            top_extension_count=settings["top_extension_count"],
            llm_status=llm_status,
            ignore_cache=controls["ignore_cache"],
            fast_mode=controls["fast_mode"],
            llm_batch_size=settings["llm_batch_size"],
            allow_raw_parent_evidence=settings["allow_raw_parent_evidence"],
        )

    rows_state = st.session_state.get("topic_naming_rows", [])
    if not rows_state:
        st.info("Generate names to populate the editable table.")
        return

    _render_naming_table(
        rows_state=rows_state,
        payload_lookup=payload_lookup,
        file_assignments=file_assignments,
        hide_ids=controls["hide_ids"],
    )


def _render_naming_settings() -> dict[str, Any]:
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
            llm_batch_size = st.number_input(
                "LLM batch size",
                min_value=1,
                max_value=50,
                value=DEFAULT_LLM_BATCH_SIZE,
                help="Used in fast mode for batch LLM naming.",
            )
            allow_raw_parent_evidence = st.checkbox(
                "Allow raw evidence for parent naming (fallback)",
                value=DEFAULT_ALLOW_RAW_PARENT_EVIDENCE,
            )

    max_path_depth_value = int(max_path_depth)
    max_path_depth_value = None if max_path_depth_value == 0 else max_path_depth_value
    root_path_value = root_path.strip() or None

    return {
        "max_keywords": int(max_keywords),
        "max_path_depth": max_path_depth_value,
        "root_path": root_path_value,
        "top_extension_count": int(top_extension_count),
        "llm_batch_size": int(llm_batch_size),
        "allow_raw_parent_evidence": allow_raw_parent_evidence,
    }


def _render_naming_controls() -> dict[str, Any]:
    controls = st.columns(5)
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
        fast_mode = st.toggle(
            "Fast mode (batch LLM naming)",
            value=True,
        )
    with controls[4]:
        generate_clicked = st.button(
            "Generate names (LLM)",
            type="primary",
        )

    return {
        "include_snippets": include_snippets,
        "hide_ids": hide_ids,
        "ignore_cache": ignore_cache,
        "fast_mode": fast_mode,
        "generate_clicked": generate_clicked,
    }


def _extract_cluster_data(
    cluster_result: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any], list[Any], dict[str, Any]]:
    payloads = cluster_result.get("payloads", [])
    checksums = cluster_result.get("checksums", [])
    clusters = cluster_result.get("clusters", [])
    parent_summaries = cluster_result.get("parent_summaries", [])
    file_assignments = cluster_result.get("file_assignments", {})
    return payloads, checksums, clusters, parent_summaries, file_assignments


def _build_payload_lookup(
    checksums: list[Any],
    payloads: list[Any],
) -> dict[str, dict[str, Any]]:
    return {
        checksum: payloads[idx] if idx < len(payloads) else {}
        for idx, checksum in enumerate(checksums)
    }


def _render_llm_estimate(
    clusters: list[Any],
    parent_summaries: list[Any],
    *,
    fast_mode: bool,
    llm_batch_size: int,
) -> None:
    if fast_mode:
        estimated_llm_calls = (
            math.ceil(len(clusters) / max(int(llm_batch_size), 1))
            + math.ceil(len(parent_summaries) / max(int(llm_batch_size), 1))
        )
    else:
        estimated_llm_calls = len(clusters) + len(parent_summaries)
    st.caption(f"LLM calls: ~{estimated_llm_calls}")


def _run_naming(
    *,
    clusters: list[Mapping[str, Any]],
    parent_summaries: list[Mapping[str, Any]],
    topic_parent_map: Mapping[int, int],
    payload_lookup: Mapping[str, dict[str, Any]],
    include_snippets: bool,
    max_keywords: int,
    max_path_depth: int | None,
    root_path: str | None,
    top_extension_count: int,
    llm_status: Mapping[str, Any],
    ignore_cache: bool,
    fast_mode: bool,
    llm_batch_size: int,
    allow_raw_parent_evidence: bool,
) -> None:
    run_id = uuid.uuid4().hex[:8]
    st.session_state["_run_id"] = run_id
    set_run_id(run_id)
    topic_naming.reset_os_keyword_metrics(clear_cache=True)
    topic_naming.reset_llm_request_metrics()
    child_profiles: list[ClusterProfile] = []
    for cluster in clusters:
        profile = _cluster_profile(
            cluster,
            payload_lookup,
            include_snippets,
            max_keywords=max_keywords,
            max_path_depth=max_path_depth,
            root_path=root_path,
            top_extension_count=top_extension_count,
        )
        child_profiles.append(profile)

    parent_profiles = _build_parent_profiles(
        parent_summaries,
        child_profiles,
        topic_parent_map,
        top_extension_count=top_extension_count,
    )

    with timed_block(
        "action.topic_discovery.generate_names_llm",
        extra={"run_id": run_id, "ignore_cache": ignore_cache},
        logger=logger,
    ):
        with timed_block(
            "step.topic_naming.run",
            extra={"run_id": run_id, "child_clusters": len(child_profiles)},
            logger=logger,
        ):
            start_time = time.perf_counter()
            rows = _build_rows(
                child_profiles=child_profiles,
                parent_profiles=parent_profiles,
                llm_status=llm_status,
                ignore_cache=ignore_cache,
                fast_mode=fast_mode,
                batch_size=llm_batch_size,
                allow_raw_parent_evidence=allow_raw_parent_evidence,
            )
            total_runtime = time.perf_counter() - start_time
            topic_naming.log_llm_request_summary(
                run_id=run_id,
                child_count=len(child_profiles),
                parent_count=len(parent_profiles),
                estimated_calls_before=len(child_profiles) + len(parent_profiles),
                total_runtime_s=total_runtime,
            )
    topic_naming.log_os_keyword_metrics(run_id=run_id)
    topic_naming.log_qdrant_embedding_metrics(run_id=run_id)
    if any(row["source"] != "llm" for row in rows):
        st.warning("LLM naming unavailable for some rows; using baseline names instead.")
    _update_session_rows(cast(Sequence[Mapping[Hashable, Any]], rows))


def _build_parent_profiles(
    parent_summaries: list[Mapping[str, Any]],
    child_profiles: list[ClusterProfile],
    topic_parent_map: Mapping[int, int],
    *,
    top_extension_count: int,
) -> list[ParentProfile]:
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
                top_extension_count=top_extension_count,
            )
        )
    return parent_profiles


def _render_naming_table(
    *,
    rows_state: list[dict[str, Any]],
    payload_lookup: Mapping[str, dict[str, Any]],
    file_assignments: Mapping[str, dict[str, Any]],
    hide_ids: bool,
) -> None:
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
            "llm_used": st.column_config.CheckboxColumn("llm_used", disabled=True),
            "fallback_reason": st.column_config.TextColumn(
                "fallback_reason",
                disabled=True,
            ),
            "error_summary": st.column_config.TextColumn(
                "error_summary",
                disabled=True,
            ),
            "cache_bypassed": st.column_config.CheckboxColumn(
                "cache_bypassed",
                disabled=True,
            ),
        },
        disabled=[
            "id",
            "level",
            "confidence",
            "warnings",
            "rationale",
            "cache_hit",
            "llm_used",
            "fallback_reason",
            "error_summary",
            "cache_bypassed",
        ],
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
                payload_lookup=dict(payload_lookup),
                file_assignments=dict(file_assignments),
                hide_ids=hide_ids,
            )
        st.success("Applied names to the dry-run move plan.")

    move_plan = st.session_state.get("topic_discovery_move_plan")
    if move_plan:
        st.subheader("Dry-run move plan")
        st.dataframe(pd.DataFrame(move_plan).head(200), use_container_width=True)


def _cluster_profile(
    cluster: Mapping[str, Any],
    payload_lookup: Mapping[str, Mapping[str, Any]],
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
        component_labels.append(f"kw_entropy_norm {keyword_entropy:.3f}")
    embedding_spread = payload.get("embedding_spread")
    if embedding_spread is not None:
        component_labels.append(f"emb_spread_norm {embedding_spread:.3f}")
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


def _mixedness_rationale(metadata: Mapping[str, Any]) -> str:
    subthemes = metadata.get("mixedness_subthemes") or []
    note = metadata.get("mixedness_note")
    lines: list[str] = []
    if subthemes:
        lines.append(f"Likely subthemes: {', '.join(str(item) for item in subthemes)}")
    if note:
        lines.append(f"Why mixed: {note}")
    return "\n".join(lines)


def _merge_rationale(*segments: str) -> str:
    return "\n".join(segment for segment in segments if segment)


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
    fast_mode: bool,
    batch_size: int,
    allow_raw_parent_evidence: bool,
) -> list[dict[str, Any]]:
    model_id = llm_status.get("current_model") or "default"
    rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    parent_differentiators: list[str | None] = []

    child_rows: list[dict[str, Any]] = []
    child_differentiators: list[str | None] = []
    child_suggestions: dict[int, Any] = {}
    child_confidence_map: dict[int, float] = {}
    if fast_mode and child_profiles:
        child_suggestions = suggest_child_names_with_llm_batch(
            child_profiles,
            model_id=model_id,
            allow_cache=True,
            ignore_cache=ignore_cache,
            batch_size=batch_size,
        )
    for profile in child_profiles:
        cache_hit = _cache_hit(profile, model_id, ignore_cache=ignore_cache)
        suggestion = child_suggestions.get(profile.cluster_id)
        if suggestion is None:
            suggestion = suggest_child_name_with_llm(
                profile,
                model_id=model_id,
                allow_cache=True,
                ignore_cache=ignore_cache,
            )
        llm_cache = (suggestion.metadata or {}).get("llm_cache", {})
        base_confidence = (
            suggestion.confidence if suggestion.confidence is not None else profile.avg_prob
        )
        child_confidence_map[profile.cluster_id] = float(base_confidence)
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
                "rationale": _merge_rationale(
                    _profile_rationale(profile),
                    _mixedness_rationale(suggestion.metadata or {}),
                ),
                "cache_hit": cache_hit,
                "source": suggestion.source,
                "llm_used": llm_cache.get("llm_used"),
                "fallback_reason": llm_cache.get("fallback_reason"),
                "error_summary": llm_cache.get("error_summary"),
                "cache_bypassed": llm_cache.get("cache_bypassed"),
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

    child_summary_map: dict[int, dict[str, Any]] = {}
    for profile, row in zip(child_profiles, child_rows, strict=False):
        base_confidence = child_confidence_map.get(profile.cluster_id, profile.avg_prob)
        child_summary_map[profile.cluster_id] = {
            "child_id": profile.cluster_id,
            "child_name": row.get("proposed_name", ""),
            "confidence": float(base_confidence),
            "keywords_top5": profile.keywords[:5],
            "flags": {
                "mixed": profile.mixedness >= 0.85,
                "low_conf": float(base_confidence) < 0.55,
            },
            "size": profile.size,
        }

    enriched_parent_profiles: list[ParentProfile] = []
    for profile in parent_profiles:
        child_summaries = [
            child_summary_map[child_id]
            for child_id in profile.cluster_ids
            if child_id in child_summary_map
        ]
        enriched_parent_profiles.append(
            replace(
                profile,
                child_summaries=child_summaries,
                allow_raw_parent_evidence=allow_raw_parent_evidence,
            )
        )

    parent_suggestions: dict[int, Any] = {}
    if fast_mode and enriched_parent_profiles:
        parent_suggestions = suggest_parent_names_with_llm_batch(
            enriched_parent_profiles,
            model_id=model_id,
            allow_cache=True,
            ignore_cache=ignore_cache,
            batch_size=batch_size,
        )

    for profile in enriched_parent_profiles:
        cache_hit = _cache_hit(profile, model_id, ignore_cache=ignore_cache)
        suggestion = parent_suggestions.get(profile.parent_id)
        if suggestion is None:
            suggestion = suggest_parent_name_with_llm(
                profile,
                model_id=model_id,
                allow_cache=True,
                ignore_cache=ignore_cache,
            )
        llm_cache = (suggestion.metadata or {}).get("llm_cache", {})
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
                "rationale": _merge_rationale(
                    _profile_rationale(profile),
                    _mixedness_rationale(suggestion.metadata or {}),
                ),
                "cache_hit": cache_hit,
                "source": suggestion.source,
                "llm_used": llm_cache.get("llm_used"),
                "fallback_reason": llm_cache.get("fallback_reason"),
                "error_summary": llm_cache.get("error_summary"),
                "cache_bypassed": llm_cache.get("cache_bypassed"),
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
            f"{format_label(parent_id, parent_name, hide_ids)} / "
            f"{format_label(topic_id, topic_name, hide_ids)}"
        )
        move_plan.append(
            {
                "checksum": checksum,
                "file": format_file_label(payload, checksum),
                "parent": format_label(parent_id, parent_name, hide_ids),
                "topic": format_label(topic_id, topic_name, hide_ids),
                "combined_label": combined,
                "prob": assignment.get("prob"),
            }
        )

    st.session_state["topic_discovery_name_map"] = {
        "parents": parent_names,
        "children": child_names,
    }
    st.session_state["topic_discovery_move_plan"] = move_plan

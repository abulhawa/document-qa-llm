import math
import uuid
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import pandas as pd
import streamlit as st

from config import logger
from ui.ingestion_ui import run_root_picker
from utils.timing import set_run_id, timed_block

from app.usecases import topic_naming_usecase

from .shared import format_file_label, format_label

DEFAULT_MAX_KEYWORDS = topic_naming_usecase.DEFAULT_MAX_KEYWORDS
DEFAULT_MAX_PATH_DEPTH = topic_naming_usecase.DEFAULT_MAX_PATH_DEPTH
DEFAULT_ROOT_PATH = topic_naming_usecase.DEFAULT_ROOT_PATH
DEFAULT_TOP_EXTENSION_COUNT = topic_naming_usecase.DEFAULT_TOP_EXTENSION_COUNT
DEFAULT_LLM_BATCH_SIZE = topic_naming_usecase.DEFAULT_LLM_BATCH_SIZE
DEFAULT_ALLOW_RAW_PARENT_EVIDENCE = (
    topic_naming_usecase.DEFAULT_ALLOW_RAW_PARENT_EVIDENCE
)
FAST_MODE_CLUSTER_THRESHOLD = topic_naming_usecase.FAST_MODE_CLUSTER_THRESHOLD


def render_naming_tab() -> None:
    st.subheader("Topic naming")
    st.caption("Generate and edit names for parent and child clusters.")

    cluster_result = st.session_state.get("topic_discovery_clusters")
    if not cluster_result:
        st.info("Run clustering in the Overview tab to generate results before naming topics.")
        return

    llm_status = topic_naming_usecase.get_llm_status()
    clusters = cluster_result.get("clusters", [])
    cluster_count = len(clusters)
    settings = _render_naming_settings()
    controls = _render_naming_controls(cluster_count=cluster_count)

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


def _render_naming_controls(*, cluster_count: int) -> dict[str, Any]:
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
            value=cluster_count > FAST_MODE_CLUSTER_THRESHOLD,
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
    if len(clusters) > FAST_MODE_CLUSTER_THRESHOLD and not fast_mode:
        st.info(
            "Large cluster count detected. Consider enabling Fast mode (batch LLM naming) "
            "to reduce total LLM calls."
        )


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
    if not fast_mode and len(clusters) > FAST_MODE_CLUSTER_THRESHOLD:
        st.warning(
            "Fast mode is disabled with a large cluster count. Naming may take longer "
            "and require many LLM calls."
        )
    rows, run_id, used_baseline = topic_naming_usecase.run_naming(
        clusters=clusters,
        parent_summaries=parent_summaries,
        topic_parent_map=topic_parent_map,
        payload_lookup=payload_lookup,
        include_snippets=include_snippets,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
        top_extension_count=top_extension_count,
        llm_status=llm_status,
        ignore_cache=ignore_cache,
        fast_mode=fast_mode,
        llm_batch_size=llm_batch_size,
        allow_raw_parent_evidence=allow_raw_parent_evidence,
    )
    st.session_state["_run_id"] = run_id
    if used_baseline:
        st.warning("LLM naming unavailable for some rows; using baseline names instead.")
    _update_session_rows(cast(Sequence[Mapping[Hashable, Any]], rows))


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


def _update_session_rows(rows: Sequence[Mapping[Hashable, Any]]) -> None:
    st.session_state["topic_naming_rows"] = topic_naming_usecase.normalize_rows(rows)


def _apply_names(
    *,
    rows: list[dict[str, Any]],
    payload_lookup: dict[str, dict[str, Any]],
    file_assignments: dict[str, dict[str, Any]],
    hide_ids: bool,
) -> None:
    name_map, move_plan = topic_naming_usecase.build_move_plan(
        rows=rows,
        payload_lookup=payload_lookup,
        file_assignments=file_assignments,
    )

    formatted_plan: list[dict[str, Any]] = []
    for entry in move_plan:
        topic_id = int(entry.get("topic_id", -1))
        parent_id = int(entry.get("parent_id", -1))
        payload = entry.get("payload") or {}
        topic_name = name_map["children"].get(topic_id, f"Topic {topic_id}")
        parent_name = name_map["parents"].get(parent_id, f"Parent {parent_id}")
        combined = (
            f"{format_label(parent_id, parent_name, hide_ids)} / "
            f"{format_label(topic_id, topic_name, hide_ids)}"
        )
        formatted_plan.append(
            {
                "checksum": entry.get("checksum"),
                "file": format_file_label(payload, entry.get("checksum")),
                "parent": format_label(parent_id, parent_name, hide_ids),
                "topic": format_label(topic_id, topic_name, hide_ids),
                "combined_label": combined,
                "prob": entry.get("prob"),
            }
        )

    st.session_state["topic_discovery_name_map"] = name_map
    st.session_state["topic_discovery_move_plan"] = formatted_plan

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

GENERIC_NAME_RE = re.compile(
    r"^(Misc|Other|Documents|Files|Review)(?:\b|\s|\u2014|-)",
    re.I,
)
FALLBACK_REASONS = {
    "llm_timeout",
    "llm_invalid_json",
    "llm_unreachable",
    "llm_model_not_loaded",
    "cache_hit_baseline",
}


def render_review_tab() -> None:
    st.subheader("Topic naming review")
    st.caption("Review naming quality and spot topics that need attention.")

    rows_state = st.session_state.get("topic_naming_rows", [])
    if not rows_state:
        st.info("Generate names in the Naming tab first.")
        return

    rows = _normalize_rows(rows_state)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No topic naming rows available.")
        return

    df = _add_status_fields(df)
    filters = _render_filters(df)
    filtered_df = _apply_filters(df, filters)
    _render_metrics(df, filtered_df, filters)

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    _render_charts(filtered_df, filters["mixedness_threshold"])
    selected_row = _render_detail_selector(filtered_df)
    _render_details(selected_row)
    _render_snapshots(filtered_df)
    _render_filtered_table(filtered_df)


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        level = _first_present(row, ["level", "cluster_level", "type"], "n/a")
        identifier = _first_present(row, ["id", "cluster_id", "parent_id"], "n/a")
        proposed_name = _first_present(row, ["proposed_name", "name", "topic_name"], "n/a")
        parent_name = _first_present(row, ["parent_name", "parent"], "")
        confidence = _first_present(row, ["confidence", "score", "name_confidence"], None)
        mixedness = _first_present(row, ["mixedness", "cluster_mixedness"], None)
        warnings = _ensure_list(_first_present(row, ["warnings", "warning"], []))
        rationale = _first_present(row, ["rationale", "explanation"], "")
        llm_used = bool(_first_present(row, ["llm_used"], False))
        cache_hit = bool(_first_present(row, ["cache_hit"], False))
        cache_bypassed = bool(_first_present(row, ["cache_bypassed"], False))
        fallback_reason = _first_present(row, ["fallback_reason"], "")
        error_summary = _first_present(row, ["error_summary"], "")
        keywords = _ensure_list(
            _first_present(row, ["keywords", "significant_terms", "terms"], [])
        )
        example_paths = _ensure_list(
            _first_present(
                row,
                ["example_paths", "representative_paths", "paths"],
                [],
            )
        )
        snippets = _ensure_list(
            _first_present(row, ["snippets", "representative_snippets"], [])
        )

        normalized.append(
            {
                "id": identifier,
                "level": level,
                "proposed_name": proposed_name,
                "parent_name": parent_name,
                "confidence": confidence,
                "mixedness": mixedness,
                "warnings": warnings,
                "rationale": rationale,
                "llm_used": llm_used,
                "cache_hit": cache_hit,
                "cache_bypassed": cache_bypassed,
                "fallback_reason": fallback_reason,
                "error_summary": error_summary,
                "keywords": keywords,
                "example_paths": example_paths,
                "snippets": snippets,
            }
        )
    return normalized


def _status_category(row: dict[str, Any]) -> str:
    if row.get("llm_used"):
        return "LLM"
    if row.get("cache_hit"):
        return "Cache"
    return "Baseline"


def _warnings_text(warnings: list[str]) -> str:
    return "; ".join([entry for entry in warnings if entry])


def _needs_review(
    row: dict[str, Any],
    mixedness_threshold: float,
    confidence_threshold: float,
) -> bool:
    mixedness = row.get("mixedness")
    confidence = row.get("confidence")
    warnings_text = _warnings_text(row.get("warnings", []))
    proposed_name = str(row.get("proposed_name") or "")
    fallback_reason = str(row.get("fallback_reason") or "")
    mixedness_flag = mixedness is not None and mixedness >= mixedness_threshold
    confidence_flag = confidence is not None and confidence <= confidence_threshold
    warning_flag = bool(re.search(r"review|mixed", warnings_text, re.I))
    name_flag = bool(GENERIC_NAME_RE.search(proposed_name))
    fallback_flag = fallback_reason in FALLBACK_REASONS
    return bool(
        mixedness_flag or confidence_flag or warning_flag or name_flag or fallback_flag
    )


def _add_status_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["status"] = df.apply(_status_category, axis=1)
    df["row_key"] = df.apply(lambda row: f"{row['level']}:{row['id']}", axis=1)
    df["warnings_text"] = df["warnings"].apply(_warnings_text)
    return df


def _render_filters(df: pd.DataFrame) -> dict[str, Any]:
    sidebar = st.sidebar
    sidebar.header("Topic naming review filters")

    level_options = sorted(df["level"].dropna().unique().tolist())
    status_options = ["LLM", "Cache", "Baseline"]

    selected_levels = sidebar.multiselect(
        "Level",
        options=level_options,
        default=level_options,
    )
    selected_status = sidebar.multiselect(
        "Status",
        options=status_options,
        default=status_options,
    )

    mixedness_threshold = sidebar.slider(
        "Needs review mixedness threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
    )
    confidence_threshold = sidebar.slider(
        "Needs review confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.01,
    )

    mixedness_range = sidebar.slider(
        "Mixedness range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )
    confidence_range = sidebar.slider(
        "Confidence range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )

    needs_review_only = sidebar.checkbox("Needs review only", value=False)
    search_query = sidebar.text_input(
        "Search proposed names / example paths",
        value="",
    )

    return {
        "selected_levels": selected_levels,
        "selected_status": selected_status,
        "mixedness_threshold": mixedness_threshold,
        "confidence_threshold": confidence_threshold,
        "mixedness_range": mixedness_range,
        "confidence_range": confidence_range,
        "needs_review_only": needs_review_only,
        "search_query": search_query,
    }


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["needs_review"] = df.apply(
        lambda row: _needs_review(
            row, filters["mixedness_threshold"], filters["confidence_threshold"]
        ),
        axis=1,
    )

    if filters["selected_levels"]:
        df = df[df["level"].isin(filters["selected_levels"])]
    if filters["selected_status"]:
        df = df[df["status"].isin(filters["selected_status"])]

    df = _apply_range_filters(
        df,
        mixedness_range=filters["mixedness_range"],
        confidence_range=filters["confidence_range"],
    )

    if filters["needs_review_only"]:
        df = df[df["needs_review"]]

    if filters["search_query"]:
        query = filters["search_query"].lower()
        search_mask = df["proposed_name"].fillna("").str.lower().str.contains(query)
        example_paths = df["example_paths"].apply(lambda paths: " ".join(paths).lower())
        search_mask |= example_paths.str.contains(query)
        df = df[search_mask]

    return df


def _apply_range_filters(
    df: pd.DataFrame,
    *,
    mixedness_range: tuple[float, float],
    confidence_range: tuple[float, float],
) -> pd.DataFrame:
    def _range_filter(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
        if min_value == 0.0 and max_value == 1.0:
            return pd.Series([True] * len(series), index=series.index)
        return series.between(min_value, max_value, inclusive="both")

    if "mixedness" in df:
        mixedness_mask = _range_filter(
            df["mixedness"].fillna(-1.0),
            mixedness_range[0],
            mixedness_range[1],
        )
        if mixedness_range != (0.0, 1.0):
            mixedness_mask &= df["mixedness"].notna()
        df = df[mixedness_mask]

    if "confidence" in df:
        confidence_mask = _range_filter(
            df["confidence"].fillna(-1.0),
            confidence_range[0],
            confidence_range[1],
        )
        if confidence_range != (0.0, 1.0):
            confidence_mask &= df["confidence"].notna()
        df = df[confidence_mask]

    return df


def _render_metrics(
    df: pd.DataFrame, filtered_df: pd.DataFrame, filters: dict[str, Any]
) -> None:
    metric_source = filtered_df if not filtered_df.empty else df
    total_items = len(metric_source)
    llm_used_pct = metric_source["llm_used"].mean() * 100 if total_items else 0.0
    cache_hit_pct = metric_source["cache_hit"].mean() * 100 if total_items else 0.0
    baseline_pct = (
        (
            ((~metric_source["llm_used"]) & (~metric_source["cache_hit"]))
            | metric_source["fallback_reason"].fillna("").astype(bool)
        ).mean()
        * 100
        if total_items
        else 0.0
    )
    high_mixedness_pct = (
        (metric_source["mixedness"].fillna(-1.0) >= filters["mixedness_threshold"]).mean()
        * 100
        if total_items
        else 0.0
    )
    median_confidence = metric_source["confidence"].median() if total_items else None
    p10_confidence = metric_source["confidence"].quantile(0.1) if total_items else None
    needs_review_count = int(metric_source["needs_review"].sum())

    metrics = st.columns(8)
    metrics[0].metric("Total items", f"{total_items}")
    metrics[1].metric("% LLM used", f"{llm_used_pct:.1f}%")
    metrics[2].metric("% Cache hit", f"{cache_hit_pct:.1f}%")
    metrics[3].metric("% Baseline/fallback", f"{baseline_pct:.1f}%")
    metrics[4].metric("% High mixedness", f"{high_mixedness_pct:.1f}%")
    metrics[5].metric(
        "Median confidence",
        f"{median_confidence:.2f}" if median_confidence is not None else "n/a",
    )
    metrics[6].metric(
        "Worst 10% confidence",
        f"{p10_confidence:.2f}" if p10_confidence is not None else "n/a",
    )
    metrics[7].metric("Needs review", f"{needs_review_count}")

    st.caption("Metrics reflect current filters.")


def _render_charts(filtered_df: pd.DataFrame, mixedness_threshold: float) -> None:
    scatter_df = filtered_df.copy()
    if len(scatter_df) > 2000:
        scatter_df = scatter_df.sample(2000, random_state=7)

    scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X("mixedness:Q", title="Mixedness", scale=alt.Scale(domain=(0, 1))),
            y=alt.Y("confidence:Q", title="Confidence", scale=alt.Scale(domain=(0, 1))),
            color=alt.Color("status:N", title="Status"),
            tooltip=[
                alt.Tooltip("id:N", title="ID"),
                alt.Tooltip("level:N", title="Level"),
                alt.Tooltip("proposed_name:N", title="Proposed"),
                alt.Tooltip("fallback_reason:N", title="Fallback"),
                alt.Tooltip("keywords:N", title="Keywords"),
                alt.Tooltip("example_paths:N", title="Examples"),
            ],
        )
        .properties(height=360)
    )

    histogram_base = alt.Chart(filtered_df).encode(
        x=alt.X(
            "mixedness:Q",
            bin=alt.Bin(maxbins=30),
            title="Mixedness",
            scale=alt.Scale(domain=(0, 1)),
        ),
        y=alt.Y("count():Q", title="Count"),
    )
    histogram = histogram_base.mark_bar(opacity=0.7)
    threshold_rule = (
        alt.Chart(pd.DataFrame({"threshold": [mixedness_threshold]}))
        .mark_rule(color="red")
        .encode(x="threshold:Q")
    )

    chart_cols = st.columns((2, 1))
    with chart_cols[0]:
        st.subheader("Mixedness vs Confidence")
        st.altair_chart(scatter, use_container_width=True)
    with chart_cols[1]:
        st.subheader("Mixedness distribution")
        st.altair_chart(histogram + threshold_rule, use_container_width=True)


def _render_detail_selector(filtered_df: pd.DataFrame) -> pd.Series:
    select_labels = (
        filtered_df.apply(
            lambda row: f"{row['level']} {row['id']} - {row['proposed_name']}",
            axis=1,
        )
        .tolist()
    )
    selection_lookup = dict(
        zip(select_labels, filtered_df["row_key"].tolist(), strict=False)
    )

    selected_label = st.selectbox(
        "Select an item for details",
        options=select_labels,
    )
    selected_key = selection_lookup.get(selected_label)
    return filtered_df.loc[filtered_df["row_key"] == selected_key].iloc[0]


def _render_details(selected_row: pd.Series) -> None:
    st.divider()

    st.subheader(
        f"{selected_row['level']} {selected_row['id']}: {selected_row['proposed_name']}"
    )

    badges = []
    if selected_row.get("llm_used"):
        badges.append("LLM")
    if selected_row.get("cache_hit"):
        badges.append("Cache hit")
    if selected_row.get("cache_bypassed"):
        badges.append("Cache bypassed")
    if not selected_row.get("llm_used") and not selected_row.get("cache_hit"):
        badges.append("Baseline")
    fallback_reason = selected_row.get("fallback_reason") or ""
    if fallback_reason:
        badges.append(f"Fallback: {fallback_reason}")

    if badges:
        st.markdown("**Status:** " + " ".join(f"`{badge}`" for badge in badges))

    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "Mixedness",
        f"{selected_row['mixedness']:.3f}"
        if pd.notna(selected_row["mixedness"])
        else "n/a",
    )
    metric_cols[1].metric(
        "Confidence",
        f"{selected_row['confidence']:.3f}"
        if pd.notna(selected_row["confidence"])
        else "n/a",
    )

    with st.expander("Warnings & rationale", expanded=False):
        warnings_value = _warnings_text(selected_row.get("warnings", [])) or "n/a"
        st.markdown(f"**Warnings:** {warnings_value}")
        st.markdown("**Rationale:**")
        st.text(selected_row.get("rationale") or "n/a")

    keyword_list = selected_row.get("keywords", [])[:15]
    st.markdown("**Keywords**")
    if keyword_list:
        st.markdown(" ".join(f"`{kw}`" for kw in keyword_list))
    else:
        st.caption("n/a")

    example_paths = selected_row.get("example_paths", [])[:10]
    st.markdown("**Example paths**")
    st.text_area(
        label="example_paths",
        value="\n".join(example_paths) if example_paths else "n/a",
        height=180,
    )

    snippets = selected_row.get("snippets", [])[:3]
    if snippets:
        with st.expander("Snippets", expanded=False):
            for snippet in snippets:
                st.text(snippet)

    st.divider()


def _render_snapshots(filtered_df: pd.DataFrame) -> None:
    snapshot_cols = st.columns([1, 2])
    filtered_rows = filtered_df.drop(columns=["warnings_text"]).to_dict(orient="records")
    snapshot_payload = json.dumps(filtered_rows, indent=2, default=str)

    with snapshot_cols[0]:
        if st.button("Export review snapshot (JSON)"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = Path(".cache") / "topic_naming" / "review_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"review_snapshot_{timestamp}.json"
            snapshot_path.write_text(snapshot_payload, encoding="utf-8")
            st.success(f"Saved snapshot to {snapshot_path}")

    with snapshot_cols[1]:
        st.download_button(
            "Download snapshot JSON",
            data=snapshot_payload,
            file_name="topic_naming_review_snapshot.json",
            mime="application/json",
        )


def _render_filtered_table(filtered_df: pd.DataFrame) -> None:
    st.subheader("Filtered rows")
    sort_choice = st.selectbox(
        "Sort by",
        options=["confidence", "mixedness", "id"],
        index=0,
    )
    sort_df = filtered_df.sort_values(by=sort_choice, ascending=True)
    st.dataframe(
        sort_df[
            [
                "id",
                "level",
                "proposed_name",
                "confidence",
                "mixedness",
                "llm_used",
                "cache_hit",
                "fallback_reason",
                "warnings_text",
            ]
        ],
        use_container_width=True,
    )

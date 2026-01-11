from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.usecases import topic_discovery_review_usecase

alt: Any = alt


def render_review_tab() -> None:
    st.subheader("Topic naming review")
    st.caption("Review naming quality and spot topics that need attention.")

    rows_state = st.session_state.get("topic_naming_rows", [])
    if not rows_state:
        st.info("Generate names in the Naming tab first.")
        return

    rows = topic_discovery_review_usecase.normalize_rows(rows_state)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No topic naming rows available.")
        return

    df = topic_discovery_review_usecase.add_status_fields(df)
    filters = _render_filters(df)
    filtered_df = topic_discovery_review_usecase.apply_filters(df, filters)
    _render_metrics(df, filtered_df, filters)

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    _render_charts(filtered_df, filters["mixedness_threshold"])
    selected_row = _render_detail_selector(filtered_df)
    _render_details(selected_row)
    _render_snapshots(filtered_df)
    _render_filtered_table(filtered_df)


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


def _render_metrics(
    df: pd.DataFrame, filtered_df: pd.DataFrame, filters: dict[str, Any]
) -> None:
    metrics_data = topic_discovery_review_usecase.build_metrics(df, filtered_df, filters)
    metrics = st.columns(8)
    metrics[0].metric("Total items", f"{metrics_data['total_items']}")
    metrics[1].metric("% LLM used", f"{metrics_data['llm_used_pct']:.1f}%")
    metrics[2].metric("% Cache hit", f"{metrics_data['cache_hit_pct']:.1f}%")
    metrics[3].metric("% Baseline/fallback", f"{metrics_data['baseline_pct']:.1f}%")
    metrics[4].metric("% High mixedness", f"{metrics_data['high_mixedness_pct']:.1f}%")
    metrics[5].metric(
        "Median confidence",
        f"{metrics_data['median_confidence']:.2f}"
        if metrics_data["median_confidence"] is not None
        else "n/a",
    )
    metrics[6].metric(
        "Worst 10% confidence",
        f"{metrics_data['p10_confidence']:.2f}"
        if metrics_data["p10_confidence"] is not None
        else "n/a",
    )
    metrics[7].metric("Needs review", f"{metrics_data['needs_review_count']}")

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
    mixedness_value = selected_row.get("mixedness")
    confidence_value = selected_row.get("confidence")
    metric_cols[0].metric(
        "Mixedness",
        f"{mixedness_value:.3f}"
        if topic_discovery_review_usecase.is_present(mixedness_value)
        else "n/a",
    )
    metric_cols[1].metric(
        "Confidence",
        f"{confidence_value:.3f}"
        if topic_discovery_review_usecase.is_present(confidence_value)
        else "n/a",
    )

    with st.expander("Warnings & rationale", expanded=False):
        warnings_value = (
            topic_discovery_review_usecase.warnings_text(
                selected_row.get("warnings", [])
            )
            or "n/a"
        )
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
    snapshot_payload = topic_discovery_review_usecase.build_snapshot_payload(filtered_df)

    with snapshot_cols[0]:
        if st.button("Export review snapshot (JSON)"):
            snapshot_path = topic_discovery_review_usecase.save_snapshot(snapshot_payload)
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

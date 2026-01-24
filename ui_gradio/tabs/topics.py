"""Topic discovery and naming tab."""

from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.express as px

from app.gradio_utils import (
    build_payload_lookup,
    dataframe_to_records,
    summarize_cluster_result,
)
from app.usecases import (
    topic_discovery_admin_usecase,
    topic_discovery_overview_usecase,
    topic_discovery_review_usecase,
    topic_naming_usecase,
)


def build_topics_tab(cluster_state: gr.State) -> None:
    naming_rows_state = gr.State([])
    review_rows_state = gr.State([])
    filtered_rows_state = gr.State([])
    review_table_headers = [
        "Level",
        "ID",
        "Proposed Name",
        "Status",
        "Mixedness",
        "Confidence",
        "Needs Review",
    ]

    gr.Markdown("## Topic Discovery Overview")
    with gr.Row():
        min_cluster_size = gr.Slider(5, 50, value=10, step=1, label="Min cluster size")
        min_samples = gr.Slider(1, 30, value=3, step=1, label="Min samples")
        use_umap = gr.Checkbox(label="Use UMAP", value=False)
    with gr.Row():
        macro_min_k = gr.Slider(2, 20, value=5, step=1, label="Macro min k")
        macro_max_k = gr.Slider(2, 30, value=10, step=1, label="Macro max k")
    with gr.Row():
        run_button = gr.Button("Run clustering", variant="primary")
        load_button = gr.Button("Load cached run")
        clear_button = gr.Button("Clear cache")
    overview_status = gr.Markdown()
    overview_json = gr.JSON(label="Cluster summary")

    def run_clustering(
        min_cluster: int,
        min_samples_value: int,
        use_umap_value: bool,
        macro_min: int,
        macro_max: int,
    ):
        settings = {
            "min_cluster_size": int(min_cluster),
            "min_samples": int(min_samples_value),
            "use_umap": use_umap_value,
            "umap_config": None,
            "macro_k_range": (int(macro_min), int(macro_max)),
        }
        result, used_cache, run_id = topic_discovery_overview_usecase.run_clustering(
            settings
        )
        if result is None:
            return None, "No clustering results available.", {}
        status = f"Run {run_id} complete."
        if used_cache:
            status += " (cache)"
        return result, status, summarize_cluster_result(result)

    def load_cached(
        min_cluster: int,
        min_samples_value: int,
        use_umap_value: bool,
        macro_min: int,
        macro_max: int,
    ):
        settings = {
            "min_cluster_size": int(min_cluster),
            "min_samples": int(min_samples_value),
            "use_umap": use_umap_value,
            "umap_config": None,
            "macro_k_range": (int(macro_min), int(macro_max)),
        }
        result = topic_discovery_overview_usecase.load_cached(settings)
        if result is None:
            return None, "No cached clustering run found.", {}
        return result, "Loaded cached clustering run.", summarize_cluster_result(result)

    def clear_cache():
        removed = topic_discovery_overview_usecase.clear_cache()
        message = "Cluster cache cleared." if removed else "No cache to clear."
        return None, message, {}

    run_button.click(
        run_clustering,
        inputs=[min_cluster_size, min_samples, use_umap, macro_min_k, macro_max_k],
        outputs=[cluster_state, overview_status, overview_json],
    )
    load_button.click(
        load_cached,
        inputs=[min_cluster_size, min_samples, use_umap, macro_min_k, macro_max_k],
        outputs=[cluster_state, overview_status, overview_json],
    )
    clear_button.click(clear_cache, outputs=[cluster_state, overview_status, overview_json])

    gr.Markdown("## Topic Naming")
    with gr.Row():
        generate_button = gr.Button("Generate Names", variant="primary")
        save_button = gr.Button("Save Labels")
    naming_table = gr.Dataframe(
        headers=["Cluster ID", "Generated Name", "User Label", "Document Count"],
        datatype=["number", "str", "str", "number"],
        interactive=True,
        row_count=0,
        column_count=(4, "fixed"),
        label="Editable cluster labels",
    )
    naming_status = gr.Markdown()

    def generate_names(cluster_result):
        if not cluster_result:
            return [], [], "Run clustering first to generate names."
        llm_status = topic_naming_usecase.get_llm_status()
        clusters = cluster_result.get("clusters", [])
        parent_summaries = cluster_result.get("parent_summaries", [])
        topic_parent_map = {
            int(key): int(value)
            for key, value in cluster_result.get("topic_parent_map", {}).items()
        }
        payload_lookup = build_payload_lookup(
            cluster_result.get("checksums", []),
            cluster_result.get("payloads", []),
        )
        rows, run_id, used_baseline = topic_naming_usecase.run_naming(
            clusters=clusters,
            parent_summaries=parent_summaries,
            topic_parent_map=topic_parent_map,
            payload_lookup=payload_lookup,
            include_snippets=False,
            max_keywords=topic_naming_usecase.DEFAULT_MAX_KEYWORDS,
            max_path_depth=topic_naming_usecase.DEFAULT_MAX_PATH_DEPTH,
            root_path=topic_naming_usecase.DEFAULT_ROOT_PATH,
            top_extension_count=topic_naming_usecase.DEFAULT_TOP_EXTENSION_COUNT,
            llm_status=llm_status,
            ignore_cache=False,
            fast_mode=True,
            llm_batch_size=topic_naming_usecase.DEFAULT_LLM_BATCH_SIZE,
            allow_raw_parent_evidence=topic_naming_usecase.DEFAULT_ALLOW_RAW_PARENT_EVIDENCE,
        )
        size_map = {cluster.get("cluster_id"): cluster.get("size", 0) for cluster in clusters}
        table_rows = [
            {
                "Cluster ID": row.get("id"),
                "Generated Name": row.get("proposed_name"),
                "User Label": "",
                "Document Count": size_map.get(row.get("id"), 0),
            }
            for row in rows
            if row.get("level") == "child"
        ]
        status = f"Generated names for {len(table_rows)} clusters (run {run_id})."
        if used_baseline:
            status += " Baseline naming used for some rows."
        return table_rows, rows, status

    def save_labels(table):
        records = dataframe_to_records(table)
        result = topic_naming_usecase.update_labels(records)
        return f"Saved {result['saved']} labels to {result['path']}."

    generate_button.click(
        generate_names,
        inputs=[cluster_state],
        outputs=[naming_table, naming_rows_state, naming_status],
    )
    save_button.click(save_labels, inputs=[naming_table], outputs=[naming_status])

    gr.Markdown("## Naming Review")
    review_status = gr.Markdown()

    with gr.Row():
        refresh_review_button = gr.Button("Load review data", variant="primary")
        export_snapshot_button = gr.Button("Export snapshot")

    with gr.Row():
        review_levels = gr.Dropdown(
            label="Levels",
            multiselect=True,
            choices=[],
        )
        review_statuses = gr.Dropdown(
            label="Status",
            multiselect=True,
            choices=["LLM", "Cache", "Baseline"],
            value=["LLM", "Cache", "Baseline"],
        )
        needs_review_only = gr.Checkbox(label="Needs review only", value=False)

    with gr.Row():
        mixedness_threshold = gr.Slider(
            0.0,
            1.0,
            value=0.85,
            step=0.01,
            label="Needs review mixedness threshold",
        )
        confidence_threshold = gr.Slider(
            0.0,
            1.0,
            value=0.55,
            step=0.01,
            label="Needs review confidence threshold",
        )

    with gr.Row():
        mixedness_range = gr.RangeSlider(
            0.0,
            1.0,
            value=(0.0, 1.0),
            step=0.01,
            label="Mixedness range",
        )
        confidence_range = gr.RangeSlider(
            0.0,
            1.0,
            value=(0.0, 1.0),
            step=0.01,
            label="Confidence range",
        )

    review_search = gr.Textbox(label="Search proposed names / example paths")

    review_metrics = gr.Markdown()
    with gr.Row():
        review_scatter = gr.Plot(label="Mixedness vs Confidence")
        review_histogram = gr.Plot(label="Mixedness distribution")

    review_detail_select = gr.Dropdown(label="Select an item for details", choices=[])
    review_detail = gr.Markdown()

    review_table = gr.Dataframe(
        headers=review_table_headers,
        datatype=["str", "str", "str", "str", "number", "number", "bool"],
        row_count=0,
        column_count=(7, "fixed"),
        interactive=False,
        label="Filtered naming review rows",
    )

    snapshot_file = gr.File(label="Snapshot file")
    snapshot_payload = gr.Code(label="Snapshot payload", language="json")

    def _load_review_data(rows_state: list[dict]) -> tuple:
        if not rows_state:
            return (
                [],
                gr.update(choices=[], value=[]),
                gr.update(value="Generate names in the Naming section first."),
            )

        normalized = topic_discovery_review_usecase.normalize_rows(rows_state)
        df = pd.DataFrame(normalized)
        if df.empty:
            return (
                [],
                gr.update(choices=[], value=[]),
                gr.update(value="No review rows available."),
            )

        df = topic_discovery_review_usecase.add_status_fields(df)
        level_options = sorted(df["level"].dropna().unique().tolist())
        return (
            df.to_dict(orient="records"),
            gr.update(choices=level_options, value=level_options),
            gr.update(
                value=(
                    f"Loaded {len(df)} naming rows for review. "
                    "Adjust filters to explore quality metrics."
                )
            ),
        )

    def _build_review_outputs(
        rows: list[dict],
        selected_levels: list[str],
        selected_status: list[str],
        mixedness_threshold_value: float,
        confidence_threshold_value: float,
        mixedness_range_value: tuple[float, float],
        confidence_range_value: tuple[float, float],
        needs_review_only_value: bool,
        search_query: str,
    ) -> tuple:
        if not rows:
            return (
                gr.update(value="No review rows loaded."),
                None,
                None,
                gr.update(choices=[], value=None),
                gr.update(value=""),
                pd.DataFrame(columns=review_table_headers),
                [],
            )

        df = pd.DataFrame(rows)
        filters = {
            "selected_levels": selected_levels,
            "selected_status": selected_status,
            "mixedness_threshold": mixedness_threshold_value,
            "confidence_threshold": confidence_threshold_value,
            "mixedness_range": mixedness_range_value,
            "confidence_range": confidence_range_value,
            "needs_review_only": needs_review_only_value,
            "search_query": search_query,
        }
        filtered_df = topic_discovery_review_usecase.apply_filters(df, filters)
        metrics_data = topic_discovery_review_usecase.build_metrics(
            df, filtered_df, filters
        )
        metrics_summary = (
            "**Metrics (current filters)**\n\n"
            f"- Total items: {metrics_data['total_items']}\n"
            f"- % LLM used: {metrics_data['llm_used_pct']:.1f}%\n"
            f"- % Cache hit: {metrics_data['cache_hit_pct']:.1f}%\n"
            f"- % Baseline/fallback: {metrics_data['baseline_pct']:.1f}%\n"
            f"- % High mixedness: {metrics_data['high_mixedness_pct']:.1f}%\n"
        )
        if metrics_data["median_confidence"] is not None:
            metrics_summary += (
                f"- Median confidence: {metrics_data['median_confidence']:.2f}\n"
                f"- Worst 10% confidence: {metrics_data['p10_confidence']:.2f}\n"
            )
        else:
            metrics_summary += (
                "- Median confidence: n/a\n- Worst 10% confidence: n/a\n"
            )
        metrics_summary += f"- Needs review count: {metrics_data['needs_review_count']}"

        if filtered_df.empty:
            return (
                gr.update(value=metrics_summary),
                None,
                None,
                gr.update(choices=[], value=None),
                gr.update(value="No rows match the current filters."),
                pd.DataFrame(columns=review_table_headers),
                [],
            )

        scatter_df = filtered_df.copy()
        if len(scatter_df) > 2000:
            scatter_df = scatter_df.sample(2000, random_state=7)

        scatter_fig = px.scatter(
            scatter_df,
            x="mixedness",
            y="confidence",
            color="status",
            hover_data=[
                "id",
                "level",
                "proposed_name",
                "fallback_reason",
                "keywords",
                "example_paths",
            ],
            height=360,
        )
        scatter_fig.update_xaxes(range=[0, 1])
        scatter_fig.update_yaxes(range=[0, 1])

        histogram_fig = px.histogram(
            filtered_df,
            x="mixedness",
            nbins=30,
            height=360,
        )
        histogram_fig.update_xaxes(range=[0, 1])
        histogram_fig.add_vline(
            x=mixedness_threshold_value,
            line_color="red",
            line_dash="dash",
        )

        selection_labels = (
            filtered_df.apply(
                lambda row: f"{row['level']} {row['id']} - {row['proposed_name']}",
                axis=1,
            )
            .tolist()
        )
        selection_lookup = dict(
            zip(selection_labels, filtered_df["row_key"].tolist(), strict=False)
        )
        default_label = selection_labels[0] if selection_labels else None
        selected_key = selection_lookup.get(default_label)
        selected_row = (
            filtered_df.loc[filtered_df["row_key"] == selected_key].iloc[0]
            if selected_key
            else None
        )

        detail_text = "Select a row to see details."
        if selected_row is not None:
            detail_text = _format_review_detail(selected_row)

        table_df = filtered_df[
            [
                "level",
                "id",
                "proposed_name",
                "status",
                "mixedness",
                "confidence",
                "needs_review",
            ]
        ].rename(
            columns={
                "level": "Level",
                "id": "ID",
                "proposed_name": "Proposed Name",
                "status": "Status",
                "mixedness": "Mixedness",
                "confidence": "Confidence",
                "needs_review": "Needs Review",
            }
        )

        return (
            gr.update(value=metrics_summary),
            scatter_fig,
            histogram_fig,
            gr.update(choices=selection_labels, value=default_label),
            gr.update(value=detail_text),
            table_df,
            filtered_df.to_dict(orient="records"),
        )

    def _format_review_detail(row: pd.Series) -> str:
        badges = []
        if row.get("llm_used"):
            badges.append("LLM")
        if row.get("cache_hit"):
            badges.append("Cache hit")
        if row.get("cache_bypassed"):
            badges.append("Cache bypassed")
        if not row.get("llm_used") and not row.get("cache_hit"):
            badges.append("Baseline")

        warnings_text = row.get("warnings_text") or ""
        return (
            f"**{row['level']} {row['id']}: {row['proposed_name']}**\n\n"
            f"- Status: {row.get('status')}\n"
            f"- Parent: {row.get('parent_name', '')}\n"
            f"- Confidence: {row.get('confidence', 'n/a')}\n"
            f"- Mixedness: {row.get('mixedness', 'n/a')}\n"
            f"- Warnings: {warnings_text or 'None'}\n"
            f"- Fallback reason: {row.get('fallback_reason') or 'n/a'}\n"
            f"- Badges: {', '.join(badges) if badges else 'None'}\n\n"
            f"**Keywords**: {', '.join(row.get('keywords', [])) or 'n/a'}\n"
            f"**Example paths**: {'; '.join(row.get('example_paths', [])) or 'n/a'}\n"
            f"**Snippets**: {'; '.join(row.get('snippets', [])) or 'n/a'}"
        )

    def _update_review_detail(selected_label: str, filtered_rows: list[dict]) -> str:
        if not selected_label or not filtered_rows:
            return "Select a row to see details."
        selection_lookup = {
            f"{row['level']} {row['id']} - {row['proposed_name']}": row["row_key"]
            for row in filtered_rows
        }
        selected_key = selection_lookup.get(selected_label)
        for row in filtered_rows:
            if row.get("row_key") == selected_key:
                return _format_review_detail(pd.Series(row))
        return "Select a row to see details."

    def _export_snapshot(filtered_rows: list[dict]) -> tuple:
        if not filtered_rows:
            return None, "No filtered rows to export.", ""
        df = pd.DataFrame(filtered_rows)
        payload = topic_discovery_review_usecase.build_snapshot_payload(df)
        snapshot_path = topic_discovery_review_usecase.save_snapshot(payload)
        return str(snapshot_path), f"Saved snapshot to {snapshot_path}.", payload

    refresh_review_button.click(
        _load_review_data,
        inputs=[naming_rows_state],
        outputs=[review_rows_state, review_levels, review_status],
    )

    review_inputs = [
        review_rows_state,
        review_levels,
        review_statuses,
        mixedness_threshold,
        confidence_threshold,
        mixedness_range,
        confidence_range,
        needs_review_only,
        review_search,
    ]

    for component in review_inputs[1:]:
        component.change(
            _build_review_outputs,
            inputs=review_inputs,
            outputs=[
                review_metrics,
                review_scatter,
                review_histogram,
                review_detail_select,
                review_detail,
                review_table,
                filtered_rows_state,
            ],
        )

    refresh_review_button.click(
        _build_review_outputs,
        inputs=review_inputs,
        outputs=[
            review_metrics,
            review_scatter,
            review_histogram,
            review_detail_select,
            review_detail,
            review_table,
            filtered_rows_state,
        ],
    )

    review_detail_select.change(
        _update_review_detail,
        inputs=[review_detail_select, filtered_rows_state],
        outputs=[review_detail],
    )

    export_snapshot_button.click(
        _export_snapshot,
        inputs=[filtered_rows_state],
        outputs=[snapshot_file, review_status, snapshot_payload],
    )

    gr.Markdown("## Admin")
    admin_status = gr.Markdown()
    admin_logs = gr.Textbox(
        label="Build logs",
        value="",
        lines=8,
        interactive=False,
    )

    with gr.Row():
        refresh_status_button = gr.Button("Refresh status")
        ensure_collection_button = gr.Button("Ensure collection")

    with gr.Row():
        k_value = gr.Number(label="Top-k chunks", value=8, precision=0)
        batch_value = gr.Number(label="Upsert batch size", value=64, precision=0)
        limit_value = gr.Number(label="Limit (0 = all)", value=0, precision=0)

    with gr.Row():
        build_missing_button = gr.Button("Build missing file vectors", variant="primary")
        rebuild_confirm = gr.Checkbox(label="Confirm rebuild all", value=False)
        rebuild_button = gr.Button("Rebuild all")

    sample_vectors_button = gr.Button("Sample 5 vectors")
    sample_vectors_table = gr.Dataframe(
        headers=[],
        row_count=0,
        column_count=(0, "dynamic"),
        interactive=False,
        label="Sample vectors",
    )

    def _render_status() -> str:
        total_files, file_vectors_count, coverage = (
            topic_discovery_admin_usecase.get_file_vector_status()
        )
        return (
            "**File vector status**\n\n"
            f"- Unique files in chunks: {total_files}\n"
            f"- File vectors: {file_vectors_count}\n"
            f"- Coverage: {coverage:.1f}%"
        )

    def _refresh_status() -> str:
        return _render_status()

    def _ensure_collection() -> str:
        topic_discovery_admin_usecase.ensure_collection()
        return "Collection is ready.\n\n" + _render_status()

    def _build_vectors(
        k: float,
        batch: float,
        limit: float,
        force: bool,
        recreate: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, str]:
        logs: list[str] = []
        if recreate:
            topic_discovery_admin_usecase.ensure_collection(recreate=True)

        def _on_progress(payload: dict) -> None:
            total = payload.get("total") or 1
            processed = payload.get("processed", 0)
            progress(processed / total)
            message = (
                f"{processed}/{total} {payload.get('checksum')} "
                f"({payload.get('status')})"
            )
            logs.append(message)

        stats = topic_discovery_admin_usecase.build_file_vectors(
            k=int(k),
            batch=int(batch),
            limit=None if int(limit) == 0 else int(limit),
            force=force,
            progress_callback=_on_progress,
        )

        summary = (
            "**Build complete**\n\n"
            f"- Processed: {stats['processed']}\n"
            f"- Created: {stats['created']}\n"
            f"- Skipped: {stats['skipped']}\n"
            f"- Errors: {stats['errors']}"
        )
        if stats.get("error_list"):
            summary += "\n\n**Errors**\n" + "\n".join(stats["error_list"])
        summary += "\n\n" + _render_status()
        return summary, "\n".join(logs[-50:])

    def _build_missing(k: float, batch: float, limit: float) -> tuple[str, str]:
        return _build_vectors(k, batch, limit, False, False)

    def _sample_vectors() -> pd.DataFrame:
        samples = topic_discovery_admin_usecase.sample_vectors(limit=5)
        return pd.DataFrame(samples)

    refresh_status_button.click(_refresh_status, outputs=[admin_status])
    ensure_collection_button.click(_ensure_collection, outputs=[admin_status])

    build_missing_button.click(
        _build_missing,
        inputs=[k_value, batch_value, limit_value],
        outputs=[admin_status, admin_logs],
    )

    def _run_rebuild(
        k: float,
        batch: float,
        limit: float,
        confirm: bool,
    ) -> tuple[str, str]:
        if not confirm:
            return "Enable confirm to rebuild.", ""
        return _build_vectors(k, batch, limit, True, True)

    rebuild_button.click(
        _run_rebuild,
        inputs=[k_value, batch_value, limit_value, rebuild_confirm],
        outputs=[admin_status, admin_logs],
    )

    sample_vectors_button.click(_sample_vectors, outputs=[sample_vectors_table])

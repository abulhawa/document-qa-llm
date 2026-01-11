"""Topic discovery and naming tab."""

from __future__ import annotations

import gradio as gr

from app.gradio_utils import (
    build_payload_lookup,
    dataframe_to_records,
    summarize_cluster_result,
)
from app.usecases import topic_discovery_overview_usecase, topic_naming_usecase


def build_topics_tab(cluster_state: gr.State) -> None:
    naming_state = gr.State([])

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
        col_count=(4, "fixed"),
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
        return table_rows, table_rows, status

    def save_labels(table):
        records = dataframe_to_records(table)
        result = topic_naming_usecase.update_labels(records)
        return f"Saved {result['saved']} labels to {result['path']}."

    generate_button.click(
        generate_names,
        inputs=[cluster_state],
        outputs=[naming_table, naming_state, naming_status],
    )
    save_button.click(save_labels, inputs=[naming_table], outputs=[naming_status])

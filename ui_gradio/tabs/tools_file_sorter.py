"""Gradio tab for the Smart File Sorter."""

from __future__ import annotations

import math
import os
import time
import uuid
from typing import List, cast

import gradio as gr
import pandas as pd

from app.usecases.file_sorter_usecase import apply_sort_plan_action, preview_sort_plan
from app.usecases.tools_usecase import (
    DEFAULT_SMART_FILE_SORTER_PRESET,
    get_smart_file_sorter_presets,
)
from core.llm import check_llm_status
from core.sync.file_sorter import SortOptions


def _normalize_weights(meta: float, content: float, keyword: float) -> tuple[float, float, float]:
    weight_total = meta + content + keyword
    if weight_total <= 0:
        return 0.0, 0.0, 0.0
    return meta / weight_total, content / weight_total, keyword / weight_total


def _update_weight_total(meta: float, content: float, keyword: float) -> str:
    total = meta + content + keyword
    return f"Total = {total:.2f} (weights are normalized to 1.00)"


def _parse_reason(reason: str) -> dict:
    parsed: dict = {}
    if not isinstance(reason, str):
        return parsed
    for chunk in reason.split(";"):
        part = chunk.strip()
        if "=" not in part:
            continue
        key, raw_value = [item.strip() for item in part.split("=", 1)]
        key = key.lower()
        if key == "llm":
            parsed[key] = raw_value
            continue
        try:
            parsed[key] = float(raw_value)
        except ValueError:
            continue
    return parsed


def _score_bucket(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "medium"
    if score >= 0.15:
        return "light"
    return "none"


def _format_reason_tokens(reason: str) -> str:
    parsed = _parse_reason(reason)
    tokens: List[str] = []
    if "llm" in parsed:
        tokens.append(f"LLM → {parsed['llm']}")
    for key, label in (("meta", "Meta"), ("content", "Content"), ("keywords", "Keywords")):
        if key in parsed:
            tokens.append(f"{label} {_score_bucket(parsed[key])}")
    return " · ".join(tokens)


def _build_display_table(dataframe: pd.DataFrame, show_advanced: bool) -> pd.DataFrame:
    columns = ["basename", "proposed_folder", "confidence", "reason_tokens", "path"]
    labels = {
        "basename": "Basename",
        "proposed_folder": "Proposed folder",
        "confidence": "Confidence",
        "reason_tokens": "Reason",
        "path": "Path",
    }
    if show_advanced:
        columns += ["meta_similarity", "content_similarity", "keyword_score", "reason"]
        labels.update(
            {
                "meta_similarity": "Meta score",
                "content_similarity": "Content score",
                "keyword_score": "Keyword score",
                "reason": "Reason (raw)",
            }
        )
    display = cast(pd.DataFrame, dataframe[columns]).rename(columns=labels)
    return cast(pd.DataFrame, display)


def _prepare_dataframe(plan: list) -> pd.DataFrame:
    df = pd.DataFrame([p.as_dict() for p in plan]) if plan else pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    df["filename"] = df["path"].apply(os.path.basename)
    df["extension"] = df["filename"].apply(lambda name: os.path.splitext(name)[1].lower())
    df["target_label_display"] = df["target_label"].fillna("Unassigned")
    df["basename"] = df["filename"]
    df["proposed_folder"] = df["target_label_display"]
    df["reason_tokens"] = df["reason"].apply(_format_reason_tokens)
    if "second_confidence" not in df.columns:
        df["second_confidence"] = pd.NA
    if "top2_margin" not in df.columns:
        df["top2_margin"] = pd.NA
    return df


def _build_summary_tables(
    df: pd.DataFrame,
    options: SortOptions | None,
    min_conf: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    total_files = len(df)
    if df.empty:
        summary_df = pd.DataFrame(
            [
                {"Metric": "Files scanned", "Value": 0},
                {f"Metric": f"Above {min_conf:.2f}", "Value": 0},
                {"Metric": "Below threshold", "Value": 0},
                {"Metric": "Ambiguous (±0.05)", "Value": 0},
                {"Metric": "Distinct targets", "Value": 0},
            ]
        )
        return summary_df, pd.DataFrame(columns=["Bucket", "Count"]), ""
    qualifying_count = int((df["confidence"] >= min_conf).sum())
    below_threshold = total_files - qualifying_count
    ambiguous_band = 0.05
    ambiguous_count = int(
        df["confidence"].between(
            max(0.0, min_conf - ambiguous_band),
            min(1.0, min_conf + ambiguous_band),
            inclusive="both",
        ).sum()
    )
    distinct_targets = int(df["target_label"].nunique())
    show_llm_used = bool(options and options.use_llm_fallback)
    llm_used = int(df["reason"].str.startswith("llm=").sum()) if show_llm_used else 0
    summary_rows = [
        {"Metric": "Files scanned", "Value": total_files},
        {"Metric": f"Above {min_conf:.2f}", "Value": qualifying_count},
        {"Metric": "Below threshold", "Value": below_threshold},
        {
            "Metric": f"Ambiguous (±{ambiguous_band:.2f})",
            "Value": ambiguous_count,
        },
        {"Metric": "Distinct targets", "Value": distinct_targets},
    ]
    summary_df = pd.DataFrame(summary_rows)
    llm_text = f"LLM used: {llm_used}" if show_llm_used else ""
    bucket_counts = (
        df["confidence"]
        .apply(_score_bucket)
        .value_counts()
        .reindex(["strong", "medium", "light", "none"])
        .fillna(0)
        .astype(int)
    )
    confidence_df = pd.DataFrame(
        {"Bucket": bucket_counts.index, "Count": bucket_counts.values}
    )
    return summary_df, confidence_df, llm_text


def _extract_filter_choices(df: pd.DataFrame) -> dict[str, list]:
    if df.empty:
        return {"targets": ["All"], "extensions": []}
    target_options = ["All"] + sorted(df["target_label_display"].unique().tolist())
    extension_series = df["extension"]
    extension_options = sorted({ext for ext in extension_series.unique() if ext})
    if "" in extension_series.unique():
        extension_options = extension_options + ["(none)"]
    return {"targets": target_options, "extensions": extension_options}


def _build_review_queue(
    df: pd.DataFrame,
    gray_low_value: float,
    gray_high_value: float,
    margin_threshold_value: float,
    suspicious_raw_value: str,
    show_advanced: bool,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=bool), "No files were included in the plan."
    suspicious_items = [
        item.strip().lower()
        for item in suspicious_raw_value.split(",")
        if item.strip()
    ]
    suspicious_extensions = {f".{item}" for item in suspicious_items if "." not in item}
    suspicious_extensions |= {
        item for item in suspicious_items if item.startswith(".") and item.count(".") == 1
    }
    suspicious_names = {
        item for item in suspicious_items if "." in item and not item.startswith(".")
    }
    filename_lower = cast(pd.Series, df["filename"]).str.lower()
    extension_lower = cast(pd.Series, df["extension"]).str.lower()
    suspicious_mask = filename_lower.isin(suspicious_names) | extension_lower.isin(
        suspicious_extensions
    )
    gray_mask = cast(pd.Series, df["confidence"]).between(
        gray_low_value, gray_high_value, inclusive="both"
    )
    top2_margin = cast(pd.Series, df["top2_margin"])
    if top2_margin.notna().any():
        margin_mask = top2_margin.isna() | (top2_margin <= margin_threshold_value)
    else:
        margin_mask = pd.Series(True, index=df.index)
    review_mask = cast(pd.Series, (gray_mask & margin_mask) | suspicious_mask)
    review_df = cast(pd.DataFrame, df.loc[review_mask]).sort_values(
        "confidence", ascending=True
    )
    caption = f"{len(review_df)} file(s) flagged for review."
    if review_df.empty:
        caption = "No files match the review queue criteria."
    review_display = _build_display_table(review_df, show_advanced)
    return review_display, review_mask, caption


def _build_plan_details(
    df: pd.DataFrame,
    review_mask: pd.Series,
    min_conf_filter_value: float,
    target_filter_value: str,
    filename_filter_value: str,
    selected_extensions: list,
    only_above_threshold_value: bool,
    only_review_queue_value: bool,
    view_mode_value: str,
    page_value: int,
    min_confidence_value: float,
    show_advanced: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    if df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "No files were included in the plan.",
            1,
        )
    filtered = cast(pd.DataFrame, df.loc[df["confidence"] >= min_conf_filter_value]).copy()
    if target_filter_value and target_filter_value != "All":
        filtered = cast(
            pd.DataFrame,
            filtered.loc[filtered["target_label_display"] == target_filter_value],
        )
    if filename_filter_value:
        filtered = cast(
            pd.DataFrame,
            filtered.loc[
                cast(pd.Series, filtered["filename"]).str.contains(
                    filename_filter_value, case=False, na=False
                )
            ],
        )
    if selected_extensions:
        extension_mask = cast(pd.Series, filtered["extension"]).isin(
            [ext for ext in selected_extensions if ext != "(none)"]
        )
        if "(none)" in selected_extensions:
            extension_mask |= cast(pd.Series, filtered["extension"]) == ""
        filtered = cast(pd.DataFrame, filtered.loc[extension_mask])
    if only_above_threshold_value:
        filtered = cast(pd.DataFrame, filtered.loc[filtered["confidence"] >= min_confidence_value])
    if only_review_queue_value and not review_mask.empty:
        filtered = cast(
            pd.DataFrame,
            filtered.loc[review_mask.reindex(filtered.index, fill_value=False)],
        )
    if filtered.empty:
        return (
            filtered,
            filtered,
            filtered,
            "No files match the current filters.",
            1,
        )
    export_df = filtered
    if view_mode_value == "By folder":
        plan_df = (
            filtered.groupby("target_label_display", dropna=False)
            .agg(
                file_count=("path", "size"),
                avg_confidence=("confidence", "mean"),
                pct_above_threshold=(
                    "confidence",
                    lambda s: (s >= min_confidence_value).mean() * 100,
                ),
            )
            .reset_index()
            .sort_values("file_count", ascending=False)
        )
    elif view_mode_value == "By confidence":
        plan_df = cast(pd.DataFrame, filtered).sort_values(
            "confidence", ascending=False
        )
    else:
        plan_df = cast(pd.DataFrame, filtered)
    page_size = 200
    total_rows = len(plan_df)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page_value = max(1, min(page_value, total_pages))
    start = (page_value - 1) * page_size
    end = min(start + page_size, total_rows)
    paged = plan_df.iloc[start:end]
    if view_mode_value == "By folder":
        display = paged
    else:
        display = _build_display_table(paged, show_advanced)
    caption = f"Showing {start + 1}-{end} of {total_rows}" if total_rows else ""
    return filtered, export_df, display, caption, page_value


def _build_move_counts(df: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {"Status": "Will move", "Count": 0},
                {"Status": "Will not move", "Count": 0},
                {"Status": "Ignored", "Count": 0},
            ]
        )
    target_mask = df["target_path"].fillna("").astype(str).str.len() > 0
    will_move_mask = target_mask & (df["confidence"] >= min_conf)
    will_not_move_mask = target_mask & (df["confidence"] < min_conf)
    will_move_count = int(will_move_mask.sum())
    will_not_move_count = int(will_not_move_mask.sum())
    ignored_count = int(len(df) - will_move_count - will_not_move_count)
    return pd.DataFrame(
        [
            {"Status": "Will move", "Count": will_move_count},
            {"Status": "Will not move", "Count": will_not_move_count},
            {"Status": "Ignored", "Count": ignored_count},
        ]
    )


def _export_csv(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    file_path = f"smart_sort_plan_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(file_path, index=False)
    return file_path


def build_tools_file_sorter_tab() -> None:
    presets = get_smart_file_sorter_presets()
    llm_status = check_llm_status()
    plan_state = gr.State(None)
    options_state = gr.State(None)
    df_state = gr.State(None)
    review_mask_state = gr.State(None)
    run_id_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Smart File Sorter")
            gr.Markdown(
                "Dry-run classifier for moving files into your 1-5 topic folders."
            )
        with gr.Column(scale=1):
            preset = gr.Dropdown(
                choices=list(presets.keys()),
                value=DEFAULT_SMART_FILE_SORTER_PRESET,
                label="Preset",
            )

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            "- Scores each file against the subfolders under your numbered topics.\n"
            "- Uses filename + parent folder names + optional content excerpt embeddings.\n"
            "- Applies keyword boosts from folder names and optional aliases.\n"
            "- Produces a dry-run plan you can review before moving files."
        )

    root_default = os.getenv("LOCAL_SYNC_ROOT", "C:\\Users\\ali_a\\My Drive")
    with gr.Row():
        root = gr.Textbox(label="Root folder to scan", value=root_default)
    gr.Markdown("Example: `C:\\Users\\ali_a\\My Drive`. All files under this root will be considered.")
    include_content = gr.Checkbox(
        label="Use file content (PDF/DOCX/TXT)", value=True
    )
    gr.Markdown(
        "When enabled, PDFs/DOCX/TXT contribute text embeddings. Large files are skipped."
    )
    max_files_choice = gr.Dropdown(
        label="Max files preset",
        choices=[50, 200, 1000, "All"],
        value="All",
    )
    gr.Markdown("Use a small limit for quick dry-runs.")
    preview_button = gr.Button("Preview classification (dry-run)", variant="primary")

    with gr.Accordion("Advanced settings", open=False):
        max_parent_levels = gr.Slider(
            label="Use up to N parent folders as hints",
            minimum=0,
            maximum=8,
            value=4,
            step=1,
        )
        gr.Markdown(
            "Depth 0 = filename only; depth 2 uses two parent folders. "
            "Example: `file.pdf Tickets Germany`."
        )
        max_content_mb = gr.Slider(
            label="Max file size for content parsing (MB)",
            minimum=1,
            maximum=200,
            value=25,
            step=1,
        )
        gr.Markdown("Files larger than this are not parsed for content embeddings.")
        max_content_chars = gr.Slider(
            label="Read up to N chars per file",
            minimum=500,
            maximum=20000,
            value=6000,
            step=500,
        )
        gr.Markdown("Only the first N characters from each file are used.")

        gr.Markdown("### Scoring weights")
        weight_meta = gr.Slider(
            label="Filename + folder embedding weight",
            minimum=0.0,
            maximum=1.0,
            value=float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_meta"]),
            step=0.01,
        )
        gr.Markdown(
            "Signals from filename + parent folders. Higher = more path-driven classification."
        )
        weight_content = gr.Slider(
            label="Content embedding weight",
            minimum=0.0,
            maximum=1.0,
            value=float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_content"]),
            step=0.01,
        )
        gr.Markdown(
            "Signals from PDF/DOCX/TXT content. Higher = more content-driven classification."
        )
        weight_keyword = gr.Slider(
            label="Keyword boost weight",
            minimum=0.0,
            maximum=1.0,
            value=float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_keyword"]),
            step=0.01,
        )
        gr.Markdown(
            "Exact keyword matches from folder names + aliases. "
            "Good for IDs, taxes, tickets, etc."
        )
        weight_total = gr.Markdown(
            _update_weight_total(
                float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_meta"]),
                float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_content"]),
                float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["weight_keyword"]),
            )
        )

        gr.Markdown("### LLM fallback (same model as Ask Your Documents)")
        use_llm_fallback = gr.Checkbox(
            label="Use LLM for low-confidence items",
            value=bool(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["use_llm_fallback"]),
            interactive=bool(llm_status.get("active")),
        )
        llm_status_text = gr.Markdown(
            "LLM server or model is not active. Load a model in Ask Your Documents."
            if not llm_status.get("active")
            else "Only applies to items below the confidence floor; uses the currently loaded model."
        )
        llm_confidence_floor = gr.Slider(
            label="Ask LLM when confidence <",
            minimum=0.0,
            maximum=1.0,
            value=float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["llm_confidence_floor"]),
            step=0.01,
        )
        gr.Markdown("LLM can overwrite the target only if confidence is below this value.")
        llm_max_items = gr.Number(
            label="Max LLM items per run",
            value=int(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["llm_max_items"]),
            precision=0,
        )
        gr.Markdown("Hard cap to prevent large LLM batches.")

        alias_map = gr.Textbox(
            label="Alias map (optional)",
            placeholder="Topic/Subfolder | passport:2, id:1.5\nMedical Records | mri, xray",
            lines=4,
        )
        gr.Markdown(
            "Format: `Target Label | keyword[:weight], keyword`. "
            "Example: `2. Personal Admin & Life/Taxes | tax:2, finanzamt`."
        )

        min_confidence = gr.Slider(
            label="Only move when confidence ≥",
            minimum=0.0,
            maximum=1.0,
            value=float(presets[DEFAULT_SMART_FILE_SORTER_PRESET]["move_threshold"]),
            step=0.01,
        )
        gr.Markdown("Only items at or above this confidence will move when you confirm.")

    progress_status = gr.Markdown()
    scan_metric = gr.Markdown()
    parse_metric = gr.Markdown()
    embed_metric = gr.Markdown()
    elapsed_metric = gr.Markdown()

    with gr.Accordion("Results", open=True):
        gr.Markdown("### Plan Summary")
        summary_table = gr.Dataframe(
            headers=["Metric", "Value"],
            datatype=["str", "str"],
            row_count=0,
            column_count=(2, "fixed"),
            interactive=False,
        )
        confidence_table = gr.Dataframe(
            headers=["Bucket", "Count"],
            datatype=["str", "number"],
            row_count=0,
            column_count=(2, "fixed"),
            interactive=False,
        )
        llm_used_text = gr.Markdown()

        gr.Markdown("### Review Queue")
        show_advanced_columns = gr.Checkbox(
            label="Show advanced columns",
            value=False,
        )
        gray_low = gr.Slider(
            label="Confidence gray zone (low)",
            minimum=0.0,
            maximum=1.0,
            value=0.55,
            step=0.01,
        )
        gray_high = gr.Slider(
            label="Confidence gray zone (high)",
            minimum=0.0,
            maximum=1.0,
            value=0.75,
            step=0.01,
        )
        margin_threshold = gr.Slider(
            label="Top-2 margin ≤",
            minimum=0.0,
            maximum=1.0,
            value=0.08,
            step=0.01,
        )
        suspicious_raw = gr.Textbox(
            label="Suspicious extensions or filenames (comma-separated)",
            value="desktop.ini, .lnk, .temp",
        )
        review_caption = gr.Markdown()
        review_table = gr.Dataframe(interactive=False)

        gr.Markdown("### Plan Details")
        min_conf_filter = gr.Slider(
            label="Minimum confidence",
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.01,
        )
        target_filter = gr.Dropdown(label="Target folder", choices=["All"], value="All")
        filename_filter = gr.Textbox(label="Filename contains", value="")
        extensions_filter = gr.CheckboxGroup(label="Extensions", choices=[], value=[])
        only_above_threshold = gr.Checkbox(
            label="Only above move threshold",
            value=False,
        )
        only_review_queue = gr.Checkbox(
            label="Only review queue",
            value=False,
        )
        view_mode = gr.Dropdown(
            label="View mode",
            choices=["By folder", "By confidence", "Raw table"],
            value="By folder",
        )
        page = gr.Number(label="Page", value=1, precision=0)
        plan_table = gr.Dataframe(interactive=False)
        csv_export = gr.File(label="Export CSV")

        gr.Markdown("### Apply Moves")
        move_counts = gr.Dataframe(
            headers=["Status", "Count"],
            datatype=["str", "number"],
            row_count=0,
            column_count=(2, "fixed"),
            interactive=False,
        )
        move_mode = gr.Radio(
            label="Mode",
            choices=["Dry-run only (default)", "Enable moving"],
            value="Dry-run only (default)",
        )
        move_warning = gr.Markdown()
        confirm_move = gr.Textbox(label="Type MOVE to confirm", value="")
        apply_button = gr.Button("Move 0 file(s)", variant="primary")
        apply_status = gr.Markdown()

    def _apply_preset(preset_name: str):
        preset_values = presets[preset_name]
        weight_text = _update_weight_total(
            float(preset_values["weight_meta"]),
            float(preset_values["weight_content"]),
            float(preset_values["weight_keyword"]),
        )
        return (
            float(preset_values["weight_meta"]),
            float(preset_values["weight_content"]),
            float(preset_values["weight_keyword"]),
            bool(preset_values["use_llm_fallback"]),
            float(preset_values["llm_confidence_floor"]),
            int(preset_values["llm_max_items"]),
            float(preset_values["move_threshold"]),
            weight_text,
        )

    def _preview_classification(
        root_value: str,
        include_content_value: bool,
        max_files_value,
        max_parent_levels_value: float,
        max_content_mb_value: float,
        max_content_chars_value: float,
        weight_meta_value: float,
        weight_content_value: float,
        weight_keyword_value: float,
        alias_map_value: str,
        use_llm_value: bool,
        llm_floor_value: float,
        llm_max_value: float,
        min_confidence_value: float,
        gray_low_value: float,
        gray_high_value: float,
        margin_threshold_value: float,
        suspicious_raw_value: str,
        show_advanced_value: bool,
        min_conf_filter_value: float,
        target_filter_value: str,
        filename_filter_value: str,
        extensions_value: list,
        only_above_threshold_value: bool,
        only_review_queue_value: bool,
        view_mode_value: str,
        page_value: float,
    ):
        run_id = uuid.uuid4().hex[:8]
        max_files = None if max_files_value == "All" else int(max_files_value)
        meta_weight, content_weight, keyword_weight = _normalize_weights(
            float(weight_meta_value),
            float(weight_content_value),
            float(weight_keyword_value),
        )
        options = SortOptions(
            root=root_value,
            include_content=bool(include_content_value),
            max_parent_levels=int(max_parent_levels_value),
            max_content_mb=int(max_content_mb_value),
            max_content_chars=int(max_content_chars_value),
            weight_meta=float(meta_weight),
            weight_content=float(content_weight),
            weight_keyword=float(keyword_weight),
            alias_map_text=alias_map_value,
            max_files=max_files,
            use_llm_fallback=bool(use_llm_value),
            llm_confidence_floor=float(llm_floor_value),
            llm_max_items=int(llm_max_value),
        )
        progress_state = {"scan": 0, "scan_total": 0, "parse": 0, "parse_total": 0, "embed": 0, "embed_total": 0}
        start_time = time.monotonic()

        def _format_count(value: int, total: int) -> str:
            return f"{value}/{total}" if total else str(value)

        def _update_progress(stage: str, payload: dict) -> None:
            if stage == "scan":
                progress_state["scan"] = payload.get("scanned", progress_state["scan"])
                progress_state["scan_total"] = payload.get("total", progress_state["scan_total"])
            elif stage == "parse":
                progress_state["parse"] = payload.get("parsed", progress_state["parse"])
                progress_state["parse_total"] = payload.get("total", progress_state["parse_total"])
            elif stage == "embed":
                progress_state["embed"] = payload.get("embedded", progress_state["embed"])
                progress_state["embed_total"] = payload.get("total", progress_state["embed_total"])

        plan = preview_sort_plan(options, run_id=run_id, progress_callback=_update_progress)
        elapsed = time.monotonic() - start_time
        df = _prepare_dataframe(plan)
        summary_df, confidence_df, llm_text = _build_summary_tables(
            df, options, min_confidence_value
        )
        review_df, review_mask, review_caption_value = _build_review_queue(
            df,
            gray_low_value,
            gray_high_value,
            margin_threshold_value,
            suspicious_raw_value,
            show_advanced_value,
        )
        filter_choices = _extract_filter_choices(df)
        resolved_target_filter = (
            target_filter_value
            if target_filter_value in filter_choices["targets"]
            else "All"
        )
        resolved_extensions = (
            extensions_value
            if extensions_value
            else filter_choices["extensions"]
        )
        filtered_df, export_df, plan_table_df, plan_caption, page_value = _build_plan_details(
            df,
            review_mask,
            min_conf_filter_value,
            resolved_target_filter,
            filename_filter_value,
            resolved_extensions,
            only_above_threshold_value,
            only_review_queue_value,
            view_mode_value,
            int(page_value),
            min_confidence_value,
            show_advanced_value,
        )
        move_table = _build_move_counts(df, min_confidence_value)
        move_label = (
            f"Move {move_table.loc[move_table['Status'] == 'Will move', 'Count'].sum()} file(s)"
        )
        csv_path = _export_csv(export_df)
        return (
            plan,
            options,
            df,
            review_mask,
            run_id,
            "Built plan with {} file(s).".format(len(df)),
            f"Scanning {_format_count(progress_state['scan'], progress_state['scan_total'])}",
            f"Parsing {_format_count(progress_state['parse'], progress_state['parse_total'])}",
            f"Embedding {_format_count(progress_state['embed'], progress_state['embed_total'])}",
            f"Elapsed {elapsed:.1f}s",
            summary_df,
            confidence_df,
            llm_text,
            review_caption_value,
            review_df,
            gr.Dropdown.update(choices=filter_choices["targets"], value=resolved_target_filter),
            gr.CheckboxGroup.update(choices=filter_choices["extensions"], value=resolved_extensions),
            plan_table_df,
            csv_path,
            move_table,
            move_label,
        )

    def _refresh_tables(
        df: pd.DataFrame | None,
        options: SortOptions | None,
        min_confidence_value: float,
        gray_low_value: float,
        gray_high_value: float,
        margin_threshold_value: float,
        suspicious_raw_value: str,
        show_advanced: bool,
        min_conf_filter_value: float,
        target_filter_value: str,
        filename_filter_value: str,
        extensions_value: list,
        only_above_threshold_value: bool,
        only_review_queue_value: bool,
        view_mode_value: str,
        page_value: float,
    ):
        if df is None:
            empty_df = pd.DataFrame()
            return (
                empty_df,
                empty_df,
                "",
                "",
                empty_df,
                pd.Series(dtype=bool),
                ["All"],
                [],
                empty_df,
                None,
                _build_move_counts(empty_df, min_confidence_value),
                "Move 0 file(s)",
            )
        summary_df, confidence_df, llm_text = _build_summary_tables(
            df,
            options,
            min_confidence_value,
        )
        review_df, review_mask_value, review_caption_value = _build_review_queue(
            df,
            gray_low_value,
            gray_high_value,
            margin_threshold_value,
            suspicious_raw_value,
            show_advanced,
        )
        filter_choices = _extract_filter_choices(df)
        _filtered_df, export_df, plan_display, plan_caption, page_value = _build_plan_details(
            df,
            review_mask_value,
            min_conf_filter_value,
            target_filter_value,
            filename_filter_value,
            extensions_value,
            only_above_threshold_value,
            only_review_queue_value,
            view_mode_value,
            int(page_value),
            min_confidence_value,
            show_advanced,
        )
        csv_path = _export_csv(export_df)
        move_table = _build_move_counts(df, min_confidence_value)
        move_label = f"Move {move_table.loc[move_table['Status'] == 'Will move', 'Count'].sum()} file(s)"
        return (
            summary_df,
            confidence_df,
            llm_text,
            review_caption_value,
            review_df,
            review_mask_value,
            filter_choices["targets"],
            filter_choices["extensions"],
            plan_display,
            csv_path,
            move_table,
            move_label,
        )

    def _apply_moves(
        plan: list | None,
        options: SortOptions | None,
        min_confidence_value: float,
        move_mode_value: str,
        confirm_value: str,
    ) -> str:
        if not plan:
            return "Missing plan options. Re-run the scan."
        if confirm_value.strip().upper() != "MOVE":
            return "Type MOVE to confirm."
        enable_moving = move_mode_value == "Enable moving"
        run_id = uuid.uuid4().hex[:8]
        result = apply_sort_plan_action(
            plan,
            min_confidence=min_confidence_value,
            dry_run=not enable_moving,
            run_id=run_id,
        )
        moved_items = result.get("moved")
        if isinstance(moved_items, list):
            moved_list = moved_items
        elif isinstance(moved_items, tuple):
            moved_list = list(moved_items)
        else:
            moved_list = []

        error_items = result.get("errors")
        if isinstance(error_items, list):
            error_list = error_items
        elif isinstance(error_items, tuple):
            error_list = list(error_items)
        else:
            error_list = []

        skipped_items = result.get("skipped")
        if isinstance(skipped_items, list):
            skipped_list = skipped_items
        elif isinstance(skipped_items, tuple):
            skipped_list = list(skipped_items)
        else:
            skipped_list = []
        lines = []
        if enable_moving:
            lines.append(f"Moved {len(moved_list)} file(s).")
        else:
            lines.append(f"Dry run: would move {len(moved_list)} file(s).")
        if error_list:
            lines.append(f"Errors: {len(error_list)}")
        if skipped_list:
            lines.append(f"Skipped: {len(skipped_list)} below threshold.")
        return "\n".join(lines)

    preset.change(
        _apply_preset,
        inputs=[preset],
        outputs=[
            weight_meta,
            weight_content,
            weight_keyword,
            use_llm_fallback,
            llm_confidence_floor,
            llm_max_items,
            min_confidence,
            weight_total,
        ],
    )

    for component in (weight_meta, weight_content, weight_keyword):
        component.change(
            _update_weight_total,
            inputs=[weight_meta, weight_content, weight_keyword],
            outputs=[weight_total],
        )

    preview_button.click(
        _preview_classification,
        inputs=[
            root,
            include_content,
            max_files_choice,
            max_parent_levels,
            max_content_mb,
            max_content_chars,
            weight_meta,
            weight_content,
            weight_keyword,
            alias_map,
            use_llm_fallback,
            llm_confidence_floor,
            llm_max_items,
            min_confidence,
            gray_low,
            gray_high,
            margin_threshold,
            suspicious_raw,
            show_advanced_columns,
            min_conf_filter,
            target_filter,
            filename_filter,
            extensions_filter,
            only_above_threshold,
            only_review_queue,
            view_mode,
            page,
        ],
        outputs=[
            plan_state,
            options_state,
            df_state,
            review_mask_state,
            run_id_state,
            progress_status,
            scan_metric,
            parse_metric,
            embed_metric,
            elapsed_metric,
            summary_table,
            confidence_table,
            llm_used_text,
            review_caption,
            review_table,
            target_filter,
            extensions_filter,
            plan_table,
            csv_export,
            move_counts,
            apply_button,
        ],
    )

    def _refresh_plan_tables(
        df: pd.DataFrame | None,
        options: SortOptions | None,
        min_confidence_value: float,
        gray_low_value: float,
        gray_high_value: float,
        margin_threshold_value: float,
        suspicious_raw_value: str,
        show_advanced_value: bool,
        min_conf_filter_value: float,
        target_filter_value: str,
        filename_filter_value: str,
        extensions_value: list,
        only_above_threshold_value: bool,
        only_review_queue_value: bool,
        view_mode_value: str,
        page_value: float,
    ):
        (
            summary_df,
            confidence_df,
            llm_text,
            review_caption_value,
            review_df,
            review_mask_value,
            targets,
            extensions,
            plan_display,
            csv_path,
            move_table,
            move_label,
        ) = _refresh_tables(
            df,
            options,
            min_confidence_value,
            gray_low_value,
            gray_high_value,
            margin_threshold_value,
            suspicious_raw_value,
            show_advanced_value,
            min_conf_filter_value,
            target_filter_value,
            filename_filter_value,
            extensions_value,
            only_above_threshold_value,
            only_review_queue_value,
            view_mode_value,
            page_value,
        )
        return (
            summary_df,
            confidence_df,
            llm_text,
            review_caption_value,
            review_df,
            review_mask_value,
            gr.Dropdown.update(choices=targets, value=target_filter_value if target_filter_value in targets else "All"),
            gr.CheckboxGroup.update(choices=extensions, value=extensions_value),
            plan_display,
            csv_path,
            move_table,
            move_label,
        )

    refresh_inputs = [
        df_state,
        options_state,
        min_confidence,
        gray_low,
        gray_high,
        margin_threshold,
        suspicious_raw,
        show_advanced_columns,
        min_conf_filter,
        target_filter,
        filename_filter,
        extensions_filter,
        only_above_threshold,
        only_review_queue,
        view_mode,
        page,
    ]
    refresh_outputs = [
        summary_table,
        confidence_table,
        llm_used_text,
        review_caption,
        review_table,
        review_mask_state,
        target_filter,
        extensions_filter,
        plan_table,
        csv_export,
        move_counts,
        apply_button,
    ]
    for component in (
        min_confidence,
        gray_low,
        gray_high,
        margin_threshold,
        suspicious_raw,
        show_advanced_columns,
        min_conf_filter,
        target_filter,
        filename_filter,
        extensions_filter,
        only_above_threshold,
        only_review_queue,
        view_mode,
        page,
    ):
        component.change(_refresh_plan_tables, inputs=refresh_inputs, outputs=refresh_outputs)

    def _update_move_warning(mode_value: str) -> str:
        if mode_value == "Enable moving":
            return "Moves are destructive. Review your plan before proceeding."
        return "Dry-run mode: no files will be moved."

    move_mode.change(_update_move_warning, inputs=[move_mode], outputs=[move_warning])

    apply_button.click(
        _apply_moves,
        inputs=[plan_state, options_state, min_confidence, move_mode, confirm_move],
        outputs=[apply_status],
    )

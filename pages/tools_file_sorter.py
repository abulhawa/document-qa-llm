import math
import os
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from core.llm import check_llm_status
from core.sync.file_sorter import SortOptions, apply_sort_plan, build_sort_plan


if st.session_state.get("_nav_context") != "hub":
    st.set_page_config(page_title="Smart File Sorter", layout="wide")

st.title("Smart File Sorter")
st.caption("Dry-run classifier for moving files into your 1-5 topic folders.")

PRESET_CONFIGS = {
    "Balanced (default)": {
        "move_threshold": 0.7,
        "weight_meta": 0.55,
        "weight_content": 0.30,
        "weight_keyword": 0.15,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.65,
        "llm_max_items": 200,
    },
    "Path-heavy (folder structure)": {
        "move_threshold": 0.75,
        "weight_meta": 0.7,
        "weight_content": 0.2,
        "weight_keyword": 0.1,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.65,
        "llm_max_items": 200,
    },
    "Content-heavy (document text)": {
        "move_threshold": 0.65,
        "weight_meta": 0.35,
        "weight_content": 0.5,
        "weight_keyword": 0.15,
        "use_llm_fallback": False,
        "llm_confidence_floor": 0.6,
        "llm_max_items": 200,
    },
    "LLM assist (low-confidence only)": {
        "move_threshold": 0.7,
        "weight_meta": 0.5,
        "weight_content": 0.3,
        "weight_keyword": 0.2,
        "use_llm_fallback": True,
        "llm_confidence_floor": 0.75,
        "llm_max_items": 300,
    },
}


def ensure_default(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def apply_preset():
    preset = PRESET_CONFIGS[st.session_state["smart_sort_preset"]]
    for key, value in preset.items():
        st.session_state[key] = value


default_preset = "Balanced (default)"
for preset_key, preset_value in PRESET_CONFIGS[default_preset].items():
    ensure_default(preset_key, preset_value)

st.selectbox(
    "Preset",
    options=list(PRESET_CONFIGS.keys()),
    index=list(PRESET_CONFIGS.keys()).index(st.session_state.get("smart_sort_preset", default_preset)),
    key="smart_sort_preset",
    on_change=apply_preset,
)

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        - Scores each file against the subfolders under your numbered topics.
        - Uses filename + parent folder names + optional content excerpt embeddings.
        - Applies keyword boosts from folder names and optional aliases.
        - Produces a dry-run plan you can review before moving files.
        """
    )

root_default = os.getenv("LOCAL_SYNC_ROOT", "C:\\Users\\ali_a\\My Drive")
llm_status = check_llm_status()

with st.form("smart_sort_config"):
    root = st.text_input("Root folder to scan", value=root_default)
    st.caption("Example: `C:\\Users\\ali_a\\My Drive`. All files under this root will be considered.")
    include_content = st.checkbox("Use file content (PDF/DOCX/TXT)", value=True)
    st.caption("When enabled, PDFs/DOCX/TXT contribute text embeddings. Large files are skipped.")
    max_files_choice = st.selectbox("Max files preset", options=[50, 200, 1000, "All"], index=3)
    st.caption("Use a small limit for quick dry-runs.")
    submitted = st.form_submit_button("Preview classification (dry-run)", type="primary")

    with st.expander("Advanced settings"):
        max_parent_levels = st.slider("Use up to N parent folders as hints", 0, 8, 4)
        st.caption(
            "Depth 0 = filename only; depth 2 uses two parent folders. Example: `file.pdf Tickets Germany`."
        )
        max_content_mb = st.slider("Max file size for content parsing (MB)", 1, 200, 25)
        st.caption("Files larger than this are not parsed for content embeddings.")
        max_content_chars = st.slider("Read up to N chars per file", 500, 20000, 6000, step=500)
        st.caption("Only the first N characters from each file are used.")

        st.subheader("Scoring weights")
        weight_meta = st.slider(
            "Filename + folder embedding weight", 0.0, 1.0, key="weight_meta"
        )
        st.caption("Signals from filename + parent folders. Higher = more path-driven classification.")
        weight_content = st.slider("Content embedding weight", 0.0, 1.0, key="weight_content")
        st.caption("Signals from PDF/DOCX/TXT content. Higher = more content-driven classification.")
        weight_keyword = st.slider("Keyword boost weight", 0.0, 1.0, key="weight_keyword")
        st.caption("Exact keyword matches from folder names + aliases. Good for IDs, taxes, tickets, etc.")
        weight_total = weight_meta + weight_content + weight_keyword
        if weight_total > 0:
            normalized_meta = weight_meta / weight_total
            normalized_content = weight_content / weight_total
            normalized_keyword = weight_keyword / weight_total
        else:
            normalized_meta = 0.0
            normalized_content = 0.0
            normalized_keyword = 0.0
        st.caption(f"Total = {weight_total:.2f} (weights are normalized to 1.00)")

        st.subheader("LLM fallback (same model as Ask Your Documents)")
        use_llm_fallback = st.checkbox(
            "Use LLM for low-confidence items",
            key="use_llm_fallback",
            disabled=not llm_status.get("active"),
        )
        st.caption("Only applies to items below the confidence floor; uses the currently loaded model.")
        if not llm_status.get("active"):
            st.caption("LLM server or model is not active. Load a model in Ask Your Documents.")
        llm_confidence_floor = st.slider(
            "Ask LLM when confidence <",
            0.0,
            1.0,
            key="llm_confidence_floor",
        )
        st.caption("LLM can overwrite the target only if confidence is below this value.")
        llm_max_items = st.number_input("Max LLM items per run", min_value=0, key="llm_max_items")
        st.caption("Hard cap to prevent large LLM batches.")

        alias_map = st.text_area(
            "Alias map (optional)",
            placeholder="Topic/Subfolder | passport:2, id:1.5\nMedical Records | mri, xray",
            height=140,
        )
        st.caption(
            "Format: `Target Label | keyword[:weight], keyword`. Example: `2. Personal Admin & Life/Taxes | tax:2, finanzamt`."
        )

        min_confidence = st.slider("Only move when confidence ≥", 0.0, 1.0, key="move_threshold")
        st.caption("Only items at or above this confidence will move when you confirm.")

if submitted:
    options = SortOptions(
        root=root,
        include_content=include_content,
        max_parent_levels=int(max_parent_levels),
        max_content_mb=int(max_content_mb),
        max_content_chars=int(max_content_chars),
        weight_meta=float(normalized_meta),
        weight_content=float(normalized_content),
        weight_keyword=float(normalized_keyword),
        alias_map_text=alias_map,
        max_files=None if max_files_choice == "All" else int(max_files_choice),
        use_llm_fallback=bool(use_llm_fallback),
        llm_confidence_floor=float(llm_confidence_floor),
        llm_max_items=int(llm_max_items),
    )
    with st.spinner("Scanning and classifying files..."):
        plan = build_sort_plan(options)
    st.session_state["smart_sort_plan"] = plan
    st.session_state["smart_sort_options"] = options
    st.success(f"Built plan with {len(plan)} file(s).")

plan: List | None = st.session_state.get("smart_sort_plan")
options: SortOptions | None = st.session_state.get("smart_sort_options")

def _score_bucket(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "medium"
    if score >= 0.15:
        return "light"
    return "none"


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
    display = dataframe[columns].rename(columns=labels)
    return display


if plan is not None:
    st.subheader("Results")
    st.subheader("Plan Summary")
    df = pd.DataFrame([p.as_dict() for p in plan]) if plan else pd.DataFrame()
    total_files = len(df)
    qualifying_count = (df["confidence"] >= min_confidence).sum() if not df.empty else 0
    below_threshold = total_files - qualifying_count
    ambiguous_band = 0.05
    ambiguous_count = (
        df["confidence"].between(
            max(0.0, min_confidence - ambiguous_band),
            min(1.0, min_confidence + ambiguous_band),
            inclusive="both",
        ).sum()
        if not df.empty
        else 0
    )
    distinct_targets = df["target_label"].nunique() if not df.empty else 0
    show_llm_used = bool(options and options.use_llm_fallback)
    llm_used = (
        df["reason"].str.startswith("llm=").sum() if show_llm_used and not df.empty else 0
    )

    metric_labels = [
        ("Files scanned", total_files),
        (f"Above {min_confidence:.2f}", qualifying_count),
        ("Below threshold", below_threshold),
        (f"Ambiguous (±{ambiguous_band:.2f})", ambiguous_count),
    ]
    if show_llm_used:
        metric_labels.append(("LLM used", llm_used))
    metric_labels.append(("Distinct targets", distinct_targets))

    columns = st.columns(len(metric_labels))
    for column, (label, value) in zip(columns, metric_labels):
        column.metric(label, value)

    if not df.empty:
        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence"),
            y=alt.Y("count()", title="Files"),
        )
        threshold = alt.Chart(pd.DataFrame({"threshold": [min_confidence]})).mark_rule(
            color="#d65f5f"
        ).encode(x="threshold:Q")
        st.altair_chart(
            (hist + threshold).properties(height=220, title="Confidence distribution"),
            use_container_width=True,
        )

    if not df.empty:
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

    st.subheader("Review Queue")
    show_advanced_columns = st.checkbox(
        "Show advanced columns",
        value=False,
        key="show_advanced_columns",
    )
    path_column_config = {
        "Path": st.column_config.TextColumn("Path", help="Full path", max_chars=60)
    }
    review_mask = pd.Series(False, index=df.index) if not df.empty else pd.Series(dtype=bool)
    if df.empty:
        st.info("No files were included in the plan.")
    else:
        gray_low, gray_high = st.slider(
            "Confidence gray zone",
            0.0,
            1.0,
            (0.55, 0.75),
            step=0.01,
            key="review_gray_zone",
        )
        margin_threshold = st.slider(
            "Top-2 margin ≤",
            0.0,
            1.0,
            0.08,
            step=0.01,
            key="review_margin_threshold",
        )
        suspicious_raw = st.text_input(
            "Suspicious extensions or filenames (comma-separated)",
            value="desktop.ini, .lnk, .temp",
            key="review_suspicious_items",
        )
        suspicious_items = [item.strip().lower() for item in suspicious_raw.split(",") if item.strip()]
        suspicious_extensions = {
            f".{item}" for item in suspicious_items if "." not in item
        }
        suspicious_extensions |= {
            item for item in suspicious_items if item.startswith(".") and item.count(".") == 1
        }
        suspicious_names = {
            item for item in suspicious_items if "." in item and not item.startswith(".")
        }

        filename_lower = df["filename"].str.lower()
        extension_lower = df["extension"].str.lower()
        suspicious_mask = filename_lower.isin(suspicious_names) | extension_lower.isin(
            suspicious_extensions
        )
        gray_mask = df["confidence"].between(gray_low, gray_high, inclusive="both")
        if df["top2_margin"].notna().any():
            margin_mask = df["top2_margin"].isna() | (df["top2_margin"] <= margin_threshold)
        else:
            margin_mask = pd.Series(True, index=df.index)
            st.caption("Top-2 margin is not available for this run.")

        review_mask = (gray_mask & margin_mask) | suspicious_mask
        review_df = df[review_mask].sort_values("confidence", ascending=True)
        st.caption(f"{len(review_df)} file(s) flagged for review.")
        if review_df.empty:
            st.info("No files match the review queue criteria.")
        else:
            review_display = _build_display_table(review_df, show_advanced_columns)
            st.dataframe(
                review_display,
                use_container_width=True,
                hide_index=True,
                column_config=path_column_config,
            )

    st.subheader("Plan Details")
    if df.empty:
        st.info("No files were included in the plan.")
    else:
        min_conf_filter = st.slider(
            "Minimum confidence",
            0.0,
            1.0,
            0.0,
            key="filter_confidence",
        )
        target_options = ["All"] + sorted(df["target_label_display"].unique())
        target_filter = st.selectbox("Target folder", options=target_options, index=0)
        filename_filter = st.text_input("Filename contains", value="", key="filter_filename")
        extension_options = sorted({ext for ext in df["extension"].unique() if ext})
        if "" in df["extension"].unique():
            extension_options = extension_options + ["(none)"]
        selected_extensions = st.multiselect(
            "Extensions",
            options=extension_options,
            default=extension_options,
            key="filter_extensions",
        )
        only_above_threshold = st.checkbox(
            f"Only above move threshold ({min_confidence:.2f})",
            value=False,
            key="filter_above_threshold",
        )
        only_review_queue = st.checkbox(
            "Only review queue",
            value=False,
            key="filter_review_queue",
        )

        filtered = df[df["confidence"] >= min_conf_filter].copy()
        if target_filter != "All":
            filtered = filtered[filtered["target_label_display"] == target_filter]
        if filename_filter:
            filtered = filtered[
                filtered["filename"].str.contains(filename_filter, case=False, na=False)
            ]
        if selected_extensions:
            extension_mask = filtered["extension"].isin(
                [ext for ext in selected_extensions if ext != "(none)"]
            )
            if "(none)" in selected_extensions:
                extension_mask |= filtered["extension"] == ""
            filtered = filtered[extension_mask]
        if only_above_threshold:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if only_review_queue and not review_mask.empty:
            filtered = filtered[review_mask.reindex(filtered.index, fill_value=False)]

        if filtered.empty:
            st.info("No files match the current filters.")
        else:
            view_mode = st.selectbox(
                "View mode",
                options=["By folder", "By confidence", "Raw table"],
                index=0,
            )

            def paginate_dataframe(dataframe: pd.DataFrame, key: str) -> pd.DataFrame:
                page_size = 200
                total_rows = len(dataframe)
                total_pages = max(1, math.ceil(total_rows / page_size))
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key=key,
                )
                start = (page - 1) * page_size
                end = min(start + page_size, total_rows)
                if total_rows:
                    st.caption(f"Showing {start + 1}-{end} of {total_rows}")
                return dataframe.iloc[start:end]

            export_df = filtered
            if view_mode == "By folder":
                summary = (
                    filtered.groupby("target_label_display", dropna=False)
                    .agg(
                        file_count=("path", "size"),
                        avg_confidence=("confidence", "mean"),
                        pct_above_threshold=(
                            "confidence",
                            lambda s: (s >= min_confidence).mean() * 100,
                        ),
                    )
                    .reset_index()
                    .sort_values("file_count", ascending=False)
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

                selected_folder = st.selectbox(
                    "Folder",
                    options=summary["target_label_display"].tolist(),
                )
                details = filtered[filtered["target_label_display"] == selected_folder]
                paged = paginate_dataframe(details, f"page_folder_{selected_folder}")
                details_display = _build_display_table(paged, show_advanced_columns)
                st.dataframe(
                    details_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config=path_column_config,
                )
                export_df = details
            else:
                if view_mode == "By confidence":
                    view_df = filtered.sort_values("confidence", ascending=False)
                else:
                    view_df = filtered
                paged = paginate_dataframe(view_df, f"page_{view_mode.lower().replace(' ', '_')}")
                view_display = _build_display_table(paged, show_advanced_columns)
                st.dataframe(
                    view_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config=path_column_config,
                )
                export_df = view_df

            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export CSV", data=csv_data, file_name="smart_sort_plan.csv")

    st.subheader("Apply Moves")
    st.warning("Moves are destructive. Review your plan before proceeding.")
    confirm = st.text_input("Type MOVE to confirm", value="")
    if st.button("Move files", type="primary"):
        if confirm.strip().upper() != "MOVE":
            st.error("Confirmation text mismatch. Type MOVE to proceed.")
        elif not options:
            st.error("Missing plan options. Re-run the scan.")
        else:
            with st.spinner("Moving files..."):
                result = apply_sort_plan(plan, min_confidence=min_confidence, dry_run=False)
            st.success(f"Moved {len(result['moved'])} file(s).")
            if result["errors"]:
                st.error(f"Errors: {len(result['errors'])}")
                st.json(result["errors"])
            if result["skipped"]:
                st.info(f"Skipped: {len(result['skipped'])} below threshold.")

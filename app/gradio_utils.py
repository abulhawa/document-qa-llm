"""Shared helpers for the Gradio UI."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime
import json
from typing import Any

from app.schemas import IngestLogEntry, SearchHit


DEFAULT_FILETYPES: tuple[str, ...] = (
    "pdf",
    "docx",
    "txt",
    "md",
    "csv",
    "pptx",
    "xlsx",
)


def to_chat_history(
    history: Sequence[Sequence[str]] | None, message: str | None = None
) -> list[dict[str, str]]:
    chat_history: list[dict[str, str]] = []
    for entry in history or []:
        if len(entry) < 2:
            continue
        user_msg, assistant_msg = entry[0], entry[1]
        if user_msg:
            chat_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            chat_history.append({"role": "assistant", "content": assistant_msg})
    if message:
        chat_history.append({"role": "user", "content": message})
    return chat_history


def stream_text(text: str, chunk_size: int = 20) -> Iterator[str]:
    if not text:
        yield ""
        return
    for idx in range(0, len(text), chunk_size):
        yield text[: idx + chunk_size]


def normalize_date_input(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def search_hits_to_rows(hits: Iterable[SearchHit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        rows.append(
            {
                "Filename": hit.filename or (hit.path.split("/")[-1] if hit.path else ""),
                "Path": hit.path,
                "Score": hit.score,
                "Date": hit.modified_at or hit.created_at,
            }
        )
    return rows


def render_search_snippet(hit: SearchHit) -> str:
    highlights = [h for h in hit.highlights if h]
    if highlights:
        return "\n\n".join(highlights)
    return hit.path or ""


def format_ingest_logs(entries: Iterable[IngestLogEntry]) -> str:
    lines = []
    for entry in entries:
        parts = [entry.status or "unknown"]
        if entry.stage:
            parts.append(entry.stage)
        if entry.reason:
            parts.append(entry.reason)
        suffix = " - ".join(parts)
        lines.append(f"{entry.path} :: {suffix}")
    return "\n".join(lines)


def summarize_cluster_result(result: Mapping[str, Any]) -> dict[str, Any]:
    checksums = result.get("checksums", [])
    labels = result.get("labels", [])
    clusters = result.get("clusters", [])
    parent_summaries = result.get("parent_summaries", [])
    total_files = len(checksums)
    outlier_count = sum(1 for label in labels if label == -1)
    cluster_count = len([cluster for cluster in clusters if cluster.get("cluster_id", -1) >= 0])
    parent_count = len(parent_summaries)
    return {
        "total_files": total_files,
        "outliers": outlier_count,
        "cluster_count": cluster_count,
        "parent_count": parent_count,
        "params": result.get("params", {}),
    }


def build_payload_lookup(
    checksums: Sequence[str], payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for checksum, payload in zip(checksums, payloads, strict=False):
        lookup[str(checksum)] = payload
    return lookup


def dataframe_to_records(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    if isinstance(df, list):
        if df and isinstance(df[0], dict):
            return df
        return [
            {"Cluster ID": row[0], "Generated Name": row[1], "User Label": row[2], "Document Count": row[3]}
            for row in df
        ]
    try:
        return json.loads(df.to_json(orient="records"))
    except AttributeError:
        return []

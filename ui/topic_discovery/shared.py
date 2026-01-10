from typing import Any


def format_file_label(payload: dict[str, Any], checksum: str) -> str:
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


def format_label(identifier: int, name: str, hide_ids: bool) -> str:
    return name if hide_ids else f"{identifier} — {name}"

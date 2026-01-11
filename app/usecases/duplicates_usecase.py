"""Use case helpers for duplicate file lookup."""

from __future__ import annotations

from app.schemas import DuplicateFileEntry, DuplicateGroup, DuplicatesResponse
from utils.file_utils import format_file_size
from utils import opensearch_utils
from utils.time_utils import format_timestamp, format_timestamp_ampm


def lookup_duplicates() -> DuplicatesResponse:
    """Fetch duplicate file groups from OpenSearch."""
    groups: list[DuplicateGroup] = []
    for checksum in opensearch_utils.get_duplicate_checksums():
        files = opensearch_utils.get_files_by_checksum(checksum)
        entries = [
            DuplicateFileEntry(
                checksum=checksum,
                path=file_data.get("path", ""),
                canonical_path=file_data.get("canonical_path"),
                location_type=file_data.get("location_type", "canonical"),
                filetype=file_data.get("filetype"),
                created_at=file_data.get("created_at"),
                modified_at=file_data.get("modified_at"),
                indexed_at=file_data.get("indexed_at"),
                num_chunks=file_data.get("num_chunks"),
                bytes=file_data.get("bytes"),
            )
            for file_data in files
        ]
        if entries:
            groups.append(DuplicateGroup(checksum=checksum, files=entries))

    return DuplicatesResponse(groups=groups)


def format_duplicate_rows(response: DuplicatesResponse) -> list[dict[str, object]]:
    """Prepare duplicate rows for tabular display."""
    rows: list[dict[str, object]] = []
    for group in response.groups:
        for entry in group.files:
            rows.append(
                {
                    "Checksum": group.checksum,
                    "Location": entry.path,
                    "Canonical Path": entry.canonical_path or entry.path,
                    "Location Type": (
                        "Alias" if entry.location_type == "alias" else "Canonical"
                    ),
                    "Filetype": entry.filetype,
                    "Created": format_timestamp_ampm(entry.created_at or ""),
                    "Modified": format_timestamp_ampm(entry.modified_at or ""),
                    "Indexed": format_timestamp(entry.indexed_at or ""),
                    "Chunks": entry.num_chunks,
                    "Size": entry.bytes or 0,
                }
            )
    return rows


def format_duplicate_size(size_bytes: int) -> str:
    """Format duplicate file sizes for display."""
    return format_file_size(size_bytes)

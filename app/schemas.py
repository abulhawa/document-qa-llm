"""UI-agnostic request/response schemas for the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class RequestContext:
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# QA (Ask Your Documents)
# -----------------------------
@dataclass
class DocumentSnippet:
    text: str
    path: str
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    page: Optional[int] = None
    location_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QARequest:
    question: str
    mode: str = "default"
    temperature: float = 0.2
    model: Optional[str] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    use_cache: bool = True
    context: Optional[RequestContext] = None


@dataclass
class QAResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    documents: List[DocumentSnippet] = field(default_factory=list)
    rewritten_question: Optional[str] = None
    clarification: Optional[str] = None
    error: Optional[str] = None


# -----------------------------
# Ingestion
# -----------------------------
@dataclass
class IngestRequest:
    paths: List[str]
    mode: Literal["ingest", "reingest", "delete"] = "ingest"
    recursive: bool = False
    context: Optional[RequestContext] = None


@dataclass
class IngestResponse:
    task_ids: List[str] = field(default_factory=list)
    queued_count: int = 0
    errors: List[str] = field(default_factory=list)


# -----------------------------
# Search
# -----------------------------
@dataclass
class SearchRequest:
    query: str
    page: int = 0
    page_size: int = 25
    sort: Literal["relevance", "modified"] = "relevance"
    path_contains: Optional[str] = None
    filetypes: List[str] = field(default_factory=list)
    modified_from: Optional[str] = None
    modified_to: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    context: Optional[RequestContext] = None


@dataclass
class SearchHit:
    path: str
    filename: Optional[str] = None
    filetype: Optional[str] = None
    score: Optional[float] = None
    bytes: Optional[int] = None
    modified_at: Optional[str] = None
    created_at: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    total: int = 0
    took_ms: int = 0
    hits: List[SearchHit] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Index Viewer
# -----------------------------
@dataclass
class IndexViewerRequest:
    path_filter: Optional[str] = None
    only_embedding_mismatches: bool = False
    include_qdrant_counts: bool = False
    page: int = 0
    page_size: int = 25
    context: Optional[RequestContext] = None


@dataclass
class IndexFileEntry:
    filename: Optional[str]
    path: str
    filetype: Optional[str] = None
    modified_at: Optional[str] = None
    created_at: Optional[str] = None
    indexed_at: Optional[str] = None
    bytes: Optional[int] = None
    num_chunks: Optional[int] = None
    qdrant_count: Optional[int] = None
    checksum: Optional[str] = None
    canonical_path: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class IndexViewerResponse:
    files: List[IndexFileEntry] = field(default_factory=list)
    total: int = 0


# -----------------------------
# Duplicates
# -----------------------------
@dataclass
class DuplicateFileEntry:
    checksum: str
    path: str
    canonical_path: Optional[str] = None
    location_type: Literal["canonical", "alias"] = "canonical"
    filetype: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    indexed_at: Optional[str] = None
    num_chunks: Optional[int] = None
    bytes: Optional[int] = None


@dataclass
class DuplicateGroup:
    checksum: str
    files: List[DuplicateFileEntry] = field(default_factory=list)


@dataclass
class DuplicatesResponse:
    groups: List[DuplicateGroup] = field(default_factory=list)


# -----------------------------
# Ingestion Logs
# -----------------------------
@dataclass
class IngestLogRequest:
    status: Optional[str] = None
    path_query: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    size: int = 200
    context: Optional[RequestContext] = None


@dataclass
class IngestLogEntry:
    path: str
    bytes: Optional[int] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    reason: Optional[str] = None
    stage: Optional[str] = None
    attempt_at: Optional[str] = None


@dataclass
class IngestLogResponse:
    logs: List[IngestLogEntry] = field(default_factory=list)


# -----------------------------
# Watchlist
# -----------------------------
@dataclass
class WatchlistPrefix:
    prefix: str
    total_files: Optional[int] = None
    indexed_files: Optional[int] = None
    remaining_files: Optional[int] = None
    quick_wins: Optional[int] = None
    last_refreshed: Optional[str] = None
    last_scanned: Optional[str] = None
    last_scan_found: Optional[int] = None
    last_scan_missing: Optional[int] = None


@dataclass
class WatchlistListRequest:
    context: Optional[RequestContext] = None


@dataclass
class WatchlistListResponse:
    prefixes: List[WatchlistPrefix] = field(default_factory=list)


@dataclass
class WatchlistUpdateRequest:
    action: Literal["add", "remove"]
    prefix: str
    context: Optional[RequestContext] = None


@dataclass
class WatchlistUpdateResponse:
    success: bool
    message: Optional[str] = None


@dataclass
class WatchlistRefreshRequest:
    prefix: str
    include_chunk_counts: bool = True
    context: Optional[RequestContext] = None


@dataclass
class WatchlistRefreshResponse:
    prefix: str
    total_files: int = 0
    indexed_files: int = 0
    remaining_files: int = 0


@dataclass
class WatchlistScanRequest:
    prefix: str
    context: Optional[RequestContext] = None


@dataclass
class WatchlistScanResponse:
    prefix: str
    found: int = 0
    marked_missing: int = 0


# -----------------------------
# File Resync
# -----------------------------
ResyncBucket = Literal["SAFE", "REVIEW", "BLOCKED", "INFO"]
ResyncActionType = Literal[
    "INGEST_NEW",
    "ADD_ALIAS",
    "REMOVE_ALIAS",
    "SET_CANONICAL",
    "DELETE_CONTENT",
]


@dataclass
class FileResyncScanRequest:
    roots: List[str]
    allowed_extensions: List[str] = field(default_factory=list)
    context: Optional[RequestContext] = None


@dataclass
class ResyncAction:
    type: ResyncActionType
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileResyncPlanItem:
    bucket: ResyncBucket
    reason: str
    checksum: str
    content_id: Optional[str]
    disk_paths: List[str]
    indexed_paths: List[str]
    actions: List[ResyncAction] = field(default_factory=list)
    explanation: str = ""
    new_checksum: Optional[str] = None


@dataclass
class FileResyncPlanResponse:
    items: List[FileResyncPlanItem] = field(default_factory=list)
    counts: Dict[ResyncBucket, int] = field(default_factory=dict)
    scanned_roots: List[str] = field(default_factory=list)
    scanned_roots_failed: List[str] = field(default_factory=list)
    generated_at: Optional[datetime] = None


@dataclass
class FileResyncApplyRequest:
    items: List[FileResyncPlanItem] = field(default_factory=list)
    ingest_missing: bool = False
    apply_safe_only: bool = True
    delete_orphaned: bool = False
    retire_replaced_content: bool = False
    context: Optional[RequestContext] = None


@dataclass
class FileResyncApplyResponse:
    ingested: int = 0
    updated_fulltext: int = 0
    updated_chunks: int = 0
    updated_qdrant: int = 0
    deleted_checksums: int = 0
    errors: List[str] = field(default_factory=list)
    counts_by_bucket: Dict[ResyncBucket, int] = field(default_factory=dict)


# -----------------------------
# Running Tasks
# -----------------------------
@dataclass
class RunningTasksRequest:
    lookback_hours: Optional[int] = None
    failed_page: int = 0
    failed_page_size: int = 25
    include_failed: bool = False
    context: Optional[RequestContext] = None


@dataclass
class TaskSummary:
    task_id: Optional[str] = None
    name: Optional[str] = None
    state: Optional[str] = None
    worker: Optional[str] = None
    args: Optional[str] = None
    kwargs: Optional[str] = None
    eta: Optional[str] = None


@dataclass
class FailedTaskEntry:
    task_id: str
    name: str
    time: str
    state: Optional[str] = None
    error: Optional[str] = None
    args: Optional[str] = None
    kwargs: Optional[str] = None


@dataclass
class RunningTasksOverview:
    counts: Dict[str, int] = field(default_factory=dict)
    queue_depth: Optional[int] = None


@dataclass
class RunningTasksResponse:
    overview: RunningTasksOverview = field(default_factory=RunningTasksOverview)
    active: List[TaskSummary] = field(default_factory=list)
    reserved: List[TaskSummary] = field(default_factory=list)
    scheduled: List[TaskSummary] = field(default_factory=list)
    failed: List[FailedTaskEntry] = field(default_factory=list)


# -----------------------------
# Tools/Admin actions
# -----------------------------
@dataclass
class ToolActionRequest:
    tool: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Optional[RequestContext] = None


@dataclass
class ToolActionResponse:
    success: bool
    message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdminActionRequest:
    action: Literal["revoke_task", "clear_finished", "reload", "shutdown"]
    task_id: Optional[str] = None
    terminate: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Optional[RequestContext] = None


@dataclass
class AdminActionResponse:
    success: bool
    message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Topic Discovery
# -----------------------------
@dataclass
class TopicDiscoveryRequest:
    action: Literal[
        "run_clustering",
        "load_cached",
        "clear_cache",
        "run_naming",
        "run_review",
    ]
    settings: Dict[str, Any] = field(default_factory=dict)
    context: Optional[RequestContext] = None


@dataclass
class TopicClusterSummary:
    cluster_id: str
    label: Optional[str] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicNameSuggestion:
    cluster_id: str
    name: str
    source: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicDiscoveryResponse:
    run_id: Optional[str] = None
    used_cache: bool = False
    clusters: List[TopicClusterSummary] = field(default_factory=list)
    naming_suggestions: List[TopicNameSuggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

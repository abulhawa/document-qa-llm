from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Set

from config import (
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    QDRANT_COLLECTION,
    QDRANT_URL,
    logger,
)
from core.opensearch_client import get_client
from ingestion.orchestrator import ingest_one
from opensearchpy import helpers
from qdrant_client import QdrantClient, models
from utils.file_utils import compute_checksum, normalize_path
from utils.opensearch_utils import (
    delete_chunks_by_checksum,
    delete_fulltext_by_checksum,
    get_fulltext_by_checksum,
)
from utils.qdrant_utils import delete_vectors_by_checksum

DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MIN_INGEST_BYTES = 1024  # guardrail against tiny/temp files
TEMP_PREFIXES = ("~$",)
TEMP_SUFFIXES = (".tmp", ".temp", ".swp", ".swx", ".part")
IGNORE_DIR_NAMES = {".git", ".obsidian", ".cache", "node_modules", "__pycache__"}
CANONICAL_PATH_RULES = """
1. Canonical path must exist on disk.
2. Canonical path must be unique per checksum.
3. Canonical path is auto-changed only when unambiguous (single disk path).
4. Otherwise, surface for REVIEW.
"""


# -----------------------------------------------------------------------------
# Reconciliation algorithm (pseudocode)
# -----------------------------------------------------------------------------
# 1. Scan disk
#    - walk roots, filter allowed extensions
#    - collect FileHit(path, size, mtime, checksum)
# 2. Load index snapshot
#    - fetch full-text docs to map checksum -> (canonical_path, aliases, id)
#    - build path -> checksum map (canonical + aliases)
#    - record duplicate index docs per checksum for BLOCKED items
# 3. Build plan
#    - group disk hits by checksum and by path
#    - for each checksum not in index: INFO/NOT_INDEXED (+ INGEST_NEW action)
#    - for each indexed checksum:
#        * compare indexed paths vs disk paths
#        * propose ADD_ALIAS for new disk locations
#        * propose REMOVE_ALIAS for aliases missing on disk within scanned roots
#        * if canonical missing and unambiguous replacement: SET_CANONICAL (SAFE)
#        * if canonical missing but ambiguous: REVIEW
#        * if no paths remain on disk inside scanned roots: REVIEW/ORPHANED with
#          optional DELETE_CONTENT action (executed only when delete_orphaned)
#    - detect path replacement (same path, different checksum) to mark REVIEW and
#      optionally flag DELETE_CONTENT when retire_replaced_content is enabled and
#      the old checksum has no other disk paths.
#    - count bucket totals (SAFE, REVIEW, BLOCKED, INFO) with explanations.
# 4. Apply plan (dry-run by default)
#    - gather actions grouped per checksum
#    - apply aliases/canonical updates via OpenSearch partial updates
#    - when canonical moves, also update chunk docs + Qdrant payloads
#    - optionally ingest missing files (ingest_missing)
#    - optionally delete orphaned/replaced content (delete_orphaned/retire option)
#    - return summary counts + errors; idempotent so a second run is a no-op.
# -----------------------------------------------------------------------------


Bucket = Literal["SAFE", "REVIEW", "BLOCKED", "INFO"]
ActionType = Literal[
    "INGEST_NEW",
    "ADD_ALIAS",
    "REMOVE_ALIAS",
    "SET_CANONICAL",
    "DELETE_CONTENT",
]


@dataclass
class FileHit:
    path: str
    size: int
    mtime: float
    checksum: str
    ext: str


@dataclass
class IndexedDoc:
    content_id: str
    checksum: str
    canonical_path: Optional[str]
    aliases: list[str]

    @property
    def paths(self) -> list[str]:
        vals = [self.canonical_path] if self.canonical_path else []
        vals.extend(self.aliases)
        return [normalize_path(p) for p in vals if p]


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]


@dataclass
class PlanItem:
    bucket: Bucket
    reason: str
    checksum: str
    content_id: Optional[str]
    disk_paths: list[str]
    indexed_paths: list[str]
    actions: list[Action] = field(default_factory=list)
    explanation: str = ""
    new_checksum: Optional[str] = None

    def as_row(self) -> dict:
        """Flatten for Streamlit table rendering."""
        return {
            "bucket": self.bucket,
            "reason": self.reason,
            "checksum": self.checksum,
            "content_id": self.content_id,
            "indexed_paths": "; ".join(self.indexed_paths),
            "disk_paths": "; ".join(self.disk_paths),
            "actions": ", ".join(sorted({a.type for a in self.actions})),
            "explanation": self.explanation,
            "new_checksum": self.new_checksum,
        }


@dataclass
class ReconciliationPlan:
    items: list[PlanItem]
    counts: Dict[Bucket, int]
    scanned_roots: list[str]
    scanned_roots_failed: list[str]
    generated_at: datetime

    def as_rows(self) -> list[dict]:
        return [item.as_row() for item in self.items]


@dataclass
class ApplyOptions:
    ingest_missing: bool = False
    apply_safe_only: bool = True
    delete_orphaned: bool = False
    retire_replaced_content: bool = False


@dataclass
class ApplyResult:
    ingested: int = 0
    updated_fulltext: int = 0
    updated_chunks: int = 0
    updated_qdrant: int = 0
    deleted_checksums: int = 0
    errors: list[str] = field(default_factory=list)
    counts_by_bucket: Dict[Bucket, int] = field(default_factory=dict)


@dataclass
class ScanResult:
    hits: list[FileHit]
    scanned_roots_successful: list[str]
    scanned_roots_failed: list[str]
    ignored_files: int = 0


def _should_ignore_file(name: str, dirpath: str) -> bool:
    if name.startswith(TEMP_PREFIXES):
        return True
    lowered = name.lower()
    if lowered.endswith(TEMP_SUFFIXES):
        return True
    parts = normalize_path(dirpath).split("/")
    return any(part in IGNORE_DIR_NAMES for part in parts if part)


def scan_files(roots: Sequence[str], allowed_exts: Iterable[str]) -> ScanResult:
    """Scan filesystem roots and return discovered FileHit entries."""

    normalized_roots = [normalize_path(r) for r in roots if r]
    allowed = {e.lower().strip() for e in allowed_exts if e}
    if not allowed:
        allowed = set(DEFAULT_ALLOWED_EXTENSIONS)

    hits: list[FileHit] = []
    ignored_files = 0
    scanned_success: list[str] = []
    scanned_failed: list[str] = []

    for root in normalized_roots:
        if not root:
            continue
        if not os.path.isdir(root):
            scanned_failed.append(root)
            logger.warning("Root not accessible: %s", root)
            continue
        scanned_success.append(root)
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if allowed and ext not in allowed:
                    continue
                if _should_ignore_file(name, dirpath):
                    ignored_files += 1
                    continue
                abs_path = normalize_path(os.path.join(dirpath, name))
                try:
                    st = os.stat(abs_path)
                except OSError as e:  # noqa: BLE001
                    logger.warning("Stat failed for %s: %s", abs_path, e)
                    continue
                if st.st_size < MIN_INGEST_BYTES:
                    ignored_files += 1
                    continue
                try:
                    checksum = compute_checksum(abs_path)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Checksum failed for %s: %s", abs_path, e)
                    continue
                hits.append(
                    FileHit(
                        path=abs_path,
                        size=st.st_size,
                        mtime=st.st_mtime,
                        checksum=checksum,
                        ext=ext.lstrip("."),
                    )
                )
    return ScanResult(
        hits=hits,
        scanned_roots_successful=scanned_success,
        scanned_roots_failed=scanned_failed,
        ignored_files=ignored_files,
    )


def _load_index_snapshot() -> tuple[dict[str, IndexedDoc], dict[str, list[str]], dict[str, str]]:
    """Return (docs_by_checksum, duplicate_ids_by_checksum, path_to_checksum)."""

    client = get_client()
    docs: dict[str, IndexedDoc] = {}
    duplicates: dict[str, list[str]] = {}
    path_to_checksum: dict[str, str] = {}

    # Pull all full-text docs because they store canonical + aliases.
    for hit in helpers.scan(
        client,
        index=FULLTEXT_INDEX,
        query={"query": {"match_all": {}}, "_source": ["path", "aliases", "checksum"]},
    ):
        source = hit.get("_source") or {}
        checksum = source.get("checksum") or hit.get("_id")
        if not checksum:
            continue
        canonical = normalize_path(source.get("path") or "") or None
        aliases = [normalize_path(p) for p in (source.get("aliases") or []) if p]
        doc = IndexedDoc(
            content_id=hit.get("_id"),
            checksum=checksum,
            canonical_path=canonical,
            aliases=aliases,
        )
        if checksum in docs:
            duplicates.setdefault(checksum, [docs[checksum].content_id]).append(doc.content_id)
        else:
            docs[checksum] = doc
        for p in doc.paths:
            path_to_checksum[p] = checksum

    return docs, duplicates, path_to_checksum


def _paths_within_roots(paths: Iterable[str], roots: Sequence[str]) -> list[str]:
    normalized_roots = [normalize_path(r) for r in roots if r]
    result = []
    for p in paths:
        np = normalize_path(p)
        if any(np.startswith(r.rstrip("/") + "/") or np == r for r in normalized_roots):
            result.append(np)
    return result


def _bucket_counts(items: Iterable[PlanItem]) -> Dict[Bucket, int]:
    counts: Dict[Bucket, int] = {"SAFE": 0, "REVIEW": 0, "BLOCKED": 0, "INFO": 0}
    for item in items:
        counts[item.bucket] = counts.get(item.bucket, 0) + 1
    return counts


def build_reconciliation_plan(
    scan_result: ScanResult | Sequence[FileHit],
    roots: Sequence[str] | None = None,
    *,
    retire_replaced_content: bool = False,
) -> ReconciliationPlan:
    """Produce a deterministic reconciliation plan."""

    docs, duplicates, _ = _load_index_snapshot()
    hits: Sequence[FileHit]
    scanned_roots: list[str]
    scanned_failed: list[str]
    if isinstance(scan_result, ScanResult):
        hits = scan_result.hits
        scanned_roots = [normalize_path(r) for r in scan_result.scanned_roots_successful]
        scanned_failed = [normalize_path(r) for r in scan_result.scanned_roots_failed]
    else:
        hits = scan_result
        scanned_roots = [normalize_path(r) for r in (roots or []) if r]
        scanned_failed = []

    disk_by_checksum: dict[str, list[str]] = {}
    disk_by_path: dict[str, str] = {}
    for hit in hits:
        disk_by_checksum.setdefault(hit.checksum, []).append(hit.path)
        disk_by_path[hit.path] = hit.checksum

    items: list[PlanItem] = []
    blocked_checksums = set(duplicates)

    # Duplicate index documents for the same checksum are BLOCKED.
    for checksum, ids in duplicates.items():
        canonical = docs.get(checksum, IndexedDoc("", checksum, None, [])).paths
        items.append(
            PlanItem(
                bucket="BLOCKED",
                reason="DUPLICATE_INDEX_DOCS",
                checksum=checksum,
                content_id=None,
                disk_paths=disk_by_checksum.get(checksum, []),
                indexed_paths=canonical,
                explanation=f"Multiple full-text docs share checksum={checksum}: {ids}",
            )
        )

    # Files present on disk but not indexed.
    for checksum, paths in disk_by_checksum.items():
        if checksum in docs:
            continue
        actions: list[Action] = []
        for p in sorted(paths):
            # Only propose ingestion when guardrails are met (size already checked in scan).
            actions.append(Action("INGEST_NEW", {"path": p}))
        items.append(
            PlanItem(
                bucket="INFO",
                reason="NOT_INDEXED",
                checksum=checksum,
                content_id=None,
                disk_paths=sorted(paths),
                indexed_paths=[],
                actions=actions,
                explanation="Checksum not found in indexes.",
            )
        )

    for checksum, doc in docs.items():
        if checksum in blocked_checksums:
            continue
        disk_paths = sorted(disk_by_checksum.get(checksum, []))
        indexed_paths = doc.paths
        actions: list[Action] = []
        reasons: list[str] = []
        bucket: Bucket = "SAFE"

        missing_paths = [p for p in indexed_paths if p not in disk_paths]
        missing_in_scanned = _paths_within_roots(missing_paths, scanned_roots)
        extra_disk_paths = [p for p in disk_paths if p not in indexed_paths]

        # New aliases to add.
        if extra_disk_paths:
            reasons.append("ADD_ALIAS")
            for path in extra_disk_paths:
                actions.append(Action("ADD_ALIAS", {"path": path}))

        # Aliases missing on disk (only remove when root was scanned).
        if missing_in_scanned:
            for path in missing_in_scanned:
                if path in doc.aliases:
                    actions.append(Action("REMOVE_ALIAS", {"path": path}))
                    reasons.append("REMOVE_ALIAS")

        # Canonical path handling (see CANONICAL_PATH_RULES above).
        canonical_missing = doc.canonical_path and doc.canonical_path in missing_in_scanned
        if canonical_missing:
            if len(disk_paths) == 1:
                new_canonical = disk_paths[0]
                actions.append(
                    Action("SET_CANONICAL", {"path": new_canonical, "previous": doc.canonical_path})
                )
                reasons.append("SET_CANONICAL")
                bucket = "SAFE"
            else:
                reasons.append("CANONICAL_AMBIGUOUS")
                bucket = "REVIEW"

        # Orphan detection (no disk paths for this checksum under scanned roots).
        if not disk_paths and missing_in_scanned:
            bucket = "REVIEW"
            reasons.append("ORPHANED_INDEX_CONTENT")
            actions.append(Action("DELETE_CONTENT", {"checksum": checksum, "conditional": True}))

        # Path replaced with different checksum at canonical path.
        if doc.canonical_path:
            disk_checksum_at_path = disk_by_path.get(doc.canonical_path)
            if disk_checksum_at_path and disk_checksum_at_path != checksum:
                reasons.append("PATH_REPLACED")
                bucket = "REVIEW"
                # Only retire when the old checksum had exactly one known path (canonical) and it was replaced.
                if (
                    retire_replaced_content
                    and not disk_paths
                    and len(indexed_paths) == 1
                ):
                    actions.append(
                        Action(
                            "DELETE_CONTENT",
                            {
                                "checksum": checksum,
                                "conditional": True,
                                "reason": "retire_replaced_content",
                            },
                        )
                    )
                items.append(
                    PlanItem(
                        bucket=bucket,
                        reason="PATH_REPLACED",
                        checksum=checksum,
                        content_id=doc.content_id,
                        disk_paths=disk_paths,
                        indexed_paths=indexed_paths,
                        actions=list(actions),
                        explanation=(
                            f"Path {doc.canonical_path} now points to checksum "
                            f"{disk_checksum_at_path}; indexed checksum is {checksum}."
                        ),
                        new_checksum=disk_checksum_at_path,
                    )
                )
                continue

        # No changes detected: skip noisy rows.
        if not actions and not reasons:
            continue

        if bucket == "SAFE" and ("CANONICAL_AMBIGUOUS" in reasons or "ORPHANED_INDEX_CONTENT" in reasons):
            bucket = "REVIEW"

        items.append(
            PlanItem(
                bucket=bucket,
                reason=";".join(sorted(set(reasons))) or "MIXED",
                checksum=checksum,
                content_id=doc.content_id,
                disk_paths=disk_paths,
                indexed_paths=indexed_paths,
                actions=actions,
                explanation="; ".join(sorted(set(reasons))) or "Metadata update required.",
            )
        )

    return ReconciliationPlan(
        items=items,
        counts=_bucket_counts(items),
        scanned_roots=scanned_roots,
        scanned_roots_failed=scanned_failed,
        generated_at=datetime.now(timezone.utc),
    )


def _update_fulltext_paths(content_id: str, canonical_path: Optional[str], aliases: list[str]) -> int:
    """Update canonical/alias paths for a single full-text document."""

    client = get_client()
    now = datetime.now(timezone.utc).isoformat()
    script_lines = [
        "if (params.aliases != null) { ctx._source.aliases = params.aliases; }",
        "else if (ctx._source.containsKey('aliases')) { ctx._source.remove('aliases'); }",
    ]
    params: Dict[str, Any] = {"aliases": aliases}
    if canonical_path is not None:
        script_lines.append(
            "ctx._source.path=params.path; ctx._source.filename=params.filename; "
            "ctx._source.path_updated_at=params.path_updated_at;"
        )
        params.update(
            {
                "path": canonical_path,
                "filename": os.path.basename(canonical_path),
                "path_updated_at": now,
            }
        )
    script = {"source": " ".join(script_lines), "lang": "painless", "params": params}
    resp = client.update(
        index=FULLTEXT_INDEX,
        id=content_id,
        body={"script": script},
        params={"refresh": "true"},
    )
    return int(resp.get("_shards", {}).get("successful", 0))


def _update_chunk_paths(checksum: str, canonical_path: str) -> int:
    """Update chunk docs to the new canonical path."""

    client = get_client()
    script = {
        "source": "ctx._source.path=params.path; ctx._source.filename=params.filename;",
        "lang": "painless",
        "params": {"path": canonical_path, "filename": os.path.basename(canonical_path)},
    }
    resp = client.update_by_query(
        index=CHUNKS_INDEX,
        body={"script": script, "query": {"term": {"checksum": checksum}}},
        params={"refresh": "true", "conflicts": "proceed"},
    )
    return int(resp.get("updated", 0))


def _update_qdrant_payload(checksum: str, new_path: str) -> int:
    """Update Qdrant payload path for a checksum."""

    client = QdrantClient(url=QDRANT_URL)
    filename = os.path.basename(new_path)
    result = client.set_payload(
        collection_name=QDRANT_COLLECTION,
        points=models.Filter(
            must=[models.FieldCondition(key="checksum", match=models.MatchValue(value=checksum))]
        ),
        payload={"path": new_path, "filename": filename},
        wait=True,
    )
    result_any: Any = result
    if isinstance(result_any, dict):
        return int(result_any.get("result", {}).get("count", 0))
    result_dict = getattr(result_any, "result", None)
    if isinstance(result_dict, dict):
        return int(result_dict.get("count", 0))
    return 0


def _apply_alias_canonical_updates(
    changes: dict[str, dict[str, Any]],
    result: ApplyResult,
) -> None:
    """Apply alias + canonical updates for each checksum."""

    for checksum, change in changes.items():
        doc = get_fulltext_by_checksum(checksum)
        if not doc or not doc.get("id"):
            result.errors.append(f"Missing full-text doc for checksum={checksum}")
            continue

        aliases: Set[str] = set(doc.get("aliases") or [])
        aliases.update(change.get("add_aliases", set()))
        aliases.difference_update(change.get("remove_aliases", set()))

        canonical_path: Optional[str] = doc.get("path")
        new_canonical = change.get("canonical_path")
        if new_canonical is not None:
            canonical_path = new_canonical
        if canonical_path and canonical_path in aliases:
            aliases.discard(canonical_path)

        try:
            updated = _update_fulltext_paths(doc["id"], canonical_path, sorted(aliases))
            result.updated_fulltext += updated
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to update full-text paths for %s: %s", checksum, e)
            result.errors.append(str(e))
            continue

        if canonical_path and canonical_path != doc.get("path"):
            try:
                result.updated_chunks += _update_chunk_paths(checksum, canonical_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to update chunk paths for %s: %s", checksum, e)
                result.errors.append(str(e))
            try:
                result.updated_qdrant += _update_qdrant_payload(checksum, canonical_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to update Qdrant payload for %s: %s", checksum, e)
                result.errors.append(str(e))


def apply_plan(plan: ReconciliationPlan, options: ApplyOptions) -> ApplyResult:
    """Execute a reconciliation plan respecting safety options."""

    result = ApplyResult(counts_by_bucket=plan.counts)
    if not plan.items:
        return result

    metadata_changes: dict[str, dict[str, Any]] = {}
    ingestion_queue: list[str] = []
    delete_checksums: list[str] = []

    for item in plan.items:
        if item.bucket == "BLOCKED":
            continue
        if options.apply_safe_only and item.bucket != "SAFE":
            continue
        for action in item.actions:
            if action.type == "INGEST_NEW":
                if options.ingest_missing:
                    payload_path = action.payload.get("path")
                    if payload_path:
                        ingestion_queue.append(payload_path)
            elif action.type == "ADD_ALIAS":
                change = metadata_changes.setdefault(
                    item.checksum, {"add_aliases": set(), "remove_aliases": set()}
                )
                change["add_aliases"].add(action.payload.get("path"))
            elif action.type == "REMOVE_ALIAS":
                change = metadata_changes.setdefault(
                    item.checksum, {"add_aliases": set(), "remove_aliases": set()}
                )
                change["remove_aliases"].add(action.payload.get("path"))
            elif action.type == "SET_CANONICAL":
                change = metadata_changes.setdefault(
                    item.checksum, {"add_aliases": set(), "remove_aliases": set()}
                )
                change["canonical_path"] = action.payload.get("path")
            elif action.type == "DELETE_CONTENT":
                if options.delete_orphaned or action.payload.get("reason") == "retire_replaced_content":
                    delete_checksums.append(action.payload.get("checksum", item.checksum))

    # Apply metadata updates first to avoid ingest collisions.
    if metadata_changes:
        _apply_alias_canonical_updates(metadata_changes, result)

    for checksum in delete_checksums:
        try:
            delete_vectors_by_checksum(checksum)
            delete_chunks_by_checksum(checksum)
            delete_fulltext_by_checksum(checksum)
            result.deleted_checksums += 1
        except Exception as e:  # noqa: BLE001
            logger.error("Delete failed for checksum=%s: %s", checksum, e)
            result.errors.append(str(e))

    for path in dict.fromkeys(ingestion_queue):
        if not path:
            continue
        try:
            ingest_one(path, force=True, replace=True, op="resync", source="resync")
            result.ingested += 1
        except Exception as e:  # noqa: BLE001
            logger.error("Ingest failed for %s: %s", path, e)
            result.errors.append(str(e))

    return result


__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "CANONICAL_PATH_RULES",
    "ScanResult",
    "FileHit",
    "IndexedDoc",
    "PlanItem",
    "Action",
    "ReconciliationPlan",
    "ApplyOptions",
    "ApplyResult",
    "scan_files",
    "compute_checksum",
    "build_reconciliation_plan",
    "apply_plan",
]

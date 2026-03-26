#!/usr/bin/env python
from __future__ import annotations

"""
Canary/full migration utility: rebuild chunks from FULLTEXT_INDEX text_full.

What it does
- Selects candidate docs from FULLTEXT_INDEX (optionally by prefix/checksum)
- Builds new chunks from text_full using deterministic policy rules
- In apply mode:
  1) deletes old vectors/chunks by checksum
  2) embeds and indexes rebuilt chunks

Notes
- Default mode is dry-run (no writes)
- This path intentionally does not preserve page numbers; it uses location_percent

Usage
- Dry-run canary:
    python scripts/rechunk_from_fulltext.py --prefix "C:/Users/ali_a/My Drive" --limit 20
- Apply canary:
    python scripts/rechunk_from_fulltext.py --prefix "C:/Users/ali_a/My Drive" --limit 20 --apply
- Target specific checksums:
    python scripts/rechunk_from_fulltext.py --checksums "abc,def,ghi" --apply
"""

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

# Ensure repo root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHUNKS_INDEX, FULLTEXT_INDEX, logger  # noqa: E402
from core.chunking import split_documents  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from utils.file_utils import format_file_size, normalize_path  # noqa: E402
from utils.opensearch_utils import delete_chunks_by_checksum, index_documents  # noqa: E402
from utils.qdrant_utils import (  # noqa: E402
    count_qdrant_chunks_by_checksum,
    delete_vectors_by_checksum,
    index_chunks_in_batches,
)

_IDENTITY_DOC_TYPES = {"cv", "cover_letter", "reference_letter"}
_SHORT_MAX_CHARS = 3000
_MEDIUM_MAX_CHARS = 20000


@dataclass(slots=True)
class FulltextCandidate:
    checksum: str
    path: str
    filetype: str
    text_full: str
    created_at: Optional[str]
    modified_at: Optional[str]
    size_bytes: int
    doc_type: Optional[str]
    person_name: Optional[str]
    authority_rank: Optional[float]


@dataclass(slots=True)
class ChunkPolicy:
    profile: str
    chunk_size: int
    chunk_overlap: int
    length_bucket: str


def _length_bucket(char_len: int) -> str:
    if char_len <= _SHORT_MAX_CHARS:
        return "short"
    if char_len <= _MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


def _resolve_policy(doc_type: Optional[str], text_full: str) -> ChunkPolicy:
    length_bucket = _length_bucket(len(text_full))
    dt = (doc_type or "").strip()
    if dt in _IDENTITY_DOC_TYPES:
        return ChunkPolicy(
            profile="profile_identity_native_400_50",
            chunk_size=400,
            chunk_overlap=50,
            length_bucket=length_bucket,
        )
    if length_bucket == "short":
        return ChunkPolicy(
            profile="profile_native_short_600_80",
            chunk_size=600,
            chunk_overlap=80,
            length_bucket=length_bucket,
        )
    return ChunkPolicy(
        profile="profile_native_default_800_100",
        chunk_size=800,
        chunk_overlap=100,
        length_bucket=length_bucket,
    )


def _build_query(prefix: Optional[str], checksums: Optional[List[str]]) -> Dict[str, Any]:
    if checksums:
        return {"terms": {"checksum": checksums}}
    if prefix:
        return {"prefix": {"path": normalize_path(prefix)}}
    return {"match_all": {}}


def _count_os_chunks_by_checksum(client: Any, checksum: str) -> int:
    resp = client.count(
        index=CHUNKS_INDEX,
        body={"query": {"term": {"checksum": {"value": checksum}}}},
    )
    return int(resp.get("count", 0))


def _select_candidates(
    *,
    prefix: Optional[str],
    checksums: Optional[List[str]],
    batch_size: int,
    limit: int,
    client: Any,
) -> tuple[List[FulltextCandidate], Dict[str, int]]:
    stats = {
        "scanned_docs": 0,
        "selected_docs": 0,
        "skipped_empty_text": 0,
        "skipped_missing_checksum": 0,
        "skipped_missing_path": 0,
    }
    out: List[FulltextCandidate] = []
    scroll_id = ""

    try:
        response = client.search(
            index=FULLTEXT_INDEX,
            body={
                "size": max(1, batch_size),
                "query": _build_query(prefix, checksums),
                "_source": [
                    "checksum",
                    "path",
                    "filetype",
                    "text_full",
                    "created_at",
                    "modified_at",
                    "size_bytes",
                    "doc_type",
                    "person_name",
                    "authority_rank",
                ],
                "sort": [{"_id": "asc"}],
            },
            params={"scroll": "2m"},
        )
        scroll_id = response.get("_scroll_id", "")

        while len(out) < limit:
            hits = response.get("hits", {}).get("hits", []) or []
            if not hits:
                break

            for hit in hits:
                if len(out) >= limit:
                    break

                stats["scanned_docs"] += 1
                source = hit.get("_source") or {}

                checksum = str(source.get("checksum") or "").strip()
                path = str(source.get("path") or "").strip()
                text_full = str(source.get("text_full") or "")

                if not checksum:
                    stats["skipped_missing_checksum"] += 1
                    continue
                if not path:
                    stats["skipped_missing_path"] += 1
                    continue
                if not text_full.strip():
                    stats["skipped_empty_text"] += 1
                    continue

                size_bytes_raw = source.get("size_bytes", 0)
                try:
                    size_bytes = int(size_bytes_raw) if size_bytes_raw is not None else 0
                except (TypeError, ValueError):
                    size_bytes = 0

                authority_raw = source.get("authority_rank")
                authority_rank: Optional[float] = None
                if authority_raw is not None:
                    try:
                        authority_rank = float(authority_raw)
                    except (TypeError, ValueError):
                        authority_rank = None

                out.append(
                    FulltextCandidate(
                        checksum=checksum,
                        path=path,
                        filetype=str(source.get("filetype") or ""),
                        text_full=text_full,
                        created_at=source.get("created_at"),
                        modified_at=source.get("modified_at"),
                        size_bytes=size_bytes,
                        doc_type=source.get("doc_type"),
                        person_name=source.get("person_name"),
                        authority_rank=authority_rank,
                    )
                )
                stats["selected_docs"] += 1

            if len(out) >= limit or not scroll_id:
                break

            response = client.scroll(scroll_id=scroll_id, params={"scroll": "2m"})
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:  # noqa: BLE001
                pass

    return out, stats


def _build_chunks(candidate: FulltextCandidate, policy: ChunkPolicy) -> List[Dict[str, Any]]:
    docs = [Document(page_content=candidate.text_full, metadata={"source": candidate.path})]
    chunks = split_documents(
        docs,
        chunk_size=policy.chunk_size,
        chunk_overlap=policy.chunk_overlap,
    )
    now_iso = datetime.now().astimezone().isoformat()
    total = max(len(chunks), 1)

    for i, chunk in enumerate(chunks):
        chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{candidate.checksum}:{i}"))
        chunk["chunk_index"] = i
        chunk["path"] = candidate.path
        chunk["checksum"] = candidate.checksum
        chunk["chunk_char_len"] = len(chunk.get("text") or "")
        chunk["filetype"] = candidate.filetype
        chunk["indexed_at"] = now_iso
        chunk["created_at"] = candidate.created_at
        chunk["modified_at"] = candidate.modified_at
        chunk["bytes"] = candidate.size_bytes
        chunk["size"] = format_file_size(candidate.size_bytes)
        chunk["page"] = None
        chunk["location_percent"] = round((i / max(total - 1, 1)) * 100.0, 2)
        if candidate.doc_type is not None:
            chunk["doc_type"] = candidate.doc_type
        if candidate.person_name is not None:
            chunk["person_name"] = candidate.person_name
        if candidate.authority_rank is not None:
            chunk["authority_rank"] = candidate.authority_rank
        # Keep policy metadata for auditability in this migration path.
        chunk["chunk_profile"] = policy.profile
        chunk["chunk_policy_version"] = "v1"
        chunk["chunk_size"] = policy.chunk_size
        chunk["chunk_overlap"] = policy.chunk_overlap
        chunk["length_bucket"] = policy.length_bucket
        chunk["extraction_mode"] = "native"
        chunk["quality_bucket"] = "medium"

    return chunks


def _index_new_chunks(chunks: List[Dict[str, Any]]) -> tuple[int, int]:
    os_indexed = 0
    os_errors = 0

    def _os_batch(group: List[Dict[str, Any]]) -> None:
        nonlocal os_indexed, os_errors
        n, errs = index_documents(group)
        os_indexed += int(n)
        err_count = len(errs)
        os_errors += err_count
        if err_count:
            raise RuntimeError(f"OpenSearch chunk indexing returned {err_count} errors")

    index_chunks_in_batches(chunks, os_index_batch=_os_batch)
    return os_indexed, os_errors


def rechunk_from_fulltext(
    *,
    prefix: Optional[str],
    checksums: Optional[List[str]],
    limit: int,
    batch_size: int,
    apply: bool,
    sample_limit: int = 20,
    client: Any = None,
) -> Dict[str, Any]:
    os_client = client or get_client()
    candidates, select_stats = _select_candidates(
        prefix=prefix,
        checksums=checksums,
        batch_size=batch_size,
        limit=max(1, limit),
        client=os_client,
    )

    summary: Dict[str, Any] = {
        **select_stats,
        "apply_mode": bool(apply),
        "rebuilt_docs": 0,
        "failed_docs": 0,
        "total_new_chunks": 0,
        "deleted_old_vectors": 0,
        "deleted_old_chunks": 0,
        "indexed_new_chunks": 0,
        "sample": [],
        "sample_truncated": 0,
        "errors": [],
    }

    for cand in candidates:
        policy = _resolve_policy(cand.doc_type, cand.text_full)
        new_chunks = _build_chunks(cand, policy)
        old_os = _count_os_chunks_by_checksum(os_client, cand.checksum)
        old_qdrant = count_qdrant_chunks_by_checksum(cand.checksum)
        summary["total_new_chunks"] += len(new_chunks)

        sample_entry = {
            "checksum": cand.checksum,
            "path": cand.path,
            "doc_type": cand.doc_type or "__missing__",
            "profile": policy.profile,
            "old_os_chunks": old_os,
            "old_qdrant_chunks": old_qdrant,
            "new_chunks": len(new_chunks),
        }
        if len(summary["sample"]) < max(0, sample_limit):
            summary["sample"].append(sample_entry)
        else:
            summary["sample_truncated"] += 1

        if not apply:
            continue

        try:
            deleted_vectors = delete_vectors_by_checksum(cand.checksum)
            if int(deleted_vectors) == 0 and isinstance(old_qdrant, int) and old_qdrant > 0:
                # Some Qdrant responses omit deleted count; fall back to pre-delete count for reporting.
                deleted_vectors = old_qdrant
            deleted_chunks = delete_chunks_by_checksum(cand.checksum)
            indexed_chunks, _ = _index_new_chunks(new_chunks)

            summary["deleted_old_vectors"] += int(deleted_vectors)
            summary["deleted_old_chunks"] += int(deleted_chunks)
            summary["indexed_new_chunks"] += int(indexed_chunks)
            summary["rebuilt_docs"] += 1
        except Exception as exc:  # noqa: BLE001
            summary["failed_docs"] += 1
            message = f"checksum={cand.checksum} path={cand.path} error={exc}"
            logger.error("Rechunk failed: %s", message)
            summary["errors"].append(message)

    return summary


def _parse_checksums(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    vals = [v.strip() for v in raw.split(",")]
    return [v for v in vals if v]


def _safe_console_text(value: Any) -> str:
    text = str(value)
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Canary/full rechunk from FULLTEXT_INDEX text_full payloads."
    )
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--checksums", type=str, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Maximum sample rows printed/stored in summary (default 20).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Apply writes (default is dry-run).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output only.",
    )
    args = ap.parse_args()

    summary = rechunk_from_fulltext(
        prefix=args.prefix,
        checksums=_parse_checksums(args.checksums),
        limit=max(1, args.limit),
        batch_size=max(1, args.batch_size),
        apply=args.apply,
        sample_limit=max(0, args.sample_limit),
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(
        "Rechunk summary: "
        f"apply_mode={summary['apply_mode']} "
        f"scanned_docs={summary['scanned_docs']} "
        f"selected_docs={summary['selected_docs']} "
        f"rebuilt_docs={summary['rebuilt_docs']} "
        f"failed_docs={summary['failed_docs']} "
        f"total_new_chunks={summary['total_new_chunks']} "
        f"deleted_old_vectors={summary['deleted_old_vectors']} "
        f"deleted_old_chunks={summary['deleted_old_chunks']} "
        f"indexed_new_chunks={summary['indexed_new_chunks']}"
    )
    print(
        "Selection skips: "
        f"empty_text={summary['skipped_empty_text']} "
        f"missing_checksum={summary['skipped_missing_checksum']} "
        f"missing_path={summary['skipped_missing_path']}"
    )
    print(f"Sample (up to {len(summary['sample'])} docs):")
    for item in summary["sample"]:
        safe_path = _safe_console_text(item["path"])
        print(
            f"  checksum={item['checksum']} profile={item['profile']} "
            f"old_os={item['old_os_chunks']} old_qdrant={item['old_qdrant_chunks']} "
            f"new={item['new_chunks']} path={safe_path}"
        )
    if summary["sample_truncated"] > 0:
        print(f"Sample truncated rows: {summary['sample_truncated']}")
    if summary["errors"]:
        print("Errors:")
        for err in summary["errors"]:
            print(f"  {err}")


if __name__ == "__main__":
    main()

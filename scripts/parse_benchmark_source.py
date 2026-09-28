"""Parse a reconstructed benchmark corpus with the application's real loaders.

The parsed corpus can contain gated OfficeQA text. This script writes a deterministic
runner-local artifact. The GitHub workflow persists successful artifacts to a private
Hugging Face Storage Bucket using short-lived OIDC credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
PARSER_INPUTS = (
    Path("scripts/parse_benchmark_source.py"),
    Path("core/file_loader.py"),
    Path("core/document_preprocessor.py"),
    Path("core/text_preprocess.py"),
    Path("requirements/parse.txt"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield dict(json.loads(line))


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _normalise_metadata(metadata: Mapping[str, Any], logical_source: str) -> dict[str, Any]:
    out = {str(k): _json_safe(v) for k, v in dict(metadata or {}).items()}
    out["source"] = logical_source
    return out


def _parser_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in PARSER_INPUTS:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("langchain-core", "langchain-community", "pypdf", "ftfy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _artifact_fingerprint(
    parent_source_fingerprint: str,
    parent_lock_sha: str,
    parser_fingerprint: str,
    package_versions: Mapping[str, str],
    python_version: str,
) -> str:
    payload = {
        "parent_source_artifact_fingerprint": parent_source_fingerprint,
        "parent_source_lock_sha256": parent_lock_sha,
        "parser_fingerprint": parser_fingerprint,
        "package_versions": dict(sorted(package_versions.items())),
        "python_version": python_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_one(task: Mapping[str, Any]) -> dict[str, Any]:
    from core.document_preprocessor import PreprocessConfig, preprocess_to_documents
    from core.file_loader import load_documents

    track = str(task["track"])
    benchmark_id = str(task["benchmark_id"])
    entry = dict(task["entry"])
    file_path = Path(str(task["file_path"]))
    source_document_id = str(entry["source_document_id"])
    ext = file_path.suffix.lower().lstrip(".")
    logical_source = f"benchmark://{benchmark_id}/{track}/{source_document_id}"

    try:
        loaded = load_documents(str(file_path))
        docs = preprocess_to_documents(
            loaded,
            source_path=logical_source,
            cfg=PreprocessConfig(),
            doc_type=ext,
        )
        if not docs:
            raise RuntimeError("loader returned no documents/pages")

        pages: list[dict[str, Any]] = []
        empty_pages = 0
        character_count = 0
        for page_index, doc in enumerate(docs):
            text = str(getattr(doc, "page_content", "") or "")
            if not text.strip():
                empty_pages += 1
            character_count += len(text)
            pages.append(
                {
                    "source_benchmark": track,
                    "source_document_id": source_document_id,
                    "page_index": page_index,
                    "text": text,
                    "metadata": _normalise_metadata(
                        getattr(doc, "metadata", {}) or {},
                        logical_source,
                    ),
                }
            )

        document = {
            "source_benchmark": track,
            "source_document_id": source_document_id,
            "source_path": str(entry["path"]),
            "filetype": ext,
            "source_sha256": entry.get("sha256"),
            "source_bytes": entry.get("bytes"),
            "page_count": len(pages),
            "empty_pages": empty_pages,
            "character_count": character_count,
            "status": "parsed",
        }
        return {"document": document, "pages": pages, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {
            "document": {
                "source_benchmark": track,
                "source_document_id": source_document_id,
                "source_path": str(entry.get("path") or ""),
                "filetype": ext,
                "source_sha256": entry.get("sha256"),
                "source_bytes": entry.get("bytes"),
                "page_count": 0,
                "empty_pages": 0,
                "character_count": 0,
                "status": "failed",
            },
            "pages": [],
            "error": {
                "source_benchmark": track,
                "source_document_id": source_document_id,
                "source_path": str(entry.get("path") or ""),
                "error_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        }


def parse_source(
    source_root: Path,
    output_root: Path,
    *,
    benchmark_id: str,
    workers: int,
    max_failures: int,
    repo_revision: str,
    parent_source_fingerprint: str,
) -> Path:
    output_dir = output_root / benchmark_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_lock = source_root / "source.lock.json"
    if not source_lock.exists():
        raise FileNotFoundError(f"missing reconstructed source lock: {source_lock}")

    all_errors: list[dict[str, Any]] = []
    track_stats: dict[str, dict[str, int]] = {}

    for track in TRACKS:
        source_track = source_root / track
        source_documents = source_track / "documents.jsonl"
        if not source_documents.exists():
            raise FileNotFoundError(f"missing {source_documents}")

        track_out = output_dir / track
        entries = sorted(
            _jsonl_rows(source_documents),
            key=lambda row: str(row["source_document_id"]),
        )
        tasks = [
            {
                "track": track,
                "benchmark_id": benchmark_id,
                "entry": entry,
                "file_path": str(source_track / str(entry["path"])),
            }
            for entry in entries
        ]

        documents: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for result in executor.map(_parse_one, tasks):
                documents.append(result["document"])
                pages.extend(result["pages"])
                if result["error"]:
                    all_errors.append(result["error"])

        _jsonl_write(track_out / "documents.jsonl", documents)
        _jsonl_write(track_out / "pages.jsonl", pages)

        evaluation_out = track_out / "evaluation"
        evaluation_out.mkdir(parents=True, exist_ok=True)
        for name in ("queries.jsonl", "qrels.jsonl", "answers.jsonl"):
            source_file = source_track / name
            if source_file.exists():
                shutil.copy2(source_file, evaluation_out / name)

        track_failures = sum(1 for row in documents if row["status"] == "failed")
        track_stats[track] = {
            "documents_expected": len(entries),
            "documents_parsed": len(entries) - track_failures,
            "failures": track_failures,
            "pages": len(pages),
            "empty_pages": sum(int(row["empty_pages"]) for row in documents),
            "characters": sum(int(row["character_count"]) for row in documents),
        }

    _jsonl_write(output_dir / "errors.jsonl", all_errors)

    files: list[dict[str, Any]] = []
    parsed_bytes = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        size = path.stat().st_size
        parsed_bytes += size
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": size,
            }
        )

    parser_fingerprint = _parser_fingerprint()
    parent_lock_sha = _sha256_file(source_lock)
    package_versions = _package_versions()
    python_version = platform.python_version()
    artifact_digest = _artifact_fingerprint(
        parent_source_fingerprint,
        parent_lock_sha,
        parser_fingerprint,
        package_versions,
        python_version,
    )
    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "artifact_type": "parsed",
        "artifact_fingerprint": artifact_digest,
        "parent_source_artifact_fingerprint": parent_source_fingerprint,
        "parent_source_lock_sha256": parent_lock_sha,
        "parser_fingerprint": parser_fingerprint,
        "repository_revision": repo_revision,
        "python_version": python_version,
        "package_versions": package_versions,
        "workers": max(1, workers),
        "stats": track_stats,
        "total_failures": len(all_errors),
        "parsed_bytes": parsed_bytes,
        "files": files,
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": False,
            "artifact_uri": None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if len(all_errors) > max_failures:
        raise RuntimeError(
            f"parse failures {len(all_errors)} exceed allowed maximum {max_failures}"
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".benchmark/parsed"))
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--parent-source-fingerprint", required=True)
    args = parser.parse_args()

    try:
        output = parse_source(
            args.source_root,
            args.output_root,
            benchmark_id=args.benchmark_id,
            workers=args.workers,
            max_failures=args.max_failures,
            repo_revision=os.environ.get("GITHUB_SHA", "local"),
            parent_source_fingerprint=args.parent_source_fingerprint,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Benchmark parse failed: {exc}", file=sys.stderr)
        return 1

    print(f"Parsed benchmark corpus: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

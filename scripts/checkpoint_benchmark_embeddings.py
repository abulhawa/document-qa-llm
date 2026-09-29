"""Checkpoint and resume benchmark embedding shards without changing artifact identity.

The semantic embedding fingerprint is owned by scripts.embed_benchmark_chunks.
This helper changes only execution: document embeddings are computed in durable
parts, uploaded immediately, and assembled into the same final shard contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts import embed_benchmark_chunks as base

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_SIZE = 2048


def _helper_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _checkpoint_signature(
    *,
    artifact_fingerprint: str,
    shard_count: int,
    checkpoint_size: int,
    batch_size: int,
    helper_fingerprint: str,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_fingerprint": artifact_fingerprint,
        "shard_count": shard_count,
        "checkpoint_size": checkpoint_size,
        "batch_size": batch_size,
        "helper_fingerprint": helper_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _part_ranges(total_rows: int, checkpoint_size: int) -> list[tuple[int, int]]:
    if total_rows < 0:
        raise ValueError("total_rows must be >= 0")
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be > 0")
    return [
        (start, min(start + checkpoint_size, total_rows))
        for start in range(0, total_rows, checkpoint_size)
    ]


def _hf(
    args: list[str],
    *,
    quiet: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for checkpoint persistence")
    env = os.environ.copy()
    # Model loading runs offline, but checkpoint persistence must reach the
    # Hugging Face bucket and OIDC endpoint.
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    kwargs: dict[str, Any] = {
        "env": env,
        "text": True,
        "check": check,
    }
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.run(["hf", "buckets", *args], **kwargs)


def _try_restore(remote_dir: str, local_dir: Path) -> bool:
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    result = _hf(["sync", remote_dir, str(local_dir)], quiet=True, check=False)
    if result.returncode != 0:
        if local_dir.exists():
            shutil.rmtree(local_dir)
        return False
    return True


def _files_manifest(root: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "sha256": base._sha256_file(root / name),
            "bytes": (root / name).stat().st_size,
        }
        for name in names
    ]


def _validate_part(
    part_dir: Path,
    *,
    artifact_fingerprint: str,
    checkpoint_signature: str,
    shard_index: int,
    shard_count: int,
    part_index: int,
    part_count: int,
    start_row: int,
    stop_row: int,
) -> dict[str, Any] | None:
    import numpy as np

    manifest_path = part_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "artifact_type": "embedding-checkpoint",
            "artifact_fingerprint": artifact_fingerprint,
            "checkpoint_signature": checkpoint_signature,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "part_index": part_index,
            "part_count": part_count,
            "start_row": start_row,
            "stop_row": stop_row,
            "chunk_embeddings": stop_row - start_row,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None

        files = {str(row["path"]): row for row in manifest.get("files", [])}
        for name in ("embeddings.npy", "records.jsonl"):
            path = part_dir / name
            meta = files.get(name)
            if not path.exists() or not meta:
                return None
            if path.stat().st_size != int(meta.get("bytes", -1)):
                return None
            if base._sha256_file(path) != meta.get("sha256"):
                return None

        vectors = np.load(
            part_dir / "embeddings.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        if vectors.ndim != 2 or vectors.shape[0] != stop_row - start_row:
            return None
        if int(manifest.get("vector_dimension", -1)) != int(vectors.shape[1]):
            return None

        records = list(base._jsonl_rows(part_dir / "records.jsonl"))
        if len(records) != stop_row - start_row:
            return None
        if [int(row["row_index"]) for row in records] != list(
            range(start_row, stop_row)
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return manifest


def _write_part(
    part_dir: Path,
    *,
    rows: list[dict[str, Any]],
    vectors: Any,
    artifact_fingerprint: str,
    checkpoint_signature: str,
    shard_index: int,
    shard_count: int,
    part_index: int,
    part_count: int,
    start_row: int,
    stop_row: int,
) -> None:
    import numpy as np

    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    np.save(part_dir / "embeddings.npy", vectors, allow_pickle=False)
    base._jsonl_write(
        part_dir / "records.jsonl",
        (
            {
                "row_index": start_row + i,
                "global_index": int(row["_global_index"]),
                "id": row["id"],
                "source_benchmark": row["source_benchmark"],
                "source_document_id": row["source_document_id"],
                "chunk_index": row["chunk_index"],
            }
            for i, row in enumerate(rows)
        ),
    )
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "embedding-checkpoint",
        "artifact_fingerprint": artifact_fingerprint,
        "checkpoint_signature": checkpoint_signature,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "part_index": part_index,
        "part_count": part_count,
        "start_row": start_row,
        "stop_row": stop_row,
        "chunk_embeddings": stop_row - start_row,
        "vector_dimension": int(vectors.shape[1]),
        "files": _files_manifest(
            part_dir,
            ("embeddings.npy", "records.jsonl"),
        ),
    }
    (part_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _upload_part(part_dir: Path, remote_part: str) -> None:
    _hf(
        [
            "sync",
            str(part_dir),
            remote_part,
            "--exclude",
            "manifest.json",
        ]
    )
    _hf(
        [
            "cp",
            str(part_dir / "manifest.json"),
            f"{remote_part}/manifest.json",
        ]
    )


def build_checkpointed_shard(
    chunks_root: Path,
    output_root: Path,
    checkpoint_root: Path,
    *,
    benchmark_id: str,
    parent_chunks_fingerprint: str,
    artifact_fingerprint: str,
    model_name: str,
    model_revision: str,
    input_format: str,
    execution_device: str,
    batch_size: int,
    shard_index: int,
    shard_count: int,
    checkpoint_size: int,
    remote_root: str,
    repo_revision: str,
    vector_provider: Any | None = None,
) -> Path:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if checkpoint_size % batch_size != 0:
        raise ValueError("checkpoint_size must be a multiple of batch_size")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard layout")

    base._validate_parent(
        chunks_root,
        benchmark_id=benchmark_id,
        parent_chunks_fingerprint=parent_chunks_fingerprint,
    )
    fingerprint, packages, python_version, code_fingerprint = base._identity(
        parent_chunks_fingerprint=parent_chunks_fingerprint,
        model_name=model_name,
        model_revision=model_revision,
        input_format=input_format,
        execution_device=execution_device,
    )
    if fingerprint != artifact_fingerprint:
        raise ValueError("embedding artifact fingerprint mismatch")

    rows = base._selected_chunks(
        chunks_root,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    texts = [str(row.get("text") or "") for row in rows]
    if any(not text for text in texts):
        raise RuntimeError("chunk artifact contains an empty text row")

    ranges = _part_ranges(len(rows), checkpoint_size)
    checkpoint_signature = _checkpoint_signature(
        artifact_fingerprint=artifact_fingerprint,
        shard_count=shard_count,
        checkpoint_size=checkpoint_size,
        batch_size=batch_size,
        helper_fingerprint=_helper_fingerprint(),
    )
    local_parts_root = (
        checkpoint_root
        / benchmark_id
        / artifact_fingerprint
        / checkpoint_signature
        / "shards"
        / f"{shard_index:03d}"
        / "parts"
    )
    remote_parts_root = (
        f"{remote_root.rstrip('/')}/work/{checkpoint_signature}/"
        f"shards/{shard_index:03d}/parts"
    )

    valid_parts: dict[int, Path] = {}
    missing_parts: list[tuple[int, int, int, Path, str]] = []
    for part_index, (start_row, stop_row) in enumerate(ranges):
        local_part = local_parts_root / f"{part_index:05d}"
        remote_part = f"{remote_parts_root}/{part_index:05d}"
        restored = _try_restore(remote_part, local_part)
        manifest = None
        if restored:
            manifest = _validate_part(
                local_part,
                artifact_fingerprint=artifact_fingerprint,
                checkpoint_signature=checkpoint_signature,
                shard_index=shard_index,
                shard_count=shard_count,
                part_index=part_index,
                part_count=len(ranges),
                start_row=start_row,
                stop_row=stop_row,
            )
        if manifest is not None:
            valid_parts[part_index] = local_part
            print(
                f"Reused checkpoint shard={shard_index} "
                f"part={part_index + 1}/{len(ranges)} "
                f"rows={start_row}:{stop_row}"
            )
        else:
            if local_part.exists():
                shutil.rmtree(local_part)
            missing_parts.append(
                (part_index, start_row, stop_row, local_part, remote_part)
            )

    model = None
    if missing_parts or shard_index == 0:
        model_snapshot = base._resolve_local_model_snapshot(
            model_name,
            model_revision,
        )
        model = SentenceTransformer(
            str(model_snapshot),
            device=execution_device,
            local_files_only=True,
        )

    for (
        part_index,
        start_row,
        stop_row,
        local_part,
        remote_part,
    ) in missing_parts:
        part_rows = rows[start_row:stop_row]
        vectors = (
            vector_provider(part_rows, model, batch_size=batch_size, input_format=input_format)
            if vector_provider is not None
            else base._encode(
                texts[start_row:stop_row], model=model, input_type="passage",
                input_format=input_format, batch_size=batch_size,
            )
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(part_rows):
            raise RuntimeError("checkpoint embedding row count mismatch")
        if not np.isfinite(vectors).all():
            raise RuntimeError(
                "checkpoint embedding matrix contains non-finite values"
            )
        _write_part(
            local_part,
            rows=part_rows,
            vectors=vectors,
            artifact_fingerprint=artifact_fingerprint,
            checkpoint_signature=checkpoint_signature,
            shard_index=shard_index,
            shard_count=shard_count,
            part_index=part_index,
            part_count=len(ranges),
            start_row=start_row,
            stop_row=stop_row,
        )
        _upload_part(local_part, remote_part)
        valid_parts[part_index] = local_part
        print(
            f"Persisted checkpoint shard={shard_index} "
            f"part={part_index + 1}/{len(ranges)} "
            f"rows={start_row}:{stop_row}"
        )

    if len(valid_parts) != len(ranges):
        raise RuntimeError("not all embedding checkpoints are available")

    vectors_by_part = [
        np.load(
            valid_parts[index] / "embeddings.npy",
            allow_pickle=False,
        )
        for index in range(len(ranges))
    ]
    if not vectors_by_part:
        raise RuntimeError("embedding shard unexpectedly contains no rows")
    vectors = np.concatenate(vectors_by_part, axis=0)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise RuntimeError("assembled embedding row count mismatch")
    if not np.isfinite(vectors).all():
        raise RuntimeError("assembled embedding matrix contains non-finite values")

    output_dir = output_root / benchmark_id
    shard_dir = output_dir / "shards" / f"{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "embeddings.npy", vectors, allow_pickle=False)
    with (shard_dir / "records.jsonl").open("w", encoding="utf-8") as out:
        for part_index in range(len(ranges)):
            records_path = valid_parts[part_index] / "records.jsonl"
            out.write(records_path.read_text(encoding="utf-8"))

    query_count = 0
    query_files: list[dict[str, Any]] = []
    if shard_index == 0:
        assert model is not None
        for track in base.TRACKS:
            query_path = (
                chunks_root
                / track
                / "evaluation"
                / "queries.jsonl"
            )
            if not query_path.exists():
                continue
            queries = list(base._jsonl_rows(query_path))
            query_vectors = base._encode(
                [str(row["text"]) for row in queries],
                model=model,
                input_type="query",
                input_format=input_format,
                batch_size=batch_size,
            )
            query_dir = output_dir / "queries" / track
            query_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                query_dir / "embeddings.npy",
                query_vectors,
                allow_pickle=False,
            )
            base._jsonl_write(
                query_dir / "records.jsonl",
                (
                    {"row_index": i, "query_id": row["query_id"]}
                    for i, row in enumerate(queries)
                ),
            )
            query_count += len(queries)
            for path in (
                query_dir / "embeddings.npy",
                query_dir / "records.jsonl",
            ):
                query_files.append(
                    {
                        "path": path.relative_to(output_dir).as_posix(),
                        "sha256": base._sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    files: list[dict[str, Any]] = []
    for path in (
        shard_dir / "embeddings.npy",
        shard_dir / "records.jsonl",
    ):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": base._sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    files.extend(query_files)
    manifest = {
        "schema_version": base.SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "artifact_type": "embedding-shard",
        "artifact_fingerprint": artifact_fingerprint,
        "parent_chunks_artifact_fingerprint": parent_chunks_fingerprint,
        "repository_revision": repo_revision,
        "embedder_fingerprint": code_fingerprint,
        "python_version": python_version,
        "package_versions": packages,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "input_format": input_format,
            "document_input_type": "passage",
            "query_input_type": "query",
            "normalize_embeddings": True,
            "execution_device": execution_device,
            "output_dtype": "float32",
        },
        "shard_index": shard_index,
        "shard_count": shard_count,
        "chunk_embeddings": len(rows),
        "query_embeddings": query_count,
        "vector_dimension": int(vectors.shape[1]),
        "files": files,
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Checkpointed shard complete: shard={shard_index}, "
        f"parts={len(ranges)}, signature={checkpoint_signature}"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".benchmark/embeddings"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(".benchmark/embedding-checkpoints"),
    )
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument(
        "--parent-chunks-fingerprint",
        required=True,
    )
    parser.add_argument("--artifact-fingerprint", required=True)
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-base",
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--input-format",
        choices=("e5", "raw"),
        default="e5",
    )
    parser.add_argument(
        "--execution-device",
        choices=("cpu",),
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=DEFAULT_CHECKPOINT_SIZE,
    )
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()

    try:
        output = build_checkpointed_shard(
            args.chunks_root,
            args.output_root,
            args.checkpoint_root,
            benchmark_id=args.benchmark_id,
            parent_chunks_fingerprint=args.parent_chunks_fingerprint,
            artifact_fingerprint=args.artifact_fingerprint,
            model_name=args.model_name,
            model_revision=args.model_revision,
            input_format=args.input_format,
            execution_device=args.execution_device,
            batch_size=args.batch_size,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            checkpoint_size=args.checkpoint_size,
            remote_root=args.remote_root,
            repo_revision=os.environ.get("GITHUB_SHA", "local"),
        )
    except Exception as exc:
        import traceback

        print(
            f"Checkpointed benchmark embedding failed: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1
    print(f"Checkpointed embedding shard output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

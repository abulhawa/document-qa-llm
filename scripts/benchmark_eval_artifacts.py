"""Artifact I/O helpers for the Benchmark 6 GitHub Actions workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
BUCKET = "hf://buckets/abulhawa/document-qa-artifacts"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy(source: str, destination: Path | str) -> None:
    if isinstance(destination, Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = str(destination)
    else:
        target = destination
    subprocess.run(["hf", "buckets", "cp", source, target], check=True)


def _resolve_pointer(base: str, requested: str, pointer_path: str, workdir: Path) -> str:
    if requested != "current":
        return requested
    local = workdir / "pointer.json"
    _copy(f"{base}/{pointer_path}", local)
    return str(_json(local)["artifact_fingerprint"])


def _append_env(path: Path | None, values: Mapping[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def prepare(args: argparse.Namespace) -> None:
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    base = f"{BUCKET}/{args.benchmark_id}"

    index_fp = _resolve_pointer(
        base,
        args.index_artifact_fingerprint,
        "indexes/current.json",
        workdir / "index-pointer",
    )
    index_manifest_path = workdir / "index-manifest.json"
    _copy(f"{base}/indexes/artifacts/{index_fp}/manifest.json", index_manifest_path)
    combined = _json(index_manifest_path)

    if combined.get("artifact_type") != "native-indexes":
        raise ValueError("selected Benchmark 5 artifact is not native-indexes")
    if combined.get("artifact_fingerprint") != index_fp:
        raise ValueError("Benchmark 5 artifact fingerprint mismatch")
    if combined.get("benchmark_id") != args.benchmark_id:
        raise ValueError("Benchmark 5 benchmark_id mismatch")
    if not combined.get("persistence", {}).get("persisted"):
        raise ValueError("Benchmark 5 artifact is not marked persisted")

    engines = combined.get("engines", {})
    os_meta = engines.get("opensearch", {})
    qd_meta = engines.get("qdrant", {})
    os_backend = os_meta.get("backend", {})
    qd_backend = qd_meta.get("backend", {})
    if str(os_backend.get("version")) != args.opensearch_version:
        raise ValueError("OpenSearch snapshot version does not match Benchmark 6")
    if str(qd_backend.get("version")) != args.qdrant_version:
        raise ValueError("Qdrant snapshot version does not match Benchmark 6")
    if os_backend.get("index_name") != args.opensearch_index:
        raise ValueError("OpenSearch index name does not match Benchmark 6 runtime")
    if qd_backend.get("collection_name") != args.qdrant_collection:
        raise ValueError("Qdrant collection name does not match Benchmark 6 runtime")

    chunks_fp = str(combined["parent_chunks_artifact_fingerprint"])
    embeddings_fp = str(combined["parent_embeddings_artifact_fingerprint"])
    chunks_root = workdir / "chunks"
    embeddings_root = workdir / "embeddings"
    _copy(f"{base}/chunks/{chunks_fp}/manifest.json", chunks_root / "manifest.json")
    _copy(
        f"{base}/embeddings/{embeddings_fp}/manifest.json",
        embeddings_root / "manifest.json",
    )
    for track in TRACKS:
        _copy(
            f"{base}/chunks/{chunks_fp}/{track}/documents.jsonl",
            chunks_root / track / "documents.jsonl",
        )
        for name in ("queries.jsonl", "qrels.jsonl"):
            _copy(
                f"{base}/chunks/{chunks_fp}/{track}/evaluation/{name}",
                chunks_root / track / "evaluation" / name,
            )
        for name in ("embeddings.npy", "records.jsonl"):
            _copy(
                f"{base}/embeddings/{embeddings_fp}/queries/{track}/{name}",
                embeddings_root / "queries" / track / name,
            )

    native_root = workdir / "native"
    os_root = native_root / "opensearch"
    qd_root = native_root / "qdrant"
    _copy(f"{os_meta['artifact_uri']}/manifest.json", os_root / "manifest.json")
    _copy(f"{os_meta['artifact_uri']}/snapshot.tar.gz", os_root / "snapshot.tar.gz")
    _copy(f"{qd_meta['artifact_uri']}/manifest.json", qd_root / "manifest.json")
    _copy(f"{qd_meta['artifact_uri']}/snapshot.snapshot", qd_root / "snapshot.snapshot")

    os_manifest = _json(os_root / "manifest.json")
    qd_manifest = _json(qd_root / "manifest.json")
    for engine, manifest, expected in (
        ("opensearch", os_manifest, os_meta),
        ("qdrant", qd_manifest, qd_meta),
    ):
        if manifest.get("engine_fingerprint") != expected.get("engine_fingerprint"):
            raise ValueError(f"{engine} engine fingerprint mismatch")
        if (
            manifest.get("parent_chunks_artifact_fingerprint")
            != combined.get("parent_chunks_artifact_fingerprint")
        ):
            raise ValueError(f"{engine} parent chunks mismatch")
        if manifest.get("benchmark_id") != args.benchmark_id:
            raise ValueError(f"{engine} benchmark_id mismatch")
    if (
        qd_manifest.get("parent_embeddings_artifact_fingerprint")
        != combined.get("parent_embeddings_artifact_fingerprint")
    ):
        raise ValueError("Qdrant parent embeddings mismatch")

    baseline_path = ""
    baseline_fp = ""
    if args.baseline:
        eval_base = f"{base}/evaluations"
        if args.baseline == "current":
            pointer = workdir / "baseline-pointer.json"
            _copy(f"{eval_base}/current/{args.tier}.json", pointer)
            baseline_fp = str(_json(pointer)["artifact_fingerprint"])
        else:
            baseline_fp = args.baseline
        local_baseline = workdir / "baseline-summary.json"
        _copy(f"{eval_base}/{baseline_fp}/summary.json", local_baseline)
        baseline_path = str(local_baseline)

    env = {
        "BENCH_WORKDIR": str(workdir),
        "BENCH_INDEX_FP": index_fp,
        "BENCH_CHUNKS_FP": chunks_fp,
        "BENCH_EMBEDDINGS_FP": embeddings_fp,
        "BENCH_OPENSEARCH_FP": os_meta["engine_fingerprint"],
        "BENCH_QDRANT_FP": qd_meta["engine_fingerprint"],
        "BENCH_BASELINE_PATH": baseline_path,
        "BENCH_BASELINE_FP": baseline_fp,
    }
    _append_env(args.github_env, env)
    _write_json(workdir / "context.json", env)
    print(json.dumps(env, sort_keys=True))


def persist(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    manifest = _json(manifest_path)
    uri = str(manifest["persistence"]["artifact_uri"])
    _copy(str(args.output_dir / "results.jsonl"), f"{uri}/results.jsonl")
    _copy(str(args.output_dir / "summary.json"), f"{uri}/summary.json")

    manifest["persistence"]["persisted"] = True
    _write_json(manifest_path, manifest)
    _copy(str(manifest_path), f"{uri}/manifest.json")

    pointer = {
        "schema_version": 1,
        "benchmark_id": manifest["benchmark_id"],
        "artifact_type": manifest["artifact_type"],
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "evaluation_signature": manifest["evaluation_signature"],
        "artifact_uri": uri,
        "tier": manifest["tier"],
        "run_id": manifest["run_id"],
    }
    pointer_path = args.output_dir / "pointer.json"
    _write_json(pointer_path, pointer)
    _copy(
        str(pointer_path),
        f"{BUCKET}/{manifest['benchmark_id']}/evaluations/current/{manifest['tier']}.json",
    )
    print(json.dumps(pointer, sort_keys=True))


def write_summary(args: argparse.Namespace) -> None:
    payload = _json(args.summary_json)
    summary = payload["summary"]
    composite = summary["composite"]
    lines = [
        "## Benchmark 6 - Retrieval evaluation",
        f"- Benchmark: `{payload['benchmark_id']}`",
        f"- Tier: `{payload['tier']}`",
        f"- Evaluation artifact: `{payload['artifact_fingerprint']}`",
        f"- Index artifact: `{payload['lineage']['index_artifact_fingerprint']}`",
        f"- Queries: {composite['queries']} ({composite['errors']} errors)",
        "",
        "| Track | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track, row in summary["per_track"].items():
        lines.append(
            f"| {track} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg_at_5']:.4f} | {row['latency_ms']['p95']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Composite macro: Recall@1 **{composite['recall_at_1']:.4f}**, "
            f"Recall@3 **{composite['recall_at_3']:.4f}**, "
            f"Recall@5 **{composite['recall_at_5']:.4f}**, "
            f"MRR **{composite['mrr']:.4f}**, "
            f"nDCG@5 **{composite['ndcg_at_5']:.4f}**.",
        ]
    )
    comparison = payload.get("baseline_comparison")
    if comparison:
        delta = comparison.get("metric_deltas", {})
        lines.extend(
            [
                "",
                f"Baseline: `{comparison.get('baseline_artifact_fingerprint')}`",
                f"Delta Recall@5 **{delta.get('recall_at_5', 0.0):+.4f}**, "
                f"MRR **{delta.get('mrr', 0.0):+.4f}**, "
                f"nDCG@5 **{delta.get('ndcg_at_5', 0.0):+.4f}**.",
            ]
        )
    with args.output.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if args.fail_on_errors and int(composite["errors"]):
        raise SystemExit(
            f"Benchmark 6 completed with {composite['errors']} retrieval errors; "
            "results were persisted for diagnosis"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--benchmark-id", required=True)
    prep.add_argument("--index-artifact-fingerprint", required=True)
    prep.add_argument("--tier", choices=("smoke", "medium", "full"), required=True)
    prep.add_argument("--baseline", default="")
    prep.add_argument("--workdir", type=Path, required=True)
    prep.add_argument("--opensearch-version", required=True)
    prep.add_argument("--qdrant-version", required=True)
    prep.add_argument("--opensearch-index", required=True)
    prep.add_argument("--qdrant-collection", required=True)
    prep.add_argument("--github-env", type=Path)

    save = subparsers.add_parser("persist")
    save.add_argument("--output-dir", type=Path, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--summary-json", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--fail-on-errors", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "persist":
        persist(args)
    else:
        write_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

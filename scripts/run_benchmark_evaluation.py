"""Run deterministic retrieval evaluation against restored Benchmark 5 indexes.

Benchmark 6 is retrieval-only. It reuses Benchmark 4 query embeddings and the
production hybrid retrieval pipeline while disabling stages that would require
an LLM, new embeddings, reranking, or answer generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
SCHEMA_VERSION = 1
BENCHMARK_SPECS = {
    "composite-v1": Path("evaluation/benchmarks/composite_v1.yaml"),
    "composite-v2": Path("evaluation/benchmarks/composite_v2.yaml"),
}
DEFAULT_TOP_K = 5
TIER_SELECTION_ALGORITHM = "tier-sha256-v1"

def _default_spec_for_benchmark(benchmark_id: str) -> Path:
    try:
        return BENCHMARK_SPECS[benchmark_id]
    except KeyError as exc:
        raise ValueError(
            f"no default benchmark spec registered for {benchmark_id!r}; pass --spec explicitly"
        ) from exc



def _install_noop_tracing() -> None:
    """Keep Benchmark 6 independent of the optional Phoenix tracing stack."""

    if "tracing" in sys.modules:
        return

    class _NoopSpan:
        def set_attribute(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_status(self, *args: Any, **kwargs: Any) -> None:
            return None

        def record_exception(self, *args: Any, **kwargs: Any) -> None:
            return None

    @contextmanager
    def _start_span(*args: Any, **kwargs: Any):
        yield _NoopSpan()

    module = types.ModuleType("tracing")
    module.start_span = _start_span
    module.record_span_error = lambda *args, **kwargs: None
    module.CHAIN = "CHAIN"
    module.LLM = "LLM"
    module.RETRIEVER = "RETRIEVER"
    module.EMBEDDING = "EMBEDDING"
    module.TOOL = "TOOL"
    module.INPUT_VALUE = "input.value"
    module.OUTPUT_VALUE = "output.value"
    module.STATUS_OK = "OK"
    module.get_current_span = lambda: _NoopSpan()
    sys.modules["tracing"] = module


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_query_key(seed: int, track: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{track}:{query_id}".encode("utf-8")).hexdigest()


def select_tier_queries(
    queries_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Derive deterministic nested tiers from the already-frozen full query set."""

    selected: dict[str, list[dict[str, Any]]] = {}
    for track in TRACKS:
        rows = [dict(row) for row in queries_by_track.get(track, [])]
        quota = int(quotas.get(track, 0))
        if quota < 0 or quota > len(rows):
            raise ValueError(
                f"invalid tier quota for {track}: {quota}; available={len(rows)}"
            )
        rows.sort(
            key=lambda row: (
                _stable_query_key(seed, track, str(row.get("query_id") or "")),
                str(row.get("query_id") or ""),
            )
        )
        selected[track] = rows[:quota]
    return selected


def _discounted_gain(relevance: float, rank: int) -> float:
    if relevance <= 0:
        return 0.0
    return (2.0**relevance - 1.0) / math.log2(rank + 1.0)


def query_metrics(
    retrieved_document_ids: Sequence[str],
    qrels: Mapping[str, float],
) -> dict[str, float]:
    """Compute document-level Hit/Recall, MRR, and graded nDCG for one query."""

    relevant = {doc_id for doc_id, score in qrels.items() if float(score) > 0.0}
    if not relevant:
        raise ValueError("query has no positive relevance judgments")

    retrieved: list[str] = []
    seen: set[str] = set()
    for doc_id in retrieved_document_ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            retrieved.append(doc_id)

    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        relevant_hits = len(set(retrieved[:k]) & relevant)
        metrics[f"hit_at_{k}"] = 1.0 if relevant_hits else 0.0
        metrics[f"recall_at_{k}"] = relevant_hits / len(relevant)

    metrics["mrr"] = next(
        (1.0 / rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant),
        0.0,
    )
    dcg = sum(
        _discounted_gain(float(qrels.get(doc_id, 0.0)), rank)
        for rank, doc_id in enumerate(retrieved[:5], 1)
    )
    ideal = sorted(
        (float(score) for score in qrels.values() if float(score) > 0.0),
        reverse=True,
    )[:5]
    idcg = sum(_discounted_gain(score, rank) for rank, score in enumerate(ideal, 1))
    metrics["ndcg_at_5"] = dcg / idcg if idcg > 0 else 0.0
    return metrics


def document_id_from_hit(hit: Mapping[str, Any], *, benchmark_id: str) -> str:
    path = str(hit.get("path") or "")
    if path.startswith("benchmark://"):
        # Incremental indexes retain the original logical source for reused rows.
        parts = path[len("benchmark://") :].split("/", 2)
        if len(parts) == 3 and parts[0] and parts[1] in TRACKS and parts[2]:
            return parts[2]
    return str(hit.get("checksum") or "").strip()


def document_identity_from_hit(hit: Mapping[str, Any], *, benchmark_id: str) -> str:
    """Use the content identity that the production retriever deduplicates on."""

    checksum = str(hit.get("checksum") or "").strip()
    if checksum:
        return checksum
    return document_id_from_hit(hit, benchmark_id=benchmark_id)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    )
    by_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[str(row["track"])].append(row)

    per_track: dict[str, Any] = {}
    for track in TRACKS:
        track_rows = by_track.get(track, [])
        successful = [row for row in track_rows if not row.get("error")]
        latencies = [float(row.get("latency_ms") or 0.0) for row in track_rows]
        averaged = {
            name: round(
                sum(float(row.get(name) or 0.0) for row in track_rows) / len(track_rows),
                6,
            )
            if track_rows
            else 0.0
            for name in metric_names
        }
        per_track[track] = {
            "queries": len(track_rows),
            "successful_queries": len(successful),
            "errors": len(track_rows) - len(successful),
            **averaged,
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "p50": round(_percentile(latencies, 50.0), 3),
                "p95": round(_percentile(latencies, 95.0), 3),
            },
        }

    active = [track for track in TRACKS if per_track[track]["queries"]]
    composite = {
        name: round(
            sum(float(per_track[track][name]) for track in active) / len(active),
            6,
        )
        if active
        else 0.0
        for name in metric_names
    }
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    successful_count = sum(1 for row in rows if not row.get("error"))
    composite.update(
        {
            "aggregation": "macro_average_across_tracks",
            "queries": len(rows),
            "successful_queries": successful_count,
            "errors": len(rows) - successful_count,
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "p50": round(_percentile(latencies, 50.0), 3),
                "p95": round(_percentile(latencies, 95.0), 3),
            },
        }
    )
    return {"per_track": per_track, "composite": composite}


def _load_spec(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark spec must be a mapping")
    return dict(value)


def _validate_lineage(
    *,
    benchmark_id: str,
    chunks_manifest: Mapping[str, Any],
    embeddings_manifest: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    chunks_fingerprint: str,
    embeddings_fingerprint: str,
    index_fingerprint: str,
) -> None:
    if (
        chunks_manifest.get("artifact_type") != "chunks"
        or chunks_manifest.get("benchmark_id") != benchmark_id
        or chunks_manifest.get("artifact_fingerprint") != chunks_fingerprint
        or not chunks_manifest.get("persistence", {}).get("persisted")
    ):
        raise ValueError("chunks artifact lineage mismatch")
    if (
        embeddings_manifest.get("artifact_type") != "embeddings"
        or embeddings_manifest.get("benchmark_id") != benchmark_id
        or embeddings_manifest.get("artifact_fingerprint") != embeddings_fingerprint
        or embeddings_manifest.get("parent_chunks_artifact_fingerprint")
        != chunks_fingerprint
        or not embeddings_manifest.get("persistence", {}).get("persisted")
    ):
        raise ValueError("embeddings artifact lineage mismatch")
    if (
        index_manifest.get("artifact_type") != "native-indexes"
        or index_manifest.get("benchmark_id") != benchmark_id
        or index_manifest.get("artifact_fingerprint") != index_fingerprint
        or index_manifest.get("parent_chunks_artifact_fingerprint")
        != chunks_fingerprint
        or index_manifest.get("parent_embeddings_artifact_fingerprint")
        != embeddings_fingerprint
        or not index_manifest.get("persistence", {}).get("persisted")
    ):
        raise ValueError("native index artifact lineage mismatch")


def _load_query_vectors(
    embeddings_root: Path,
    selected_query_ids: Mapping[str, set[str]],
) -> dict[tuple[str, str], Any]:
    import numpy as np

    vectors: dict[tuple[str, str], Any] = {}
    for track in TRACKS:
        wanted = selected_query_ids.get(track, set())
        if not wanted:
            continue
        root = embeddings_root / "queries" / track
        matrix = np.load(root / "embeddings.npy", allow_pickle=False, mmap_mode="r")
        records = _jsonl_rows(root / "records.jsonl")
        if matrix.ndim != 2 or matrix.shape[0] != len(records):
            raise ValueError(f"query embedding row mismatch for {track}")
        for record in records:
            query_id = str(record["query_id"])
            if query_id in wanted:
                vector = matrix[int(record["row_index"])]
                if not np.isfinite(vector).all():
                    raise ValueError(f"non-finite query embedding: {track}/{query_id}")
                vectors[(track, query_id)] = vector
        missing = sorted(
            wanted - {query_id for current_track, query_id in vectors if current_track == track}
        )
        if missing:
            raise ValueError(f"missing query embeddings for {track}: {missing[:5]}")
    return vectors


def _load_document_identities(chunks_root: Path) -> dict[tuple[str, str], str]:
    identities: dict[tuple[str, str], str] = {}
    for track in TRACKS:
        for row in _jsonl_rows(chunks_root / track / "documents.jsonl"):
            document_id = str(row["source_document_id"])
            identity = str(row.get("source_sha256") or document_id).strip()
            if not identity:
                raise ValueError(f"missing document identity: {track}/{document_id}")
            identities[(track, document_id)] = identity
    return identities


def _load_qrels(
    chunks_root: Path,
    document_identities: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], dict[str, float]]:
    qrels: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for track in TRACKS:
        path = chunks_root / track / "evaluation" / "qrels.jsonl"
        for row in _jsonl_rows(path):
            query_id = str(row["query_id"])
            document_id = str(row["document_id"])
            score = float(row.get("score", 0.0))
            identity = document_identities.get((track, document_id))
            if identity is None:
                if score <= 0.0:
                    continue
                raise ValueError(
                    f"positive qrel references unknown document: "
                    f"{track}/{query_id}/{document_id}"
                )
            current = qrels[(track, query_id)].get(identity)
            if current is None or score > current:
                qrels[(track, query_id)][identity] = score
    return dict(qrels)


def _make_deps(query_text: str, query_vector: Any):
    import core.vector_store as vector_store

    normalized_query_text = query_text.strip()
    from core.opensearch_store import search as keyword_retriever
    from core.retrieval.types import RetrievalDeps

    def semantic_retriever(request_query: str, top_k: int):
        if request_query != normalized_query_text:
            raise RuntimeError(
                "Benchmark 6 deterministic profile only supports the exact query"
            )
        original_embed = vector_store.embed_texts

        def fixed_embed(
            texts: list[str],
            *args: Any,
            **kwargs: Any,
        ) -> list[list[float]]:
            if (
                kwargs.get("input_type", "query") != "query"
                or len(texts) != 1
                or texts[0] != normalized_query_text
            ):
                raise RuntimeError("unexpected embedding request during Benchmark 6")
            return [query_vector.tolist()]

        vector_store.embed_texts = fixed_embed
        try:
            hits = vector_store.retrieve_top_k(request_query, top_k)
        finally:
            vector_store.embed_texts = original_embed
        if hits and any("status" in hit and not hit.get("id") for hit in hits):
            raise RuntimeError(f"semantic retriever returned an error payload: {hits[:1]}")
        return hits

    return RetrievalDeps(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        embed_texts=None,
        cross_encoder=None,
        sibling_chunk_fetcher=None,
    )


def _retrieval_config(top_k: int):
    from core.retrieval.types import RetrievalConfig

    return RetrievalConfig(
        top_k=top_k,
        top_k_each=max(20, top_k * 4),
        enable_variants=False,
        enable_query_planning=False,
        enable_hyde=False,
        enable_mmr=False,
        enable_rerank=False,
        sibling_expansion_enabled=False,
        abstention_enabled=False,
        sim_threshold=0.0,
    )


def _config_payload(cfg: Any) -> dict[str, Any]:
    return {
        "profile": "deterministic-core-retrieval-v1",
        "relevance_identity": "source_sha256_with_document_id_fallback",
        "top_k": int(cfg.top_k),
        "top_k_each": int(cfg.top_k_each),
        "enable_variants": bool(cfg.enable_variants),
        "enable_query_planning": bool(cfg.enable_query_planning),
        "enable_hyde": bool(cfg.enable_hyde),
        "enable_mmr": bool(cfg.enable_mmr),
        "enable_rerank": bool(cfg.enable_rerank),
        "sibling_expansion_enabled": bool(cfg.sibling_expansion_enabled),
        "abstention_enabled": bool(cfg.abstention_enabled),
        "fusion_weight_vector": float(cfg.fusion_weight_vector),
        "fusion_weight_bm25": float(cfg.fusion_weight_bm25),
        "anchored_lexical_bias_enabled": bool(cfg.anchored_lexical_bias_enabled),
        "anchored_fusion_weight_vector": float(cfg.anchored_fusion_weight_vector),
        "anchored_fusion_weight_bm25": float(cfg.anchored_fusion_weight_bm25),
    }


def _zero_metrics() -> dict[str, float]:
    return {
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
        "ndcg_at_5": 0.0,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _install_noop_tracing()
    from core.retrieval.pipeline import retrieve

    spec = _load_spec(args.spec)
    if str(spec.get("benchmark_id")) != args.benchmark_id:
        raise ValueError("benchmark spec ID mismatch")
    if args.tier not in spec.get("tiers", {}):
        raise ValueError(f"unknown evaluation tier: {args.tier}")

    chunks_manifest = _json(args.chunks_root / "manifest.json")
    embeddings_manifest = _json(args.embeddings_root / "manifest.json")
    index_manifest = _json(args.index_manifest)
    _validate_lineage(
        benchmark_id=args.benchmark_id,
        chunks_manifest=chunks_manifest,
        embeddings_manifest=embeddings_manifest,
        index_manifest=index_manifest,
        chunks_fingerprint=args.chunks_fingerprint,
        embeddings_fingerprint=args.embeddings_fingerprint,
        index_fingerprint=args.index_fingerprint,
    )

    queries_by_track = {
        track: _jsonl_rows(args.chunks_root / track / "evaluation" / "queries.jsonl")
        for track in TRACKS
    }
    tier = dict(spec["tiers"][args.tier]["queries"])
    seed = int(spec.get("selection_seed", 0))
    selected = select_tier_queries(queries_by_track, quotas=tier, seed=seed)
    expected_total = int(spec["tiers"][args.tier]["total_queries"])
    if sum(len(rows) for rows in selected.values()) != expected_total:
        raise ValueError("tier query count does not match benchmark specification")

    selected_ids = {
        track: {str(row["query_id"]) for row in rows}
        for track, rows in selected.items()
    }
    vectors = _load_query_vectors(args.embeddings_root, selected_ids)
    document_identities = _load_document_identities(args.chunks_root)
    qrels = _load_qrels(args.chunks_root, document_identities)
    cfg = _retrieval_config(args.top_k)

    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        for query in selected[track]:
            query_id = str(query["query_id"])
            query_text = str(query["text"])
            relevance = qrels.get((track, query_id), {})
            if not any(score > 0 for score in relevance.values()):
                raise ValueError(f"query has no positive qrels: {track}/{query_id}")

            started = time.perf_counter()
            error: str | None = None
            hits: list[Mapping[str, Any]] = []
            try:
                output = retrieve(
                    query_text,
                    cfg=cfg,
                    deps=_make_deps(query_text, vectors[(track, query_id)]),
                    query_plan=None,
                )
                hits = list(output.documents)
            except Exception as exc:  # noqa: BLE001
                error = f"{exc.__class__.__name__}: {exc}"
                print(
                    f"[benchmark6] retrieval error track={track} query_id={query_id}: {error}",
                    flush=True,
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            retrieved_ids = [
                document_id_from_hit(hit, benchmark_id=args.benchmark_id)
                for hit in hits
            ]
            retrieved_identities = [
                document_identity_from_hit(hit, benchmark_id=args.benchmark_id)
                for hit in hits
            ]
            rows.append(
                {
                    "track": track,
                    "query_id": query_id,
                    "retrieved_document_ids": retrieved_ids[: args.top_k],
                    "retrieved_document_identities": retrieved_identities[: args.top_k],
                    "latency_ms": round(latency_ms, 3),
                    "error": error,
                    **(
                        query_metrics(retrieved_identities, relevance)
                        if error is None
                        else _zero_metrics()
                    ),
                }
            )

    summary = summarize_results(rows)
    selected_query_ids = {
        track: [str(row["query_id"]) for row in selected[track]] for track in TRACKS
    }
    evaluation_config = _config_payload(cfg)
    functional_identity = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": args.benchmark_id,
        "index_artifact_fingerprint": args.index_fingerprint,
        "chunks_artifact_fingerprint": args.chunks_fingerprint,
        "embeddings_artifact_fingerprint": args.embeddings_fingerprint,
        "tier": args.tier,
        "tier_selection_algorithm": TIER_SELECTION_ALGORITHM,
        "selection_seed": seed,
        "selected_query_ids": selected_query_ids,
        "evaluation_config": evaluation_config,
        "repository_revision": args.repository_revision,
    }
    evaluation_signature = _canonical_sha256(functional_identity)
    artifact_fingerprint = _canonical_sha256(
        {"evaluation_signature": evaluation_signature, "run_id": args.run_id}
    )
    artifact_uri = f"{args.remote_base.rstrip('/')}/{artifact_fingerprint}"
    created_at = datetime.now(timezone.utc).isoformat()

    result_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": args.benchmark_id,
        "artifact_type": "retrieval-evaluation",
        "artifact_fingerprint": artifact_fingerprint,
        "evaluation_signature": evaluation_signature,
        "run_id": args.run_id,
        "tier": args.tier,
        "lineage": {
            "index_artifact_fingerprint": args.index_fingerprint,
            "chunks_artifact_fingerprint": args.chunks_fingerprint,
            "embeddings_artifact_fingerprint": args.embeddings_fingerprint,
        },
        "repository_revision": args.repository_revision,
        "created_at": created_at,
        "evaluation_config": evaluation_config,
        "selection": {
            "algorithm": TIER_SELECTION_ALGORITHM,
            "seed": seed,
            "query_ids": selected_query_ids,
        },
        "summary": summary,
    }

    if args.baseline_summary and args.baseline_summary.exists():
        baseline = _json(args.baseline_summary)
        if baseline.get("benchmark_id") != args.benchmark_id:
            raise ValueError("baseline benchmark_id mismatch")
        if baseline.get("tier") != args.tier:
            raise ValueError("baseline tier mismatch")
        if baseline.get("selection", {}).get("query_ids") != selected_query_ids:
            raise ValueError("baseline selected query IDs do not match this run")
        base_metrics = baseline.get("summary", {}).get("composite", {})
        current_metrics = summary["composite"]
        result_payload["baseline_comparison"] = {
            "baseline_artifact_fingerprint": baseline.get("artifact_fingerprint"),
            "metric_deltas": {
                name: round(
                    float(current_metrics.get(name, 0.0))
                    - float(base_metrics.get(name, 0.0)),
                    6,
                )
                for name in (
                    "hit_at_1",
                    "hit_at_3",
                    "hit_at_5",
                    "recall_at_1",
                    "recall_at_3",
                    "recall_at_5",
                    "mrr",
                    "ndcg_at_5",
                )
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "results.jsonl", rows)
    _write_json(args.output_dir / "summary.json", result_payload)
    manifest = {
        **functional_identity,
        "artifact_type": "retrieval-evaluation",
        "artifact_fingerprint": artifact_fingerprint,
        "evaluation_signature": evaluation_signature,
        "run_id": args.run_id,
        "created_at": created_at,
        "files": [
            {
                "path": name,
                "sha256": _sha256_file(args.output_dir / name),
                "bytes": (args.output_dir / name).stat().st_size,
            }
            for name in ("results.jsonl", "summary.json")
        ],
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": False,
            "artifact_uri": artifact_uri,
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument("--tier", choices=("smoke", "medium", "full"), default="smoke")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--chunks-fingerprint", required=True)
    parser.add_argument("--embeddings-fingerprint", required=True)
    parser.add_argument("--index-fingerprint", required=True)
    parser.add_argument(
        "--repository-revision",
        default=os.environ.get("GITHUB_SHA", "local"),
    )
    parser.add_argument(
        "--run-id",
        default=(
            f"{os.environ.get('GITHUB_RUN_ID', 'local')}-"
            f"{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    args = parser.parse_args()
    if args.spec is None:
        args.spec = _default_spec_for_benchmark(args.benchmark_id)
    if args.top_k < 5:
        parser.error("--top-k must be at least 5 to calculate @5 metrics")
    manifest = run_evaluation(args)
    print(
        json.dumps(
            {
                "artifact_fingerprint": manifest["artifact_fingerprint"],
                "evaluation_signature": manifest["evaluation_signature"],
                "artifact_uri": manifest["persistence"]["artifact_uri"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

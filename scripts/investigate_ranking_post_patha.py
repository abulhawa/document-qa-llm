#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Ensure repo root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FULLTEXT_INDEX
from core.opensearch_client import get_client
from core.retrieval.pipeline import retrieve
from core.retrieval.types import RetrievalConfig


TOKEN_RE = re.compile(r"[a-z0-9]+")

RANK_BUCKET_1 = "rank_1"
RANK_BUCKET_2_3 = "rank_2_to_3"
RANK_BUCKET_4_10 = "rank_4_to_10"
RANK_BUCKET_11_TO_DEPTH = "rank_11_to_depth"
RANK_BUCKET_NOT_RETRIEVED = "not_retrieved_within_depth"

LIKELY_CORRECT = "likely_correct_despite_rank_miss"
POSSIBLY_CORRECT_TOP3 = "possibly_correct_with_top3_context"
POSSIBLY_CORRECT_SIMILAR = "possibly_correct_due_to_high_similarity"
UNLIKELY_OR_UNKNOWN = "unlikely_or_unknown"

DEFAULT_PROBE_DEPTH = 60
EQUIV_SUPPORT_CONTAINMENT_MIN = 0.82
EQUIV_SUPPORT_SEQUENCE_MIN = 0.72
EQUIV_SUPPORT_ALT_CONTAINMENT_MIN = 0.75
EQUIV_SUPPORT_ALT_JACCARD_MIN = 0.50
EQUIV_SUPPORT_ALT_SEQUENCE_MIN = 0.65
DEFAULT_SUPPORT_LABELS_PATH = Path("tests/fixtures/retrieval_eval_answer_support_labels.json")


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def rank_bucket(rank: int | None, depth: int) -> str:
    if rank is None or rank > depth:
        return RANK_BUCKET_NOT_RETRIEVED
    if rank == 1:
        return RANK_BUCKET_1
    if rank <= 3:
        return RANK_BUCKET_2_3
    if rank <= 10:
        return RANK_BUCKET_4_10
    return RANK_BUCKET_11_TO_DEPTH


def text_similarity_metrics(text_a: str, text_b: str) -> dict[str, float | bool]:
    sample_a = _sample_text(text_a)
    sample_b = _sample_text(text_b)
    tokens_a = set(TOKEN_RE.findall(sample_a.lower()))
    tokens_b = set(TOKEN_RE.findall(sample_b.lower()))

    jaccard = 0.0
    containment = 0.0
    if tokens_a and tokens_b:
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = inter / union if union > 0 else 0.0
        containment = inter / min(len(tokens_a), len(tokens_b))

    seq_ratio = SequenceMatcher(None, sample_a, sample_b).ratio() if (sample_a and sample_b) else 0.0
    near_duplicate = (
        containment >= 0.95
        or seq_ratio >= 0.95
        or (containment >= 0.90 and jaccard >= 0.85 and seq_ratio >= 0.85)
    )
    return {
        "jaccard": round(jaccard, 4),
        "containment_min": round(containment, 4),
        "sequence_ratio": round(seq_ratio, 4),
        "near_duplicate": near_duplicate,
    }


def auto_answer_likelihood(
    *,
    hit_at_3: bool,
    expected_rank: int | None,
    near_duplicate: bool,
    similarity_containment: float,
) -> str:
    if near_duplicate:
        return LIKELY_CORRECT
    if hit_at_3 and expected_rank is not None and expected_rank <= 3:
        return POSSIBLY_CORRECT_TOP3
    if expected_rank is not None and expected_rank <= 10 and similarity_containment >= 0.80:
        return POSSIBLY_CORRECT_SIMILAR
    return UNLIKELY_OR_UNKNOWN


def is_equivalent_answer_support(similarity: Mapping[str, Any] | None) -> bool:
    if not similarity:
        return False
    if bool(similarity.get("near_duplicate")):
        return True
    containment = float(similarity.get("containment_min", 0.0) or 0.0)
    sequence_ratio = float(similarity.get("sequence_ratio", 0.0) or 0.0)
    jaccard = float(similarity.get("jaccard", 0.0) or 0.0)
    if containment >= EQUIV_SUPPORT_CONTAINMENT_MIN and sequence_ratio >= EQUIV_SUPPORT_SEQUENCE_MIN:
        return True
    if (
        containment >= EQUIV_SUPPORT_ALT_CONTAINMENT_MIN
        and jaccard >= EQUIV_SUPPORT_ALT_JACCARD_MIN
        and sequence_ratio >= EQUIV_SUPPORT_ALT_SEQUENCE_MIN
    ):
        return True
    return False


def _sample_text(text: str, max_chars: int = 20000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _dedupe_str_checksums(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        checksum = value.strip()
        if not checksum or checksum in seen:
            continue
        seen.add(checksum)
        out.append(checksum)
    return out


def _load_support_label_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = _load_json(path)
    raw_overrides = data.get("overrides", {})
    if not isinstance(raw_overrides, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for query_id, override in raw_overrides.items():
        if not isinstance(query_id, str) or not isinstance(override, dict):
            continue
        out[query_id] = override
    return out


def resolve_support_expectations(
    *,
    strict_expected_checksums: Sequence[str],
    label_override: Mapping[str, Any] | None,
) -> tuple[list[str], str | None]:
    strict = _dedupe_str_checksums(strict_expected_checksums)
    if not label_override:
        preferred = strict[0] if strict else None
        return strict, preferred

    raw_mode = str(label_override.get("answer_support_mode") or "merge").strip().lower()
    mode = raw_mode if raw_mode in {"merge", "replace"} else "merge"
    extra_checksums = _dedupe_str_checksums(
        label_override.get("answer_support_checksums") or []
    )
    if mode == "replace":
        support = extra_checksums
    else:
        support = _dedupe_str_checksums([*strict, *extra_checksums])

    preferred_override = label_override.get("preferred_checksum")
    preferred_checksum: str | None = None
    if isinstance(preferred_override, str) and preferred_override.strip():
        preferred_checksum = preferred_override.strip()
    if preferred_checksum is None:
        preferred_checksum = strict[0] if strict else None
    return support, preferred_checksum


def _fetch_fulltext_by_checksum(
    os_client: Any,
    checksum: str,
) -> dict[str, Any] | None:
    try:
        response = os_client.get(index=FULLTEXT_INDEX, id=checksum)
        source = response.get("_source", {}) or {}
        source["id"] = response.get("_id") or checksum
        return source
    except Exception:
        pass

    try:
        response = os_client.search(
            index=FULLTEXT_INDEX,
            body={
                "size": 1,
                "query": {"term": {"checksum": {"value": checksum}}},
            },
        )
    except Exception:
        return None
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    hit = hits[0]
    source = hit.get("_source", {}) or {}
    source["id"] = hit.get("_id") or checksum
    return source


def _build_probe_cfg(patha_cfg: Mapping[str, Any], probe_depth: int) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=max(probe_depth, 1),
        top_k_each=max(probe_depth * 4, 40),
        enable_variants=False,
        enable_mmr=bool(patha_cfg.get("enable_mmr", True)),
        anchored_exact_only=bool(patha_cfg.get("anchored_exact_only", True)),
        anchored_lexical_bias_enabled=bool(
            patha_cfg.get("anchored_lexical_bias_enabled", True)
        ),
        anchored_fusion_weight_vector=float(
            patha_cfg.get("anchored_fusion_weight_vector", 0.4)
        ),
        anchored_fusion_weight_bm25=float(
            patha_cfg.get("anchored_fusion_weight_bm25", 0.6)
        ),
        fusion_weight_vector=float(patha_cfg.get("fusion_weight_vector", 0.7)),
        fusion_weight_bm25=float(patha_cfg.get("fusion_weight_bm25", 0.3)),
    )


def _expected_rank(docs: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> int | None:
    expected_set = set(expected)
    if not expected_set:
        return None
    for idx, doc in enumerate(docs, start=1):
        checksum = doc.get("checksum")
        if isinstance(checksum, str) and checksum in expected_set:
            return idx
    return None


def _doc_by_checksum(
    docs: Sequence[Mapping[str, Any]],
    checksum: str | None,
) -> Mapping[str, Any] | None:
    if not checksum:
        return None
    for doc in docs:
        if doc.get("checksum") == checksum:
            return doc
    return None


def _feature_snapshot(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    fields = (
        "checksum",
        "path",
        "filename",
        "doc_type",
        "person_name",
        "retrieval_score",
        "score_vector",
        "score_bm25",
        "_bm25_variant_weight",
        "authority_rank",
        "_profile_intent_adjustment",
        "modified_at",
    )
    return {field: doc.get(field) for field in fields}


def _rank_value_for_delta(rank: int | None, depth: int) -> int:
    if rank is None:
        return depth + 1
    return rank


def _most_common_wrong_top1(
    failed_rows: Sequence[Mapping[str, Any]],
    metadata_cache: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in failed_rows:
        checksum = row.get("top1_checksum")
        if isinstance(checksum, str) and checksum:
            counter[checksum] += 1

    records: list[dict[str, Any]] = []
    for checksum, count in counter.most_common():
        if count < 2:
            continue
        meta = metadata_cache.get(checksum) or {}
        records.append(
            {
                "checksum": checksum,
                "count_as_wrong_top1": count,
                "path": meta.get("path"),
                "filename": meta.get("filename"),
                "doc_type": meta.get("doc_type"),
            }
        )
    return records


def _query_rows_by_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if isinstance(query_id, str):
            out[query_id] = row
    return out


def _best_similarity_to_expected(
    *,
    doc_checksum: str,
    doc_text: str,
    expected_checksums: Sequence[str],
    expected_text_by_checksum: Mapping[str, str],
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]],
) -> tuple[str | None, dict[str, float | bool] | None]:
    best_expected_checksum: str | None = None
    best_similarity: dict[str, float | bool] | None = None
    for expected_checksum in expected_checksums:
        expected_text = expected_text_by_checksum.get(expected_checksum) or ""
        if not expected_text:
            continue
        cache_key = (doc_checksum, expected_checksum)
        if cache_key not in similarity_cache:
            similarity_cache[cache_key] = text_similarity_metrics(doc_text, expected_text)
        similarity = similarity_cache[cache_key]
        if best_similarity is None:
            best_similarity = similarity
            best_expected_checksum = expected_checksum
            continue
        if float(similarity.get("containment_min", 0.0) or 0.0) > float(
            best_similarity.get("containment_min", 0.0) or 0.0
        ):
            best_similarity = similarity
            best_expected_checksum = expected_checksum
    return best_expected_checksum, best_similarity


def find_first_answer_support(
    *,
    docs: Sequence[Mapping[str, Any]],
    expected_checksums: Sequence[str],
    metadata_cache: Mapping[str, Mapping[str, Any] | None],
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]],
) -> dict[str, Any]:
    expected_set = set(expected_checksums)
    expected_text_by_checksum: dict[str, str] = {}
    for checksum in expected_checksums:
        meta = metadata_cache.get(checksum) or {}
        text = str(meta.get("text_full") or "")
        if text:
            expected_text_by_checksum[checksum] = text

    for rank, doc in enumerate(docs, start=1):
        checksum = doc.get("checksum")
        if not isinstance(checksum, str) or not checksum:
            continue
        if checksum in expected_set:
            return {
                "rank": rank,
                "checksum": checksum,
                "support_type": "strict",
                "matched_expected_checksum": checksum,
                "similarity": None,
            }
        meta = metadata_cache.get(checksum) or {}
        text = str(meta.get("text_full") or "")
        if not text:
            continue
        matched_expected, similarity = _best_similarity_to_expected(
            doc_checksum=checksum,
            doc_text=text,
            expected_checksums=expected_checksums,
            expected_text_by_checksum=expected_text_by_checksum,
            similarity_cache=similarity_cache,
        )
        if is_equivalent_answer_support(similarity):
            return {
                "rank": rank,
                "checksum": checksum,
                "support_type": "equivalent",
                "matched_expected_checksum": matched_expected,
                "similarity": similarity,
            }

    return {
        "rank": None,
        "checksum": None,
        "support_type": None,
        "matched_expected_checksum": None,
        "similarity": None,
    }


def investigate_ranking_post_patha(
    *,
    patha_runbook_path: Path,
    fixture_path: Path,
    output_path: Path,
    probe_depth: int,
    support_labels_path: Path | None = DEFAULT_SUPPORT_LABELS_PATH,
) -> dict[str, Any]:
    runbook = _load_json(patha_runbook_path)
    _load_json(fixture_path)
    support_label_overrides = _load_support_label_overrides(support_labels_path)
    rows = [
        row
        for row in runbook.get("rows", [])
        if isinstance(row, dict) and row.get("mode") == "positive"
    ]
    row_by_id = _query_rows_by_id(rows)
    failed_rows = [row for row in rows if not bool(row.get("hit_at_1"))]
    os_client = get_client()
    metadata_cache: dict[str, dict[str, Any] | None] = {}
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]] = {}

    def cached_fulltext(checksum: str) -> dict[str, Any] | None:
        if checksum not in metadata_cache:
            metadata_cache[checksum] = _fetch_fulltext_by_checksum(os_client, checksum)
        return metadata_cache[checksum]

    base_cfg = _build_probe_cfg(runbook.get("config", {}), probe_depth=probe_depth)
    ablation_cfgs = {
        "base": base_cfg,
        "no_authority": replace(base_cfg, authority_boost_enabled=False),
        "no_recency": replace(base_cfg, recency_boost_enabled=False),
        "no_profile_intent": replace(base_cfg, profile_intent_boost_enabled=False),
        "no_boosts": replace(
            base_cfg,
            authority_boost_enabled=False,
            recency_boost_enabled=False,
            profile_intent_boost_enabled=False,
        ),
    }

    base_probe_docs: dict[str, list[dict[str, Any]]] = {}
    per_query_rank_rows: list[dict[str, Any]] = []
    rank_buckets: Counter[str] = Counter()
    rr_values: list[float] = []
    support_hit1_probe_count = 0
    support_hit3_probe_count = 0
    support_top1_equivalent_count = 0
    support_top3_queries = 0
    preferred_source_top1_when_support_top1 = 0
    preferred_source_in_top3_when_support_top3 = 0

    for row in rows:
        query_id = str(row.get("query_id"))
        query_text = str(row.get("query") or "")
        strict_expected = _dedupe_str_checksums(row.get("expected_checksums") or [])
        label_override = support_label_overrides.get(query_id)
        support_expected, preferred_checksum = resolve_support_expectations(
            strict_expected_checksums=strict_expected,
            label_override=label_override,
        )
        retrieval = retrieve(query_text, cfg=base_cfg)
        docs = list(retrieval.documents)
        base_probe_docs[query_id] = docs

        for checksum in strict_expected:
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)
        for checksum in support_expected:
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)
        for doc in docs:
            checksum = doc.get("checksum")
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)

        rank = _expected_rank(docs, strict_expected)
        preferred_rank = _expected_rank(docs, [preferred_checksum]) if preferred_checksum else None
        rr = reciprocal_rank(rank)
        rr_values.append(rr)
        rank_buckets[rank_bucket(rank, probe_depth)] += 1

        top1 = docs[0] if docs else None
        matched_expected = None
        if rank is not None:
            matched_expected = docs[rank - 1].get("checksum")

        first_support = find_first_answer_support(
            docs=docs,
            expected_checksums=support_expected,
            metadata_cache=metadata_cache,
            similarity_cache=similarity_cache,
        )
        support_rank = first_support.get("rank")
        support_hit_at_1 = support_rank == 1
        support_hit_at_3 = isinstance(support_rank, int) and support_rank <= 3
        if support_hit_at_1:
            support_hit1_probe_count += 1
            if first_support.get("support_type") == "equivalent":
                support_top1_equivalent_count += 1
            if preferred_checksum and first_support.get("checksum") == preferred_checksum:
                preferred_source_top1_when_support_top1 += 1
        if support_hit_at_3:
            support_hit3_probe_count += 1
            support_top3_queries += 1
            if preferred_rank is not None and preferred_rank <= 3:
                preferred_source_in_top3_when_support_top3 += 1

        per_query_rank_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "archived_hit_at_1": bool(row.get("hit_at_1")),
                "archived_hit_at_3": bool(row.get("hit_at_3")),
                "strict_expected_checksums_probe": strict_expected,
                "answer_support_checksums_probe": support_expected,
                "support_label_override_applied": bool(label_override),
                "preferred_expected_checksum": preferred_checksum,
                "preferred_expected_rank_probe": preferred_rank,
                "expected_rank_probe": rank,
                "reciprocal_rank_probe": round(rr, 4),
                "rank_bucket_probe": rank_bucket(rank, probe_depth),
                "top1_checksum_probe": top1.get("checksum") if top1 else None,
                "top1_features_probe": _feature_snapshot(top1),
                "matched_expected_checksum_probe": matched_expected,
                "matched_expected_features_probe": _feature_snapshot(
                    docs[rank - 1] if rank is not None and rank - 1 < len(docs) else None
                ),
                "answer_support_rank_probe": support_rank,
                "answer_support_hit_at_1_probe": support_hit_at_1,
                "answer_support_hit_at_3_probe": support_hit_at_3,
                "answer_support_first_doc_checksum_probe": first_support.get("checksum"),
                "answer_support_match_type_probe": first_support.get("support_type"),
                "answer_support_matched_expected_checksum_probe": first_support.get(
                    "matched_expected_checksum"
                ),
                "answer_support_similarity_probe": first_support.get("similarity"),
                "reviewer_answer_support_override": None,
                "reviewer_answer_support_notes": "",
            }
        )

    total_pos = len(rows)
    hit1_probe = rank_buckets.get(RANK_BUCKET_1, 0)
    hit3_probe = hit1_probe + rank_buckets.get(RANK_BUCKET_2_3, 0)
    mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
    failed_query_ids = [str(row.get("query_id")) for row in failed_rows]
    ablation_per_query: list[dict[str, Any]] = []
    ablation_deltas: dict[str, list[int]] = defaultdict(list)

    for query_id in failed_query_ids:
        row = row_by_id[query_id]
        query_text = str(row.get("query") or "")
        strict_expected = _dedupe_str_checksums(row.get("expected_checksums") or [])
        base_rank = _expected_rank(base_probe_docs.get(query_id, []), strict_expected)
        ablation_result = {
            "query_id": query_id,
            "query": query_text,
            "ranks": {},
            "rank_delta_vs_base": {},
        }
        base_rank_value = _rank_value_for_delta(base_rank, probe_depth)
        ablation_result["ranks"]["base"] = base_rank

        for name, cfg in ablation_cfgs.items():
            if name == "base":
                continue
            docs = list(retrieve(query_text, cfg=cfg).documents)
            rank = _expected_rank(docs, strict_expected)
            ablation_result["ranks"][name] = rank
            delta = _rank_value_for_delta(rank, probe_depth) - base_rank_value
            ablation_result["rank_delta_vs_base"][name] = delta
            ablation_deltas[name].append(delta)

        ablation_per_query.append(ablation_result)

    ablation_summary = []
    for name in ("no_authority", "no_recency", "no_profile_intent", "no_boosts"):
        deltas = ablation_deltas.get(name, [])
        improved = sum(1 for d in deltas if d < 0)
        worsened = sum(1 for d in deltas if d > 0)
        unchanged = sum(1 for d in deltas if d == 0)
        avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
        ablation_summary.append(
            {
                "ablation": name,
                "queries": len(deltas),
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "avg_rank_delta_vs_base": round(avg_delta, 3),
            }
        )

    similarity_rows: list[dict[str, Any]] = []
    similarity_counter: Counter[str] = Counter()
    likely_correct_count = 0

    for row in failed_rows:
        query_id = str(row.get("query_id") or "")
        strict_expected = _dedupe_str_checksums(row.get("expected_checksums") or [])
        label_override = support_label_overrides.get(query_id)
        support_expected, _ = resolve_support_expectations(
            strict_expected_checksums=strict_expected,
            label_override=label_override,
        )
        top1_checksum = row.get("top1_checksum")
        if not isinstance(top1_checksum, str) or not top1_checksum:
            continue

        top1_meta = cached_fulltext(top1_checksum) or {}
        top1_text = str(top1_meta.get("text_full") or "")

        expected_text_by_checksum: dict[str, str] = {}
        best_expected_meta: dict[str, Any] | None = None
        for checksum in support_expected:
            expected_meta = cached_fulltext(checksum)
            if not expected_meta:
                continue
            expected_text = str(expected_meta.get("text_full") or "")
            if expected_text:
                expected_text_by_checksum[checksum] = expected_text

        best_expected_checksum, best_similarity = _best_similarity_to_expected(
            doc_checksum=top1_checksum,
            doc_text=top1_text,
            expected_checksums=support_expected,
            expected_text_by_checksum=expected_text_by_checksum,
            similarity_cache=similarity_cache,
        )
        if best_expected_checksum:
            best_expected_meta = cached_fulltext(best_expected_checksum) or {}

        near_duplicate = bool(best_similarity.get("near_duplicate")) if best_similarity else False
        containment = float(best_similarity.get("containment_min", 0.0)) if best_similarity else 0.0
        expected_rank_probe = _expected_rank(base_probe_docs.get(query_id, []), strict_expected)
        support_rank_probe_computed = _expected_rank(
            base_probe_docs.get(query_id, []), support_expected
        )
        support_probe_row = next(
            (row_diag for row_diag in per_query_rank_rows if row_diag.get("query_id") == query_id),
            {},
        )
        support_rank_probe = support_probe_row.get("answer_support_rank_probe")
        if support_rank_probe is None:
            support_rank_probe = support_rank_probe_computed
        support_hit3_probe_row = bool(
            isinstance(support_rank_probe, int) and support_rank_probe <= 3
        )
        if bool(support_probe_row.get("answer_support_hit_at_3_probe")):
            support_hit3_probe_row = True

        likelihood = auto_answer_likelihood(
            hit_at_3=support_hit3_probe_row,
            expected_rank=expected_rank_probe,
            near_duplicate=near_duplicate,
            similarity_containment=containment,
        )
        similarity_counter[likelihood] += 1
        if likelihood == LIKELY_CORRECT:
            likely_correct_count += 1

        similarity_rows.append(
            {
                "query_id": query_id,
                "query": row.get("query"),
                "top1_checksum": top1_checksum,
                "top1_path": top1_meta.get("path"),
                "top1_doc_type": top1_meta.get("doc_type"),
                "strict_expected_checksums_probe": strict_expected,
                "answer_support_checksums_probe": support_expected,
                "best_expected_checksum": best_expected_checksum,
                "best_expected_path": (best_expected_meta or {}).get("path"),
                "best_expected_doc_type": (best_expected_meta or {}).get("doc_type"),
                "similarity": best_similarity,
                "expected_rank_probe": expected_rank_probe,
                "answer_support_rank_probe": support_probe_row.get("answer_support_rank_probe"),
                "answer_support_match_type_probe": support_probe_row.get(
                    "answer_support_match_type_probe"
                ),
                "hit_at_3_archived": bool(row.get("hit_at_3")),
                "answer_support_hit_at_3_probe": support_hit3_probe_row,
                "auto_answer_likelihood": likelihood,
                "reviewer_answer_correct": None,
                "reviewer_notes": "",
            }
        )

    hard_negatives = _most_common_wrong_top1(failed_rows, metadata_cache)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "patha_runbook": str(patha_runbook_path),
            "fixture": str(fixture_path),
            "support_labels": str(support_labels_path) if support_labels_path else None,
        },
        "methodology": {
            "deterministic_probe": True,
            "note": (
                "Deep ranking probe runs retrieval with variants disabled (exact query only) "
                "to avoid rewrite-model variance while investigating ranking behavior."
            ),
            "support_labels_overrides_count": len(support_label_overrides),
            "answer_support_policy": {
                "strict_support": "retrieved checksum in expected_checksums",
                "equivalent_support": (
                    "retrieved doc text is near-duplicate or exceeds configured "
                    "containment/sequence similarity thresholds vs expected docs"
                ),
                "equivalent_thresholds": {
                    "containment_min": EQUIV_SUPPORT_CONTAINMENT_MIN,
                    "sequence_ratio_min": EQUIV_SUPPORT_SEQUENCE_MIN,
                    "alt_containment_min": EQUIV_SUPPORT_ALT_CONTAINMENT_MIN,
                    "alt_jaccard_min": EQUIV_SUPPORT_ALT_JACCARD_MIN,
                    "alt_sequence_ratio_min": EQUIV_SUPPORT_ALT_SEQUENCE_MIN,
                },
            },
            "probe_depth": probe_depth,
            "probe_config": {
                "top_k": base_cfg.top_k,
                "top_k_each": base_cfg.top_k_each,
                "enable_variants": base_cfg.enable_variants,
                "enable_mmr": base_cfg.enable_mmr,
                "anchored_fusion_weight_vector": base_cfg.anchored_fusion_weight_vector,
                "anchored_fusion_weight_bm25": base_cfg.anchored_fusion_weight_bm25,
            },
        },
        "archived_patha_metrics": runbook.get("summary", {}),
        "probe_ranking_metrics": {
            "positive_total": total_pos,
            "hit_at_1_probe": hit1_probe,
            "hit_at_1_probe_rate": round((hit1_probe / total_pos), 4) if total_pos else 0.0,
            "hit_at_3_probe": hit3_probe,
            "hit_at_3_probe_rate": round((hit3_probe / total_pos), 4) if total_pos else 0.0,
            "source_strict_hit_at_1_probe": hit1_probe,
            "source_strict_hit_at_1_probe_rate": round((hit1_probe / total_pos), 4)
            if total_pos
            else 0.0,
            "source_strict_hit_at_3_probe": hit3_probe,
            "source_strict_hit_at_3_probe_rate": round((hit3_probe / total_pos), 4)
            if total_pos
            else 0.0,
            "answer_support_hit_at_1_probe": support_hit1_probe_count,
            "answer_support_hit_at_1_probe_rate": round(
                (support_hit1_probe_count / total_pos), 4
            )
            if total_pos
            else 0.0,
            "answer_support_hit_at_3_probe": support_hit3_probe_count,
            "answer_support_hit_at_3_probe_rate": round(
                (support_hit3_probe_count / total_pos), 4
            )
            if total_pos
            else 0.0,
            "answer_support_lift_vs_strict_at_1": support_hit1_probe_count - hit1_probe,
            "answer_support_lift_vs_strict_at_3": support_hit3_probe_count - hit3_probe,
            "mrr_probe": round(mrr, 4),
            "rank_bucket_counts": {
                RANK_BUCKET_1: rank_buckets.get(RANK_BUCKET_1, 0),
                RANK_BUCKET_2_3: rank_buckets.get(RANK_BUCKET_2_3, 0),
                RANK_BUCKET_4_10: rank_buckets.get(RANK_BUCKET_4_10, 0),
                RANK_BUCKET_11_TO_DEPTH: rank_buckets.get(RANK_BUCKET_11_TO_DEPTH, 0),
                RANK_BUCKET_NOT_RETRIEVED: rank_buckets.get(RANK_BUCKET_NOT_RETRIEVED, 0),
            },
            "attribution_quality": {
                "support_top1_queries": support_hit1_probe_count,
                "preferred_source_top1_when_support_top1": preferred_source_top1_when_support_top1,
                "preferred_source_top1_rate_when_support_top1": round(
                    (preferred_source_top1_when_support_top1 / support_hit1_probe_count), 4
                )
                if support_hit1_probe_count
                else 0.0,
                "support_top1_equivalent_source_count": support_top1_equivalent_count,
                "support_top3_queries": support_top3_queries,
                "preferred_source_in_top3_when_support_top3": preferred_source_in_top3_when_support_top3,
                "preferred_source_in_top3_rate_when_support_top3": round(
                    (preferred_source_in_top3_when_support_top3 / support_top3_queries), 4
                )
                if support_top3_queries
                else 0.0,
            },
        },
        "per_query_rank_diagnostics": per_query_rank_rows,
        "ablation_analysis": {
            "scope": "queries where archived hit@1=false",
            "query_count": len(failed_query_ids),
            "summary": ablation_summary,
            "per_query": ablation_per_query,
        },
        "content_similarity_analysis": {
            "scope": "archived hit@1 failures",
            "summary": {
                "queries_analyzed": len(similarity_rows),
                "auto_answer_likelihood_counts": dict(similarity_counter),
                "likely_correct_despite_rank_miss": likely_correct_count,
            },
            "failed_query_rows": similarity_rows,
            "repeated_wrong_top1_hard_negatives": hard_negatives,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return output


def _default_output(patha_runbook: Path) -> Path:
    return patha_runbook.with_name(f"{patha_runbook.stem}_ranking_investigation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Investigate residual ranking weakness after Path A without changing production behavior."
    )
    parser.add_argument(
        "--patha-runbook",
        type=Path,
        default=Path("docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json"),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/retrieval_eval_queries.json"),
    )
    parser.add_argument(
        "--support-labels",
        type=Path,
        default=DEFAULT_SUPPORT_LABELS_PATH,
        help=(
            "Optional JSON file containing per-query answer-support overrides "
            "(equivalent source checksums, preferred checksum)."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--probe-depth", type=int, default=DEFAULT_PROBE_DEPTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output(args.patha_runbook)
    output = investigate_ranking_post_patha(
        patha_runbook_path=args.patha_runbook,
        fixture_path=args.fixture,
        output_path=output_path,
        probe_depth=max(args.probe_depth, 1),
        support_labels_path=args.support_labels,
    )
    metrics = output.get("probe_ranking_metrics", {})
    print(f"Wrote: {output_path}")
    print(
        "Probe metrics: "
        f"hit@1={metrics.get('hit_at_1_probe')}/{metrics.get('positive_total')}, "
        f"hit@3={metrics.get('hit_at_3_probe')}/{metrics.get('positive_total')}, "
        f"support_hit@3={metrics.get('answer_support_hit_at_3_probe')}/{metrics.get('positive_total')}, "
        f"MRR={metrics.get('mrr_probe')}"
    )


if __name__ == "__main__":
    main()

"""Deterministic source-selection helpers for the composite benchmark.

These functions deliberately know nothing about the Document QA retriever. Benchmark
membership is based only on upstream IDs/labels plus a fixed hash ordering.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

SELECTION_ALGORITHM = "stable-sha256-v1"
DEFAULT_SEED = 20260928


def stable_rank_key(value: str, *, namespace: str, seed: int = DEFAULT_SEED) -> str:
    payload = f"{SELECTION_ALGORITHM}\0{seed}\0{namespace}\0{value}".encode("utf-8")
    return sha256(payload).hexdigest()


def stable_take(
    values: Iterable[str],
    count: int,
    *,
    namespace: str,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    unique = sorted(set(values))
    if len(unique) < count:
        raise ValueError(f"requested {count} items from a pool of {len(unique)}")
    return sorted(
        unique,
        key=lambda value: (stable_rank_key(value, namespace=namespace, seed=seed), value),
    )[:count]


def balanced_take(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    id_key: str,
    stratum: Callable[[Mapping[str, Any]], str],
    namespace: str,
    seed: int = DEFAULT_SEED,
) -> list[Mapping[str, Any]]:
    """Round-robin across strata with stable ordering inside every stratum."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(stratum(record))].append(record)

    ordered_groups: dict[str, list[Mapping[str, Any]]] = {}
    for name, group in groups.items():
        ordered_groups[name] = sorted(
            group,
            key=lambda row: (
                stable_rank_key(str(row[id_key]), namespace=f"{namespace}:{name}", seed=seed),
                str(row[id_key]),
            ),
        )

    names = sorted(
        ordered_groups,
        key=lambda name: (stable_rank_key(name, namespace=f"{namespace}:strata", seed=seed), name),
    )
    selected: list[Mapping[str, Any]] = []
    offsets = {name: 0 for name in names}

    while len(selected) < count:
        progressed = False
        for name in names:
            offset = offsets[name]
            group = ordered_groups[name]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            offsets[name] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"requested {count} records from a pool of {len(records)}")

    return selected


def select_open_ragbench(
    queries: Mapping[str, Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, Any]],
    all_document_ids: Iterable[str],
    *,
    positive_documents: int = 80,
    hard_negative_documents: int = 120,
    questions_per_document: int = 2,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    doc_to_queries: dict[str, list[str]] = defaultdict(list)
    all_gold_docs: set[str] = set()

    for query_id, rel in qrels.items():
        doc_id = str(rel["doc_id"])
        all_gold_docs.add(doc_id)
        query = queries.get(query_id)
        if not query or query.get("source") != "text":
            continue
        doc_to_queries[doc_id].append(query_id)

    eligible_docs = [
        doc_id for doc_id, query_ids in doc_to_queries.items()
        if len(query_ids) >= questions_per_document
    ]
    selected_docs = stable_take(
        eligible_docs,
        positive_documents,
        namespace="open-ragbench:positive-docs",
        seed=seed,
    )

    selected_queries: list[str] = []
    for doc_id in selected_docs:
        query_ids = doc_to_queries[doc_id]
        by_type: dict[str, list[str]] = defaultdict(list)
        for query_id in query_ids:
            by_type[str(queries[query_id].get("type", "unknown"))].append(query_id)

        chosen: list[str] = []
        if questions_per_document >= 2 and by_type.get("extractive") and by_type.get("abstractive"):
            chosen.extend(
                stable_take(
                    by_type["extractive"],
                    1,
                    namespace=f"open-ragbench:{doc_id}:extractive",
                    seed=seed,
                )
            )
            chosen.extend(
                stable_take(
                    by_type["abstractive"],
                    1,
                    namespace=f"open-ragbench:{doc_id}:abstractive",
                    seed=seed,
                )
            )

        remaining = questions_per_document - len(chosen)
        if remaining:
            pool = [query_id for query_id in query_ids if query_id not in chosen]
            chosen.extend(
                stable_take(
                    pool,
                    remaining,
                    namespace=f"open-ragbench:{doc_id}:fallback",
                    seed=seed,
                )
            )
        selected_queries.extend(chosen)

    hard_pool = set(map(str, all_document_ids)) - all_gold_docs
    selected_hard_negatives = stable_take(
        hard_pool,
        hard_negative_documents,
        namespace="open-ragbench:hard-negative-docs",
        seed=seed,
    )

    return {
        "positive_document_ids": selected_docs,
        "hard_negative_document_ids": selected_hard_negatives,
        "query_ids": selected_queries,
    }


_YEAR_RE = re.compile(r"(?<!\\d)((?:19|20)\\d{2})(?!\\d)")


def source_decade(source_files: Sequence[str]) -> str:
    years: list[int] = []
    for source_file in source_files:
        match = _YEAR_RE.search(source_file)
        if match:
            years.append(int(match.group(1)))
    if not years:
        return "unknown"
    return str((min(years) // 10) * 10)


def select_officeqa(
    questions: Sequence[Mapping[str, Any]],
    all_corpus_files: Iterable[str],
    *,
    easy_questions: int = 25,
    hard_questions: int = 25,
    target_total_documents: int = 150,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    selected_questions: list[Mapping[str, Any]] = []

    for difficulty, quota in (("easy", easy_questions), ("hard", hard_questions)):
        pool = [row for row in questions if str(row["difficulty"]).lower() == difficulty]
        selected_questions.extend(
            balanced_take(
                pool,
                quota,
                id_key="uid",
                stratum=lambda row: source_decade(row["source_files"]),
                namespace=f"officeqa:{difficulty}",
                seed=seed,
            )
        )

    positive_files = {
        str(source_file)
        for row in selected_questions
        for source_file in row["source_files"]
    }

    if len(positive_files) > target_total_documents:
        raise ValueError(
            "selected OfficeQA source documents exceed target_total_documents; "
            "increase the target instead of dropping gold documents"
        )

    all_files = set(map(str, all_corpus_files))
    candidates = all_files - positive_files
    positive_decades = {source_decade([name]) for name in positive_files}
    same_decade = [
        name for name in candidates
        if source_decade([name]) in positive_decades
    ]

    needed = target_total_documents - len(positive_files)
    chosen_distractors: list[str] = []
    same_decade_count = min(needed, len(same_decade))
    if same_decade_count:
        chosen_distractors.extend(
            stable_take(
                same_decade,
                same_decade_count,
                namespace="officeqa:distractors:same-decade",
                seed=seed,
            )
        )

    remaining = needed - len(chosen_distractors)
    if remaining:
        fallback_pool = candidates - set(chosen_distractors)
        chosen_distractors.extend(
            stable_take(
                fallback_pool,
                remaining,
                namespace="officeqa:distractors:fallback",
                seed=seed,
            )
        )

    return {
        "query_ids": [str(row["uid"]) for row in selected_questions],
        "positive_document_ids": sorted(positive_files),
        "distractor_document_ids": chosen_distractors,
    }


def select_nfcorpus(
    test_query_ids: Iterable[str],
    *,
    queries: int = 100,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    return stable_take(
        test_query_ids,
        queries,
        namespace="nfcorpus:test-queries",
        seed=seed,
    )


def select_miracl(
    query_ids: Iterable[str],
    qrels: Sequence[Mapping[str, Any]],
    *,
    language: str,
    queries: int = 50,
    hard_negatives_per_query: int = 20,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    selected_queries = stable_take(
        query_ids,
        queries,
        namespace=f"miracl:{language}:queries",
        seed=seed,
    )
    selected_set = set(selected_queries)

    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in qrels:
        query_id = str(row["query-id"])
        if query_id in selected_set:
            by_query[query_id].append(row)

    selected_qrels: list[dict[str, Any]] = []
    document_ids: set[str] = set()

    for query_id in selected_queries:
        rows = by_query.get(query_id, [])
        positives = [row for row in rows if int(row["score"]) > 0]
        negatives = [row for row in rows if int(row["score"]) <= 0]

        if not positives:
            raise ValueError(f"MIRACL query {query_id!r} has no positive qrels")

        negative_ids = stable_take(
            [str(row["corpus-id"]) for row in negatives],
            min(hard_negatives_per_query, len(negatives)),
            namespace=f"miracl:{language}:{query_id}:negatives",
            seed=seed,
        )
        keep_negative_ids = set(negative_ids)

        for row in positives:
            doc_id = str(row["corpus-id"])
            document_ids.add(doc_id)
            selected_qrels.append(
                {"query-id": query_id, "corpus-id": doc_id, "score": int(row["score"])}
            )
        for row in negatives:
            doc_id = str(row["corpus-id"])
            if doc_id not in keep_negative_ids:
                continue
            document_ids.add(doc_id)
            selected_qrels.append(
                {"query-id": query_id, "corpus-id": doc_id, "score": int(row["score"])}
            )

    return {
        "query_ids": selected_queries,
        "document_ids": sorted(document_ids),
        "qrels": selected_qrels,
    }

from evaluation.benchmarks.selection import (
    balanced_take,
    select_miracl,
    select_nfcorpus,
    select_officeqa,
    select_open_ragbench,
    stable_take,
)


def test_stable_take_is_order_independent():
    values = ["c", "a", "b", "d"]
    assert stable_take(values, 3, namespace="x") == stable_take(
        reversed(values), 3, namespace="x"
    )


def test_balanced_take_spreads_across_strata():
    rows = [
        {"id": "a1", "group": "a"},
        {"id": "a2", "group": "a"},
        {"id": "b1", "group": "b"},
        {"id": "b2", "group": "b"},
    ]
    selected = balanced_take(
        rows,
        2,
        id_key="id",
        stratum=lambda row: row["group"],
        namespace="balanced",
    )
    assert {row["group"] for row in selected} == {"a", "b"}


def test_open_ragbench_uses_text_queries_and_non_gold_hard_negatives():
    queries = {
        "q1": {"source": "text", "type": "extractive"},
        "q2": {"source": "text", "type": "abstractive"},
        "q3": {"source": "text-image", "type": "extractive"},
        "q4": {"source": "text", "type": "extractive"},
        "q5": {"source": "text", "type": "abstractive"},
    }
    qrels = {
        "q1": {"doc_id": "p1"},
        "q2": {"doc_id": "p1"},
        "q3": {"doc_id": "p1"},
        "q4": {"doc_id": "p2"},
        "q5": {"doc_id": "p2"},
    }
    result = select_open_ragbench(
        queries,
        qrels,
        ["p1", "p2", "n1", "n2", "n3"],
        positive_documents=2,
        hard_negative_documents=2,
        questions_per_document=2,
    )
    assert set(result["positive_document_ids"]) == {"p1", "p2"}
    assert "q3" not in result["query_ids"]
    assert set(result["hard_negative_document_ids"]).issubset({"n1", "n2", "n3"})


def test_officeqa_keeps_all_gold_docs_and_fills_distractors():
    rows = [
        {"uid": "e1", "difficulty": "easy", "source_files": ["tb_1941_01.pdf"]},
        {"uid": "e2", "difficulty": "easy", "source_files": ["tb_1951_01.pdf"]},
        {"uid": "h1", "difficulty": "hard", "source_files": ["tb_1942_01.pdf"]},
        {"uid": "h2", "difficulty": "hard", "source_files": ["tb_1952_01.pdf"]},
    ]
    corpus = [
        "tb_1941_01.pdf",
        "tb_1942_01.pdf",
        "tb_1943_01.pdf",
        "tb_1951_01.pdf",
        "tb_1952_01.pdf",
        "tb_1953_01.pdf",
    ]
    result = select_officeqa(
        rows,
        corpus,
        easy_questions=2,
        hard_questions=2,
        target_total_documents=6,
    )
    assert len(result["query_ids"]) == 4
    assert set(result["positive_document_ids"]) == {
        "tb_1941_01.pdf",
        "tb_1942_01.pdf",
        "tb_1951_01.pdf",
        "tb_1952_01.pdf",
    }
    assert len(result["distractor_document_ids"]) == 2


def test_nfcorpus_selection_is_fixed_size():
    assert len(select_nfcorpus([f"q{i}" for i in range(200)], queries=100)) == 100


def test_miracl_keeps_positives_and_caps_hard_negatives():
    qrels = [
        {"query-id": "q1", "corpus-id": "p1", "score": 1},
        {"query-id": "q1", "corpus-id": "n1", "score": 0},
        {"query-id": "q1", "corpus-id": "n2", "score": 0},
        {"query-id": "q2", "corpus-id": "p2", "score": 1},
        {"query-id": "q2", "corpus-id": "n3", "score": 0},
        {"query-id": "q2", "corpus-id": "n4", "score": 0},
    ]
    result = select_miracl(
        ["q1", "q2"],
        qrels,
        language="de",
        queries=2,
        hard_negatives_per_query=1,
    )
    scores_by_query = {}
    for row in result["qrels"]:
        scores_by_query.setdefault(row["query-id"], []).append(row["score"])
    assert all(1 in scores for scores in scores_by_query.values())
    assert all(scores.count(0) == 1 for scores in scores_by_query.values())

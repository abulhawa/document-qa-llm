"""The larger Open RAGBench corpus preserves the frozen v1 evaluation set."""

import gzip
import json
from pathlib import Path


def test_composite_v2_extends_v1_without_changing_queries_or_other_tracks():
    root = Path("evaluation/benchmarks")
    with gzip.open(root / "composite_v1.lock.json.gz", "rt", encoding="utf-8") as stream:
        old = json.load(stream)
    new = json.loads((root / "composite_v2.lock.json").read_text(encoding="utf-8"))

    assert new["benchmark_id"] == "composite-v2"
    assert new["sources"] == old["sources"]
    assert new["selection_seed"] == old["selection_seed"]
    for track in ("officeqa", "nfcorpus", "miracl_de", "miracl_ar"):
        assert new["selection"][track] == old["selection"][track]

    previous = old["selection"]["open_ragbench"]
    expanded = new["selection"]["open_ragbench"]
    assert expanded["positive_document_ids"] == previous["positive_document_ids"]
    assert expanded["query_ids"] == previous["query_ids"]
    assert len(expanded["positive_document_ids"]) == 80
    assert len(expanded["query_ids"]) == 160
    assert len(expanded["hard_negative_document_ids"]) == 920
    assert set(previous["hard_negative_document_ids"]) <= set(expanded["hard_negative_document_ids"])
    assert len(set(expanded["positive_document_ids"] + expanded["hard_negative_document_ids"])) == 1000


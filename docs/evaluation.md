# Evaluation guide

The repository separates retrieval evaluation from QA context-handoff evaluation. Existing JSON/CSV files under `docs/runbooks/` are historical experiment artifacts, not general product benchmarks.

## Retrieval evaluation

`scripts/run_retrieval_eval.py` evaluates the checked-in query fixture and can report expected-document Hit@k and MRR-style ranking measurements. It requires populated backends matching the fixture corpus. Review `--help` before running:

```bash
python scripts/run_retrieval_eval.py --help
```

Record the exact revision, fixture, service/model versions, flags, corpus state, output artifact, and latency. Compare a candidate against the same baseline rather than comparing unrelated runs.

## QA handoff and answer support

`scripts/run_qa_handoff_eval.py` compares context-selection strategies and support labels without treating retrieval success as final-answer correctness:

```bash
python scripts/run_qa_handoff_eval.py --help
```

When changing retrieval, ranking, grounding, or prompting, report the relevant subset of Hit@1/3/5, MRR, nDCG (if available), answer support, citation accuracy, refusal/failure counts, and latency. A change should not be described as an improvement without a reproducible before/after result.

## Safe fixtures

Only commit synthetic, licensed, or otherwise redistributable fixtures. Remove personal paths, document contents, credentials, and identifiers. Store large/generated run outputs outside Git unless a small artifact is intentionally retained as a documented baseline.

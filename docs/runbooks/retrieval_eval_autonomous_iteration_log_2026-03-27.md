# Retrieval Iteration Log (2026-03-27)

## Iteration 0 - Baseline + diagnosis setup
- Hypothesis: Establish a clean baseline and identify dominant miss buckets before changing logic.
- Files changed: none
- Commands:
  - `python -` (inline eval driver) -> `docs/runbooks/retrieval_eval_autonomous_baseline_2026-03-27.json`
  - `python scripts/investigate_ranking_post_patha.py --patha-runbook docs/runbooks/retrieval_eval_autonomous_baseline_2026-03-27.json --fixture tests/fixtures/retrieval_eval_queries.json --output docs/runbooks/retrieval_eval_autonomous_baseline_2026-03-27_ranking_investigation.json --probe-depth 60`
- Eval delta: baseline reference (`Hit@1=9/20`, `Hit@3=14/20`).
- Decision: keep as baseline artifacts.

## Iteration 1 - Parameter-only sweep
- Hypothesis: Small fusion/MMR/rescue/candidate-breadth tuning can recover misses without code changes.
- Files changed: none
- Commands:
  - `python -` (inline parameter sweep) -> `docs/runbooks/retrieval_eval_autonomous_param_sweep_2026-03-27.json`
- Eval delta: no improvement over baseline (`Hit@1=9/20`, `Hit@3=14/20` best).
- Decision: discard as optimization path.

## Iteration 2 - Global near-duplicate threshold increase
- Hypothesis: Raising near-duplicate collapse similarity threshold globally will preserve relevant sibling chunks.
- Files changed: none (runtime-only experiment)
- Commands:
  - `python -` (inline targeted eval variants with global `sim_threshold` changes)
- Eval delta: `Hit@3` could reach 15/20 but introduced regression (`Q01`).
- Decision: discard global change due regression risk.

## Iteration 3 - Canonical-query-scoped near-duplicate threshold
- Hypothesis: Apply higher duplicate threshold only on canonical anchored/semi-anchored queries to avoid global regressions.
- Files changed:
  - `core/retrieval/types.py`
  - `core/retrieval/pipeline.py`
  - `tests/test_retrieval_pipeline.py`
- Commands:
  - `pytest -q tests/test_retrieval_pipeline.py -k "higher_canonical_sim_threshold or keeps_default_sim_threshold or sim_threshold_default"`
  - `pytest -q tests/test_retrieval_pipeline.py`
  - `python -` (inline eval driver) -> `docs/runbooks/retrieval_eval_autonomous_postfix_2026-03-27_canonical_collapse_v1.json`
  - `python scripts/investigate_ranking_post_patha.py --patha-runbook docs/runbooks/retrieval_eval_autonomous_postfix_2026-03-27_canonical_collapse_v1.json --fixture tests/fixtures/retrieval_eval_queries.json --output docs/runbooks/retrieval_eval_autonomous_postfix_2026-03-27_canonical_collapse_v1_ranking_investigation.json --probe-depth 60`
  - `python -` (inline ambiguity review export) -> `docs/runbooks/retrieval_eval_autonomous_postfix_2026-03-27_canonical_collapse_v1_ambiguity_review.json`
- Eval delta: improved to `Hit@1=9/20`, `Hit@3=16/20` (wins: `Q03`, `Q09`; no losses).
- Decision: keep.

# Retrieval Eval Reliability Runbook (2026-03-28)

## Scope
- Keep sibling expansion patch `e522cdf` unchanged.
- Diagnose full probe timeout and make benchmarking reliable in this environment.
- Measure sibling expansion `OFF` vs `ON` on:
  - full gold set (`tests/fixtures/retrieval_eval_queries.json`)
  - targeted profile `when/where` subset.

## Timeout diagnosis
- Reproduced long probe runtime on `2026-03-28`:
  - command: `python scripts/investigate_ranking_post_patha.py --patha-runbook docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_candidate.json --fixture tests/fixtures/retrieval_eval_queries.json --output docs/runbooks/_tmp_probe_runtime_check.json --cleaned-strict-output docs/runbooks/_tmp_probe_runtime_check_cleaned.json --probe-depth 60`
  - wall time: `~399s`
- Dominant cost driver:
  - repeated high-depth retrieval calls in probe mode (`top_k=60`, `top_k_each=240`) with heavy semantic/embedding work.
  - this probe artifact includes `12` failed queries with `4` ablations each (`48` additional retrieval passes) on top of the base pass.
- Not the primary bottleneck:
  - rerank calls (`0` in this config),
  - sibling expansion fetch cost (small relative share),
  - harness bookkeeping.

## Reliability changes
- Added stable benchmark harness:
  - `scripts/run_retrieval_eval.py`
  - replaces ad-hoc inline eval for this workflow.
- Adds per-query timing diagnostics:
  - `stage_timings_ms`
  - `stage_call_counts`
  - `stage_error_counts`
  - `query_duration_ms`
  - soft-timeout flags/reporting.
- Adds A/B mode in one command:
  - sibling expansion `off|on|both`.
- Adds targeted subset + lightweight end-to-end QA probe:
  - subset definition: profile-scoped positive queries with explicit `when/where` phrasing.

## Run command
- `python scripts/run_retrieval_eval.py --output docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare.json --sibling-expansion-mode both`

## Artifacts
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_off.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_off.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_on.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_on.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare.json`

## Results snapshot
- Full gold set:
  - strict `hit@1`: `9 -> 9` (`delta 0`)
  - strict `hit@3`: `16 -> 15` (`delta -1`)
  - final-context support hits: `16 -> 16` (`delta 0`)
- Targeted profile `when/where` subset:
  - query ids: `Q01`
  - support hit: `1 -> 1`
  - QA answered-without-error: `1 -> 1`
- Timing:
  - dominant stage group in both runs: `live_index_queries`
  - rerank stage cost: `0` (disabled)
  - context expansion cost present only in `ON` run and remains bounded.

## Notes
- The strict `hit@3` regression (`Q01`) is due to sibling insertion changing top-3 strict checksum placement; support-context coverage for the targeted query remains unchanged.
- Harness now completes reliably in this environment with explicit timing and timeout reporting.

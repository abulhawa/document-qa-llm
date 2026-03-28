# QA Handoff Policy + Q04 Stage Plan (2026-03-28)

## Scope
- Preserve sibling expansion behavior.
- Keep rerank disabled.
- Compare end-to-end QA handoff policies:
  - `top3`
  - `top5`
  - `dynamic` token-budget packing (retrieval depth 7, budget 1200 est. tokens, min chunks 3)
- Isolate `Q04` failure stage and propose a narrow fix path.

## Commands
- Clean comparison run (throttled to avoid external LLM rate-limit artifacts):
  - `python scripts/run_qa_handoff_eval.py --output docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean.json --strategies top3,top5,dynamic --dynamic-token-budget 1200 --dynamic-retrieval-top-k 7 --dynamic-min-chunks 3 --per-query-sleep-seconds 3.0`

- Q04 stage artifact:
  - `docs/runbooks/q04_retrieval_stage_investigation_2026-03-28.json`

## Results (Clean Run)
Artifacts:
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top3.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top5.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_dynamic.json`

Positive queries (`n=20`):
- `top3`
  - answered_without_error: `11/20` (`0.55`)
  - support context hit: `16/20` (`0.80`)
  - answered_with_support: `9/20` (`0.45`)

- `top5`
  - answered_without_error: `12/20` (`0.60`)
  - support context hit: `18/20` (`0.90`)
  - answered_with_support: `11/20` (`0.55`)

- `dynamic`
  - answered_without_error: `14/20` (`0.70`)
  - support context hit: `19/20` (`0.95`)
  - answered_with_support: `13/20` (`0.65`)

Deltas:
- `top5 vs top3`
  - answered_without_error: `+1`
  - support_context_hits: `+2`
  - answered_with_support: `+2`

- `dynamic vs top5`
  - answered_without_error: `+2`
  - support_context_hits: `+1`
  - answered_with_support: `+2`

Controls (`n=3`):
- answered_without_error: `0/3` for all strategies (no regression on out-of-corpus abstention behavior in this run).

Targeted profile/timeline when/where subset (`Q01`):
- support hit: `1/1` for `top3`, `top5`, `dynamic`.
- answered_without_error: `1/1` for `top3`, `top5`, `dynamic`.

## Policy Recommendation
- Promote dynamic token-budget packing as the default QA handoff policy with the measured winning configuration:
  - retrieval depth: `7`
  - token budget: `1200` (estimated)
  - min chunks: `3`
- Keep top-5 available as a configurable fallback policy, but not the default while dynamic remains superior.

## Q04 Stage Isolation
Query:
- `Q04`: "In Ali's most recent CV contact section, which city is listed?"

Observed stage behavior (`rerank off`, sibling expansion on):
- Final top-7 has expected checksum only at rank 6 in benchmark labels.
- `top3` and `top5` handoff miss benchmark support; dynamic includes it at rank 6 but answer still falls back.
- Semantic top-120 has no CV/resume candidate.
- Keyword top-120 first CV/resume appears at rank 45.
- Current benchmark `Q04` expected checksums resolve to non-CV research docs in the present index snapshot (see artifact lookup block).

Conclusion:
- Immediate blocker is benchmark integrity drift between fixture labels and the current index snapshot.
- Until expected checksum labels are reconciled with intended CV/resume family targets, Q04 misses are not reliable retrieval-signal misses.

## Stage-Local Fix Proposal (Next Narrow Change)
- Do not re-enable reranking.
- Add a benchmark-integrity gate before the next retrieval optimization cycle:
  - verify expected checksums resolve to intended doc families in the active index snapshot,
  - explicitly flag fixture/index drift,
  - separate integrity failures from true retrieval failures in report summaries.
- Defer Q04-specific retrieval tuning until integrity status is clean.
- Next safe expansion target after integrity cleanup:
  - broaden profile-oriented fact coverage from only `when/where` to nearby education/employment/profile fact intents,
  - benchmark that expansion as a separate cycle (not implemented in this step).

# Cross-Encoder Reranker Decision (2026-03-28)

## Decision
- Keep cross-encoder reranking **disabled** for the current corpus (`RETRIEVAL_ENABLE_RERANK=false`).
- Keep reranker integration code available for future experiments, but do not tune this path now.

## Evidence
Source artifacts:
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare.json`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_off.json`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_on.json`

Measured (`top_k=7`, sibling expansion enabled):

- Strict `hit@n` (OFF -> ON):
  - `@1`: `9/20 -> 5/20`
  - `@3`: `15/20 -> 9/20`
  - `@5`: `18/20 -> 12/20`
  - `@7`: `19/20 -> 13/20`

- Support `hit@n` (OFF -> ON):
  - `@1`: `10/20 -> 6/20`
  - `@3`: `16/20 -> 10/20`
  - `@5`: `18/20 -> 12/20`
  - `@7`: `19/20 -> 13/20`

- Latency:
  - rerank stage total: `0.0 ms` (OFF) vs `1511.768 ms` (ON)

- Query-level impact:
  - improved: `Q04`
  - regressed: `Q03,Q05,Q09,Q14,Q15,Q16,Q20`
  - targeted profile/timeline when/where subset (`Q01`) stayed solved in both modes.

## Policy Outcome
- Current production-like recommendation remains:
  - sibling expansion: **enabled**
  - cross-encoder rerank: **disabled**
  - handoff policy work: prioritize top-`5` vs dynamic-budget packing evaluation over reranker tuning.

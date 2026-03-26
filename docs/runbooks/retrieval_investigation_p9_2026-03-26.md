# P9 Retrieval Investigation and Decision Memo (2026-03-26)

## Scope and method

- Fixture: `tests/fixtures/retrieval_eval_queries.json` (20 positive, 3 control).
- Retrieval run mode baseline: `top_k=3`, `enable_variants=true`, `enable_mmr=true`.
- Groq constraints considered:
  - `12k` tokens/minute
  - `30` requests/minute
  - `100k` tokens/day
- To stay within limits, investigation reused cache (`use_cache=true`) and limited variant-enabled sweeps.
- Machine-readable artifact:
  - `docs/runbooks/retrieval_investigation_p9_2026-03-26.json`

## Stage attribution summary

| Run | hit@1 (pos/20) | hit@3 (pos/20) | hit@3 rate | control with results (of 3) | clarify count |
|---|---:|---:|---:|---:|---:|
| full_current | 2 | 8 | 0.40 | 0 | 0 |
| no_variants | 1 | 12 | 0.60 | 0 | 0 |
| no_mmr | 2 | 7 | 0.35 | 0 | 0 |
| no_boosts | 3 | 10 | 0.50 | 0 | 0 |
| semantic_only | 0 | 7 | 0.35 | 0 | 0 |
| bm25_only | 6 | 12 | 0.60 | 0 | 0 |
| no_abstention | 2 | 8 | 0.40 | 3 | 0 |

## Findings by investigation area

### 1) Retrieval stage attribution

- Variants are currently a net negative on fixture hit@3 (`0.40` -> `0.60` when disabled).
- BM25 signal is materially stronger than semantic on this corpus (`bm25_only hit@3=0.60` vs `semantic_only=0.35`).
- MMR is currently mildly beneficial (`no_mmr` slightly worse than `full_current`).
- Current boosts (authority/recency/profile) are mixed and likely too broad (`no_boosts hit@3=0.50` > `full_current=0.40`).

### 2) Query rewrite quality

- Rewrite summary on fixture:
  - `clarify_count=0`
  - `error_count=0`
  - `anchored_total=22/23`
- Clarification over-triggering is fixed, but drift risk remains:
  - Several failed positives recover when variants are disabled, indicating rewritten variants can displace better exact candidates.

### 3) Data and metadata quality

- Fulltext docs scanned: `2142`.
- `doc_type` distribution now has broad deterministic coverage:
  - `research_paper=484`, `course_material=209`, `government_form=172`, `insurance_letter=106`, `cv=100`, etc.
  - `other=846` (still high), `__missing__=122`.
- `doc_type_source`:
  - `rule=1174`
  - `fallback=846`
  - `__missing__=122`
- `doc_type_confidence` stats over classified docs:
  - `count=2020`, `min=0.25`, `max=0.98`, `avg~0.60`.

### 4) Infrastructure behavior

- Root cause of recurring chunk-text warning was validated:
  - Old request shape: `mget` with `{"ids":[...], "_source":["text"]}` returns `400 parsing_exception`.
  - New request shape: `mget` with `{"ids":[...]}` + `params={"_source_includes":"text"}` succeeds.
- This is fixed in code and no longer depends on warning-path fallback behavior.

### 5) Control-query behavior

- Abstention gate impact is clear:
  - `no_abstention`: `control_with_results=3/3`
  - `full_current`: `control_with_results=0/3`
- Current bounded abstention policy is effective on fixture controls.

## Failure taxonomy (full_current)

- Positive misses at hit@3: `12/20`.
- Missed query IDs:
  - `Q02, Q03, Q04, Q05, Q06, Q07, Q10, Q12, Q14, Q15, Q17, Q18`
- Control false positives: none after abstention (`[]`).

## Decision memo

### Decision

- **Path A (targeted incremental fixes) selected now**, with a strict gate to Path B if targets are not met quickly.

### Why Path A first

- Control behavior is corrected and stable.
- Large gain is still available without broad architecture change:
  - Variants weighting/usage appears to be the highest-impact current bottleneck.
  - BM25 already outperforms semantic on this fixture and can be leveraged with low blast radius.
- A broader RAG redesign should be triggered only if targeted adjustments fail against explicit thresholds.

### Path A implementation priorities (next 1-2 PRs)

1. Variant safety gating:
   - Use exact-only by default for anchored/title-style queries, or reduce rewritten variant BM25 weight further.
2. Lexical-first routing for title/ID/numeric queries:
   - Increase BM25 influence on anchored non-profile queries.
3. Boost retuning:
   - Narrow authority/recency/profile boosts so they cannot outweigh relevance on non-profile queries.
4. Re-run fixture and compare against current post-fix snapshot.

### Trigger for Path B (broader RAG revision)

- If targeted Path A changes do not reach both thresholds after one focused iteration:
  - `positive_hit_at_3 >= 0.55`
  - `positive_hit_at_1 >= 0.20`
- Then proceed with Path B plan (retriever routing/reranking redesign) as a separate phased track.

## Path A execution update (2026-03-26, same-day follow-up)

- Implemented in retrieval pipeline:
  - Anchored-query exact-only variant gate (skip rewrite variants for strongly anchored queries).
  - Anchored-query lexical-first fusion bias (`w_vec=0.4`, `w_bm25=0.6`).
- Post-change fixture rerun (`top_k=3`, `enable_variants=true`, `enable_mmr=true`):
  - `positive_hit_at_1=5/20` (`0.25`)
  - `positive_hit_at_3=12/20` (`0.60`)
  - `positive_clarify_count=0`
  - `control_with_results=0/3`
- Artifacts:
  - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json`
  - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.csv`
- Gate outcome:
  - Path A thresholds met (`hit@3 >= 0.55` and `hit@1 >= 0.20`), so Path B is not triggered at this stage.

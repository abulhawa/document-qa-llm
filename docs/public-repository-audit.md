# Public repository privacy audit

Audit date: 2026-07-31. This is a tracked-working-tree pattern review, not a guarantee that the repository or its history contains no sensitive data.

## Method and finding

`git grep` was used to inventory tracked evaluation files containing personal Windows home paths or corpus-revealing names (owner-name/CV/resume, tax-statement, receipt/provider, or personal-drive terms). Exactly **50 tracked files** matched that scope. No artifact was deleted and Git history was not rewritten.

The classification below is a recommendation requiring owner approval. Ignore rules only protect new untracked files; they do not sanitize these files or historical objects.

## Redact in place (3)

These human-authored plans contain useful technical reasoning. Replace personal paths and filenames with neutral examples while preserving conclusions:

- `docs/plans/document_qa_optimization_plan.md`
- `docs/plans/financial_tax_retrieval_plan.md`
- `docs/runbooks/qa_handoff_policy_and_q04_plan_2026-03-28.md`

## Replace with synthetic identifiers (3)

These active fixture/template files should retain their schemas and referential relationships. Replace personal filenames, checksums, and query references consistently with stable synthetic identifiers:

- `tests/fixtures/financial_eval_queries.json`
- `tests/fixtures/retrieval_eval_queries.json`
- `docs/runbooks/retrieval_eval_scoring_template.csv`

## Remove in a future normal commit (44)

These generated historical outputs repeat sensitive corpus metadata at scale. Per-occurrence redaction risks making snapshots internally inconsistent. After owner approval, retain a synthetic aggregate report if needed and remove:

- `docs/runbooks/financial_eval_2026-03-28.json`
- `docs/runbooks/q04_retrieval_stage_investigation_2026-03-28.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_dynamic.csv`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_dynamic.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top3.csv`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top3.json`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top5.csv`
- `docs/runbooks/qa_handoff_eval_2026-03-28_compare_clean_top5.json`
- `docs/runbooks/retrieval_eval_baseline_2026-03-26.csv`
- `docs/runbooks/retrieval_eval_baseline_2026-03-26.json`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_off.csv`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_off.json`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_on.csv`
- `docs/runbooks/retrieval_eval_cross_encoder_2026-03-28_hitn_compare_cross_on.json`
- `docs/runbooks/retrieval_eval_financial_rollout_2026-03-28_baseline_off.csv`
- `docs/runbooks/retrieval_eval_financial_rollout_2026-03-28_baseline_off.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.csv`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1_ranking_investigation.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1_residual_failure_analysis.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_v3.csv`
- `docs/runbooks/retrieval_eval_postfix_2026-03-26_v3.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_anchorfix_rerun.csv`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_anchorfix_rerun.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_anchorfix_rerun_ranking_investigation.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_anchorfix_rerun_ranking_investigation_strict_canonical_cleaned_residuals.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_candidate.csv`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_candidate.json`
- `docs/runbooks/retrieval_eval_postfix_2026-03-27_patha_v2_candidate_ranking_investigation.json`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_baseline_off.csv`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_baseline_off.json`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_planning_hyde_off.csv`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_planning_hyde_off.json`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_planning_off.csv`
- `docs/runbooks/retrieval_eval_query_planning_hit5_2026-03-28_compare_planning_off.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_off.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_off.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_on.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_compare_on.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_hitn_compare_off.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_hitn_compare_off.json`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_hitn_compare_on.csv`
- `docs/runbooks/retrieval_eval_sibling_expansion_2026-03-28_hitn_compare_on.json`
- `docs/runbooks/retrieval_investigation_p9_2026-03-26.json`

## Decisions required from the repository owner

1. Decide whether per-query historical outputs have enough public value to justify a reviewed synthetic replacement.
2. Approve redaction/replacement identifiers and rerun evaluations if the active fixtures change; old and synthetic metrics must not be presented as directly comparable without verification.
3. Decide separately whether already-published history needs a coordinated purge. A normal deletion does not remove historical objects, while a history rewrite disrupts clones and references.

No private keys or populated API credentials were identified by the earlier pattern scan. A dedicated history-aware secret scan and manual review are still recommended before publication.

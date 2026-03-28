# Financial Tax Retrieval Implementation Log (2026-03-28)

## Phase 1 - Schema Contract and Mapping Safety

### Scope completed
- Added additive config support for sidecar financial records index naming.
- Added non-destructive mapping ensure logic for:
  - document/chunk financial metadata fields
  - sidecar financial records index creation + mapping compatibility checks
- Wired financial records index settings into shared index ensure utility.

### Files changed
- `config.py`
- `utils/opensearch_utils.py`
- `utils/opensearch/indexes.py`
- `tests/test_financial_mappings.py`

### Commands executed
- `pytest tests\\test_opensearch_identity_mappings.py tests\\test_financial_mappings.py -q`

### Validation results
- `7 passed in 0.95s`

### Mappings/backfills/payload updates applied
- Runtime mapping/backfill operations: not applied (code-path and tests only in this phase).

### Residual risk before next phase
- Mapping compatibility for deeply customized pre-existing sidecar index subfield schemas is guarded by type checks, but only top-level field compatibility is enforced in v1.

## Phase 2 - Ingest Extraction Split (Metadata vs Records)

### Scope completed
- Added dedicated ingestion financial extractor module with:
  - deterministic date/amount/counterparty extraction
  - source-family classification
  - document-level financial metadata synthesis
  - conservative duplicate merge that preserves `source_links`
  - optional LLM fallback path for likely-financial ambiguous/missing-field cases
- Added sidecar financial-records store module with upsert and merge behavior.
- Integrated financial enrichment into ingestion orchestration:
  - lightweight document/chunk financial metadata attachment
  - sidecar normalized financial record upsert after fulltext indexing
  - non-finance ingestion remains functional when finance enrichment is unavailable.

### Files changed
- `ingestion/financial_extractor.py`
- `ingestion/financial_records_store.py`
- `ingestion/orchestrator.py`
- `tests/test_financial_extractor.py`
- `tests/test_financial_records_store.py`
- `tests/test_ingest_financial.py`

### Commands executed
- `pytest tests\\test_financial_extractor.py tests\\test_financial_records_store.py tests\\test_ingest_fulltext.py tests\\test_ingestion_extra.py tests\\test_ingest_financial.py -q`

### Validation results
- `16 passed in 4.41s`

### Mappings/backfills/payload updates applied
- Runtime backfill/payload-update operations: not applied in this phase.
- Ingest runtime now writes additive financial metadata + sidecar records when available.

### Residual risk before next phase
- LLM fallback path is present but intentionally conservative; real-world prompt quality and extraction breadth still depend on corpus formats and will be further validated in backfill/eval phases.

## Phase 3 - Backfill Existing Corpus (No Re-Embedding)

### Scope completed
- Added text-first financial backfill utility:
  - reads existing fulltext documents as authoritative source
  - extracts financial metadata + normalized records
  - updates fulltext/chunk financial metadata (additive, non-destructive controls)
  - updates Qdrant payloads with lightweight routing/filter fields only
  - supports dry-run/apply, overwrite, doc-type and source-family cohort filters
- Added focused unit tests for backfill behavior and counters.
- Updated ingestion package init to lazy-load orchestrator to keep script imports lightweight and stable.

### Files changed
- `scripts/backfill_financial_metadata.py`
- `tests/test_backfill_financial_metadata.py`
- `tests/test_backfill_identity_metadata.py`
- `ingestion/__init__.py`

### Commands executed
- `pytest tests\\test_backfill_financial_metadata.py tests\\test_backfill_identity_metadata.py -q`
- `python scripts\\backfill_financial_metadata.py --dry-run --limit 5 --batch-size 50`
- `python scripts\\backfill_financial_metadata.py --dry-run --batch-size 200` (timed out on broad sweep attempt)

### Validation results
- Tests: `6 passed in 0.20s`
- Canary dry-run output:
  - `scanned_fulltext_docs=5`
  - `processed_docs=4`
  - `fulltext_would_update=4`
  - `chunk_would_update_calls=4`
  - `qdrant_payload_would_update_points=29`
  - `errors=0`

### Mappings/backfills/payload updates applied
- Live mapping updates applied during canary execution:
  - added financial metadata fields to `documents` and `documents_full_text`
  - created sidecar index `financial_records`
- Live data backfill apply: not executed in this runbook phase due stability constraints during broad-run attempt.

### Residual risk before next phase
- Full broad backfill apply remains pending; only canary dry-run was completed successfully before system instability surfaced during broader execution.

## Phase 4 - Financial Query Class + Retrieval Gating

### Scope completed
- Added explicit finance-query classification with planner outputs:
  - `financial_query_mode`
  - `target_entity`
  - `target_year`
  - `target_concept`
- Wired finance intent into query planning and retrieval types.
- Added finance retrieval gating path with:
  - preferred finance/admin family prioritization
  - suppressed family removal
  - year-aware filtering using `mentioned_years`, `tax_years_referenced`, `document_date`, and `transaction_dates`
  - deterministic fallback ladder with stage metadata
  - residual-budget-only fallback contribution policy
- Preserved non-finance retrieval baseline path behind `financial_query_mode` and `financial_enable_gating`.
- Added compatibility hardening for lightweight test/runtime environments:
  - lazy `qa_pipeline.answer_question` import in package init
  - lazy `core.query_rewriter` LLM import
  - retrieval output metadata fallback compatibility (`stage_metadata` optional).

### Files changed
- `core/financial_query.py`
- `core/query_rewriter.py`
- `core/retrieval/pipeline.py`
- `core/retrieval/types.py`
- `qa_pipeline/__init__.py`
- `qa_pipeline/coordinator.py`
- `qa_pipeline/retrieve.py`
- `qa_pipeline/types.py`
- `tests/test_financial_query.py`
- `tests/test_retrieval_pipeline.py`
- `tests/test_query.py`

### Commands executed
- `pytest tests\\test_financial_query.py tests\\test_retrieval_pipeline.py tests\\test_query.py tests\\test_prompt_builder.py -q`

### Validation results
- `76 passed in 0.30s`

### Mappings/backfills/payload updates applied
- None in this phase (retrieval/query-path logic only).

### Residual risk before next phase
- Source-family inference remains heuristic for unknown doc types when metadata is sparse.
- Year filtering quality depends on metadata/backfill completeness from Phase 2/3.

## Phase 5 - Evidence-First Answer Contract

### Scope completed
- Added sidecar financial record retrieval wrapper for answer synthesis.
- Added evidence-first finance answer builder with candidate buckets:
  - clearly supported year-scoped expenses/payments
  - possible but ambiguous items
  - mentioned but not confirmed as paid in target year
- Enforced honesty and confidence policies:
  - no tax/deductibility assertions without direct evidence
  - low-confidence/no-evidence records remain ambiguous
  - cross-year records moved to not-confirmed bucket
- Added fallback disclosures:
  - retrieval fallback-stage disclosure
  - normalized sidecar coverage incomplete disclosure
  - chunk/document fallback synthesis when sidecar coverage is partial/absent
- Wired finance answer short-circuit into coordinator when finance retrieval mode is active.
- Extended response/context schema with additive `financial_answer_metadata`.

### Files changed
- `core/financial_records.py`
- `qa_pipeline/financial_answer.py`
- `qa_pipeline/coordinator.py`
- `qa_pipeline/types.py`
- `app/schemas.py`
- `app/usecases/qa_usecase.py`
- `tests/test_financial_answer.py`
- `tests/test_query.py`

### Commands executed
- `pytest tests\\test_financial_query.py tests\\test_financial_answer.py tests\\test_retrieval_pipeline.py tests\\test_query.py tests\\test_prompt_builder.py -q`

### Validation results
- `80 passed in 0.36s`

### Mappings/backfills/payload updates applied
- None in this phase.

### Residual risk before next phase
- Answer completeness remains bounded by sidecar record coverage until broad Phase 3 apply backfill is completed.

## Phase 6 - Validation and Rollout Gates

### Scope completed
- Added finance benchmark fixture with hand-labeled synthetic finance retrieval cases:
  - suppression-vs-preferred-family conflicts
  - year-scoped queries
  - fallback ladder activation
  - no-entity finance query behavior
- Added finance benchmark runner:
  - baseline (`financial_enable_gating=false`) vs gated (`true`) comparison
  - explicit gate checks for suppressed-source removal, preferred-family dominance, year leakage, and fallback logging
- Added benchmark regression test for finance eval harness.
- Tightened retrieval gating implementation:
  - explicit doc-type-to-family mapping (including `book`, `publication`, `reference`, `course_material`, etc.)
  - prevent year-relaxation fallback when strict primary budget is already met
  - exposed `year_relaxation_allowed` in stage metadata
- Added fallback policy regression test.

### Files changed
- `scripts/run_financial_eval.py`
- `tests/fixtures/financial_eval_queries.json`
- `tests/test_run_financial_eval.py`
- `core/retrieval/pipeline.py`
- `tests/test_retrieval_pipeline.py`
- `docs/runbooks/financial_eval_2026-03-28.json`

### Commands executed
- `pytest tests\\test_run_financial_eval.py tests\\test_financial_query.py tests\\test_financial_answer.py tests\\test_retrieval_pipeline.py tests\\test_query.py tests\\test_prompt_builder.py -q`
- `pytest tests\\test_backfill_financial_metadata.py tests\\test_backfill_identity_metadata.py tests\\test_run_financial_eval.py tests\\test_financial_query.py tests\\test_financial_answer.py tests\\test_retrieval_pipeline.py tests\\test_query.py tests\\test_prompt_builder.py -q`
- `python scripts\\run_financial_eval.py --output docs/runbooks/financial_eval_2026-03-28.json`

### Validation results
- Focused suite: `82 passed in 0.42s`
- Expanded finance/regression suite: `88 passed in 0.49s`
- Finance benchmark gates: all pass.

### Benchmark summary (from `financial_eval_2026-03-28.json`)
- Baseline (`financial_enable_gating=false`):
  - `suppressed_docs_topk_total=4`
  - `avg_preferred_ratio_topk=0.5667`
  - `year_leakage_docs_topk_total=3`
- Gated (`financial_enable_gating=true`):
  - `suppressed_docs_topk_total=0`
  - `avg_preferred_ratio_topk=1.0`
  - `year_leakage_docs_topk_total=3`
  - `fallback_logged_count=4/4`
- Deltas:
  - suppressed docs in top-k: `-4`
  - preferred-family ratio: `+0.4333`
  - year leakage docs in top-k: `0` (not worse)

### Mappings/backfills/payload updates applied
- No new mapping/backfill/payload write operations in this phase.

### Residual risk before next phase
- Benchmark harness is synthetic/offline to remain deterministic in this environment; live-index full-corpus finance eval remains recommended once dependency-complete runtime is available.
- Broad Phase 3 apply backfill is still pending from earlier stability constraints.

## Phase 7 - Deferred Track (OCR/Image)

### Scope completed
- Kept OCR/image ingestion deferred, unchanged, and out-of-scope as required.

### Files changed
- None.

### Commands executed
- None.

### Validation results
- N/A (deferred by plan).

### Mappings/backfills/payload updates applied
- None.

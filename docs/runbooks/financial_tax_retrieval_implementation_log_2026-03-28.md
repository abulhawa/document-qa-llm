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

## Operational Closure Pass (2026-03-28, post-Phase 7)

### Scope completed
- Executed broad Phase 3 apply backfill on live corpus (not dry-run), with controlled batches, retries, and a final full-pass apply.
- Added backfill operational safety improvements:
  - `--skip` cursor support for batch windows.
  - `--scroll-keepalive` control and limit-aware scroll termination to avoid expired scroll-context crashes.
- Added live-backend mode to finance eval runner and executed live corpus benchmark on finance/tax query cohort.
- Executed full-corpus retrieval benchmark on live backends after broad backfill.
- Standardized reproducible operational/test execution path through Docker `celery` image with repo bind mount.

### Files changed
- `scripts/backfill_financial_metadata.py`
- `scripts/run_financial_eval.py`
- `tests/test_backfill_financial_metadata.py`
- `tests/test_run_financial_eval.py`
- `docs/runbooks/financial_eval_live_2026-03-28.json`
- `docs/runbooks/retrieval_eval_financial_rollout_2026-03-28_baseline_off.json`
- `docs/runbooks/retrieval_eval_financial_rollout_2026-03-28_baseline_off.csv`

### Commands executed (operational)
- Canary apply:
  - `python scripts\backfill_financial_metadata.py --limit 50 --batch-size 50`
- Controlled broad apply windows:
  - `python scripts\backfill_financial_metadata.py --skip 50 --limit 400 --batch-size 200`
  - `python scripts\backfill_financial_metadata.py --skip 450 --limit 400 --batch-size 200`
  - `python scripts\backfill_financial_metadata.py --skip 850 --limit 400 --batch-size 200`
  - `python scripts\backfill_financial_metadata.py --skip 1250 --limit 400 --batch-size 200`
  - `python scripts\backfill_financial_metadata.py --skip 1650 --limit 400 --batch-size 200`
  - `python scripts\backfill_financial_metadata.py --skip 2050 --limit 400 --batch-size 200`
- Retry after scroll-context failures:
  - `python scripts\backfill_financial_metadata.py --skip 850 --limit 400 --batch-size 200`
- Full broad apply (final):
  - `python scripts\backfill_financial_metadata.py --batch-size 200 --scroll-keepalive 30m`
- Live finance eval:
  - `python scripts\run_financial_eval.py --mode live --live-fixture tests/fixtures/retrieval_eval_queries.json --live-target-areas tax_docs finance_docs --live-top-k 5 --live-fallback-budget 2 --output docs/runbooks/financial_eval_live_2026-03-28.json`
- Full-corpus live benchmark:
  - `python scripts\run_retrieval_eval.py --fixture tests/fixtures/retrieval_eval_queries.json --support-labels tests/fixtures/retrieval_eval_answer_support_labels.json --output docs/runbooks/retrieval_eval_financial_rollout_2026-03-28.json --sibling-expansion-mode off --query-planning-mode baseline`

### Commands executed (reproducible Docker runtime)
- Runtime probe:
  - `docker compose ps`
  - `docker compose exec -T celery python -c "import langchain_community,opensearchpy,qdrant_client; print('deps-ok')"`
- Reproducible bind-mounted execution pattern (current repo + container deps):
  - `docker compose run --rm --no-deps -T -v "${PWD}:/app" celery sh -lc "<bootstrap && command>"`
  - helper wrapper added: `scripts/docker_celery_repo_exec.ps1`
    - example: `.\scripts\docker_celery_repo_exec.ps1 -RunCmd "python -m pytest tests/test_run_financial_eval.py -q"`
- Final regression sweep (containerized):
  - `docker compose run --rm --no-deps -T -v "${PWD}:/app" celery sh -lc "python -m pip install -q pytest opentelemetry-api opentelemetry-sdk arize-phoenix-otel && python -m pytest tests/test_financial_mappings.py tests/test_financial_extractor.py tests/test_financial_records_store.py tests/test_ingest_financial.py tests/test_backfill_financial_metadata.py tests/test_backfill_identity_metadata.py tests/test_financial_query.py tests/test_financial_answer.py tests/test_retrieval_pipeline.py tests/test_query.py tests/test_prompt_builder.py tests/test_run_financial_eval.py -q"`
- Live eval and full benchmark re-run from containerized bind-mounted path:
  - `docker compose run --rm --no-deps -T -v "${PWD}:/app" celery sh -lc "python -m pip install -q opentelemetry-api opentelemetry-sdk arize-phoenix-otel && python scripts/run_financial_eval.py --mode live --live-fixture tests/fixtures/retrieval_eval_queries.json --live-target-areas tax_docs finance_docs --live-top-k 5 --live-fallback-budget 2 --output docs/runbooks/financial_eval_live_2026-03-28.json"`
  - `docker compose run --rm --no-deps -T -v "${PWD}:/app" celery sh -lc "python -m pip install -q opentelemetry-api opentelemetry-sdk arize-phoenix-otel && python scripts/run_retrieval_eval.py --fixture tests/fixtures/retrieval_eval_queries.json --support-labels tests/fixtures/retrieval_eval_answer_support_labels.json --output docs/runbooks/retrieval_eval_financial_rollout_2026-03-28.json --sibling-expansion-mode off --query-planning-mode baseline"`

### Backfill apply evidence
- Canary apply (`--limit 50`):
  - `scanned_fulltext_docs=50`
  - `processed_docs=47`
  - `fulltext_updates=47`
  - `chunk_update_calls=47`
  - `chunk_docs_updated=2437`
  - `records_extracted=193`
  - `sidecar_records_created=193`
  - `errors=0`
- Controlled windows:
  - `skip=50 limit=400`: `processed_docs=372`, `fulltext_updates=372`, `chunk_docs_updated=11425`, `sidecar_created=335`, `sidecar_updated=1`, `errors=0`
  - `skip=450 limit=400`: `processed_docs=365`, `fulltext_updates=365`, `chunk_docs_updated=21517`, `sidecar_created=633`, `sidecar_updated=17`, `errors=0`
  - `skip=850 limit=400`: initial run failed with `NotFoundError ... No search context found` after long processing window; retried after script hardening
  - `skip=1250 limit=400`: `processed_docs=374`, `fulltext_updates=374`, `chunk_docs_updated=8244`, `sidecar_created=662`, `sidecar_updated=101`, `errors=0`
  - `skip=1650 limit=400`: initial run failed with `NotFoundError ... No search context found` after long processing window
  - `skip=2050 limit=400`: `processed_docs=86`, `fulltext_updates=86`, `chunk_docs_updated=1276`, `sidecar_created=89`, `sidecar_updated=13`, `errors=0`
  - Retry `skip=850 limit=400` after patch: `processed_docs=142`, `fulltext_updates=142`, `chunk_docs_updated=3770`, `sidecar_created=0`, `sidecar_updated=332`, `errors=0`
- Final full apply (host runtime):
  - `scanned_fulltext_docs=2142`
  - `processed_docs=1982`
  - `skipped_no_fulltext_text=160`
  - `fulltext_updates=1982`
  - `chunk_update_calls=1982`
  - `chunk_docs_updated=84516`
  - `records_extracted=3647`
  - `sidecar_records_created=169`
  - `sidecar_records_updated=3478`
  - `errors=0`
- Final full apply (containerized bind-mounted runtime):
  - `scanned_fulltext_docs=2142`
  - `processed_docs=1982`
  - `skipped_no_fulltext_text=160`
  - `fulltext_updates=1982`
  - `chunk_update_calls=1982`
  - `chunk_docs_updated=84516`
  - `records_extracted=3647`
  - `sidecar_records_created=0`
  - `sidecar_records_updated=3647`
  - `errors=0`

### No re-embedding + lightweight payload verification
- Qdrant point cardinality remained stable during migration (`qdrant_points_count=84516`), indicating payload updates only.
- Chunk/OpenSearch update path uses additive metadata fields only; no vector generation APIs are called by backfill.
- Qdrant payload sample confirms only lightweight routing/filter metadata fields were added (e.g., `financial_metadata_version`, `financial_metadata_source`, `financial_record_type`, `is_financial_document`, date/year/amount/counterparty fields) and no full transaction arrays were stamped (`transactions`/`financial_records` absent in payload sample).
- Sidecar normalized records are stored in `financial_records` index (count after rollout: `3363`), preserving separation of concerns.

### Live finance eval (real backends) results
- Source: `tests/fixtures/retrieval_eval_queries.json` filtered by `target_areas in {tax_docs, finance_docs}` -> `query_ids=["Q18","Q19","Q20"]`.
- Baseline (`financial_enable_gating=false`):
  - `suppressed_docs_topk_total=8`
  - `avg_preferred_ratio_topk=0.2`
  - `year_leakage_docs_topk_total=3`
  - `fallback_used_count=0`
- Gated (`financial_enable_gating=true`):
  - `suppressed_docs_topk_total=0`
  - `avg_preferred_ratio_topk=1.0`
  - `year_leakage_docs_topk_total=2`
  - `fallback_used_count=1`
  - `fallback_logged_count=3/3`
- Deltas:
  - suppressed family leakage: `-8`
  - preferred-family ratio: `+0.8`
  - year leakage docs: `-1`
- Gates: all pass.

### Full-corpus live benchmark (post-backfill)
- Output: `docs/runbooks/retrieval_eval_financial_rollout_2026-03-28_baseline_off.json` + `.csv`
- Summary:
  - `positive_hit_at_1_rate=0.45`
  - `positive_hit_at_3_rate=0.8`
  - `positive_support_hit_at_1_rate=0.5`
  - `positive_support_hit_at_3_rate=0.8`
  - `control_with_results=0` (out-of-corpus controls stayed suppressed)
  - `queries_with_errors=0`
  - `queries_exceeding_soft_timeout=0`

### Final regression sweep
- Containerized finance/regression suite result: `101 passed in 5.40s`.
- Focused updated-file suite result (host runtime): `6 passed in 2.65s`.
- Backfill unit suite after scroll hardening: `3 passed in 0.07s`.

### Mappings/backfills/payload updates applied
- Applied live additive financial mappings/index ensures during backfill runs.
- Executed broad apply updates on:
  - `documents_full_text` financial metadata fields
  - `documents` (chunk index) financial metadata fields
  - `financial_records` sidecar index records
  - Qdrant `document_chunks` payload metadata fields
- No index deletion, no volume deletion, no source-file re-read migration path, and no re-embedding migration step were executed.

### Residual risk and rollout status
- Residual risk:
  - Deterministic extractor can still over-interpret noisy numeric/date spans in some documents (example payload sample shows an anomalous historical date token); this affects metadata quality, not migration safety.
  - `qdrant_payload_updated_points` counter remains conservative because Qdrant set-payload response does not always return per-operation count; verification is done via payload field counts and sample inspection.
- Rollout status: **rollout-ready for text-first financial retrieval scope** under current plan constraints.
  - Broad apply backfill: completed.
  - Live finance eval: completed with gate improvements.
  - Full-corpus benchmark: completed.
  - Final regression sweep in reproducible Docker-backed execution path: completed.

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

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

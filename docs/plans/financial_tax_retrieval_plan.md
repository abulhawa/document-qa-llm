# Financial Query Reliability Plan (Text-First, OCR-Later)

Last updated: 2026-03-28
Status: Proposed and saved for phased execution

## Summary

This plan upgrades the text-based ingestion and QA pipeline so the app can answer year-scoped finance/tax evidence questions such as:

"What expenses did Ali make in 2022 that can help in his tax returns for that year?"

The implementation starts with metadata + evidence extraction for already indexed text documents (`pdf`, `docx`, `txt`). OCR/image ingestion remains deferred and is not the first step.

This plan is intentionally structured as the first domain-specific implementation on top of a reusable enrichment and retrieval-policy framework. Finance/tax is the initial slice because it is the current highest-value use case, but the underlying architecture should be reusable for future domains such as profile/CV, contracts, travel, or other evidence-aggregation queries.

## Locked Decisions

1. Extractor mode: Hybrid (deterministic first, LLM fallback only when needed).
2. `potentially_deductible` is not a primary ingest-truth field and is not a strict retrieval prerequisite.
3. Data model is split into two layers:
4. Document-level financial metadata for filtering/routing.
5. Normalized financial records for evidence-level answering and aggregation.
6. Retrieval mode: strict+fallback with explicit financial-family gating and source-family suppression.
7. Answer contract: evidence-first candidate buckets, not legal-judgment buckets.
8. Answer honesty rule: never assert tax relevance/deductibility without supporting evidence; otherwise mark as tentative/unclear.
9. Core infrastructure should be generic where it governs orchestration, routing, provenance, storage pattern, and fallback behavior.
10. Finance/tax remains the only implemented domain schema in v1; future domains should plug into the same framework rather than trigger a full redesign.
11. Fallback retrieval uses residual-budget contribution only; strict pass owns the primary candidate budget.
12. Confidence-to-bucket assignment is explicit and conservative; duplicate merging must not auto-inflate confidence.

## Reusable Core vs Domain-Specific Scope

### Reusable core infrastructure to build now

1. Document enrichment pipeline interfaces.
2. Document/domain classification hook.
3. Sidecar evidence-record storage pattern.
4. Query-class routing mechanism.
5. Retrieval policy framework with strict pass, fallback ladder, and traceable stage metadata.
6. Answer-contract framework.
7. Provenance, confidence, and version/source handling.

### Finance-specific implementation in v1

1. Finance document/source families.
2. Document-level financial metadata fields.
3. Normalized financial-record schema.
4. Finance-query planner outputs and retrieval gating rules.
5. Finance/tax evidence answer buckets.

### Explicit non-goal

Do not attempt a universal semantic metadata schema that tries to serve finance, CVs, contracts, travel, and other future domains with one flat field set. Reuse the framework, not a fake one-size-fits-all meaning layer.

## Implementation Safety Constraints

1. All new finance/tax-specific retrieval and answer behavior must activate only when `financial_query_mode=true`.
2. Existing non-finance QA behavior must remain unchanged by default.
3. New metadata fields are additive only; do not rename or repurpose existing fields.
4. Sidecar financial records must not become a required dependency for unrelated queries.
5. Preserve the current retrieval path as an explicit baseline path for ablation and regression comparison.
6. Roll out in phases exactly as planned; do not implement retrieval gating or finance answer bucketing before schema, ingest extraction, and backfill are in place.
7. Add regression tests proving non-finance queries still use legacy behavior and keep unchanged routing and answer contracts.
8. Benchmark before and after each major phase, especially before enabling Phase 4 retrieval gating broadly.
9. For text-first phases, existing fulltext documents are the authoritative source for enrichment backfill.
10. During initial migration, do not re-read original files for text extraction unless a document is missing or known to have incomplete indexed text.
11. Fulltext docs are enriched in place, chunks receive only lightweight additive routing/filter fields, and normalized financial records are created in the sidecar store.

## Data Model and Storage Decisions

### Document-level financial metadata (fulltext + chunk payloads)

1. `doc_type`
2. `is_financial_document`
3. `document_date`
4. `mentioned_years` or `content_years`
5. `transaction_dates` (document-level summary when derivable)
6. `amounts` (document-level summary)
7. `counterparties` (document-level summary)
8. Optional soft signals: `tax_relevance_signals`, `expense_category`, `financial_record_type`
9. Provenance/version fields: `financial_metadata_version`, `financial_metadata_source`

### Normalized financial-record layer (sidecar evidence store)

1. `record_type`
2. `date`
3. `amount`
4. `currency`
5. `counterparty`
6. `description`
7. `confidence`
8. `document_id`
9. `checksum`
10. `chunk_id` or span reference
11. `extraction_method` (deterministic, llm, hybrid)
12. `source_text_span` or equivalent evidence linkage for debugging and traceability
13. `source_links` list for merged duplicates, preserving all contributing sources

### Storage split (explicit)

1. Chunk payloads store lightweight filter/routing metadata only.
2. Fulltext docs store document-level financial metadata summaries.
3. Extracted transaction-like records live in a sidecar financial-record index/table linked by `document_id` and `chunk_id`/span.
4. Do not stamp full transaction arrays onto every chunk payload.

### v1 source-family cohorts for finance-query gating

1. Preferred strict-gating families:
2. `tax_document`, `bank_statement`, `receipt`, `invoice`, `payment_confirmation`, `school_fee_letter`, `official_letter`
3. Suppressed families:
4. `book`, `course_material`, `publication`, `cv`, `reference`, `archive_misc`

## Phase Plan

### Phase 1: Schema Contract and Mapping Safety

1. Define and freeze v1 document-level financial metadata schema with richer date fields:
2. `document_date`, `mentioned_years`, `transaction_dates`, optional `tax_years_referenced`.
3. Define and freeze v1 sidecar normalized financial-record schema.
4. Add non-destructive OpenSearch mapping ensure function for document-level fields.
5. Add sidecar index mapping setup for normalized records.
6. Keep the existing safety model: add missing mappings only, fail on incompatible types.

Exit criteria:

1. Mapping/index ensure functions are idempotent.
2. No ingest behavior change yet.

### Phase 2: Ingest Extraction Split (Metadata vs Records)

1. Add a dedicated financial extraction module under ingestion boundaries.
2. Hybrid extraction policy:
3. Deterministic parsing first for dates, amounts, currencies, counterparties.
4. LLM fallback only when key fields are missing/low confidence.
5. Conflict policy: deterministic values win for numeric/date conflicts; LLM fills semantic gaps.
6. Persist outputs separately:
7. Document-level metadata to fulltext + chunk payloads.
8. Normalized financial records to sidecar store with document/chunk linkage.
9. Add duplicate-merge policy for repeated financial records across chunks/documents:
10. Merge repeated records into one canonical record.
11. Preserve all contributing evidence links (`document_id`, `chunk_id`, `source_text_span`, `extraction_method`, `confidence`) in `source_links`.
12. Duplicate merge confidence policy is conservative: merged confidence cannot exceed the highest supported source confidence unless independently re-scored.

Exit criteria:

1. New ingests produce both metadata layer and sidecar evidence layer.
2. Existing identity metadata behavior remains unchanged.

### Phase 3: Backfill Existing Corpus (No Re-Embedding)

1. Add backfill scripts patterned after existing metadata/backfill utilities.
2. Use existing fulltext docs as the authoritative text source for backfill enrichment.
3. Do not re-read source files during initial migration except for missing fulltext docs or known incomplete indexed text.
4. Backfill document-level financial metadata into fulltext and chunks.
5. Run extraction backfill to populate sidecar normalized financial records for existing docs.
6. Update Qdrant payloads only with lightweight document-level filter fields.
7. Include dry-run/apply, cohort targeting, and overwrite controls.

Exit criteria:

1. Historical docs receive document-level metadata without re-embedding.
2. Sidecar financial records are populated for targeted historical cohorts.
3. Backfill logs include processed/updated/skipped/error counters.

### Phase 4: Financial Query Class + Retrieval Gating

1. Add explicit finance-query planner outputs:
2. `financial_query_mode` (boolean)
3. `target_entity`
4. `target_year`
5. `target_concept` (for example expenses/payments/tax-relevant items)
6. If `financial_query_mode=true` but no explicit `target_entity` is found, do not require entity filtering.
7. In that case, use source-family + year + concept constraints and treat entity cues only as soft ranking features when available.
8. Add strict retrieval pass for finance/tax evidence queries:
9. Strongly prefer or restrict to preferred financial/admin families in v1 cohorts.
10. Apply explicit suppression for v1 suppressed families.
11. Apply year-aware filtering using richer date model (`document_date`, `mentioned_years`, `transaction_dates`, optional `tax_years_referenced`).
12. Do not require deductible boolean flags in strict pass.
13. Controlled fallback ladder:
14. Relax family restriction gradually.
15. Relax year constraints gradually.
16. Always log and surface fallback stage.
17. Keep source-family suppression active so clearly irrelevant academic/reference/archive docs are pushed out for finance/tax evidence queries.
18. Candidate-budget policy:
19. Strict pass fills the primary candidate budget first.
20. Fallback stages may only contribute a capped residual budget.
21. Suppressed families remain suppressed during fallback unless explicitly overridden for investigation mode.

Exit criteria:

1. Finance/tax queries are handled as an explicit query class.
2. Strict pass minimizes cross-domain junk sources.
3. Fallback behavior is deterministic and traceable.

### Phase 5: Evidence-First Answer Contract

1. Replace legal-judgment buckets with evidence-first candidate buckets:
2. Clearly supported 2022 expenses.
3. Possible but ambiguous 2022 expenses.
4. Mentioned items not confirmed as paid in 2022.
5. For each item, include evidence fields where available:
6. Date, amount, currency, counterparty, source.
7. Optional note per item: possible tax relevance and why unclear.
8. If fallback retrieval was used, include fallback disclosure in answer metadata/context.
9. Honesty rule in answer synthesis:
10. Do not claim tax relevance or deductibility unless evidence supports it.
11. Otherwise label the statement as tentative/unclear.
12. Confidence-to-bucket policy:
13. `clearly supported 2022 expenses` requires direct evidence linkage and at least medium confidence.
14. Low-confidence records may only appear in `possible but ambiguous 2022 expenses`.
15. Duplicate merging must not inflate confidence automatically.

Exit criteria:

1. Responses are evidence-grounded and aggregation-friendly.
2. "I don't know" behavior remains intact when evidence is insufficient.

### Phase 6: Validation and Rollout Gates

1. Add focused tests for:
2. Document-level schema extraction and merge policy.
3. Sidecar normalized record extraction and linkage integrity.
4. Duplicate-merge behavior for repeated records while retaining all source links.
5. Mapping safety and backfill dry-run/apply behavior.
6. Finance query-class parsing outputs, including no-entity behavior when `financial_query_mode=true`.
7. Retrieval family gating, fallback ladder, and suppression behavior.
8. Prompt output conformance to evidence-first buckets and honesty rule.
9. Add canary benchmark criteria beyond generic accuracy:
10. Top-k finance query results are dominated by preferred financial/admin families.
11. Irrelevant academic/reference/archive sources are sharply reduced.
12. For finance evidence queries, irrelevant non-financial docs should not appear in top-5, ideally not top-10.
13. Year-scoped queries show reduced cross-year leakage.
14. Fallback usage is logged, reviewable, and explainable.
15. Rollout order:
16. Phase 1 and Phase 2.
17. Phase 3 canary then full.
18. Phase 4 and Phase 5.
19. Final regression sweep.

Exit criteria:

1. Targeted tests pass.
2. Canary metrics meet junk-source and year-leakage gates.
3. No regression in existing retrieval quality checks.

### Phase 7: Deferred Track (Not First): OCR and Image Ingestion

1. OCR pipeline remains out of first execution sequence.
2. OCR starts only after text-first phases are stable and benchmarked.

## Public Interface and Type Impact

1. Retrieval context should expose document-level financial metadata fields needed for filtering and answer rendering.
2. Finance-query planner output should surface `financial_query_mode`, `target_entity`, `target_year`, and `target_concept`.
3. Retrieval internals should carry applied filter/fallback stage metadata.
4. Sidecar financial record retrieval APIs should be introduced without breaking existing retrieval interfaces.
5. Core interface names and extension points should remain domain-neutral where possible so future domains can add their own schemas without rewriting the framework.

## Assumptions

1. Corpus quality can improve materially via text-first evidence extraction before OCR.
2. Existing ingestion/backfill patterns in this repository remain the baseline implementation style.
3. Financial evidence records and document-level metadata are treated as separate layers by design.

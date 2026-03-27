# document_qa Optimization Plan (v4)

Last updated: 2026-03-27 (post-Path-A residual + ranking investigation)
Purpose: Convert the external draft into an execution-ready plan for this repository.

## 1. Goals

- Reduce hallucinated answers in QA responses.
- Reduce duplicate and near-duplicate retrieval noise.
- Reduce identity mixing across similar CV/resume documents.
- Keep blast radius low and preserve existing behavior unless explicitly changed.

## 2. Non-goals for first milestone

- No broad refactor of retrieval or ingestion architecture.
- No default provider migration in the same PR as QA quality fixes.
- No forced data migration (re-ingest) until quality fixes are validated.
- No new Gradio UI work; continue on Streamlit UI only.

## 3. Corrections to the original draft

- Path correction: use `qa_pipeline/coordinator.py` (not root `coordinator.py`).
- Avoid "replace entire file" instructions; use focused edits only.
- Keep Groq migration separate from retrieval/prompt quality fixes.
- Treat chunk size changes as a migration step, not a quick default flip.

## 4. Baseline and acceptance criteria

Run baseline before any code changes:

```powershell
pytest tests/test_llm_module.py tests/test_llm_module_extra.py tests/test_retrieval_pipeline.py tests/test_search_dedup.py -q
```

Define a small manual QA set (5-10 questions) from known docs and record:

- Grounded answer rate (answer can be traced to retrieved chunks).
- "Not found" behavior quality for out-of-scope questions.
- Duplicate-source rate in top-k retrieval.
- Checksum hit-rate on the standardized fixture:
  - `tests/fixtures/retrieval_eval_queries.json`
  - `docs/runbooks/retrieval_eval_scoring_template.csv`

Milestone is successful when:

- Existing targeted tests pass.
- New regression tests for each changed behavior pass.
- Manual QA set shows lower hallucination and fewer duplicate sources.

Baseline snapshot (2026-03-26, checksum fixture run):

- Retrieval run mode: `top_k=3`, `enable_variants=true`, `enable_mmr=true`.
- Source fixture: `tests/fixtures/retrieval_eval_queries.json` (23 queries; 20 positive, 3 control).
- Results:
  - `positive_hit_at_1=2/20` (`0.10`)
  - `positive_hit_at_3=5/20` (`0.25`)
  - `positive_clarify_count=2`
  - `control_with_results=2/3` and `control_clarify_count=1/3`
- Artifacts:
  - `docs/runbooks/retrieval_eval_baseline_2026-03-26.json`
  - `docs/runbooks/retrieval_eval_baseline_2026-03-26.csv`

## 5. Implementation plan (phased, low blast radius)

### Current execution order (updated 2026-03-26)

- P0 is complete in code.
- P4 is complete in code.
- P1 is complete in code with targeted regression tests passing.
- P2 is complete in code; one-time metadata backfill has been executed on the existing corpus.
- P3 is complete in code with targeted regression tests passing.
- P5 code changes are complete (`800/100` defaults + migration runbook); full operational re-ingest rollout is pending.
- P6 dynamic chunking policy is proposed (not yet implemented).
- Retrieval evaluation fixture and checksum-based scoring template are committed.
- P7 doc-type coverage expansion is complete in code with targeted `__missing__/other` cohort backfill executed.
- Query-rewriter anchor fallback, bounded abstention gate, and OpenSearch chunk-text fetch request fix are complete in code.
- Post-fix checksum baseline rerun is complete and archived.
- P9 full retrieval investigation + RAG revision decision gate is complete and archived.
- Path A retrieval tuning from the P9 decision memo is complete in code (anchored exact-only variants + anchored lexical-first fusion) with fixture rerun archived.
- P8 OCR quality/cost track is proposed (not yet implemented).
- Manual QA gates for P1 and P2 have passed.
- Active sequence: `P0 -> P4 -> P1 -> P2 -> P3 -> P5 (rollout) -> P6 -> P7 -> post-fix eval baseline -> P9 -> Path A follow-up -> P8`.
- Retrieval scoring now includes bounded authority and bounded recency boosts.
- Guardrail unchanged: keep P4 isolated from retrieval/prompt quality changes.

---

### Immediate actions from baseline (2026-03-26)

Rationale:

- Baseline fixture quality is currently low (`hit@1=0.10`, `hit@3=0.25`).
- Positive queries still trigger clarifications (`2/20`).
- Out-of-corpus control queries still return corpus docs (`2/3`).
- Runtime warning observed repeatedly during retrieval pass:
  - `OpenSearch unavailable while fetching chunk text; proceeding with Qdrant payloads.`

Next actions (short-horizon, before OCR rollout):

1. P7 deterministic doc-type expansion + targeted backfill (`__missing__`, `other` cohorts).
2. Query-rewriter tightening:
   - Reduce over-clarification for anchored queries.
   - Add fallback path: if rewrite asks for clarification but query has strong anchors, run exact retrieval anyway.
3. Add bounded abstention gate for out-of-corpus queries so control questions do not surface arbitrary docs.
4. Investigate and fix the recurrent OpenSearch chunk-text warning path.
5. Re-run the same checksum baseline and compare deltas.
6. Start P8 OCR canary only after post-fix baseline shows measurable retrieval gains.

Execution update (2026-03-26, post-fix run):

- Completed:
  - P7 deterministic classifier expansion + targeted backfill on `doc_type in {__missing__, other}`.
  - Query-rewriter tightening with anchored-clarify fallback.
  - Bounded abstention gate for out-of-corpus/live-intent controls.
  - OpenSearch chunk-text fetch fix (`mget` payload corrected; warning-noise path removed at source).
  - Post-fix checksum baseline rerun.
- Post-fix retrieval snapshot (`top_k=3`, `enable_variants=true`, `enable_mmr=true`):
  - `positive_hit_at_1=2/20` (`0.10`)
  - `positive_hit_at_3=8/20` (`0.40`)
  - `positive_clarify_count=0`
  - `control_with_results=0/3`
  - `control_clarify_count=0/3`
- Artifacts:
  - `docs/runbooks/retrieval_eval_postfix_2026-03-26_v3.json`
  - `docs/runbooks/retrieval_eval_postfix_2026-03-26_v3.csv`
- Decision for next step:
  - Proceed to P8 OCR canary planning and implementation, with retrieval gates monitored against the archived Path A fixture snapshot.

---

### Phase P0: Hallucination guardrails (highest impact, low risk)

Scope:

- Lower default QA temperature from `0.7` to `0.1`.
- Fix prompt consistency between completion and chat mode.
- Remove brittle stop tokens that can truncate structured prompts.
- Raise noisy prompt-length warning threshold.

Files:

- `qa_pipeline/coordinator.py`
- `qa_pipeline/prompt_builder.py`
- `core/llm.py`
- Optional UI defaults if QA sliders exist (`pages/`)

Changes:

- `answer_question(... temperature=0.1)` in `qa_pipeline/coordinator.py`.
- In `qa_pipeline/prompt_builder.py`, ensure chat mode uses the strong QA system prompt (currently a hardcoded generic system message is used).
- Keep completion mode format, but embed explicit "answer only from context" constraints.
- In `core/llm.py`:
  - Remove `###` and `---` from stop tokens.
  - Keep provider-safe stop handling.
  - Increase `PROMPT_LENGTH_WARN_THRESHOLD` (e.g., to `2000`).

Tests:

- Add/update prompt-builder tests for both modes.
- Update LLM module tests for stop-token behavior and payload shape.

Suggested run:

```powershell
pytest tests/test_llm_module.py tests/test_llm_module_extra.py tests/test_query.py -q
```

Exit gate:

- No regression in LLM tests.
- Prompt tests confirm identical instruction intent in both modes.

---

### Phase P1: Duplicate suppression hardening

Status (2026-03-26):

- Code changes complete.
- Targeted automated tests passing.
- Manual QA exit gate passed.

Scope:

- Fix exact-dedup edge case when checksum is missing.
- Tune near-duplicate threshold for CV-like content.
- Prevent duplicate top-up from undoing dedup quality.
- Deduplicate BM25 variant results by ID before fusion.

Files:

- `core/retrieval/fusion.py`
- `core/retrieval/types.py`
- `core/retrieval/pipeline.py`

Changes:

- `dedup_by_checksum`: do not treat `None` checksum as a shared dedup key.
- `sim_threshold`: tune from `0.90` to `0.82` in `RetrievalConfig`.
- Restrict duplicate top-up logic in pipeline (avoid reflooding when enough non-duplicate hits exist).
- Dedup BM25 hits across variants by `_id`, keeping highest score.

Tests:

- Add retrieval pipeline regression tests for:
  - checksum `None` behavior
  - variant BM25 dedup
  - top-up gating behavior
- Keep existing retrieval tests passing.

Suggested run:

```powershell
pytest tests/test_retrieval_pipeline.py tests/test_search_dedup.py tests/test_query.py -q
```

Validation snapshot (2026-03-26):

- `pytest tests/test_retrieval_pipeline.py tests/test_search_dedup.py tests/test_query.py -q` -> `12 passed`.
- Manual QA gate: passed.

Exit gate:

- Fewer duplicate sources in manual QA retrieval snapshots.
- No loss of top-k recall for non-duplicate corpora.

---

### Phase P2: Identity disambiguation via metadata (incremental)

Status (2026-03-26):

- Code changes complete.
- Backfill path implemented with non-destructive mapping checks (`put_mapping` only).
- One-time backfill run completed on existing indexed data with `errors=0`.
- Manual QA exit gate passed.

Scope:

- Add metadata plumbing end-to-end first.
- Add classification at ingest in a controlled manner.
- Add small authority boost only after metadata is present.
- Add bounded recency boost using existing `modified_at` metadata.

Files:

- `qa_pipeline/types.py`
- `qa_pipeline/retrieve.py`
- `qa_pipeline/prompt_builder.py`
- `ingestion/orchestrator.py`
- `ingestion/doc_classifier.py` (new)
- `core/retrieval/pipeline.py`
- `core/retrieval/types.py`
- `utils/opensearch_utils.py`
- `scripts/backfill_identity_metadata.py` (new)

Changes (recommended order):

1. Metadata pass-through:
   - Add optional fields to `RetrievedDocument` (`doc_type`, `person_name`, `authority_rank`).
   - Pass metadata through in `qa_pipeline/retrieve.py`.
2. Source labeling:
   - In `qa_pipeline/prompt_builder.py`, annotate numbered sources with metadata only if present.
3. Ingest-time classification:
   - Add `ingestion/doc_classifier.py`.
   - Start with deterministic heuristics fallback; optional LLM path behind a flag.
   - Write metadata into chunk docs and full-text docs in `ingestion/orchestrator.py`.
4. Retrieval re-weighting:
   - Apply small authority boost in `core/retrieval/pipeline.py` only when metadata exists.
   - Keep boost bounded to avoid reordering unrelated results.
   - Apply bounded recency boost from `modified_at` in `core/retrieval/pipeline.py` (decay + cap).
5. Mapping safety + migration:
   - Ensure explicit identity field mappings exist in OpenSearch (`utils/opensearch_utils.py`).
   - Fail fast on incompatible field types to avoid risky writes.
   - Run one-time metadata backfill via `scripts/backfill_identity_metadata.py`.

One-time migration for already indexed docs:

- Run a one-time backfill after deploying P2 so historical indexed docs get the same metadata:
  - Dry run: `python scripts/backfill_identity_metadata.py --dry-run`
  - Apply: `python scripts/backfill_identity_metadata.py`
- The backfill performs non-destructive mapping checks/additions (`put_mapping` only) before writes.
- Use `--overwrite` only if you intentionally want to recompute and replace existing identity metadata.

Tests:

- Add ingestion tests for classifier integration and metadata persistence.
- Add retrieval tests confirming metadata pass-through, bounded authority boost, and bounded recency boost.
- Add migration tests for mapping safety checks and backfill behavior.

Suggested run:

```powershell
pytest tests/test_ingestion_extra.py tests/test_ingest_fulltext.py tests/test_retrieval_pipeline.py tests/test_query.py tests/test_prompt_builder.py -q
pytest tests/test_backfill_identity_metadata.py tests/test_opensearch_identity_mappings.py tests/test_retrieval_pipeline.py tests/test_query.py -q
```

Validation snapshot (2026-03-26):

- `pytest tests/test_ingestion_extra.py tests/test_ingest_fulltext.py tests/test_retrieval_pipeline.py tests/test_query.py tests/test_prompt_builder.py -q` -> `26 passed`.
- `pytest tests/test_backfill_identity_metadata.py tests/test_opensearch_identity_mappings.py tests/test_retrieval_pipeline.py tests/test_query.py -q` -> `21 passed`.
- One-time run: `python scripts/backfill_identity_metadata.py` -> `scanned_fulltext_docs=2142`, `classified_docs=122`, `chunk_docs_updated=1327`, `errors=0`.
- Manual QA gate: passed.

Exit gate:

- Personal-document queries show lower cross-person mixing in manual QA set.
- Ingest path remains idempotent and backward compatible.

---

### Phase P3: Optional grounding check (feature-flagged)

Status (2026-03-26):

- Code changes complete.
- Targeted automated tests passing.
- Feature-flagged behavior validated: default off, deterministic metadata when enabled.

Scope:

- Add a post-answer groundedness signal without blocking responses by default.

Files:

- `qa_pipeline/grounding.py` (new)
- `qa_pipeline/coordinator.py`
- `qa_pipeline/types.py` (optional fields)

Changes:

- Implement lightweight text-overlap grounding score.
- Add `is_grounded` and `grounding_score` to response context.
- Gate behavior with env/config flag (default off).

Tests:

- Add unit tests for marker handling, empty context, and overlap threshold behavior.
- Add coordinator tests for flag-on/flag-off integration path.

Suggested run:

```powershell
pytest tests/test_query.py tests/test_llm_module.py tests/test_retrieval_pipeline.py -q
```

Validation snapshot (2026-03-26):

- `pytest tests/test_grounding.py tests/test_query.py tests/test_llm_module.py tests/test_retrieval_pipeline.py -q` -> `25 passed`.

Exit gate:

- No user-visible behavior change when flag is off.
- When enabled, groundedness metadata is populated deterministically.

---

### Phase P4: Provider migration (Groq) as separate track

Scope:

- Add Groq compatibility without breaking current local provider behavior.
- Keep migration isolated from QA quality PRs.

Files:

- `config.py`
- `.env.example`
- `core/llm.py`

Changes:

- Add optional Groq config values (`USE_GROQ`, `GROQ_API_KEY`, base/model settings).
- Inject auth header and endpoint switching in LLM client.
- Keep TGW-specific model management logic intact for non-Groq mode.

Tests:

- Add LLM tests for header injection and endpoint selection under `USE_GROQ`.
- Keep existing LLM tests passing in default mode.

Suggested run:

```powershell
pytest tests/test_llm_module.py tests/test_llm_module_extra.py tests/test_llm_cache.py -q
```

Exit gate:

- Default mode unchanged.
- Groq mode works with explicit env flags.

---

### Phase P5: Chunk-size migration (in progress: code complete, rollout pending)

Status (2026-03-26):

- Runtime default chunking updated to `800/100` in config and `.env.example`.
- Operational runbook for staged migration added.
- Full re-ingest rollout and post-migration quality comparison are pending.

Scope:

- Evaluate chunk size change from `400/50` to `800/100` only after QA quality gains are measured.

Files:

- `config.py`
- Operational runbook/docs for re-ingestion

Changes:

- Update defaults only with explicit migration decision.
- Re-ingest corpus in a controlled window.

Validation snapshot (2026-03-26):

- `pytest tests/test_config_utils.py tests/test_ingestion_extra.py tests/test_ingest_fulltext.py -q` -> `12 passed`.
- Runbook added: `docs/runbooks/chunk_size_migration.md`.

Risks:

- Mixed old/new chunking can degrade retrieval consistency.
- Migration increases operational load and index churn.

Exit gate:

- Re-ingest completed.
- Retrieval quality check repeated and compared to pre-migration baseline.

## 6. PR slicing proposal

1. PR-01: P0 hallucination guardrails.
2. PR-02: P4 Groq migration.
3. PR-03: P1 duplicate suppression.
4. PR-04: P2 metadata pass-through + source labeling.
5. PR-05: P2 ingest classifier integration.
6. PR-06: P2 authority weighting.
7. PR-07: P2 recency weighting.
8. PR-08: P2 mapping safety + one-time backfill.
9. PR-09: P3 grounding (flagged).
10. PR-10: P5 chunk migration + re-ingest runbook.
11. PR-11: P6 dynamic chunking policy + metadata + staged rollout.
12. PR-12: P7 doc-type expansion + bounded classifier enrichment.
13. PR-13: P8 OCR pipeline (quality + cost controls) + staged rollout.
14. PR-14: P9 full retrieval investigation + RAG revision decision memo.

Each PR should include:

- Focused code diffs only.
- Targeted regression tests for behavior changed in that PR.
- Short rollback note (what config/commit reverts the behavior).

## 7. What to defer unless blocked

- Broad prompt rewrites that change response style across the app.
- Full retrieval redesign.
- Topic discovery clustering redesign (separate backlog item).
- Any unrelated lint/cleanup-only refactors.

## 8. Incidental findings

1. Severity: Low
   Category: Documentation quality
   Files: `docs/plans/document_qa_optimization_plan.md` (previous draft)
   Rationale: The imported draft had encoding artifacts and some path mismatches.
   Recommended disposition: now
   Possible backlog duplicate: lint/cleanup debt (partial overlap)

2. Severity: Medium
   Category: Planning risk
   Files: original draft sections proposing full-file rewrites
   Rationale: Full rewrites increase blast radius and regression risk for this codebase.
   Recommended disposition: now
   Possible backlog duplicate: oversized file refactors (yes)

## 9. Phase P6: Dynamic chunking policy (proposed)

Status (2026-03-26):

- Proposed only; no code changes merged yet.
- Scope is intentionally incremental and deterministic.
- Migration strategy selected: fulltext-first rechunk (index-only where possible) to reduce operational cost.
- Step 1 started: dry-run audit script added (`scripts/audit_fulltext_rechunk_candidates.py`).
- Step 2 started and validated on canary cohort (10 docs) using fulltext-only rechunk script.

Objective:

- Move from one global chunk profile to a small ruleset driven by document signals.
- Improve precision on structured identity docs while preserving context on dense prose.
- Prepare for future OCR ingestion without forcing an OCR implementation in this phase.

Policy inputs (v1):

- `doc_type` from existing classifier (`cv`, `cover_letter`, `reference_letter`, or missing).
- `extraction_mode` (`native` now; `ocr` reserved for future image/scanned extraction).
- `quality_bucket` (`high`/`medium`/`low`; default `medium` when unavailable).
- `length_bucket` from `text_full` character length (example cutoffs: `short <= 3000`, `medium <= 20000`, `long > 20000`).

Policy outputs:

- `chunk_size`
- `chunk_overlap`
- `chunk_profile` (named rule key used for audit/debug)
- `chunk_policy_version` (start at `v1`)

Initial rule table (v1):

- `doc_type in {cv, cover_letter, reference_letter}` and `extraction_mode=native` -> `400/50` (`profile_identity_native`).
- `doc_type missing` and `extraction_mode=native` and `length_bucket=short` -> `600/80` (`profile_native_short`).
- `doc_type missing` and `extraction_mode=native` and `length_bucket in {medium,long}` -> `800/100` (`profile_native_default`).
- `extraction_mode=ocr` and `quality_bucket=high` -> `700/120` (`profile_ocr_high`).
- `extraction_mode=ocr` and `quality_bucket in {medium,low}` -> `500/120` (`profile_ocr_noisy`).

Implementation slices (low blast radius):

1. Policy resolver:
   - Add `ingestion/chunk_policy.py` with deterministic `resolve_chunk_policy(...)`.
2. Ingestion wiring:
   - In `ingestion/orchestrator.py`, compute policy once per file and pass `chunk_size`/`chunk_overlap` into chunking.
   - Keep current default behavior when policy data is missing.
3. Metadata persistence:
   - Write `chunk_policy_version`, `chunk_profile`, `chunk_size`, `chunk_overlap`, `extraction_mode`, `quality_bucket`, `length_bucket` into chunk docs.
   - Write the same policy metadata (where relevant) into full-text docs for auditability.
4. Config safety:
   - Add feature flag `DYNAMIC_CHUNKING_ENABLED` (default `false`) and retain global `CHUNK_SIZE/CHUNK_OVERLAP` fallback.
5. Operations:
   - Add runbook for staged re-ingest by prefix/doc bucket and rollback toggle.

Testing plan:

- Unit tests for `resolve_chunk_policy(...)` covering all rule branches and fallback behavior.
- Ingestion tests confirming selected policy values are persisted to chunks/full-text metadata.
- Regression tests ensuring `DYNAMIC_CHUNKING_ENABLED=false` preserves existing chunk behavior.

Suggested run:

```powershell
pytest tests/test_ingestion_extra.py tests/test_ingest_fulltext.py tests/test_config_utils.py -q
```

Rollout and migration:

- Treat as versioned migration: all new chunks must carry `chunk_policy_version=v1`.
- Start with `DYNAMIC_CHUNKING_ENABLED=false` in production-like environments.
- Enable for a narrow prefix/bucket first (canary), compare QA metrics, then broaden rollout.
- Avoid untracked mixed states; if policy changes, bump version (`v2`) and re-ingest targeted cohorts.

Fulltext-first migration strategy (selected):

1. Step 1: Audit candidates from `documents_full_text` (started)
   - Dry-run only; no writes.
   - Identify `text_full` non-empty docs as index-only rechunk candidates.
   - Command:
     - `python scripts/audit_fulltext_rechunk_candidates.py --prefix "C:/Users/ali_a/My Drive"`
2. Step 2: Canary rechunk from `text_full` for a small cohort
   - Rebuild chunks from stored `text_full`, using policy-selected chunk profile.
   - Use `location_percent` as the positional anchor (page numbers intentionally not required in this path).
   - Commands executed:
     - Dry-run: `python scripts/rechunk_from_fulltext.py --prefix "C:/Users/ali_a/My Drive" --limit 20`
     - Apply canary: `python scripts/rechunk_from_fulltext.py --prefix "C:/Users/ali_a/My Drive" --limit 10 --apply`
   - Apply snapshot:
     - `selected_docs=10`, `rebuilt_docs=10`, `failed_docs=0`
     - `total_new_chunks=161`, `deleted_old_chunks=333`, `indexed_new_chunks=161`
   - Post-check:
     - Re-running dry-run on the same 10-doc cohort now shows `old_os_chunks == new_chunks` and `old_qdrant_chunks == new_chunks` for all 10 docs.
3. Step 3: Expand rechunk to all eligible non-empty `text_full` docs
   - Delete old vectors/chunks by checksum, then re-embed/re-index.
   - Keep full-text docs as source-of-truth metadata and text payload.
   - Command executed:
     - `python scripts/rechunk_from_fulltext.py --prefix "C:/Users/ali_a/My Drive" --limit 1979 --apply --sample-limit 20`
   - Apply snapshot:
     - `scanned_docs=2139`, `selected_docs=1979`, `rebuilt_docs=1979`, `failed_docs=0`
     - `total_new_chunks=84516`, `deleted_old_chunks=166692`, `indexed_new_chunks=84516`
   - Post-check:
     - Re-running dry-run after rollout on a 20-doc sample shows `old_os_chunks == new_chunks` and `old_qdrant_chunks == new_chunks` per sampled doc.
4. Step 4: Handle non-eligible docs separately
   - `text_full` empty docs (e.g., scanned/image-heavy) are deferred to OCR ingestion flow.
5. Step 5: Validate and close
   - Compare retrieval quality/grounding before vs after for canary and full rollout.
   - Confirm no regressions in duplicate suppression and identity disambiguation behaviors.

Exit gate:

- Retrieval precision improves on identity-style queries without degrading dense-prose QA quality.
- No regressions when dynamic policy is disabled.
- Operational runbook validated on at least one staged re-ingest cycle.

## 10. Phase P7: Doc-type coverage expansion (proposed)

Status (2026-03-26):

- Proposed only; no code changes merged yet.
- Current fulltext snapshot still has high `doc_type=__missing__` share.

Objective:

- Reduce `doc_type=__missing__` substantially so retrieval policy can help non-CV corpora too.
- Improve routing/chunk-policy behavior for legal/admin/finance/research-style documents.

Scope:

- Expand ingestion taxonomy beyond identity docs.
- Keep classification deterministic-first.
- Use optional LLM enrichment only for uncertain cases, with strict budget and confidence gates.

Proposed taxonomy (v1):

- `cv`, `cover_letter`, `reference_letter`
- `research_paper`, `technical_report`, `course_material`
- `contract`, `policy`, `invoice`, `payroll`
- `insurance_letter`, `government_form`, `academic_record`
- `other` (explicit fallback instead of silent missing)

Implementation slices (low blast radius):

1. Deterministic classifier expansion:
   - Extend `ingestion/doc_classifier.py` with path/title/text pattern rules for the v1 taxonomy.
   - Emit explicit `other` when no rule matches.
2. Confidence + uncertainty handling:
   - Return `doc_type_confidence` and `doc_type_source` (`rule`, `llm`, `fallback`).
3. Optional Groq enrichment for uncertain docs:
   - Add feature flag (default off), only for docs below deterministic confidence threshold.
   - Cache classifier decisions by checksum to avoid repeat calls.
4. Backfill/migration:
   - Add targeted backfill mode for `doc_type in {null,__missing__,other}` cohorts only.
   - Keep mapping updates non-destructive.

Testing plan:

- Unit tests for rule coverage per new class.
- Ingestion tests for metadata persistence of `doc_type`, `doc_type_confidence`, and `doc_type_source`.
- Backfill tests for idempotency and safe overwrite behavior.

Suggested run:

```powershell
pytest tests/test_ingestion_extra.py tests/test_backfill_identity_metadata.py tests/test_retrieval_pipeline.py -q
```

Exit gate:

- Relative reduction in `__missing__` doc type by at least 25% on the indexed corpus.
- No regression on CV/profile retrieval quality.
- Non-profile fixture queries show improved hit@3 in baseline comparison.

## 11. Phase P8: OCR ingestion (quality-first, cost-bounded) (proposed)

Status (2026-03-26):

- Proposed only; no OCR extraction pipeline merged yet.
- `text_full` empty docs remain deferred and are the main OCR target cohort.

Objective:

- Extract usable text from scanned/image-heavy documents with high quality while minimizing paid usage.

Scope:

- Native extraction remains first path.
- OCR only for docs/pages with missing or low-quality native text.
- Groq usage is optional and bounded; prioritize free/local OCR for bulk processing.

Pipeline policy (v1):

1. Candidate gating:
   - OCR candidates are docs with empty `text_full` or very low text-density after native extraction.
2. Tiered OCR:
   - Tier 1 (default): local OCR engine (no API cost) for all candidate pages.
   - Tier 2 (optional): Groq-assisted OCR/recovery only for low-confidence pages from tier 1.
3. Quality controls:
   - Preprocessing: orientation fix, denoise, deskew, contrast normalization.
   - Confidence scoring per page and document-level `quality_bucket`.
4. Cost controls:
   - Daily/page-level cap for Groq OCR fallback.
   - Hard stop when budget quota is reached.

Metadata additions:

- `extraction_mode` (`native`, `ocr_local`, `ocr_groq`)
- `ocr_confidence_mean`, `ocr_confidence_min`
- `ocr_page_count`, `ocr_fallback_used`

Implementation slices:

1. Add OCR module and orchestrator hook (feature-flagged, default off).
2. Persist OCR quality/cost metadata into fulltext/chunk docs.
3. Add runbook for staged OCR rollout by prefix and page budget.
4. Integrate OCR outputs into dynamic chunking policy inputs (`extraction_mode`, `quality_bucket`).

Testing plan:

- Unit tests for OCR gating and fallback policy.
- Ingestion tests for OCR metadata persistence.
- End-to-end canary tests on a small scanned cohort.

Suggested run:

```powershell
pytest tests/test_ingestion_extra.py tests/test_ingest_fulltext.py tests/test_config_utils.py -q
```

Exit gate:

- At least 80% of previously empty-text docs in canary yield non-empty text after OCR.
- OCR-enabled retrieval improves hit@3 on scanned-doc queries vs baseline.
- Groq OCR usage stays within configured daily limits.

## 12. Phase P9: Full retrieval investigation and RAG revision decision (completed)

Status (2026-03-26):

- Investigation report and decision memo completed:
  - `docs/runbooks/retrieval_investigation_p9_2026-03-26.md`
  - `docs/runbooks/retrieval_investigation_p9_2026-03-26.json`
- Decision outcome: Path A (targeted incremental fixes) selected first.
- Path A follow-up iteration completed in code:
  - Anchored exact-only variant gate.
  - Anchored lexical-first fusion bias.
- Post-Path-A fixture rerun:
  - `positive_hit_at_1=5/20` (`0.25`)
  - `positive_hit_at_3=12/20` (`0.60`)
  - `control_with_results=0/3`
  - Artifacts:
    - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json`
    - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.csv`
- Residual-failure sidecar analysis rerun completed (2026-03-27) under benchmark-separated framing:
  - Artifact:
    - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1_residual_failure_analysis.json`
  - Schema marker:
    - `schema_version=residual_failure_analysis.v2` (compatibility note included in artifact).
  - Query-type counts:
    - `canonical_document_query=19`
    - `multi_source_factual_query=1`
    - `ambiguous_reviewer_needed=0`
  - Scope: `14` benchmark-selected residual misses (`selected_failure_mode_summary: strict_retrieval=14`).
  - Primary buckets:
    - `relevant doc ranked below 1 but within top-3`: `6/14` (`42.86%`)
    - `relevant doc retrieved but ranked below top-3`: `7/14` (`50.00%`)
    - `relevant doc not retrieved despite text being available`: `1/14` (`7.14%`)
    - `likely text extraction / OCR gap`: `0/14` (`0.00%`)
  - OCR canary recommendation: **NO** (no measured text-extraction/OCR-driven residual misses).
- Ranking-focused follow-up investigation rerun completed (2026-03-27) under benchmark-separated framing:
  - Artifact:
    - `docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1_ranking_investigation.json`
  - Schema marker:
    - `schema_version=ranking_investigation.v3` (adds `probe_vs_eval_comparison` and `strict_canonical_ranking_diagnosis` blocks; legacy flat probe keys removed).
  - Deterministic deep-rank probe (`exact-query`, `probe_depth=40`) metrics:
    - `strict_retrieval`: `hit@1=4/20` (`0.20`), `hit@3=6/20` (`0.30`), `MRR=0.3261`
    - `answer_support`: `hit@1=5/20` (`0.25`), `hit@3=7/20` (`0.35`), `MRR=0.3661`
  - Query-type breakouts now emitted in both metric blocks:
    - `canonical_document_query (n=19)`: strict `hit@1=4/19`, strict `hit@3=6/19`; answer-support identical (`4/19`, `6/19`)
    - `multi_source_factual_query (n=1)`: strict `hit@1=0/1`, strict `hit@3=0/1`; answer-support `hit@1=1/1`, `hit@3=1/1`
    - `ambiguous_reviewer_needed (n=0)`: no samples in this fixture pass
  - Comparison vs prior mixed framing artifact (`HEAD` runbook before separation rerun):
    - Prior mixed probe metrics: `hit@1=4/20`, `hit@3=6/20`, `MRR=0.3261`
    - New strict metrics are unchanged (`4/20`, `6/20`, `0.3261`) -> no strict lift from relabeling; ranking weakness remains.
    - New answer-support metrics improved (`hit@1: 4->5`, `hit@3: 6->7`, `MRR: 0.3261->0.3661`) due benchmark separation/manual support labeling.
    - Benchmark-selected residual misses dropped `15->14`, explained by one multi-source query (`Q01`) moving from strict miss to answer-support success.
  - Probe-vs-eval methodology discrepancy explanation (`probe_vs_eval_comparison` block):
    - Archived Path A eval profile: `variants_enabled=true`, `rewrites_enabled=true`, `exact_query_probing=false`, `candidate_depth=3`.
    - Strict investigation probe profile: `variants_enabled=false`, `rewrites_enabled=false`, `exact_query_probing=true`, `candidate_depth=40`.
    - Deterministic strict metric delta captured in artifact: eval `hit@1=5/20`, `hit@3=12/20` vs probe `hit@1=4/20`, `hit@3=6/20`.
    - Query-level disagreement is fully one-sided (`archived_only_hit@3=6`, `probe_only_hit@3=0`), consistent with methodology mismatch rather than a benchmark-labeling bug.
    - Mechanism recorded explicitly: retrieval ties MMR depth to `top_k`; moving from `top_k=3` to `top_k=40` changes top-3 ordering.
  - Remaining true ranking failures:
    - Benchmark-selected ranking failures are entirely strict (`14`, mode=`strict_retrieval`).
    - Canonical strict misses remain high: `15/19` on probe hit@1 (query IDs: `Q02,Q03,Q04,Q05,Q06,Q07,Q09,Q10,Q11,Q12,Q13,Q14,Q15,Q19,Q20`).
    - Ablation signal over benchmark-selected failures (`14`): `no_boosts` improved `5/14`, `no_authority` improved `4/14`, `no_recency` improved `3/14`.
  - Strict canonical ranking-cause attribution (`strict_canonical_ranking_diagnosis` block):
    - Bucket counts over `15` strict canonical hit@1 misses:
      - `ambiguous/manual review`: `7`
      - `vector dominance`: `5`
      - `title/filename underweighting`: `2`
      - `candidate generation miss`: `1`
      - `sibling/near-duplicate collision`: `0`
      - `chunk aggregation bias`: `0`
      - `doc-type prior suppression`: `0`
    - Largest bucket: `ambiguous/manual review` (`7/15`, `46.67%`).
    - Largest actionable bucket: `vector dominance` (`5/15`, `33.33%`).
  - Next tuning target (strict canonical only):
    - Prioritize a low-blast-radius lexical-priority calibration for strict canonical misses that are currently `vector dominance`, while keeping answer-support framing and OCR scope unchanged.
- Path B trigger was not met after this iteration (thresholds achieved), so broader RAG redesign is deferred.

Objective:

- Run a full, evidence-based investigation of why retrieval quality is poor.
- Decide whether incremental fixes are sufficient or a broader RAG pipeline revision is required.

Investigation scope:

1. Retrieval stage attribution:
   - Measure per-stage impact (rewrite, semantic, BM25, fusion, dedup, MMR, boosts).
   - Identify where relevance is being lost for failed queries.
2. Query rewrite quality:
   - Quantify clarify over-triggering and rewrite drift on positive fixture queries.
   - Verify fallback behavior for anchored queries.
3. Data/metadata quality:
   - Analyze missing `doc_type`, noisy metadata, and checksum-family effects.
4. Infrastructure behavior:
   - Root-cause recurring OpenSearch chunk-text warning during retrieval.
   - Confirm whether this affects fusion quality or only payload enrichment.
5. Control-query behavior:
   - Measure false-positive retrieval on out-of-corpus controls.
   - Evaluate abstention policy requirements.

Deliverables:

- Investigation report with:
  - failure taxonomy by query class
  - stage-level metrics and bottlenecks
  - prioritized remediation options and estimated impact
- Decision memo:
  - Path A: targeted incremental fixes only
  - Path B: broader RAG revision (retriever architecture/routing/reranking redesign)

Revision policy:

- A broader RAG revision is allowed if investigation evidence shows incremental fixes cannot meet quality targets.
- If Path B is selected, create a separate phased plan with rollback strategy and explicit acceptance metrics.

Exit gate:

- Investigation report and decision memo completed.
- Clear go/no-go decision for full RAG revision with measurable success criteria.

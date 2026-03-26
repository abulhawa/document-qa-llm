# document_qa Optimization Plan (v2)

Last updated: 2026-03-26
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

Milestone is successful when:

- Existing targeted tests pass.
- New regression tests for each changed behavior pass.
- Manual QA set shows lower hallucination and fewer duplicate sources.

## 5. Implementation plan (phased, low blast radius)

### Current execution order (updated 2026-03-26)

- P0 is complete in code.
- P4 is moved ahead of P1-P3 for this environment so manual QA can run without the local TGW setup.
- P1 is complete in code with targeted regression tests passing.
- Active sequence: `P0 -> P4 -> P1 -> manual QA (P1 exit gate) -> P2 -> P3 -> P5`.
- Guardrail unchanged: keep P4 isolated from retrieval/prompt quality changes.

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
- Manual QA exit gate still pending.

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

Exit gate:

- Fewer duplicate sources in manual QA retrieval snapshots.
- No loss of top-k recall for non-duplicate corpora.

---

### Phase P2: Identity disambiguation via metadata (incremental)

Scope:

- Add metadata plumbing end-to-end first.
- Add classification at ingest in a controlled manner.
- Add small authority boost only after metadata is present.

Files:

- `qa_pipeline/types.py`
- `qa_pipeline/retrieve.py`
- `qa_pipeline/prompt_builder.py`
- `ingestion/orchestrator.py`
- `ingestion/doc_classifier.py` (new)
- `core/retrieval/pipeline.py`

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

One-time migration for already indexed docs:

- Run a one-time backfill after deploying P2 so historical indexed docs get the same metadata:
  - Dry run: `python scripts/backfill_identity_metadata.py --dry-run`
  - Apply: `python scripts/backfill_identity_metadata.py`
- The backfill performs non-destructive mapping checks/additions (`put_mapping` only) before writes.
- Use `--overwrite` only if you intentionally want to recompute and replace existing identity metadata.

Tests:

- Add ingestion tests for classifier integration and metadata persistence.
- Add retrieval tests confirming metadata pass-through and bounded authority boost.

Suggested run:

```powershell
pytest tests/test_ingestion_extra.py tests/test_ingest_fulltext.py tests/test_retrieval_pipeline.py tests/test_query.py -q
```

Exit gate:

- Personal-document queries show lower cross-person mixing in manual QA set.
- Ingest path remains idempotent and backward compatible.

---

### Phase P3: Optional grounding check (feature-flagged)

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

### Phase P5: Chunk-size migration (deferred until after P0-P2 validation)

Scope:

- Evaluate chunk size change from `400/50` to `800/100` only after QA quality gains are measured.

Files:

- `config.py`
- Operational runbook/docs for re-ingestion

Changes:

- Update defaults only with explicit migration decision.
- Re-ingest corpus in a controlled window.

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
7. PR-07: P3 grounding (flagged).
8. PR-08: P5 chunk migration + re-ingest runbook.

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

# Evaluation guide

The repository separates software tests from retrieval and QA quality evaluation. Existing JSON/CSV files under `docs/runbooks/` are historical experiment artifacts, not general product benchmarks.

## Retrieval evaluation

`scripts/run_retrieval_eval.py` evaluates the checked-in query fixture and can report expected-document Hit@k and MRR-style ranking measurements. It requires populated backends matching the fixture corpus. Review `--help` before running:

```bash
python scripts/run_retrieval_eval.py --help
```

Record the exact revision, fixture, service/model versions, flags, corpus state, output artifact, and latency. Compare a candidate against the same baseline rather than comparing unrelated runs.

## QA handoff and answer support

`scripts/run_qa_handoff_eval.py` compares context-selection strategies and support labels without treating retrieval success as final-answer correctness:

```bash
python scripts/run_qa_handoff_eval.py --help
```

When changing retrieval, ranking, grounding, or prompting, report the relevant subset of Hit@1/3/5, MRR, nDCG (if available), answer support, citation accuracy, refusal/failure counts, and latency. A change should not be described as an improvement without a reproducible before/after result.

## CI and evaluation strategy

The project is heavily containerized for local deployment, but CI does not need to reproduce the full deployment stack for every change. Keep four concerns separate:

| Layer | Purpose | Typical trigger | Target wall-clock time |
|---|---|---|---:|
| Unit/UI CI | Verify application logic and deterministic behavior | Every PR and push | under 5 minutes |
| Container integration smoke | Verify Docker/service wiring and ingestion plumbing | Relevant infrastructure/backend changes | 5-10 minutes |
| Retrieval regression evaluation | Measure retrieval quality against a fixed public benchmark snapshot | Relevant retrieval changes or manual run | 5-15 minutes |
| Full raw-document evaluation | Rebuild from source documents through parsing, chunking, embedding and indexing | Manual, release, or ingestion/model changes | 30+ minutes is acceptable |

The slowest layer should not block ordinary development unless the change actually affects that layer.

### Current GitHub Actions behavior

The current workflows are intentionally lighter than the full Docker Compose deployment:

- `.github/workflows/tests.yml` runs Python directly on the GitHub runner and excludes `tests/e2e/`.
- `.github/workflows/e2e.yml` is manually triggered.
- The current E2E job starts real OpenSearch and Qdrant containers.
- Streamlit, the stub embedding service, the stub LLM, Playwright, and the test process run directly on the GitHub runner.
- The E2E workflow seeds OpenSearch and Qdrant directly. It does not exercise Redis, the Celery worker container, real document ingestion, the real embedding model, the reranker, Phoenix, Flower, Docker Compose networking, or production-style volume mapping.
- The stub embedder validates application wiring, not semantic retrieval quality.

This means the existing E2E workflow is best understood as an application smoke test against real storage backends, not a full deployment or ML-quality test.

The E2E backend versions should also be kept aligned with the deployment versions in `docker-compose.yml`. At the time this guide was updated, the E2E workflow used older OpenSearch and Qdrant image tags than the main Compose file.

## Public benchmark evaluation

Use an established public, redistributable benchmark as the main retrieval-quality reference rather than personal documents. The benchmark should pass through the same retrieval code used by the application.

Keep retrieval evaluation separate from generation evaluation:

1. Retrieval evaluation measures whether the correct document/chunk is retrieved and ranked well.
2. QA evaluation measures whether retrieved evidence is handed off correctly and the generated answer is supported.
3. A later answer-generation benchmark may add correctness, faithfulness, citation quality, abstention, and latency.

For retrieval, prefer standard metrics such as:

- Recall/Hit@1, @3, and @5
- MRR
- nDCG@5
- latency

Report results against a fixed benchmark revision and fixed evaluation snapshot.

## Versioned evaluation snapshots

Parsing, chunking, and document embeddings are expensive but deterministic for a fixed corpus and configuration. They should not be recomputed on every retrieval experiment.

Treat the evaluation pipeline as dependency-aware stages:

```text
raw benchmark documents
        ↓
parsing
        ↓
parsed documents
        ↓
chunking
        ↓
canonical chunks
        ↓
embedding
        ↓
vectors
        ↓
index population
        ↓
retrieval/ranking experiments
        ↓
metrics
```

A retrieval-only change should reuse all compatible upstream artifacts.

Each snapshot should record enough lineage to determine whether it is still valid. At minimum include:

- benchmark name and revision/subset
- source-document checksums
- parser implementation/configuration fingerprint
- preprocessing fingerprint
- chunking strategy, size, and overlap
- embedding model name and model revision
- embedding dimension and normalization settings
- repository revision used to build the snapshot
- snapshot schema/version

A cache key or snapshot ID should be derived from the inputs that affect the generated artifact, not from a manually chosen label alone.

### Invalidation rules

Use the narrowest rebuild that preserves correctness:

| Change | Reuse | Rebuild |
|---|---|---|
| Retrieval fusion, MMR, filters, top-k, ranking policy | parsed docs, chunks, embeddings | indexes/query run as needed |
| Query rewriting/planning | parsed docs, chunks, embeddings | query run |
| Reranker | parsed docs, chunks, document embeddings | reranking/query run |
| Embedding model/config | parsed docs, chunks | embeddings and vector index |
| Chunker/preprocessing | parsed docs when compatible | chunks, embeddings, indexes |
| Parser/loader | source documents only | parsing and everything downstream |

Do not silently reuse an incompatible snapshot. A benchmark run should fail clearly if the required snapshot fingerprint does not match.

## Manual evaluation-snapshot build workflow

A dedicated manual GitHub Actions workflow is the preferred way to rebuild the expensive evaluation snapshot when ingestion-facing behavior changes.

Conceptually:

```text
workflow_dispatch
      ↓
download fixed public benchmark source
      ↓
run real parser/preprocessor/chunker
      ↓
run real embedding model
      ↓
build canonical snapshot
      ↓
validate manifest/fingerprints
      ↓
publish immutable versioned snapshot
```

This workflow should be separate from normal PR CI. Rebuilding the snapshot is an explicit maintenance action, not a hidden side effect of a retrieval test.

Useful manual inputs may include:

- benchmark subset: smoke / medium / full
- force rebuild: true / false
- benchmark revision
- optional snapshot label for human readability

The generated snapshot should include the canonical chunks, metadata/manifest, qrels/query files required by the benchmark adapter, and precomputed document embeddings when licensing permits.

### Snapshot storage

Do not rely on the ordinary GitHub Actions cache as the only source of truth for evaluation snapshots. Actions caches are optimized for build acceleration and may be evicted.

Prefer an immutable/versioned artifact store for durable benchmark snapshots. GitHub Container Registry as an OCI artifact is a reasonable long-term choice because snapshots can be versioned and fetched by CI without committing large generated data to Git. A simpler workflow artifact can be used during initial development, but it should not be treated as permanent storage.

Normal CI may still use `actions/cache` for disposable acceleration such as Python packages, model downloads, Docker layers, or a recently fetched snapshot.

## Trigger policy

Avoid running evaluation because an unrelated file changed.

Examples:

- Documentation or cosmetic UI changes: unit/UI checks only.
- Retrieval-policy changes: unit tests plus the small cached retrieval benchmark.
- Parser/chunker/embedding changes: unit tests; mark the current evaluation snapshot incompatible and rebuild it explicitly before comparing quality.
- Deployment/Docker changes: container integration smoke.
- Release candidate: run the larger/full evaluation appropriate to the release.

Independent jobs should run in parallel where possible so wall-clock CI time is determined by the slowest required layer rather than the sum of all layers.

## Container integration direction

A future container integration workflow should test the architecture that the current E2E job intentionally bypasses:

```text
small test document
      ↓
Redis/Celery
      ↓
real worker container
      ↓
real parser/chunker
      ↓
stub embedding service
      ↓
OpenSearch + Qdrant
      ↓
assert indexed state
```

The embedding and LLM services may remain deterministic stubs in this workflow because its purpose is deployment/integration correctness, not ML quality.

The retrieval benchmark should use real embeddings, but it does not need Streamlit, Celery, Redis, Flower, Phoenix, or an LLM when evaluating retrieval only.

## Safe fixtures and artifacts

Only commit synthetic, licensed, or otherwise redistributable fixtures. Remove personal paths, document contents, credentials, and identifiers. Store large/generated run outputs outside Git unless a small artifact is intentionally retained as a documented baseline.

Benchmark results should state whether they came from:

- a cached retrieval snapshot
- a full raw-document rebuild
- the small smoke subset
- the larger/full benchmark

This prevents a fast regression run from being mistaken for a complete end-to-end evaluation.

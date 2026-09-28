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
| Retrieval regression evaluation | Measure retrieval quality against a fixed benchmark snapshot | Relevant retrieval changes or manual run | 5-15 minutes |
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

The E2E backend versions should eventually be aligned with the deployment versions in `docker-compose.yml`. That change is intentionally separate from the benchmark skeleton so the existing runtime is not changed before review.

## Mixed public benchmark

Use a fixed mixed benchmark instead of personal documents and instead of relying on only one domain.

The initial benchmark mix should draw selected, license-compatible material from established sources such as:

- Open RAGBench for scientific/technical PDF retrieval and hard negatives.
- OfficeQA for office, government, financial, table-heavy document questions.
- Selected BEIR datasets for additional retrieval domains when the source format fits the evaluation goal.
- Additional benchmarks may be added later only when they cover a real gap.

Do not merge these sources into an opaque dataset. Every document and query must retain provenance:

```text
source_benchmark
source_version
source_document_id
source_query_id
domain
question_type
modality
qrels
license/provenance metadata
```

Report per-source metrics first. A composite score may be reported as a convenience, but it must not hide weak performance on one benchmark behind an easier benchmark.

The benchmark source selection must be deterministic and versioned. Re-running the source stage with the same source revisions, filters, and selection seed must produce the same corpus manifest.

## Evaluation artifact DAG

The full evaluation system is a dependency-aware artifact pipeline:

```text
1. source corpus
      ↓
2. parsed documents
      ↓
3. chunked documents
      ↓
4. embeddings
      ↓
5. OpenSearch + Qdrant indexes
      ↓
6. evaluation results
```

Every stage is reproducible, but ordinary evaluation should start from the newest compatible downstream artifact instead of recomputing upstream work.

### Stage 1: source corpus

The source artifact contains the fixed mixed benchmark inputs needed to reproduce the evaluation corpus:

```text
benchmark-source/
├── manifest.json
├── queries/
├── qrels/
├── provenance/
└── documents/ or source references permitted by each benchmark
```

Raw PDFs should be cached where licensing/access terms permit. For gated or restricted sources, store them privately or retain reproducible source references/checksums rather than republishing them.

### Stage 2: parsed documents

Rebuild when the source corpus, parser, loader, or parser-affecting preprocessing changes.

Typical contents:

```text
parsed/
├── documents.jsonl
├── pages.jsonl
├── metadata.jsonl
└── manifest.json
```

### Stage 3: chunks

Rebuild when chunking or chunk-affecting preprocessing changes.

Typical contents:

```text
chunks/
├── chunks.jsonl
├── chunk_metadata.jsonl
└── manifest.json
```

For the normal case, parsing and chunking should be reused for a long time rather than repeated during every retrieval experiment.

### Stage 4: embeddings

Rebuild when chunks, the embedding model/revision, dimensionality, normalization, or embedding configuration changes.

Typical contents:

```text
embeddings/
├── embeddings.npy
├── chunk_ids.json
└── manifest.json
```

The chunk artifact plus embeddings form the portable canonical representation used to rebuild search-engine-specific indexes.

### Stage 5: native search indexes

Build native OpenSearch and Qdrant snapshots from the chunk and embedding artifacts.

These snapshots are a late-stage performance cache:

```text
chunks + embeddings
        ↓
OpenSearch snapshot
Qdrant snapshot
```

They are deliberately not the only canonical representation. If an engine snapshot becomes incompatible, the indexes can be rebuilt from portable chunks and embeddings without reparsing PDFs or recomputing document embeddings.

Pin backend versions rather than using floating `latest` tags. The benchmark skeleton currently targets:

```text
OpenSearch 3.8.0
Qdrant 1.19.1
```

Backend upgrades are explicit maintenance changes. They should not occur automatically simply because a newer image exists.

### Stage 6: evaluation

The common retrieval evaluation path should restore compatible native indexes, run the application retrieval pipeline, and calculate metrics.

Normal retrieval evaluation should not require:

- PDF parsing
- chunking
- document embedding
- Celery
- Redis
- Streamlit
- Flower
- Phoenix
- an answer-generation LLM

Use the real retrieval code and real document embeddings. Generation evaluation remains a separate layer.

## Artifact lineage and fingerprints

Every stage must include a manifest that identifies its direct parent and all configuration that can affect its output.

At minimum record:

- artifact type and schema version
- parent artifact ID/fingerprint
- benchmark source names, revisions, filters, and subset
- source-document checksums where available
- parser implementation/configuration fingerprint
- preprocessing fingerprint
- chunking strategy, size, and overlap
- embedding model name and revision
- embedding dimension and normalization
- OpenSearch/Qdrant versions and index schema fingerprint for index artifacts
- repository revision used to build the artifact
- creation timestamp for traceability, but not as the compatibility key

Compatibility must be determined from functional inputs, not from a manually chosen artifact name or timestamp.

If a required parent fingerprint does not match, the downstream workflow must fail clearly rather than silently use stale data.

## Invalidation and rebuild rules

Use the narrowest rebuild that preserves correctness:

| Change | First stage to rerun |
|---|---|
| Retrieval fusion, MMR, filters, top-k, ranking policy | evaluation |
| Query rewriting/planning | evaluation |
| Reranker | evaluation, unless its model artifact is separately cached |
| OpenSearch/Qdrant mapping/index configuration | index |
| OpenSearch/Qdrant version | index |
| Embedding model/configuration | embed |
| Chunker/chunk-affecting preprocessing | chunk |
| Parser/loader | parse |
| Benchmark source selection/revision | source |

After rerunning a stage, all downstream artifacts are considered stale until deliberately rebuilt.

Do not automatically cascade expensive downstream rebuilds while the implementation is still being reviewed. Manual stage boundaries keep Actions usage predictable and avoid rebuilding several expensive layers while an upstream change is still experimental.

## Manual workflow skeleton

The repository contains manual-only skeleton workflows for each stage:

```text
.github/workflows/benchmark-source.yml
.github/workflows/benchmark-parse.yml
.github/workflows/benchmark-chunk.yml
.github/workflows/benchmark-embed.yml
.github/workflows/benchmark-index.yml
.github/workflows/benchmark-eval.yml
```

At this stage they are intentionally non-operational planning shells. They expose the intended manual inputs and dependency boundaries but do not download corpora, models, build snapshots, publish artifacts, or run evaluation.

This allows the artifact contracts and storage choices to be reviewed before CI begins consuming significant time or storage.

## Artifact storage

Do not use ordinary GitHub Actions cache as the only durable source of benchmark artifacts. Actions cache is useful for disposable acceleration but can be evicted.

The intended storage model is:

```text
durable/versioned artifact store
        +
Actions cache for local acceleration
```

GitHub Container Registry using OCI artifacts is the preferred long-term candidate for versioned source/parsed/chunk/embedding/index artifacts. Workflow artifacts may be used while the design is being developed, but should not become the permanent system of record.

The exact publication mechanism remains intentionally unimplemented until benchmark licensing, artifact sizes, retention, and access requirements are reviewed.

## Trigger policy

Avoid running evaluation because an unrelated file changed.

Examples:

- Documentation or cosmetic UI changes: unit/UI checks only.
- Retrieval-policy changes: unit tests plus the small cached retrieval benchmark when enabled.
- Index mapping or backend-version changes: rebuild index artifact, then evaluate.
- Embedding changes: rebuild embeddings, index, then evaluate.
- Chunking changes: rebuild chunks, embeddings, index, then evaluate.
- Parser changes: rebuild parsed documents and every downstream stage.
- Benchmark composition changes: rebuild from source.
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

The retrieval benchmark should use real embeddings, but it does not need the full application stack when evaluating retrieval only.

## Safe fixtures and artifacts

Only commit synthetic, licensed, or otherwise redistributable fixtures. Remove personal paths, document contents, credentials, and identifiers. Store large/generated run outputs outside Git unless a small artifact is intentionally retained as a documented baseline.

Benchmark results should state:

- source benchmark and subset
- artifact fingerprints used
- backend/model versions
- whether native indexes were restored or rebuilt
- whether the run used the smoke, medium, or full evaluation set

This prevents a fast regression run from being mistaken for a complete end-to-end evaluation.

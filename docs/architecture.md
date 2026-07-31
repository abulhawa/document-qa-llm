# Architecture

## Responsibility boundaries

- `main.py`, `pages/`, `ui/`, and `app/usecases/` implement Streamlit presentation and application use cases.
- `ingestion/` loads, classifies, preprocesses, chunks, and stores documents. `worker/` exposes that work through Celery.
- `core/retrieval/` assembles lexical and dense candidates, fusion, deduplication, optional variants/MMR, and optional reranking.
- `qa_pipeline/` coordinates rewriting, retrieval, context packing, prompt construction, generation, grounding checks, and response types.
- OpenSearch holds lexical/full-text and operational indexes. Qdrant holds dense vectors. Redis is the Celery broker/result backend.
- `tracing.py` sends OpenTelemetry spans to Phoenix when configured.

## Data flow

### Ingestion

1. A user selects local files in Streamlit.
2. The application queues work through Celery/Redis.
3. The worker loads and preprocesses supported files, then creates overlapping chunks.
4. The embedding API produces dense vectors.
5. Chunk/full-text metadata is written to OpenSearch and vectors/metadata to Qdrant.

### Question answering

1. The QA use case selects retrieval and context-handoff settings.
2. Query rewriting may clarify/rewrite a question; query planning and HyDE are optional and disabled by default.
3. Hybrid retrieval obtains OpenSearch and Qdrant candidates, fuses and deduplicates them, and can apply MMR/reranking when configured.
4. The handoff policy packs retrieved text into the prompt budget.
5. The LLM generates an answer. The UI displays source metadata associated with the retrieved context.
6. Optional grounding logic can reject insufficiently supported output; it is disabled by default.

## Trust boundaries and limitations

- Local operation does not itself provide authorization. Anyone able to reach exposed services or read the mounted host directory may access data.
- Retrieved documents and model output are untrusted inputs. Prompt injection and unsupported answers remain possible.
- Citations identify supplied sources; they do not prove every answer statement is entailed by those sources.
- Traces, logs, OpenSearch, Qdrant, caches, and backups can contain document-derived data.
- Checked-in evaluation results apply only to their fixtures/configurations and should not be generalized to a new corpus.

For code-level functions and known coupling, see [`pipeline_map.md`](pipeline_map.md).

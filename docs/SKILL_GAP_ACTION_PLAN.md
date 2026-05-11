# Skill Gap Action Plan
# Document QA — Re-entry and Portfolio Development

Last updated: 2026-05-11
Status: Active

---

## How to Use This Document

This is your single re-entry point after any break.
Read the orientation section first, then go to the skill gap you want to work on.
Each gap has a concrete implementation task tied directly to this codebase.
No courses. No abstract exercises. Every task produces something defensible in an interview.

---

## Orientation: Where the Project Stands Right Now

Before doing anything else, read these two files to rebuild context:

1. `docs/pipeline_map.md` — full architecture, all pipeline stages, file ownership
2. `docs/plans/document_qa_optimization_plan.md` — what has been done, what is pending

### What is already done (strong portfolio evidence)

- Hybrid retrieval: dense (Qdrant) + lexical (OpenSearch) with fusion
- Two-stage reranking with cross-encoder
- Dynamic chunking by doc_type
- Query rewriting and query planning
- Financial document enrichment with suppression bug fix
- Qdrant payload slimmed to minimal keys (id, checksum, path)
- Grounding check (feature-flagged)
- Retrieval evaluation: Hit@1, Hit@3, MRR with fixture and baselines
- Documented baselines and before/after comparisons
- Celery async ingestion, Docker Compose, CI workflows

### What retrieval looks like right now

Post Path-A fix baseline (2026-03-27):
- hit@1 = 5/20 (0.25)
- hit@3 = 12/20 (0.60)
- control_with_results = 0/3 (abstention working)

This is real, documented, defensible evidence. It is a portfolio asset.

### What is still pending (your actual gaps)

See the five gaps below. Each one maps to a concrete task in this project.

---

## Gap 1: RAG / LLM Agent Orchestration

### Why it matters for interviews

Every AI engineering role asks about LangGraph, agent loops, and RAG pipelines.
Your project has retrieval and generation but no explicit agent layer.
This gap is the most visible one on your CV right now.

### What you already have

- `qa_pipeline/coordinator.py` — orchestrates retrieve → prompt → generate
- `core/query_rewriter.py` — rewrites queries before retrieval
- `qa_pipeline/grounding.py` — checks answer support after generation

This is a two-step linear pipeline. An agent would make it dynamic.

### Concrete task: Add a LangGraph agent loop to the QA pipeline

**What to build:**

A simple agent that decides, after retrieval, whether it has enough evidence to answer
or needs to re-retrieve with a different query. Three nodes:

1. `retrieve` — runs current hybrid retrieval
2. `evaluate_evidence` — checks if retrieved chunks are sufficient (use grounding score threshold)
3. `generate` — builds prompt and calls LLM

If `evaluate_evidence` returns insufficient, loop back to `retrieve` with a reformulated
query (max 2 iterations to prevent infinite loops).

**Files to create/modify:**

- Create `qa_pipeline/agent.py` — LangGraph graph definition
- Modify `qa_pipeline/coordinator.py` — add flag `USE_AGENT_LOOP` to switch between linear and agent path
- Update `config.py` — add `USE_AGENT_LOOP=false` (default off, safe)

**Install:**

```bash
pip install langgraph
```

**Estimated effort:** 2-3 days

**Interview explanation after completing this:**
> I added a LangGraph agent loop to the QA pipeline that evaluates evidence quality
> after retrieval and re-retrieves with a reformulated query if grounding confidence
> is below threshold. I kept the linear path as the default and made the agent opt-in
> via config to avoid regressions.

---

## Gap 2: Inference Serving (vLLM / HuggingFace TGI)

### Why it matters

Senior ML/NLP roles increasingly expect you to know how models are served in production,
not just how they are called via API. vLLM and TGI are the two dominant open-source
inference servers. Right now your project uses local Text-Generation-WebUI or Groq —
neither of which comes up in interviews.

### What you already have

- `core/llm.py` — LLM client with Groq and local TGW support
- `embedder_api_multilingual/app.py` — FastAPI embedding service (you already built a model-serving API)
- `docker-compose.yml` — service definitions

### Concrete task: Add vLLM as an optional LLM backend

**What to build:**

vLLM exposes an OpenAI-compatible API. Your `core/llm.py` already switches between
providers via config. Adding vLLM is a config + documentation task, not a deep code change.

1. Add `USE_VLLM` config flag and `VLLM_BASE_URL` to `config.py` and `.env.example`
2. In `core/llm.py`, add a vLLM branch that sets the base URL to your vLLM instance
   (same OpenAI-compatible format as Groq)
3. Add a `vllm` service to `docker-compose.yml` (commented out, opt-in)
4. Write `docs/runbooks/vllm_setup.md` explaining how to run a model with vLLM locally
   and point the project at it

**Then run it:** Start vLLM locally with any small open model (e.g. `mistralai/Mistral-7B-Instruct-v0.2`),
point your project at it, ask a question, confirm it works end-to-end.

**Estimated effort:** 1 day for code + config, 1 day to actually run it end-to-end

**Interview explanation after completing this:**
> I added vLLM as an optional inference backend. Since vLLM exposes an OpenAI-compatible
> API, the integration was a config-level change in the LLM client. I documented the setup
> in a runbook and verified it end-to-end with Mistral-7B locally. The key tradeoff versus
> TGI is that vLLM uses PagedAttention for higher throughput, while TGI has better support
> for production features like continuous batching and quantization.

---

## Gap 3: Open-Source Model Fine-Tuning (LoRA / QLoRA)

### Why it matters

Fine-tuning is the gap that separates ML engineers who only use APIs from those who can
adapt models. Even a small, documented fine-tuning experiment shows you understand the
process end-to-end.

### What you already have

- Your Arabic NLP background (700M posts, custom FastText embeddings) — strong base
- The Document QA project has domain-specific data: your personal document corpus
- Retrieval evaluation failures = ready-made hard negatives for reranker fine-tuning

### Concrete task: Fine-tune a small reranker or embedder on your document corpus

Pick one of the two options below.

**Option A (easier, directly relevant to this project):**

Fine-tune a cross-encoder reranker on hard negatives from your retrieval evaluation failures.

Your retrieval eval already has 20 positive queries with known relevant documents.
The residual failures (hit@3 misses) are your hard negatives. Use these to create a small
fine-tuning dataset and fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` with LoRA.

Files to create:
- `scripts/build_finetuning_dataset.py` — extract (query, positive_chunk, negative_chunk)
  triples from eval fixtures and residual failure analysis
- `scripts/finetune_reranker.py` — LoRA fine-tuning script using HuggingFace PEFT
- `docs/runbooks/reranker_finetuning.md` — dataset size, training config, before/after metrics

**Option B (stronger CV signal, connects Arabic NLP background):**

Fine-tune a small Arabic text classifier using your existing Arabic NLP expertise.
Pick any small Arabic dataset (e.g. sentiment or dialect identification from HuggingFace),
fine-tune `aubmindlab/bert-base-arabertv2` with QLoRA, document the experiment.

This connects your PhD-era Arabic NLP work to modern fine-tuning tooling — a strong narrative.

**Install:**

```bash
pip install peft transformers accelerate bitsandbytes
```

**Estimated effort:** 3-4 days

**Interview explanation after completing this (Option A):**
> I used the retrieval evaluation failures from my Document QA project as hard negatives
> to fine-tune the cross-encoder reranker with LoRA. The dataset was small — around 60
> triples — but the exercise demonstrated the full fine-tuning loop: dataset construction,
> PEFT config, training, and before/after evaluation on the retrieval fixture.

---

## Gap 4: MLOps — Experiment Tracking

### Why it matters

You already have evaluation scripts and documented baselines. This is further along than
most candidates. The missing piece is that your baselines live in JSON files in a runbooks
folder, not in a proper experiment tracker. That is a single step away.

### What you already have

- `scripts/run_retrieval_eval.py` — retrieval evaluation with Hit@k and MRR
- `scripts/run_qa_handoff_eval.py` — QA evaluation
- `docs/runbooks/retrieval_eval_*.json` — archived baseline snapshots
- This is already good evaluation discipline. It just needs MLflow on top.

### Concrete task: Add MLflow experiment tracking to retrieval evaluation

**What to build:**

1. Add MLflow logging to `scripts/run_retrieval_eval.py`:
   - Log parameters: `top_k`, `enable_variants`, `enable_mmr`, `enable_rerank`, `chunk_size`
   - Log metrics: `hit@1`, `hit@3`, `hit@5`, `MRR`, `control_false_positive_rate`
   - Log artifacts: the output JSON and CSV

2. Add `mlflow` service to `docker-compose.yml` (local tracking server)

3. Run your existing baselines through MLflow — your Path A before/after comparison
   becomes a tracked experiment with a UI

4. Write `docs/runbooks/experiment_tracking.md` — how to run an eval and view results

**Install:**

```bash
pip install mlflow
```

**Estimated effort:** 1 day

**Interview explanation after completing this:**
> I added MLflow tracking to the retrieval evaluation scripts. Every eval run now logs
> retrieval config parameters and Hit@k/MRR metrics, making before/after comparisons
> reproducible and auditable. This was how I tracked the Path A improvement — hit@3
> going from 0.25 to 0.60 — without manually diffing JSON files.

---

## Gap 5: The Career Break Narrative on the CV

### Why it matters

The gap is visible. "Integration in Germany and German study" reads as passive.
What you actually did during the break is technically substantial. The CV does not
reflect that yet.

### What is true and defensible

During the career break you:
- Built a full hybrid RAG pipeline (Document QA) with documented retrieval improvements
- Built and shipped an Android app to Google Play (GermanVerbMaster)
- Built a voice companion PWA (Rafiq) for Arabic-speaking users navigating German bureaucracy
- Achieved B1 German certification, preparing for B2
- Ran structured retrieval evaluations with before/after baselines

None of this is on the CV in a way that competes with the career break label.

### Concrete task: Rewrite the career break section

Replace the current one-liner with this:

```
Career break since Apr 2024: independent technical development alongside
German-language study (B1 certified May 2025, B2 April 2026).

Independent projects:
- Document QA / RAG pipeline: hybrid retrieval (Qdrant + OpenSearch), two-stage
  reranking, dynamic chunking, query planning, and retrieval evaluation (Hit@3
  improved from 0.25 to 0.60 across a structured fixture of 20 queries).
- GermanVerbMaster: Android app (Kotlin/Jetpack Compose) published to Google Play,
  Supabase backend, 3,000+ word corpus with CEFR-level filtering.
- Rafiq: Arabic-language voice companion PWA (Flask + Groq + Edge TTS) for Arabic
  speakers navigating German bureaucracy.
```

**Estimated effort:** 1 hour

---

## Execution Order

| Order | Gap | Task | Effort | Impact |
|---|---|---|---|---|
| 1 | Career break narrative | Rewrite CV section | 1 hour | Immediate — removes passive framing today |
| 2 | MLOps | Add MLflow to retrieval eval | 1 day | Makes existing baselines portfolio-grade |
| 3 | Inference serving | Add vLLM backend | 2 days | Closes most-asked infrastructure gap |
| 4 | RAG / agent orchestration | LangGraph agent loop | 3 days | Closes highest-visibility CV gap |
| 5 | Fine-tuning | Reranker or Arabic classifier | 4 days | Strongest signal for ML engineering roles |

Total estimated effort: ~11 focused days.

---

## Quick Re-entry Checklist

Use this after any break before starting a session.

Before starting:
1. Read this file (5 minutes)
2. Confirm the system still works:
   ```bash
   python scripts/run_retrieval_eval.py
   ```
3. Check which gap you are currently working on
4. Pick up from the last incomplete task in that gap

Before ending any session:
1. Commit what you changed with a clear message
2. Update the status line of the gap you worked on in this file
3. Note one thing you can now explain in an interview that you could not before

---

## Interview Question Map

| Question you will get | Which gap covers it | Where to point |
|---|---|---|
| Walk me through your RAG pipeline | All + existing work | `docs/pipeline_map.md` |
| Why hybrid retrieval? | Existing work | `core/retrieval/fusion.py` |
| How did you evaluate retrieval quality? | Gap 4 + existing | `scripts/run_retrieval_eval.py` |
| What is LangGraph and have you used it? | Gap 1 | `qa_pipeline/agent.py` |
| How do you serve models in production? | Gap 2 | `core/llm.py`, `docs/runbooks/vllm_setup.md` |
| Have you fine-tuned any models? | Gap 3 | `scripts/finetune_reranker.py` |
| How do you track experiments? | Gap 4 | MLflow UI, `scripts/run_retrieval_eval.py` |
| What did you do during your career break? | Gap 5 | CV career break section |
| How did you use AI assistance responsibly? | All | `AGENTS.md` |

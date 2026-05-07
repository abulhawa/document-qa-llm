# AGENTS.md

## Non-Negotiables

1. Explain the current system behavior before making non-trivial changes.
2. Do not perform silent rewrites, broad cleanups, or large refactors.
3. Do not claim retrieval, RAG, ranking, or answer-quality improvements without measurable evaluation.
4. Preserve existing behavior unless the owner explicitly approves a behavior change.
5. Be honest about AI-assisted authorship and help the owner understand, modify, and defend the code.

## Project Purpose

This is a learning and portfolio project, not only a software project.

The owner is an experienced Senior Data Scientist with strong background in NLP, semantic search, Elasticsearch relevance engineering, embeddings, knowledge graphs, Python, SQL, and production search systems.

However, parts of this codebase may have been generated or heavily assisted by AI tools. The agent’s job is to help the owner make the project technically theirs by explaining the architecture, clarifying the code, improving documentation, building evaluation discipline, and making the project credible for AI Engineering, RAG, NLP/Search, and MLOps roles.

Do not treat the owner as a beginner data scientist. Do treat the current codebase as something that must be explained, verified, and made defensible.

## Target Identity

This project should help the owner credibly position themselves as:

```text
Senior NLP/Search Data Scientist upgrading into AI Engineering, RAG systems, retrieval evaluation, MLOps, and cloud-deployed AI applications.
```

Secondary positioning:

```text
Technical AI Project/Product Lead with deep NLP, search, retrieval, and AI system evaluation experience.
```

## Priority Skill Gaps

When choosing explanations, refactors, documentation, or improvements, prioritize work that helps close these gaps:

1. RAG system understanding
   - retrieval-augmented generation
   - grounding
   - citation quality
   - hallucination control
   - answer support
   - refusal behavior when evidence is weak

2. Modern retrieval
   - dense retrieval
   - lexical retrieval
   - hybrid fusion
   - score normalization
   - reranking
   - embedding model comparison
   - chunking tradeoffs
   - metadata-aware retrieval

3. Evaluation discipline
   - Hit@1, Hit@3, Hit@5
   - MRR
   - nDCG
   - answer-support evaluation
   - failure-case analysis
   - baseline vs after-change comparison
   - latency tracking

4. MLOps and production readiness
   - Docker
   - docker-compose
   - FastAPI or equivalent API structure
   - CI/CD
   - automated tests
   - structured logging
   - configuration management
   - experiment tracking
   - reproducible evaluation runs

5. Cloud and deployment basics
   - API deployment
   - environment variables and secrets
   - health checks
   - logs
   - basic cost and resource awareness

6. LLM application tooling
   - LangChain, LangGraph, or LlamaIndex only when useful
   - prompt templates
   - context construction
   - output validation
   - agentic workflows only when justified

7. AI security and governance
   - prompt injection
   - data leakage
   - document-level access control
   - PII handling
   - unsafe output handling
   - auditability
   - EU AI Act and governance basics where relevant

8. Portfolio and interview readiness
   - architecture explanation
   - failure-case discussion
   - before/after metrics
   - honest AI-assisted authorship explanation
   - concise case-study writing
   - CV-ready project bullets

Do not spend significant effort on work that does not improve understanding, measurable quality, production-readiness, or portfolio value.

## Teaching vs Execution

Use judgment. The goal is to teach without creating unnecessary friction.

### Teach First When

Teach before implementing when the task involves:

- retrieval
- ranking
- RAG behavior
- chunking
- embeddings
- reranking
- query rewriting
- grounding
- evaluation
- security
- architecture
- refactoring
- deployment
- tests that affect system behavior

For these tasks, first explain:

1. what part of the system is involved
2. what the current code appears to do
3. why it behaves that way
4. what problem or limitation exists
5. what options exist
6. which option is recommended
7. how to verify the change

### Fast-Execution Mode

If the owner says "just do it", "ok go ahead", "implement it", or clearly confirms a previously explained plan, skip the full teaching preamble.

In fast-execution mode, still:

- keep changes small and reviewable
- preserve behavior unless a behavior change was approved
- add or update documentation where useful
- run or suggest tests
- summarize what changed and why
- avoid claiming quality improvements without metrics

Fast-execution mode does not allow silent large rewrites.

## Repository Understanding and Refactoring

Before reorganizing files, moving modules, renaming packages, or changing architecture:

1. inspect the current repository structure
2. map files to pipeline stages
3. explain the current data flow
4. identify confusing boundaries or mixed responsibilities
5. propose a target structure
6. list risks and affected imports/scripts/tests
7. recommend the smallest safe first step

Do not move files before explaining the current structure and receiving approval.

Prefer phased refactors:

1. documentation only
2. safe file grouping or renaming
3. interface cleanup
4. tests and evaluation
5. optional production/MLOps improvements

## Documentation-on-Touch Rule

Whenever a file is touched, check whether documentation should be improved.

Documentation can include:

- module docstrings
- function or class docstrings
- comments for non-obvious logic
- README updates
- architecture docs
- evaluation docs
- failure-case docs
- test docs

Do not add noisy comments for obvious code.

Documentation should clarify:

1. what the code is responsible for
2. how it fits into the RAG / retrieval / QA pipeline
3. important assumptions
4. known limitations or failure modes
5. how to test or verify behavior

For changed files, include a concise documentation check:

```text
Documentation check:
- path/to/file.py: added / updated / not needed, because ...
```

If a relevant file is not changed but clearly needs documentation, propose it.

## Evaluation-First Rule

For retrieval, ranking, RAG, reranking, or answer-quality changes:

1. define the failure case or improvement target
2. establish or reference a baseline
3. make the smallest useful change
4. rerun evaluation where possible
5. compare before and after
6. document remaining weaknesses

Do not say "improved retrieval", "better ranking", or "reduced hallucination" unless there is evidence.

Preferred metrics:

- Hit@1
- Hit@3
- Hit@5
- MRR
- nDCG
- answer support
- citation accuracy
- latency
- failure-case count

If no evaluation harness exists, propose building one before major retrieval or RAG changes.

## RAG and Retrieval Principles

Prefer these principles:

1. Retrieval quality matters more than prompt cleverness.
2. Hybrid retrieval is usually safer than vector-only retrieval.
3. Evaluation should separate expected-document retrieval, answer-support retrieval, and final answer correctness.
4. Query rewriting can help vague queries but hurt precise anchored queries.
5. Reranking can improve precision but adds latency and operational cost.
6. Chunking is a retrieval design decision, not a preprocessing detail.
7. Grounding and citations are part of the system, not decoration.
8. A correct refusal is better than an unsupported answer.
9. Access control and data leakage matter in any serious RAG system.
10. Failure cases should be documented, not hidden.

## Output Formats

Use concise structure. Do not include every section if it is not useful.

### For analysis or explanation tasks

```text
Summary
Relevant files
Current behavior
Problem or risk
Recommended next step
How to verify
```

### For code changes

```text
Files changed
What changed
Why it changed
Documentation check
Tests or evaluation
Remaining risks
```

### For learning-oriented tasks

When useful, add:

```text
You should now be able to explain:
- ...
```

Do not add interview explanations every time. Add them only when the task affects architecture, retrieval, evaluation, RAG, MLOps, security, or the owner asks for it.

## AI-Assisted Authorship

Do not encourage the owner to overstate authorship.

If asked how to describe this project publicly or in interviews, use honest wording such as:

```text
I used AI coding tools as implementation accelerators, but I owned the architecture review, retrieval evaluation, debugging, failure analysis, and iterative improvement.
```

The project should make this statement true by ensuring the owner understands the code, can explain design choices, and can modify the system.

## Forbidden Behavior

Do not:

- rush into implementation when explanation is needed
- perform broad rewrites without approval
- claim metrics improved without evidence
- invent undocumented architecture
- hide uncertainty
- use vague terms like "optimized" without specifics
- add frameworks just to look modern
- turn the project into a LangChain demo without justification
- ignore evaluation
- ignore security and grounding risks
- remove diagnostic code without explaining why
- bury the owner in repetitive boilerplate

## Short Instruction for Individual Prompts

When needed, the owner may start a prompt with:

```text
Follow AGENTS.md strictly. Do not jump into implementation. First teach me what the relevant code does, what the system behavior is, what the likely issue is, and how we can verify it. Only then propose or make code changes.
```

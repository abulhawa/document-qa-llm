# Composite v2 source expansion

`composite-v2` expands the Open RAGBench arXiv PDF corpus from 200 to all 1,000
PDFs at the same pinned upstream revision used by `composite-v1`. It retains
the same 80 positive PDFs and 160 text questions. The existing 120 hard
v1 distractors are retained; 800 additional PDFs are added. All
other benchmark tracks and their query/document selections stay fixed.

The upstream arXiv subset has only 604 PDFs that are never gold for **any**
upstream question, so the v1 pool rule cannot supply 920 distractors.
For v2, a distractor is defined relative to the **selected 160 questions**.
Some added PDFs are relevant to unselected upstream questions, but none is a
gold document for a selected question. Selection does not use semantic or
lexical similarity, so these PDFs are called *distractors* rather than *hard
negatives*. This distinction is explicit in the v2 composition file and lock.

The v2 lock copies the four unchanged tracks and pinned upstream revisions
from the frozen v1 lock. Its Open RAGBench selection contains the same v1
positives and queries plus all remaining PDFs. This avoids accidental drift
if upstream datasets change later.

## Incremental processing

The cloud workflows have a `composite-v2` path that uses the persisted
`composite-v1` artifacts as its predecessor. The v1 artifacts remain in place.

1. **Source:** Copy unchanged documents from the v1 source artifact and download
   the 800 newly selected PDFs. The four unchanged tracks are copied in full.
2. **Parse:** Reuse a v1 document when its source hash, file type, parser code,
   parser settings, Python version, and relevant package versions match. Parse
   only the remaining documents.
3. **Chunk:** Reuse a v1 chunk set when the parsed content and chunker signature
   match. Split only the remaining documents.
4. **Embed:** Reuse vectors by exact track, document, chunk index, chunk ID, and
   text hash when the model and runtime signature match. Encode only misses.
   Existing checkpoints still allow an interrupted shard to resume.
5. **Index:** Restore the v1 OpenSearch and Qdrant snapshots. Retain existing
   entries for reused chunks, insert new entries, and delete entries no longer
   in the target corpus. If the embedding signature changes, Qdrant updates
   the existing points with the new vectors. The backend versions and index
   contracts must match the predecessor snapshots.

Each v2 stage writes a full artifact manifest with a `reuse` summary. The
v2 paths of reused chunks retain their original `benchmark://composite-v1/`
source URI so their native index IDs stay stable; the evaluator accepts those
source URIs when scoring v2. New chunks use v2 source URIs.

Run the five benchmark workflows in order: Source Corpus, Parse, Chunk, Embed,
then Index. Select `composite-v2` in each job. For Source Corpus, use
`evaluation/benchmarks/composite_v2.yaml` and
`evaluation/benchmarks/composite_v2.lock.json`. For Chunk, Embed, and Index,
pass the **v2** parent artifact fingerprints from the preceding persisted
manifests. Their form defaults still name v1 fingerprints. Run Benchmark 6
retrieval evaluation with the resulting v2 index and the same 160 questions.

The fixed query set makes a v1/v2 comparison useful for measuring retrieval
under a larger corpus. Report corpus sizes alongside metrics because the two
runs are different benchmark conditions.

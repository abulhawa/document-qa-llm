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

## Processing policy

The source lock is a membership definition. The existing v1 parse, chunk,
embedding, and index workflows still fingerprint entire parent artifacts and
would recompute unchanged documents for v2. Do not launch those legacy jobs
for `composite-v2`. Use the verified cache indexes described in
[Reusable benchmark cache indexes](benchmark_cache.md) when implementing v2
stage readers: match source hashes and stage signatures, reuse existing output
for unchanged documents and chunks, and compute only cache misses. Preserve
all v1 bucket artifacts. A separate v2 manifest should record reused and new
entries, checksums, and any cache misses or ambiguous embedding text keys.

The fixed query set makes a v1/v2 comparison useful for measuring retrieval
under a larger corpus. Report the corpus sizes alongside metrics, since the
two runs are different benchmark conditions.


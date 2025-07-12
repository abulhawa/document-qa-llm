import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DOCQA_TEST_MODE"] = "1"

from faiss_store import rebuild_faiss_index, query_faiss

print("Rebuilding FAISS index...")
index = rebuild_faiss_index()
assert index is not None, "Index rebuild failed."

print("Querying FAISS...")
results = query_faiss("What is this document about?", top_k=3)
for r in results:
    print(f"{r['score']:.3f} → {r['chunk']['content'][:80]}...")

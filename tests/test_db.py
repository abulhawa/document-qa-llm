import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db import create_tables, get_indexed_chunk_count, get_all_chunks_with_embeddings

print("Creating DB tables...")
create_tables()

print("Chunks indexed:", get_indexed_chunk_count())

chunks = get_all_chunks_with_embeddings()
print(f"Retrieved {len(chunks)} chunks with embeddings")
if chunks:
    print("Sample chunk:", chunks[0])

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion import ingest

# Test a local file
print("Testing file ingestion...")
ingest("./tests/sample_docs/example.pdf")  # Path to a known good test file

# Test a folder
print("Testing folder ingestion...")
ingest("./tests/sample_docs")
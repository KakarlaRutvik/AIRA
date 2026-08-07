"""
Phase 2 + 3 Integration Test: Real Document Retrieval

Loads the actual sample documents, builds a real FAISS index, and runs
realistic cross-document queries to determine whether nomic-embed-text's
earlier "hub" behavior was a small-sample artifact or a real problem.

Usage:
    python src/test_real_retrieval.py [model_name]

Default model: nomic-embed-text
"""

import sys

sys.path.insert(0, "src")

from document_loader import load_and_chunk_directory
from vector_store import build_index, search

EMBED_MODEL = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text"

print("=" * 70)
print(f"Real Retrieval Test — embedding model: {EMBED_MODEL}")
print("=" * 70)

print("\n[1] Loading and chunking real documents...")
chunks, sources = load_and_chunk_directory("data/documents")
print(f"    Loaded {len(chunks)} chunks from {len(set(sources))} files")

print("\n[2] Building FAISS index (this will take a moment)...")
# vector_store.py's embed_document uses EMBED_MODEL global - patch it if a
# different model was passed on the command line
import vector_store
vector_store.EMBED_MODEL = EMBED_MODEL

index, indexed_chunks = build_index(chunks)
print(f"    Index built with {index.ntotal} vectors")

# Test queries, each one designed to have a CLEAR correct source file.
# Format: (query, expected_source_filename_substring)
test_cases = [
    ("How much is the home office stipend for new remote employees?", "remote_work_policy"),
    ("What is the rate limit for free tier accounts?", "skycart_api_docs"),
    ("Who is responsible for the LinkedIn campaign?", "q3_marketing_meeting_notes"),
    ("What happens if I exceed the API rate limit?", "skycart_api_docs"),
    ("How many times per year must remote employees visit headquarters?", "remote_work_policy"),
    ("What was the email open rate in Q2?", "q3_marketing_meeting_notes"),
    ("What security requirements apply to remote employees?", "remote_work_policy"),
    ("What status code do webhooks need to return?", "skycart_api_docs"),
]

print("\n[3] Running real cross-document queries...\n")

results_summary = []
for query, expected_file in test_cases:
    top_chunks = search(index, indexed_chunks, query, k=1)
    if not top_chunks:
        results_summary.append((query, expected_file, "NO RESULT", False))
        continue

    top_chunk = top_chunks[0]
    # find which source file this chunk came from
    chunk_index = indexed_chunks.index(top_chunk)
    actual_source = sources[chunk_index]

    passed = expected_file in actual_source
    results_summary.append((query, expected_file, actual_source, passed))

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Query: {query}")
    print(f"         Expected source: {expected_file}  |  Got: {actual_source}")
    print(f"         Top chunk: {top_chunk[:100]}...")
    print()

print("=" * 70)
passed_count = sum(1 for r in results_summary if r[3])
total = len(results_summary)
print(f"RESULT: {passed_count}/{total} queries retrieved from the correct source document")
print("=" * 70)

if passed_count == total:
    print(f"\nAll queries correctly retrieved from their expected source.")
    print(f"'{EMBED_MODEL}' is working well on realistic document data.")
elif passed_count >= total * 0.75:
    print(f"\nMost queries passed. Minor misses may just be edge cases (overlapping")
    print(f"chunk boundaries, ambiguous phrasing) rather than a systemic problem.")
else:
    print(f"\nSignificant retrieval failures on real data. Consider switching")
    print(f"embedding models: python src/test_real_retrieval.py mxbai-embed-large")
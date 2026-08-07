"""
Phase 2: Embedding + FAISS Core

This is the foundational module for the whole project. It handles:
  - Converting text into vectors using nomic-embed-text (via Ollama)
  - Building a FAISS index from those vectors
  - Searching the index to retrieve the most relevant chunks for a query
  - Saving/loading the index + metadata to/from disk

Usage (standalone test):
    python src/vector_store.py
"""

import os
import pickle

import faiss
import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"


def embed_document(text: str) -> list[float]:
    """Embed a piece of text that will be STORED (uses the document prefix)."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=f"search_document: {text}")
    return resp["embedding"]


def embed_query(text: str) -> list[float]:
    """Embed a piece of text that is a SEARCH QUERY (uses the query prefix).

    Nomic's model is trained with different prefixes for stored documents vs.
    search queries. Using the right one for each improves retrieval quality.
    """
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=f"search_query: {text}")
    return resp["embedding"]


def build_index(chunks: list[str]) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Build a FAISS index from a list of text chunks.

    Returns the index plus the original chunks list (kept position-aligned
    with the index so we can map a search result back to its source text).

    IMPORTANT: vectors are L2-normalized before indexing. nomic-embed-text
    vectors are not guaranteed unit length, and raw Euclidean distance on
    un-normalized vectors is dominated by magnitude rather than direction
    (meaning) -- this causes wrong, seemingly random nearest matches.
    Normalizing first makes L2 distance behave consistently with cosine
    similarity ranking.
    """
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")

    vectors = np.array([embed_document(c) for c in chunks], dtype="float32")
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, chunks


def add_to_index(index: faiss.IndexFlatL2, chunks: list[str], new_chunks: list[str]) -> list[str]:
    """Add new chunks to an existing index in place. Returns the updated chunks list."""
    new_vectors = np.array([embed_document(c) for c in new_chunks], dtype="float32")
    faiss.normalize_L2(new_vectors)
    index.add(new_vectors)
    chunks.extend(new_chunks)
    return chunks


def search(index: faiss.IndexFlatL2, chunks: list[str], query: str, k: int = 3) -> list[str]:
    """Search the index for the k most relevant chunks to the query."""
    if index.ntotal == 0:
        return []
    k = min(k, index.ntotal)
    qvec = np.array([embed_query(query)], dtype="float32")
    faiss.normalize_L2(qvec)
    _, indices = index.search(qvec, k)
    return [chunks[i] for i in indices[0] if i != -1]


def save_index(index: faiss.IndexFlatL2, chunks: list[str], path_prefix: str) -> None:
    """Save the FAISS index and its metadata (chunks) to disk.

    Creates two files: {path_prefix}.faiss and {path_prefix}.pkl
    """
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    faiss.write_index(index, f"{path_prefix}.faiss")
    with open(f"{path_prefix}.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index(path_prefix: str) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Load a previously saved FAISS index and its metadata from disk."""
    index = faiss.read_index(f"{path_prefix}.faiss")
    with open(f"{path_prefix}.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Vector Store Standalone Test")
    print("=" * 60)

    sample_chunks = [
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Python is a popular programming language known for its readability.",
        "The mitochondria is the powerhouse of the cell.",
        "FAISS is a library for efficient similarity search developed by Meta.",
        "The Great Wall of China is over 13,000 miles long.",
    ]

    print("\n[1] Building index from sample chunks...")
    index, chunks = build_index(sample_chunks)
    print(f"    OK - Index built with {index.ntotal} vectors, dimension {index.d}")

    print("\n[2] Saving index to disk (vectorstore/test_index)...")
    save_index(index, chunks, "vectorstore/test_index")
    print("    OK - Saved vectorstore/test_index.faiss and .pkl")

    print("\n[3] Loading index back from disk...")
    loaded_index, loaded_chunks = load_index("vectorstore/test_index")
    print(f"    OK - Loaded index with {loaded_index.ntotal} vectors")

    print("\n[4] Running test queries...")
    test_queries = [
        ("Where is the Eiffel Tower?", "Eiffel Tower"),
        ("What does FAISS do?", "FAISS"),
        ("Tell me about cell biology", "mitochondria"),
    ]

    all_passed = True
    for query, expected_keyword in test_queries:
        results = search(loaded_index, loaded_chunks, query, k=2)
        top_result = results[0] if results else ""
        passed = expected_keyword.lower() in top_result.lower()
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] Query: '{query}'")
        print(f"           Top match: {top_result[:70]}...")

    print("\n" + "=" * 60)
    if all_passed:
        print("All retrieval tests PASSED. Phase 2 core logic is working correctly.")
    else:
        print("Some retrieval tests FAILED. Check embedding quality / chunk content.")
    print("=" * 60)
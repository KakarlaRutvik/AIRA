"""
Debug script: prints raw cosine similarity between each query and every
chunk, so we can see the actual numbers instead of just the top-1 result.

Usage:
    python src/debug_embeddings.py
"""

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"

sample_chunks = [
    "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    "Python is a popular programming language known for its readability.",
    "The mitochondria is the powerhouse of the cell.",
    "FAISS is a library for efficient similarity search developed by Meta.",
    "The Great Wall of China is over 13,000 miles long.",
]

queries = [
    "Where is the Eiffel Tower?",
    "What does FAISS do?",
    "Tell me about cell biology",
]


def embed_document(text):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=f"search_document: {text}")
    return np.array(resp["embedding"], dtype="float32")


def embed_query(text):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=f"search_query: {text}")
    return np.array(resp["embedding"], dtype="float32")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("=" * 70)
print("Embedding chunks...")
print("=" * 70)
chunk_vecs = []
for c in sample_chunks:
    v = embed_document(c)
    chunk_vecs.append(v)
    print(f"  norm={np.linalg.norm(v):.4f}  first5={v[:5]}  | {c[:50]}")

print("\n" + "=" * 70)
print("Checking if chunk embeddings are actually distinct from each other")
print("=" * 70)
for i in range(len(sample_chunks)):
    for j in range(i + 1, len(sample_chunks)):
        sim = cosine_sim(chunk_vecs[i], chunk_vecs[j])
        print(f"  sim(chunk{i}, chunk{j}) = {sim:.4f}")

print("\n" + "=" * 70)
print("Query similarity breakdown")
print("=" * 70)
for q in queries:
    qvec = embed_query(q)
    print(f"\nQuery: '{q}'  (norm={np.linalg.norm(qvec):.4f})")
    sims = [(cosine_sim(qvec, chunk_vecs[i]), sample_chunks[i]) for i in range(len(sample_chunks))]
    sims.sort(reverse=True)
    for sim, text in sims:
        print(f"    {sim:.4f}  | {text[:60]}")


# ---------------------------------------------------------------------------
# Round 2: realistic paragraph-length chunks (what Phase 3 will actually feed in)
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("ROUND 2: realistic paragraph-length chunks (~80-100 words each)")
print("=" * 70)

long_chunks = [
    (
        "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars "
        "in Paris, France. It was designed by engineer Gustave Eiffel and completed in "
        "1889 as the entrance arch for the World's Fair. Standing at 330 meters tall, "
        "it was the tallest man-made structure in the world for over 40 years and "
        "remains one of the most visited paid monuments globally, attracting millions "
        "of tourists every year to its observation decks."
    ),
    (
        "Python is a high-level, general-purpose programming language known for its "
        "clean and readable syntax, which makes it popular among beginners and "
        "experienced developers alike. It supports multiple programming paradigms, "
        "including procedural, object-oriented, and functional programming. Python has "
        "a vast ecosystem of libraries for web development, data science, machine "
        "learning, and automation, and is widely used by companies like Google, "
        "Instagram, and Netflix."
    ),
    (
        "Mitochondria are membrane-bound organelles found in most eukaryotic cells, "
        "often referred to as the powerhouse of the cell because they generate most "
        "of the cell's supply of adenosine triphosphate, or ATP, used as a source of "
        "chemical energy. Mitochondria have their own DNA, separate from the cell's "
        "nucleus, and are believed to have originated from ancient bacteria that formed "
        "a symbiotic relationship with early eukaryotic cells."
    ),
    (
        "FAISS, short for Facebook AI Similarity Search, is an open-source library "
        "developed by Meta for efficient similarity search and clustering of dense "
        "vectors. It supports searching in sets of vectors of any size, even ones that "
        "do not fit in RAM, and provides several indexing structures that trade off "
        "between search speed, accuracy, and memory usage, making it a popular choice "
        "for building retrieval-augmented generation systems."
    ),
    (
        "The Great Wall of China is a series of fortifications built across the "
        "historical northern borders of China to protect against invasions from "
        "various nomadic groups. Stretching over 13,000 miles when including all its "
        "branches, it was built over many centuries by different Chinese dynasties, "
        "with the most famous sections dating to the Ming Dynasty. It is one of the "
        "most impressive architectural feats in human history."
    ),
]

print("\nEmbedding longer chunks...")
long_vecs = [embed_document(c) for c in long_chunks]

print("\nPairwise similarity between long chunks:")
for i in range(len(long_chunks)):
    for j in range(i + 1, len(long_chunks)):
        sim = cosine_sim(long_vecs[i], long_vecs[j])
        print(f"  sim(chunk{i}, chunk{j}) = {sim:.4f}")

print("\nQuery ranking against LONG chunks:")
for q in queries:
    qvec = embed_query(q)
    print(f"\nQuery: '{q}'")
    sims = [(cosine_sim(qvec, long_vecs[i]), long_chunks[i]) for i in range(len(long_chunks))]
    sims.sort(reverse=True)
    for sim, text in sims:
        print(f"    {sim:.4f}  | {text[:70]}...")


# ---------------------------------------------------------------------------
# Round 3: try the newer /api/embed batch endpoint (ollama.embed) instead of
# the older /api/embeddings endpoint (ollama.embeddings), with NO prefix,
# to isolate whether the endpoint or the prefix is causing the issue.
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("ROUND 3: newer ollama.embed() endpoint, NO search_document/search_query prefix")
print("=" * 70)

resp = ollama.embed(model=EMBED_MODEL, input=long_chunks)
new_chunk_vecs = [np.array(v, dtype="float32") for v in resp["embeddings"]]

print("\nPairwise similarity (Round 3):")
for i in range(len(long_chunks)):
    for j in range(i + 1, len(long_chunks)):
        sim = cosine_sim(new_chunk_vecs[i], new_chunk_vecs[j])
        print(f"  sim(chunk{i}, chunk{j}) = {sim:.4f}")

print("\nQuery ranking (Round 3, no prefix):")
for q in queries:
    qresp = ollama.embed(model=EMBED_MODEL, input=[q])
    qvec = np.array(qresp["embeddings"][0], dtype="float32")
    print(f"\nQuery: '{q}'")
    sims = [(cosine_sim(qvec, new_chunk_vecs[i]), long_chunks[i]) for i in range(len(long_chunks))]
    sims.sort(reverse=True)
    for sim, text in sims:
        print(f"    {sim:.4f}  | {text[:70]}...")
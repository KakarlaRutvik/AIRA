"""
Phase 9: Conversation Memory

Applies the EXACT SAME embedding + FAISS pattern from Phase 2 (vector_store.py)
to conversation history instead of documents. After every exchange, the
question+answer pair is embedded and stored in a SEPARATE index from your
documents - so document search and memory search never interfere with
each other.

This is why vector_store.py was written generically (embed_document,
build_index, search work on ANY list of text chunks) rather than hardcoded
to documents - the exact same functions work here with zero changes.

Usage (standalone test):
    python src/memory.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vector_store import embed_document, embed_query, save_index, load_index
import faiss
import numpy as np

MEMORY_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "memory_index")


class ConversationMemory:
    """A second, separate FAISS index dedicated to past Q&A exchanges.

    Kept entirely independent from the document index - this index only
    ever contains conversation history, never document chunks.
    """

    def __init__(self):
        self.index = None  # created lazily on first add() - FAISS needs a known dimension
        self.entries = []  # parallel list: entries[i] is the text for vector i

    def add_exchange(self, query: str, answer: str):
        """Embed a (query, answer) pair and add it to the memory index."""
        entry_text = f"Q: {query}\nA: {answer}"
        vec = np.array([embed_document(entry_text)], dtype="float32")
        faiss.normalize_L2(vec)

        if self.index is None:
            dim = vec.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(vec)
        self.entries.append(entry_text)

    def search(self, query: str, k: int = 2) -> list[str]:
        """Return the k most relevant past exchanges to this new query.
        Returns an empty list if memory is empty (nothing stored yet)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        qvec = np.array([embed_query(query)], dtype="float32")
        faiss.normalize_L2(qvec)
        _, indices = self.index.search(qvec, k)
        return [self.entries[i] for i in indices[0] if i != -1]

    def save(self):
        if self.index is not None:
            save_index(self.index, self.entries, MEMORY_INDEX_PATH)

    def load(self):
        if os.path.exists(f"{MEMORY_INDEX_PATH}.faiss"):
            self.index, self.entries = load_index(MEMORY_INDEX_PATH)


# ---------------------------------------------------------------------------
# Standalone test: simulate a short multi-turn conversation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 9: Conversation Memory Test")
    print("=" * 70)

    memory = ConversationMemory()

    print("\n[1] Simulating a conversation - storing exchanges as they happen...")
    exchanges = [
        ("How much is the home office stipend?", "The home office stipend is $750 for new remote employees."),
        ("What's the rate limit for the SkyCart API free tier?", "100 requests per minute, 10,000 per day."),
        ("What did the marketing team decide about paid social?", "They shifted 15% of the budget into a new LinkedIn campaign."),
    ]

    for q, a in exchanges:
        memory.add_exchange(q, a)
        print(f"    Stored: Q: {q[:50]}...")

    print(f"\n    Memory now has {memory.index.ntotal} exchanges stored.")

    print("\n[2] Testing recall with NEW questions that reference earlier turns...")
    test_queries = [
        "What did I ask about earlier regarding remote work costs?",
        "Remind me what the API rate limits were",
        "What was decided about the ad budget?",
    ]

    for q in test_queries:
        results = memory.search(q, k=1)
        print(f"\n    New question: {q}")
        if results:
            print(f"    Recalled: {results[0][:100]}...")
        else:
            print("    No relevant memory found.")

    print("\n[3] Saving memory to disk and reloading (persistence check)...")
    memory.save()
    reloaded = ConversationMemory()
    reloaded.load()
    print(f"    Reloaded memory has {reloaded.index.ntotal} exchanges - "
          f"{'MATCH' if reloaded.index.ntotal == memory.index.ntotal else 'MISMATCH'}")

    print("\n" + "=" * 70)
    print("If recalled memories above correctly match their related topic,")
    print("conversation memory is working.")
    print("=" * 70)
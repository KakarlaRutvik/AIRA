"""
Phase 4: RAG Answer Generation — Gemini Only

Full question -> retrieve -> answer loop, hardcoded to Gemini (no router,
no agent yet). This validates retrieval + generation quality together
before we add routing/agent complexity in Phase 5/6.

Usage:
    python src/rag.py "your question here"
    python src/rag.py   # runs a fixed set of demo questions
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from document_loader import load_and_chunk_directory
from vector_store import build_index, search
from llm_router import LLMRouter

GEMINI_MODEL = "gemini-flash-latest"  # kept for reference; actual model choice now lives in llm_router.py


def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant answering questions using ONLY the context below.

If the answer is not contained in the context, say clearly that you don't have
that information in the provided documents. Do not make up information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def rag_answer(query: str, index, indexed_chunks: list[str], router: LLMRouter, k: int = 3) -> tuple[str, str]:
    """Full RAG loop: retrieve relevant chunks, then ask the router to answer using them.

    Returns (answer_text, source) where source is "cloud" or "local" -
    telling you whether Gemini or the local Ollama model actually answered.
    """
    context_chunks = search(index, indexed_chunks, query, k=k)

    if not context_chunks:
        return "No documents have been indexed yet, so I have no context to answer from.", "none"

    prompt = build_prompt(query, context_chunks)
    answer, source = router.invoke(prompt)
    return answer, source


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4: RAG Answer Generation (Gemini only)")
    print("=" * 70)

    print("\n[1] Loading and indexing documents from data/documents ...")
    chunks, sources = load_and_chunk_directory("data/documents")
    if not chunks:
        print("No documents found in data/documents. Add files and re-run.")
        sys.exit(1)

    index, indexed_chunks = build_index(chunks)
    print(f"    Indexed {index.ntotal} chunks from {len(set(sources))} files")

    print("\n[2] Setting up LLM Router (Gemini primary, Ollama fallback)...")
    router = LLMRouter()
    if router.gemini_client is None:
        print("    No Gemini key found — running in offline-only mode (Ollama only).")
    else:
        print(f"    Ready. Will try Gemini first, fall back to llama3.2:3b if needed.")

    # If a question was passed on the command line, just answer that one.
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuestion: {query}")
        print("-" * 70)
        answer, source = rag_answer(query, index, indexed_chunks, router)
        print(f"[Answered by: {'🟢 Gemini (cloud)' if source == 'cloud' else '🟡 llama3.2:3b (local)'}]\n")
        print(answer)
        sys.exit(0)

    # Otherwise run a fixed demo set covering: answerable, and NOT answerable
    # (to check the model doesn't hallucinate when info isn't in the docs).
    demo_questions = [
        "How much is the home office stipend, and what does it cover?",
        "What should I do if I hit the SkyCart API rate limit?",
        "What did the team decide about the paid social budget?",
        "What is the CEO's favorite programming language?",  # NOT in any doc
    ]

    for q in demo_questions:
        print(f"\nQuestion: {q}")
        print("-" * 70)
        answer, source = rag_answer(q, index, indexed_chunks, router)
        print(f"[Answered by: {'🟢 Gemini (cloud)' if source == 'cloud' else '🟡 llama3.2:3b (local)'}]\n")
        print(answer)
        print()
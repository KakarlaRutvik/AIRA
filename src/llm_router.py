"""
Phase 5: LLM Router — Gemini (primary) + Ollama (fallback)

Wraps both Gemini and llama3.2:3b behind ONE interface: router.invoke(prompt).
Tries Gemini first (better quality). If Gemini fails for ANY reason -
no internet, rate limit (429), invalid/expired key, server error - it
silently falls back to the local Ollama model instead of crashing.

This is a FAILOVER pattern, not a division of labor: both models do the
exact same job (answer the prompt). One is primary, one is backup.

Usage:
    python src/llm_router.py "your question here"
    python src/llm_router.py    # runs a fixed demo prompt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import ollama
from google import genai

GEMINI_MODEL = "gemini-flash-latest"
OLLAMA_MODEL = "llama3.2:3b"


def get_api_key() -> str | None:
    """Returns the Gemini API key, or None if not configured (offline-only mode)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY"):
                        api_key = line.strip().split("=", 1)[1]
        except FileNotFoundError:
            pass
    if not api_key or api_key == "your_actual_key_here":
        return None
    return api_key


class LLMRouter:
    """
    One interface, two possible engines underneath.

    router.invoke(prompt) -> (answer_text, source)
        source is "cloud" if Gemini answered, "local" if Ollama answered.
    """

    def __init__(self):
        api_key = get_api_key()
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

    def invoke(self, prompt: str) -> tuple[str, str]:
        # Try Gemini first, if we have a client at all
        if self.gemini_client is not None:
            try:
                resp = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                return resp.text, "cloud"
            except Exception as e:
                print(f"    [router] Gemini failed ({type(e).__name__}: {e}) — falling back to local model.")

        # Fallback: local Ollama model. This path also runs directly if
        # no Gemini key was configured at all (fully offline mode).
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["message"]["content"], "local"
        except Exception as e:
            # Both engines failed — this is a real, unrecoverable error.
            raise RuntimeError(
                f"Both Gemini and Ollama failed to respond. "
                f"Ollama error: {type(e).__name__}: {e}"
            )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5: LLM Router Test")
    print("=" * 70)

    router = LLMRouter()

    if router.gemini_client is None:
        print("\n[Notice] No Gemini API key found — router will use Ollama only.")
    else:
        print("\n[Notice] Gemini client configured. Router will try Gemini first, "
              "then fall back to Ollama automatically if it fails.")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "In one sentence, what is the capital of France?"

    print(f"\nQuery: {query}")
    print("-" * 70)

    answer, source = router.invoke(query)

    print(f"\nAnswered by: {'🟢 Gemini (cloud)' if source == 'cloud' else '🟡 llama3.2:3b (local)'}")
    print(f"Answer: {answer}")
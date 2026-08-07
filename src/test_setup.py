"""
Phase 1: Environment Verification
Run this BEFORE writing any other project code.
It confirms: Ollama server is up, both required local models are pulled,
embeddings work, and the Gemini API responds.

Usage:
    python src/test_setup.py
"""

import os
import sys

REQUIRED_OLLAMA_MODELS = ["llama3.2:3b", "nomic-embed-text"]


def check_ollama_server():
    print("\n[1/4] Checking Ollama server is running...")
    try:
        import ollama
        ollama.list()
        print("    OK - Ollama server is reachable.")
        return True
    except Exception as e:
        print(f"    FAIL - Could not reach Ollama server: {e}")
        print("    Fix: run 'ollama serve' in a separate terminal, then retry.")
        return False


def check_ollama_models():
    print("\n[2/4] Checking required Ollama models are pulled...")
    import ollama
    try:
        installed = [m["model"] for m in ollama.list()["models"]]
    except Exception as e:
        print(f"    FAIL - Could not list models: {e}")
        return False

    all_ok = True
    for model in REQUIRED_OLLAMA_MODELS:
        match = any(model in m for m in installed)
        if match:
            print(f"    OK - {model} found")
        else:
            print(f"    MISSING - {model} not found. Run: ollama pull {model}")
            all_ok = False
    return all_ok


def check_ollama_chat():
    print("\n[3/4] Testing local LLM chat (llama3.2:3b)...")
    import ollama
    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        content = resp["message"]["content"]
        print(f"    OK - Model responded: {content.strip()[:50]}")
        return True
    except Exception as e:
        print(f"    FAIL - llama3.2:3b chat failed: {e}")
        return False


def check_ollama_embeddings():
    print("\n[4/4a] Testing local embeddings (nomic-embed-text)...")
    import ollama
    try:
        resp = ollama.embeddings(
            model="nomic-embed-text",
            prompt="search_document: This is a test sentence.",
        )
        vec = resp["embedding"]
        print(f"    OK - Got embedding vector of dimension {len(vec)}")
        return True
    except Exception as e:
        print(f"    FAIL - nomic-embed-text embedding failed: {e}")
        return False


def check_gemini():
    print("\n[4/4b] Testing Gemini API...")
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
        print("    SKIP - No valid GEMINI_API_KEY found in environment or .env")
        print("    This is OK for offline-only development, but required before Phase 4/5.")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-flash-latest",
            contents="Reply with exactly: OK",
        )
        print(f"    OK - Gemini responded: {resp.text.strip()[:50]}")
        return True
    except Exception as e:
        print(f"    FAIL - Gemini API call failed: {e}")
        return False


def main():
    print("=" * 60)
    print("ARIA - Phase 1: Environment Verification")
    print("=" * 60)

    results = {
        "ollama_server": check_ollama_server(),
    }

    if results["ollama_server"]:
        results["ollama_models"] = check_ollama_models()
        results["ollama_chat"] = check_ollama_chat()
        results["ollama_embeddings"] = check_ollama_embeddings()
    else:
        print("\nSkipping remaining Ollama checks since server is unreachable.")
        results["ollama_models"] = False
        results["ollama_chat"] = False
        results["ollama_embeddings"] = False

    results["gemini"] = check_gemini()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "PASS" if v is True else ("SKIPPED" if v is None else "FAIL")
        print(f"  {k:20s}: {status}")

    hard_requirements = ["ollama_server", "ollama_models", "ollama_chat", "ollama_embeddings"]
    if all(results[r] for r in hard_requirements):
        print("\nAll core (offline) requirements met. Safe to proceed to Phase 2.")
        if results["gemini"] is not True:
            print("Note: Gemini is not yet working - fine for now, required by Phase 4/5.")
        sys.exit(0)
    else:
        print("\nFix the FAIL items above before proceeding to Phase 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
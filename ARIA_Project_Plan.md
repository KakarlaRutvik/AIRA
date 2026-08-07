# ARIA — Autonomous Research & Task Agent

**A personal, offline-first AI assistant combining Agentic AI, MCP, LangChain, Vector Databases (FAISS), and Gemini API**

---

## 1. Project Overview

ARIA is a personal AI assistant that can:
- Answer questions from your own documents (RAG)
- Remember past conversations (long-term memory via vector search)
- Perform simple actions like saving/reading notes (via MCP)
- Decide which tool to use for a given query (agentic behavior)
- Keep working fully offline when the internet or Gemini API is unavailable

**Core problem it solves:** Information trapped in personal documents/notes that's hard to search conversationally, with resilience against API rate limits and no-internet situations.

**Honest scope:** This is a personal knowledge assistant / learning project, not a production system. Its value is in demonstrating end-to-end understanding of RAG, agentic tool-calling, MCP, and graceful degradation — not in solving a novel real-world problem at scale.

---

## 2. Tech Stack

| Layer | Technology | Details |
|---|---|---|
| Cloud LLM | Gemini API | `gemini-2.5-flash-lite` (swap to `gemini-3.1-flash-lite` for longer support window) |
| Local LLM | Ollama | `llama3.2:3b` |
| Embeddings | Ollama | `nomic-embed-text:latest` (768-dim, prefix-aware: `search_document:` / `search_query:`) |
| Vector DB | FAISS | `faiss-cpu`, two separate indices (documents + conversation memory) |
| Orchestration | LangChain | Agent executor, tool wrapping, prompt templates |
| Agent tools | Custom | Document search tool, calculator tool, MCP filesystem tool |
| MCP | Python MCP SDK | Local server exposing `read_note` / `write_note`, stdio transport |
| UI | Streamlit | Chat interface, status indicator, file upload, offline toggle |

**Why this stack, specifically:**
- `nomic-embed-text` via Ollama is used instead of Hugging Face `sentence-transformers` to keep the whole offline stack running through **one tool** (Ollama serves both the LLM and the embedding model), reducing dependencies and failure points for a 24-hour build.
- Gemini is the **preferred** LLM (better reasoning quality); Ollama is the **fallback** (always available, no rate limits, no internet needed). This is a failover pattern, not a division of labor — both models perform the same jobs (agent routing, tool calls, answer synthesis), just one is primary and one is backup.

---

## 3. Architecture Diagram

```
                     ┌─────────────────────┐
                     │      User Query       │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  LangChain Agent      │
                     │  (LLM Router:         │
                     │   Gemini → Ollama)    │
                     └──────────┬───────────┘
                                │  decides which tool to use
              ┌─────────────────┼─────────────────┐
              │                 │                 │
      ┌───────▼──────┐  ┌───────▼───────┐  ┌──────▼───────┐
      │ MCP Server:   │  │ Calculator     │  │  RAG Tool     │
      │ read_note /   │  │ Tool           │  │ (FAISS +      │
      │ write_note    │  │                │  │  nomic-embed) │
      └───────────────┘  └────────────────┘  └───────┬───────┘
                                                       │
                                            ┌──────────▼──────────┐
                                            │ Documents Index +     │
                                            │ Conversation Memory   │
                                            │ Index (separate FAISS)│
                                            └───────────────────────┘
```

---

## 4. Development Plan — 10 Phases

### Phase 1: Environment Verification (~0.5 hr)
Confirm Ollama (`llama3.2:3b`, `nomic-embed-text`) and Gemini API all respond correctly in isolated test scripts before writing any real logic.

### Phase 2: Embedding + FAISS Core (~2 hrs)
Build the foundational module: text → embedding (via `nomic-embed-text`, with `search_document:`/`search_query:` prefixes) → FAISS index → similarity search. This is the project's foundation — validate thoroughly before moving on.

### Phase 3: Document Loading + Chunking (~1.5 hrs)
Load `.txt`/`.pdf` files, split with `RecursiveCharacterTextSplitter` (~500 tokens, 50 overlap), feed into Phase 2's indexing pipeline.

### Phase 4: RAG Answer Generation — Gemini Only (~1.5 hrs)
Full question → retrieve → answer loop, hardcoded to Gemini first (no router yet), to validate retrieval quality in isolation.

### Phase 5: LLM Router (Gemini + Ollama Fallback) (~1.5 hrs)
Wrap Gemini and Ollama behind one `.invoke()` interface with try/except failover. Test by disconnecting wifi and confirming the same query still returns an answer via `llama3.2:3b`.

### Phase 6: LangChain Agent + Tool Calling (~3–4 hrs, largest phase)
Wrap RAG as a Tool, add a calculator tool, build the agent executor. **Build and validate with Gemini only first**, then swap in Ollama and expect/fix tool-calling format issues (small local models are the weakest link here).

### Phase 7: MCP Filesystem Server (~2.5 hrs)
Build a standalone MCP server (`read_note`, `write_note` tools, stdio transport). Test it standalone before wiring into the agent as a third tool.

### Phase 8: Conversation Memory (~1.5 hrs)
Second, separate FAISS index for past Q&A pairs. After each turn, embed and store the exchange; retrieve relevant past turns on new queries.

### Phase 9: Streamlit UI (~2 hrs)
Chat interface, sidebar status (🟢 Online / 🟡 Offline), file upload widget, manual "force offline" toggle.

### Phase 10: Testing, Hardening & Documentation (~2 hrs)
Run the full test suite (see below), fix crashes, document known limitations, prepare a 2-minute demo script.

**Time budget total: ~19 hrs, leaving ~5 hrs buffer** (expect to use most of it in Phase 6).

---

## 5. Test Cases

### A. RAG / Retrieval
| # | Test | Expected Result |
|---|---|---|
| 1 | Question answerable from uploaded doc | Correct answer, right chunk retrieved |
| 2 | Question not in any doc | Model says it doesn't know, no hallucination |
| 3 | Duplicate content uploaded twice | No redundant near-identical chunks flooding context |
| 4 | Very long document (50+ pages) | Retrieval finds relevant chunk anywhere in doc |
| 5 | Query with typos/vague phrasing | Retrieval still finds relevant chunk |

### B. LLM Router / Failover
| # | Test | Expected Result |
|---|---|---|
| 6 | Internet + valid Gemini key | Gemini handles response, status shows Online |
| 7 | Wifi disconnected mid-session | Falls back to Ollama, status flips to Offline, no crash |
| 8 | Invalid/expired Gemini key | Graceful fallback, no raw error shown to user |
| 9 | Gemini rate limit (429) | Falls back once, doesn't retry infinitely |
| 10 | Manual "force offline" toggle | Skips Gemini entirely |
| 11 | Ollama not running | Clear error shown, no silent hang |

### C. Agent Decision-Making
| # | Test | Expected Result |
|---|---|---|
| 12 | "What's 2384 * 17?" | Calculator tool used, not LLM's own math |
| 13 | "Save this summary to notes.txt" | MCP write_note called, confirms success |
| 14 | "What did I ask earlier about X?" | Retrieves from memory index correctly |
| 15 | Multi-tool query ("summarize doc and save it") | Chains retrieval → synthesis → write_note |
| 16 | "Hi, how are you?" | No tool triggered, direct answer |

### D. MCP Server
| # | Test | Expected Result |
|---|---|---|
| 17 | Read nonexistent file | Clean error, no crash/traceback shown |
| 18 | Write filename with invalid characters | Sanitized or rejected gracefully |
| 19 | MCP server crashes mid-request | Agent times out, reports failure (doesn't hang) |
| 20 | Concurrent read/write | No file corruption |

### E. Edge Cases / Stress
| # | Test | Expected Result |
|---|---|---|
| 21 | Empty query | Handled without crash |
| 22 | Extremely long query (2000+ words) | Truncated/handled, no context overflow crash |
| 23 | Non-English query | Behavior noted (small local models often degrade here) |
| 24 | 10 rapid-fire queries | No memory leak, no FAISS index corruption |

---

## 6. Challenges & Drawbacks (Known Going In)

### Technical
1. **Small local models are unreliable at structured tool-calling** — `llama3.2:3b` often fails to output the exact format LangChain agents expect. This is the single biggest time risk.
2. **Ollama + LangChain agent compatibility is finicky** — some agent types assume OpenAI-style function calling; may need a simpler ReAct-style parser for local models.
3. **Embedding model lock-in** — if you switch embedding models later, the FAISS index must be fully rebuilt (dimension mismatch).
4. **MCP subprocess overhead** — debugging across a process boundary (stdio transport) is slower than in-process debugging.
5. **Latency stacking** — router decision → retrieval → synthesis → memory store = multiple sequential steps; local CPU inference can feel slow (5–15+ sec/query).

### Design / Quality
6. **Local model answers are visibly weaker** — decide whether to show this honestly in demos or only demo with Gemini online.
7. **No memory pruning strategy** — conversation memory grows unbounded; old irrelevant turns can compete with recent relevant ones.
8. **FAISS has no built-in crash safety** — a crash mid-write can corrupt the index file. Acceptable risk for a demo, not for production.
9. **No security on MCP filesystem tool** — fine for personal/local use, unsafe if ever exposed beyond localhost.

### Scope / Time
10. **Fallback plan if behind schedule:** if Phase 6 (agent + local tool-calling) isn't working by hour ~15, cut Gemini entirely and ship Ollama-only — a working single-LLM agent beats a broken dual-LLM one.

---

## 7. Honest Project Pitch (for interviews/portfolio)

> "ARIA is a personal knowledge assistant — narrow in scope, but built to explore how RAG, agentic tool-calling, and MCP behave together, including graceful degradation when cloud APIs are unavailable. It's less about solving a novel problem and more about demonstrating I can build and reason about a full agentic AI system end-to-end, including its real failure modes."

Avoid overselling this as "solving real-world AI accessibility" — the honest, narrower framing is more credible and defensible under technical scrutiny.

---

## 8. Learning Resources Referenced

- **Transformers/attention intuition:** "The Illustrated Transformer" by Jay Alammar (visual, non-math-heavy first read)
- **Hands-on embeddings:** Build understanding by embedding sentences with `nomic-embed-text` and manually computing cosine similarity between related/unrelated pairs — more effective than pure theory at this stage.

---

## 9. Key Code Snippets Reference

```python
# Embedding with task-type prefixes (Nomic-specific best practice)
def embed_doc(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=f"search_document: {text}")["embedding"]

def embed_query(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=f"search_query: {text}")["embedding"]

# LLM Router with failover
class LLMRouter:
    def invoke(self, prompt):
        try:
            resp = gemini_model.generate_content(prompt)
            return resp.text, "cloud"
        except Exception:
            resp = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"], "local"
```

---

*Document generated as the master reference for the ARIA project build.*

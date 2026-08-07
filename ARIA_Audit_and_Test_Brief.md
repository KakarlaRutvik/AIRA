# ARIA Project — Audit & Test Brief

**Purpose of this document:** hand this to a fresh Claude session (or read it
yourself before a review pass) to systematically find bugs, verify claims,
and run through every test case that matters. This document is written to
be self-contained — no other conversation context should be required to act
on it.

---

## 1. Project Status Summary

**Not fully complete.** 9.5 of 11 planned phases are done.

| Phase | File(s) | Status |
|---|---|---|
| 1. Environment Verification | `test_setup.py` | ✅ Confirmed working |
| 2. Embedding + FAISS Core | `vector_store.py` | ✅ Confirmed working (after normalization bug fix) |
| 3. Document Loading + Chunking | `document_loader.py` | ✅ Confirmed working |
| 4. RAG Answer Generation | `rag.py` | ✅ Confirmed working |
| 5. LLM Router (failover) | `llm_router.py` | ✅ Confirmed working |
| 6. LangChain Agent + Tools | `agent.py` | ✅ Confirmed working (after LangChain 1.0 migration) |
| 7. External Fallback Tools | `external_tools.py` | ✅ Confirmed working |
| 8. MCP Filesystem Server | `mcp_server.py`, `test_mcp_client.py` | ✅ Confirmed working (after FastMCP import fix) |
| 9. Conversation Memory | `memory.py` | ✅ Built, standalone test passed; integration into agent.py done but not re-verified after Phase 10 refactor |
| 10. Streamlit UI | `app.py` | 🔧 Built. Critical nested-`asyncio.run()` bug found and fixed. **NOT YET re-tested end-to-end after the fix.** |
| 11. Testing & Hardening | — | ⬜ Not started |

**Immediate next action for any reviewer:** run `streamlit run src/app.py` and
verify a full conversation works (see Section 5, UI test cases) before
assuming Phase 10 is actually done.

---

## 2. Architecture

```
User (Streamlit UI: app.py)
        |
        v
agent.answer_query(query, force_offline)  <- single source of truth, used by BOTH app.py and CLI
        |
        +- tries: build_agent() -> Gemini (gemini-flash-lite-latest) via ChatGoogleGenerativeAI
        |           |
        |           +-- fails (rate limit / network / etc.)
        |           v
        +- falls back: build_ollama_agent() -> llama3.2:3b via ChatOllama
                    |
                    v
        LangGraph agent (langchain.agents.create_agent) with tools:
          - document_search   -> vector_store.search() over FAISS doc index
          - calculator        -> safe AST-based math eval
          - web_search        -> ddgs (DuckDuckGo)
          - youtube_search    -> yt-dlp search
          - recall_memory     -> vector_store-style search over a SEPARATE memory FAISS index
          - read_note/write_note/list_notes -> MCP tools, loaded via langchain_mcp_adapters
                                                 from mcp_server.py (separate subprocess, stdio transport)
```

Two FAISS indices exist and must stay separate:
- **Document index**: built from `data/documents/*.txt|.pdf|.md`, module-level globals `_INDEX`/`_CHUNKS` in `agent.py`, populated by `load_document_index()`.
- **Memory index**: `ConversationMemory` instance `_memory` in `agent.py`, populated automatically after every `answer_query()` call.

---

## 3. Full File Manifest

| File | Purpose |
|---|---|
| `src/test_setup.py` | Phase 1: verifies Ollama server, both local models, Gemini API all respond |
| `src/vector_store.py` | Phase 2: embed_document/embed_query (nomic-embed-text via Ollama), build_index/search/save_index/load_index (FAISS) |
| `src/document_loader.py` | Phase 3: loads .txt/.pdf/.md, chunks via `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| `src/rag.py` | Phase 4: standalone RAG CLI (retrieve + Gemini answer), uses llm_router now |
| `src/llm_router.py` | Phase 5: `LLMRouter` class, Gemini-primary/Ollama-fallback for plain RAG use |
| `src/agent.py` | Phase 6/7/8/9/10 core: all tools, `build_agent`/`build_ollama_agent`, `answer_query` (shared async entrypoint), `run_agent` (CLI wrapper) |
| `src/external_tools.py` | Phase 7: `web_search` (ddgs), `youtube_search` (yt-dlp) |
| `src/mcp_server.py` | Phase 8: FastMCP server exposing read_note/write_note/list_notes over stdio |
| `src/test_mcp_client.py` | Phase 8: standalone MCP client test, no agent involved |
| `src/memory.py` | Phase 9: `ConversationMemory` class, separate FAISS index for Q&A history |
| `src/app.py` | Phase 10: Streamlit chat UI wrapping `agent.answer_query()` |
| `src/pipeline_visualizer.py` | Debug/education tool: Streamlit walkthrough of Phases 1-4 |
| `src/master_visualizer.py` | Debug/education tool: Streamlit walkthrough of Phases 1-9, includes live MCP + agent demos |
| `src/debug_embeddings.py` | Debug tool used during embedding-quality investigation (not part of the app) |
| `data/documents/*.txt` | 3 sample docs: remote work policy, SkyCart API docs, Q3 marketing meeting notes |
| `notes/` | Runtime output dir for MCP write_note/read_note |
| `vectorstore/` | Runtime output dir for saved FAISS indices |

---

## 4. Known Bugs Already Found & Fixed (chronological — do NOT re-suggest these)

1. **Gemini SDK deprecation** — `google.generativeai` fully deprecated, switched to `google.genai` (`from google import genai`, `client.models.generate_content(...)`).
2. **Model access restriction** — `gemini-2.5-flash-lite` returned 404 "no longer available to new users", switched to `gemini-flash-latest`, then later to `gemini-flash-lite-latest` (see bug #9).
3. **FAISS un-normalized vectors** — raw `IndexFlatL2` without normalization caused nonsensical nearest-neighbor results (one "hub" chunk dominated unrelated queries) on a small synthetic 5-sentence test set. Fixed by adding `faiss.normalize_L2()` before every `index.add()`/`index.search()`. **Root cause was later confirmed to be small-sample size, not a real bug** — validated with 8/8 correct retrievals on real 24-chunk document data.
4. **`langchain.text_splitter` deprecated** — switched to `langchain_text_splitters.RecursiveCharacterTextSplitter`.
5. **`langchain_community` deprecated** — removed entirely; `document_loader.py` now reads `.txt`/`.md` with plain `open()` and `.pdf` with `pypdf.PdfReader` directly.
6. **`youtubesearchpython` package pulled from PyPI** (unmaintained) — switched to `yt-dlp` with `ytsearch3:{query}` syntax.
7. **LangChain 1.0 removed `AgentExecutor`/`create_tool_calling_agent`** entirely — migrated to `langchain.agents.create_agent` (LangGraph-based), which changed the invocation shape from `executor.invoke({"input": ...})` to `agent.invoke({"messages": [...]})`.
8. **Agent appeared to hang on a calculator-only query** — root cause was an ambiguous system prompt causing unnecessary `document_search`/web escalation on pure math queries. Fixed by adding explicit "math -> calculator directly, never document_search" instruction, plus a `recursion_limit: 10` safety net.
9. **`mcp.server.fastmcp` removed in `mcp` SDK 2.0.0** — switched `mcp_server.py` to standalone `from fastmcp import FastMCP` package.
10. **Gemini 429 rate limit crashed the whole script** — `gemini-flash-latest` resolved to a preview model (`gemini-3.6-flash`) with a 5 req/min free-tier limit. Fixed two ways: (a) switched to `gemini-flash-lite-latest` for a higher free quota, (b) added a real try/except fallback to `build_ollama_agent()` in `run_agent`/`answer_query` - the agent now has the same Gemini-to-Ollama resilience the plain RAG router already had.
11. **MCP tools crashed with `NotImplementedError: StructuredTool does not support sync invocation`** — `langchain-mcp-adapters` wraps MCP tools as async-only (`_arun` only, no `_run`). Fixed by switching agent invocation from `agent.invoke()` to `await agent.ainvoke()` everywhere.
12. **Nested `asyncio.run()` crash in the Streamlit UI** — `app.py` calls `asyncio.run(answer_query(...))`; inside that (already-running) event loop, the old `build_ollama_agent()` tried to call `asyncio.run(load_mcp_tools())` again, which Python forbids. Fixed by splitting MCP tool loading into `_get_mcp_tools_async()` (used via `await` inside `answer_query`) and `_get_mcp_tools_sync()` (used only from plain CLI script context with no loop running), with tools passed explicitly into `build_agent(mcp_tools=...)`/`build_ollama_agent(mcp_tools=...)` rather than each function fetching them independently.
13. **Structured content blocks instead of plain text** — Gemini's newer response format sometimes returns `message.content` as a list of `{'type': 'text', 'text': ...}` blocks instead of a plain string. Fixed with `_extract_text()` helper used everywhere final answer text is read.

---

## 5. Full Test Case Matrix

### 5.1 Environment (Phase 1)
| # | Test | Last known result |
|---|---|---|
| 1 | Ollama server reachable | Pass |
| 2 | `llama3.2:3b` and `nomic-embed-text` both pulled | Pass |
| 3 | Local chat call succeeds | Pass |
| 4 | Local embedding call returns 768-dim vector | Pass |
| 5 | Gemini API call succeeds with current key | Pass (after SDK/model fixes) |

### 5.2 Retrieval Core (Phase 2/3)
| # | Test | Last known result |
|---|---|---|
| 6 | Real documents load and chunk correctly | Pass (24 chunks from 3 files) |
| 7 | 8 realistic cross-document queries retrieve from correct source file | 8/8 Pass |
| 8 | Small synthetic (5 one-line) test set retrieval | Failed initially, root-caused to small-sample "hub" artifact, not re-tested since — worth re-verifying this doesn't resurface on new real documents someone adds later |
| 9 | FAISS save/load round-trip preserves index | Pass (used in Phase 2 standalone test) |

### 5.3 RAG + Router (Phase 4/5)
| # | Test | Last known result |
|---|---|---|
| 10 | Gemini answers correctly from retrieved context | Pass |
| 11 | Model refuses to answer when info is not in context (no hallucination) | Pass (explicitly tested with an out-of-scope question) |
| 12 | Forced-offline mode falls back to Ollama and still answers | Pass (manually tested by disconnecting wifi, per user report) |
| 13 | Router correctly reports which engine answered (cloud/local) | Pass |

### 5.4 Agent + Tools (Phase 6/7)
| # | Test | Last known result |
|---|---|---|
| 14 | Pure math query goes to calculator tool, not document_search | Pass |
| 15 | Document question goes to document_search, correct answer | Pass |
| 16 | Casual greeting: no tool called | Pass |
| 17 | Multi-step query (doc lookup + math) chains tools correctly | Pass (webhook status code question) |
| 18 | Doc-search-miss escalates to youtube_search for conceptual question | Pass |
| 19 | Doc-search-miss escalates to web_search for factual question | Pass |
| 20 | Agent never calls more than one of document_search/web_search/youtube_search per topic | NOT explicitly stress-tested - only verified on the specific test queries used so far |
| 21 | recursion_limit correctly prevents infinite tool-loop hangs | Added defensively after a suspected hang; the actual hang was later diagnosed as unnecessary tool escalation (see bug #8), so the recursion_limit itself has never actually been triggered/verified in practice |

### 5.5 MCP (Phase 8)
| # | Test | Last known result |
|---|---|---|
| 22 | MCP server standalone: list tools | Pass |
| 23 | MCP server standalone: write_note | Pass |
| 24 | MCP server standalone: read_note | Pass |
| 25 | MCP server standalone: list_notes | Pass |
| 26 | MCP server standalone: read nonexistent file, clean error, no crash | Pass |
| 27 | Agent (via natural language) correctly calls write_note | Pass |
| 28 | Agent (via natural language) correctly calls read_note | Pass |
| 29 | Path traversal protection (_safe_path strips ../) | NOT TESTED - _safe_path() was written defensively but no test case has ever attempted a path traversal filename (e.g. ../../etc/passwd) to confirm it actually blocks escape |
| 30 | Concurrent read/write to the same note (race condition) | NOT TESTED |
| 31 | MCP server subprocess crash mid-request | NOT TESTED |

### 5.6 Memory (Phase 9)
| # | Test | Last known result |
|---|---|---|
| 32 | Standalone ConversationMemory add/search test (simulated 3-turn conversation) | Pass |
| 33 | Standalone memory save/load persistence round-trip | Pass |
| 34 | Agent auto-stores every real exchange via answer_query | Code present, not explicitly re-verified after the Phase 10 async refactor |
| 35 | Agent correctly recalls an earlier exchange via recall_memory tool, mid real conversation | Was in the Phase 9 test query list but the actual terminal output for that specific query was never pasted back for confirmation |
| 36 | Memory does NOT get confused with document search (separate index stays separate) | NOT EXPLICITLY TESTED - no test case asks something that could plausibly match both a document chunk AND a past exchange, to confirm the agent picks the right tool |

### 5.7 UI (Phase 10) — HIGHEST PRIORITY TO VERIFY
| # | Test | Last known result |
|---|---|---|
| 37 | App starts, loads/indexes documents on first run | Not confirmed since last edit |
| 38 | Chat input, Gemini answers, green badge shows | NOT CONFIRMED post-fix |
| 39 | Force-offline toggle, Ollama answers, yellow badge shows | NOT CONFIRMED post-fix |
| 40 | A query requiring web/youtube fallback works from the UI (this is exactly the query that originally triggered the nested-asyncio crash) | NOT CONFIRMED post-fix - this is the single most important test to run next |
| 41 | File upload + re-index button works | NOT TESTED AT ALL |
| 42 | Conversation history persists and displays correctly across multiple turns in one session | NOT TESTED |
| 43 | "Clear conversation" button works | NOT TESTED |
| 44 | Reasoning trace expander shows correct tool calls | NOT TESTED |
| 45 | A query that saves a note, then a later query that reads it back, both work from the UI | NOT TESTED |
| 46 | Rapid-fire multiple queries in a row don't break session state or duplicate history entries | NOT TESTED |

### 5.8 Cross-cutting / Stress
| # | Test | Last known result |
|---|---|---|
| 47 | Empty query submitted | NOT TESTED |
| 48 | Extremely long query (2000+ words) | NOT TESTED |
| 49 | Non-English query | NOT TESTED |
| 50 | What happens if data/documents/ is empty on startup (no docs at all) | document_loader.py has a warning path for this, but not verified through the full agent/UI stack |
| 51 | What happens if Ollama is not running AND Gemini is also failing (both engines down) | LLMRouter in llm_router.py explicitly raises a clear error for this case; NOT verified whether agent.py's answer_query does the same or crashes uglier, since its except block doesn't have a second-level try/except around the Ollama fallback itself |

---

## 6. Specific Instructions for a Reviewing Claude

If you are a fresh Claude session reading this file to audit the project:

1. **Start with Section 5.7 (UI tests)** — this is the least-verified, highest-risk area. Ask the user to run `streamlit run src/app.py` and walk through tests #37-46 in order.
2. **Do not re-litigate Section 4's fixed bugs** — those have confirmed root causes and fixes already. If a similar-looking symptom appears, check whether it's actually a regression of one of these before treating it as new.
3. **Pay special attention to test #51** — trace through `agent.py`'s `answer_query()` function: if `build_agent()` fails AND `build_ollama_agent()` also fails (e.g., Ollama not running), does the exception propagate cleanly with a useful message, or does it produce a confusing nested traceback? This has never been tested and is a plausible real gap.
4. **Test #29 (path traversal) is a real, unverified security claim** — `mcp_server.py`'s `_safe_path()` uses `os.path.basename()` to strip directory components. Verify this actually works by attempting `write_note("../../../tmp/evil.txt", "test")` through the standalone MCP client and confirming it either sanitizes correctly to a safe filename in `notes/` or rejects it — don't just trust the code comment.
5. **When proposing a fix, check the "Known Bugs Already Found & Fixed" section first** to avoid suggesting something already tried and superseded (e.g., don't suggest going back to `youtubesearchpython` or `AgentExecutor`).
6. **Environment specifics observed during development** (useful for reproducing issues): macOS, Python 3.11, `langchain` 1.3.14, `langchain-google-genai` 4.3.2, `langchain-ollama` 1.1.0, `langchain-mcp-adapters` 0.3.1, `mcp` 1.29.0 (client) / `fastmcp` 3.4.5 (server), `google-genai` (current, not deprecated `google-generativeai`), Ollama 0.24.0.

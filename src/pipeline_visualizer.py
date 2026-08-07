"""
ARIA Master Visualizer
=======================
A single, comprehensive, step-by-step walkthrough of the ENTIRE project so far:
phases 1-8. Every step uses your real project code - nothing here is simulated.

Usage:
    streamlit run src/master_visualizer.py
"""

import io
import os
import sys
import time

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from document_loader import load_and_chunk_directory
from vector_store import build_index, search, embed_document, embed_query
from llm_router import LLMRouter

st.set_page_config(page_title="ARIA Master Visualizer", layout="wide")

STEPS = [
    "0. Project Overview & Phase Tracker",
    "1. Load & Chunk Documents",
    "2. Embeddings & Dimensionality (768-D explained)",
    "3. Build the FAISS Index",
    "4. Explore All Vectors (full 768-dim)",
    "5. Retrieval in Action",
    "6. Gemini API — Deep Dive",
    "7. Offline Mode — LLM Router Failover",
    "8. MCP — What It Is & Live Demo",
    "9. Agentic AI — Tool-Calling Demo",
    "10. Test Case Summary",
]

DOCS_DIR = "data/documents"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "step": 0, "chunks": None, "sources": None, "index": None,
    "query": "How much is the home office stipend?", "retrieved": None,
    "answer": None, "gemini_raw": None, "router": None, "force_offline": False,
    "mcp_tools_listed": None, "mcp_write_result": None, "mcp_read_result": None,
    "agent": None, "agent_query": "What's 2384 * 17?", "agent_trace": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 Steps")
for i, label in enumerate(STEPS):
    if i == st.session_state.step:
        st.sidebar.markdown(f"**➡️ {label}**")
    elif i < st.session_state.step:
        st.sidebar.markdown(f"✅ {label}")
    else:
        st.sidebar.markdown(f"⬜ {label}")

st.sidebar.divider()
col_back, col_next = st.sidebar.columns(2)
if col_back.button("⬅️ Back", use_container_width=True, disabled=st.session_state.step == 0):
    st.session_state.step -= 1
    st.rerun()
if col_next.button("Next ➡️", use_container_width=True, disabled=st.session_state.step == len(STEPS) - 1):
    st.session_state.step += 1
    st.rerun()

st.title("🔍 ARIA Master Visualizer")
st.caption("Every step below uses your REAL project code. Nothing here is a mockup.")


# ---------------------------------------------------------------------------
# Helper: heatmap image
# ---------------------------------------------------------------------------
def heatmap_image(arr, height=1.2, xlabel="Dimension"):
    import matplotlib.pyplot as plt
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(12, height))
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
    ax.set_xlabel(xlabel)
    if arr.shape[0] > 1:
        ax.set_ylabel("Vector ID")
        ax.set_yticks(range(arr.shape[0]))
    else:
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, fraction=0.05)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


def manual_pca(X, n_components=2):
    """Simple PCA using numpy SVD - no sklearn dependency needed."""
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


# ===========================================================================
# STEP 0: Overview & Phase Tracker
# ===========================================================================
def render_overview():
    st.header("Project Status: 11 Phases Total")
    phases = [
        ("1", "Environment Verification", "done"),
        ("2", "Embedding + FAISS Core", "done"),
        ("3", "Document Loading + Chunking", "done"),
        ("4", "RAG Answer Generation (Gemini)", "done"),
        ("5", "LLM Router (Gemini + Ollama fallback)", "done"),
        ("6", "LangChain Agent + Tool Calling", "done"),
        ("7", "External Fallback Tools (Web + YouTube)", "done"),
        ("8", "MCP Filesystem Server", "in progress"),
        ("9", "Conversation Memory", "not started"),
        ("10", "Streamlit UI (final app)", "not started"),
        ("11", "Testing & Hardening", "not started"),
    ]
    for num, name, status in phases:
        icon = {"done": "✅", "in progress": "🔧", "not started": "⬜"}[status]
        st.markdown(f"{icon} **Phase {num}:** {name}")

    st.divider()
    st.subheader("What you're about to walk through")
    st.markdown("""
    This visualizer covers Phases 1-8 end to end, using your real documents,
    real embeddings, real FAISS index, real Gemini API, real local Ollama
    fallback, real MCP server, and real LangChain agent.

    Click **Next ➡️** in the sidebar to begin.
    """)


# ===========================================================================
# STEP 1: Load & Chunk
# ===========================================================================
def render_load_chunk():
    st.header("Step 1: Loading & Chunking Documents")
    if st.session_state.chunks is None:
        with st.spinner("Loading and chunking..."):
            chunks, sources = load_and_chunk_directory(DOCS_DIR)
            st.session_state.chunks = chunks
            st.session_state.sources = sources

    chunks, sources = st.session_state.chunks, st.session_state.sources
    if not chunks:
        st.error(f"No documents found in {DOCS_DIR}/. Add files and restart.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Files loaded", len(set(sources)))
    c2.metric("Total chunks", len(chunks))
    c3.metric("Avg chunk length", f"{int(np.mean([len(c) for c in chunks]))} chars")

    idx = st.slider("Preview a chunk", 0, len(chunks) - 1, 0)
    st.markdown(f"**Source:** `{sources[idx]}`")
    st.code(chunks[idx], language=None)


# ===========================================================================
# STEP 2: Embeddings & Dimensionality explained
# ===========================================================================
def render_embedding_concept():
    st.header("Step 2: What Is a 768-Dimensional Embedding?")

    st.markdown("""
    ### The core idea, scaled up from something familiar

    | Dimensions | What a point looks like | Can we draw it? |
    |---|---|---|
    | 2D | `(x, y)` | Yes - a dot on a page |
    | 3D | `(x, y, z)` | Yes - a dot in space |
    | 768D | `(x₁, x₂, ..., x₇₆₈)` | Not directly - but the MATH works identically |

    A 768-dimensional vector is just a point with 768 coordinates instead of 2 or 3.
    Humans can't visualize 768 axes, but distance and similarity between two such
    points are calculated with the exact same formulas as 2D/3D distance - just
    summed across more numbers.

    ### What's a heatmap, concretely?

    A heatmap turns a list of numbers into colors, so you can scan 768 numbers
    at a glance instead of reading them one by one. Below, embed one of your
    real chunks and see both forms side by side.
    """)

    if not st.session_state.chunks:
        st.warning("Go back to Step 1 first.")
        return

    chunks = st.session_state.chunks
    idx = st.slider("Pick a chunk to embed", 0, len(chunks) - 1, 0, key="embed_slider")
    st.code(chunks[idx][:200] + "...", language=None)

    if st.button("🧬 Embed this chunk"):
        with st.spinner("Calling nomic-embed-text..."):
            vec = np.array(embed_document(chunks[idx]))
        st.session_state["_demo_vec"] = vec

    if "_demo_vec" in st.session_state:
        vec = st.session_state["_demo_vec"]
        c1, c2 = st.columns(2)
        c1.metric("Dimensions", len(vec))
        c2.metric("Norm (length)", f"{np.linalg.norm(vec):.3f}")

        st.subheader("As raw numbers (first 20 of 768)")
        st.code(str([round(v, 3) for v in vec[:20]]) + " ...")

        st.subheader("As a heatmap (all 768 dimensions)")
        st.image(heatmap_image(vec, xlabel="Dimension (0-767)"), use_container_width=True)
        st.caption("Red = higher value, blue = lower value. Same 768 numbers as above, just colored.")

    st.divider()
    st.subheader("Seeing 768D in 2D: PCA projection of ALL your chunks")
    st.markdown("""
    PCA (Principal Component Analysis) finds the 2 directions in 768D space
    that capture the MOST variation between your chunks, and projects every
    vector onto just those 2 - so you can see clustering with your own eyes.
    """)
    if st.button("📊 Project all chunks to 2D"):
        with st.spinner("Embedding all chunks and running PCA..."):
            vectors = np.array([embed_document(c) for c in chunks])
            projected = manual_pca(vectors, n_components=2)

        import matplotlib.pyplot as plt
        sources = st.session_state.sources
        unique_sources = list(set(sources))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
        color_map = {s: colors[i] for i, s in enumerate(unique_sources)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for s in unique_sources:
            mask = [src == s for src in sources]
            pts = projected[mask]
            ax.scatter(pts[:, 0], pts[:, 1], label=s, s=80, color=color_map[s])
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(fontsize=8)
        ax.set_title("Your chunks, 768D -> 2D via PCA")
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        st.caption("Dots of the same color = same source file. Nearby dots = similar meaning. "
                   "This is the same clustering FAISS uses internally, just squashed to 2D for your eyes.")


# ===========================================================================
# STEP 3: Build FAISS
# ===========================================================================
def render_faiss_build():
    st.header("Step 3: Building the FAISS Index")
    st.markdown("Every chunk is embedded, vectors are normalized, and loaded into a FAISS `IndexFlatL2`.")

    if not st.session_state.chunks:
        st.warning("Go back to Step 1 first.")
        return

    if st.session_state.index is None:
        if st.button("🏗️ Build the index"):
            progress = st.progress(0, text="Embedding chunks...")
            chunks = st.session_state.chunks
            vectors = []
            for i, c in enumerate(chunks):
                vectors.append(embed_document(c))
                progress.progress((i + 1) / len(chunks), text=f"Chunk {i+1}/{len(chunks)}")
            import faiss
            arr = np.array(vectors, dtype="float32")
            faiss.normalize_L2(arr)
            index = faiss.IndexFlatL2(arr.shape[1])
            index.add(arr)
            st.session_state.index = index
            st.rerun()
    else:
        index = st.session_state.index
        c1, c2 = st.columns(2)
        c1.metric("Vectors indexed", index.ntotal)
        c2.metric("Dimension", index.d)
        st.success("✅ Index built and held in memory.")
        if st.button("🔄 Rebuild"):
            st.session_state.index = None
            st.rerun()


# ===========================================================================
# STEP 4: Explore all vectors
# ===========================================================================
def render_vector_explorer():
    st.header("Step 4: Every Vector, Full 768 Dimensions, With IDs")

    if st.session_state.index is None:
        st.warning("Go back to Step 3 first.")
        return

    index, chunks, sources = st.session_state.index, st.session_state.chunks, st.session_state.sources
    all_vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    st.image(heatmap_image(all_vectors, height=max(2, all_vectors.shape[0] * 0.3)), use_container_width=True,
              caption="Each row = one chunk's full vector. Similar rows = similar meaning.")

    st.dataframe([
        {"ID": i, "Source": sources[i], "Norm": f"{np.linalg.norm(all_vectors[i]):.4f}",
         "Preview": chunks[i][:60] + "..."}
        for i in range(index.ntotal)
    ], use_container_width=True, hide_index=True)

    selected_id = st.selectbox("Inspect one vector fully", range(index.ntotal),
                                 format_func=lambda i: f"ID {i} — {sources[i]}")
    vec = all_vectors[selected_id]
    with st.expander(f"All {len(vec)} raw values for ID {selected_id}"):
        st.dataframe({"dimension": list(range(len(vec))), "value": [float(v) for v in vec]},
                     use_container_width=True, hide_index=True, height=400)


# ===========================================================================
# STEP 5: Retrieval
# ===========================================================================
def render_retrieval():
    st.header("Step 5: Retrieval in Action")

    if st.session_state.index is None:
        st.warning("Go back to Step 3 first.")
        return

    query = st.text_input("Your question:", value=st.session_state.query)
    k = st.slider("k (chunks to retrieve)", 1, 5, 3)

    if st.button("🔎 Search"):
        st.session_state.query = query
        qvec = np.array([embed_query(query)], dtype="float32")
        import faiss
        faiss.normalize_L2(qvec)
        distances, indices = st.session_state.index.search(qvec, k)
        chunks, sources = st.session_state.chunks, st.session_state.sources
        st.session_state.retrieved = [(chunks[i], sources[i], float(distances[0][j]))
                                        for j, i in enumerate(indices[0])]

    if st.session_state.retrieved:
        for rank, (chunk, source, dist) in enumerate(st.session_state.retrieved, 1):
            with st.container(border=True):
                st.markdown(f"**#{rank}** — `{source}` — distance: `{dist:.4f}`")
                st.write(chunk[:250] + "...")


# ===========================================================================
# STEP 6: Gemini deep dive
# ===========================================================================
def render_gemini_deep_dive():
    st.header("Step 6: Gemini API — Deep Dive")
    st.markdown("""
    ```
    Your machine                              Google's servers
    ─────────────                              ────────────────
    1. Build prompt (context + question)
    2. client.models.generate_content(...)
             │  HTTPS POST, JSON body, API key in header
             ▼
                                       3. Gemini runs inference
             ▲
             │  HTTPS response, JSON body
    4. SDK parses response.text
    ```
    **Only the prompt TEXT crosses the network.** Your documents, FAISS index,
    and vectors never leave your machine - retrieval already happened locally.
    """)

    if not st.session_state.retrieved:
        st.warning("Go back to Step 5 and run a search first.")
        return

    context = "\n\n---\n\n".join([c for c, _, _ in st.session_state.retrieved])
    prompt = f"Answer using ONLY this context:\n\n{context}\n\nQUESTION: {st.session_state.query}\nANSWER:"

    with st.expander("See the exact prompt sent"):
        st.code(prompt, language=None)

    if st.button("✨ Send to Gemini"):
        if st.session_state.router is None:
            st.session_state.router = LLMRouter()
        with st.spinner("Calling Gemini..."):
            answer, source = st.session_state.router.invoke(prompt)
            st.session_state.answer = answer
            st.session_state.answer_source = source

    if st.session_state.answer:
        badge = "🟢 Gemini (cloud)" if st.session_state.get("answer_source") == "cloud" else "🟡 llama3.2:3b (local)"
        st.success(f"Answered by: {badge}")
        st.markdown(st.session_state.answer)


# ===========================================================================
# STEP 7: Offline mode / router failover
# ===========================================================================
def render_offline_mode():
    st.header("Step 7: Offline Mode — LLM Router Failover")
    st.markdown("""
    This is Phase 5: **one interface, two engines**. The router tries Gemini
    first (better quality). If Gemini fails for ANY reason - no internet,
    rate limit, invalid key - it silently falls back to your local
    `llama3.2:3b` model via Ollama, so the app never goes down.
    """)

    st.session_state.force_offline = st.toggle(
        "🔌 Force offline mode (simulate no internet / no Gemini key)",
        value=st.session_state.force_offline,
    )

    query = st.text_input("Ask something:", value="What is the capital of France?", key="offline_query")

    if st.button("Ask (respecting offline toggle)"):
        if st.session_state.force_offline:
            import ollama
            with st.spinner("Forced offline - calling llama3.2:3b directly..."):
                resp = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": query}])
                st.session_state.answer = resp["message"]["content"]
                st.session_state.answer_source = "local"
        else:
            if st.session_state.router is None:
                st.session_state.router = LLMRouter()
            with st.spinner("Trying Gemini first..."):
                answer, source = st.session_state.router.invoke(query)
                st.session_state.answer = answer
                st.session_state.answer_source = source

    if st.session_state.answer:
        badge = "🟢 Gemini (cloud)" if st.session_state.get("answer_source") == "cloud" else "🟡 llama3.2:3b (local)"
        st.info(f"Answered by: {badge}")
        st.write(st.session_state.answer)

    st.caption("Try toggling offline mode ON and OFF and asking the same question - "
               "notice the badge changes but you always get an answer.")


# ===========================================================================
# STEP 8: MCP explanation + live demo
# ===========================================================================
def render_mcp():
    st.header("Step 8: MCP — What It Is, and a Live Demo")
    st.markdown("""
    ### MCP vs. a normal function call

    Every tool so far (`document_search`, `calculator`, `web_search`) is just
    a Python function living INSIDE your agent's process. **MCP is different:**
    it's a standardized protocol for exposing tools from a SEPARATE process,
    which any compatible client can talk to - not just your agent.

    ```
    Your Agent (client)                    mcp_server.py (separate process)
    ────────────────────                    ──────────────────────────────
    1. Launches mcp_server.py as
       a subprocess
    2. Sends "list tools" request  ──────►
                                    ◄──────  Returns: read_note, write_note, list_notes
    3. Sends "call write_note(...)" ──────►
                                    ◄──────  Actually writes the file, returns result
    ```

    The two processes talk over **stdin/stdout** (stdio transport) using the
    MCP protocol - JSON messages back and forth - not a normal Python import.
    """)

    st.divider()
    st.subheader("Live demo: talk to the real MCP server")

    if st.button("📋 List tools on the MCP server"):
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _list():
            params = StdioServerParameters(command=sys.executable, args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    return [(t.name, t.description) for t in resp.tools]

        with st.spinner("Launching MCP server subprocess and listing tools..."):
            st.session_state.mcp_tools_listed = asyncio.run(_list())

    if st.session_state.mcp_tools_listed:
        for name, desc in st.session_state.mcp_tools_listed:
            st.markdown(f"- **{name}**: {desc}")

    st.divider()
    note_name = st.text_input("Note filename", value="demo_note.txt")
    note_content = st.text_area("Note content", value="This note was written via the real MCP server.")

    if st.button("💾 Write this note via MCP"):
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _write():
            params = StdioServerParameters(command=sys.executable, args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("write_note", arguments={"filename": note_name, "content": note_content})
                    return result.content[0].text

        with st.spinner("Calling write_note on the MCP server..."):
            st.session_state.mcp_write_result = asyncio.run(_write())

    if st.session_state.mcp_write_result:
        st.success(st.session_state.mcp_write_result)

    if st.button("📖 Read it back via MCP"):
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _read():
            params = StdioServerParameters(command=sys.executable, args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("read_note", arguments={"filename": note_name})
                    return result.content[0].text

        with st.spinner("Calling read_note on the MCP server..."):
            st.session_state.mcp_read_result = asyncio.run(_read())

    if st.session_state.mcp_read_result:
        st.info(f"Content read back: {st.session_state.mcp_read_result}")


# ===========================================================================
# STEP 9: Agentic AI demo
# ===========================================================================
def render_agent_demo():
    st.header("Step 9: Agentic AI — Watch the Agent Decide")
    st.markdown("""
    **What makes this "agentic":** the LLM doesn't just generate text - it
    DECIDES which tool to use (document search, calculator, web search,
    YouTube search, or MCP notes) based on the query, then actually executes
    that tool, reads the result, and decides what to do next.
    """)

    if st.button("🤖 Build the agent (loads all tools including MCP)"):
        with st.spinner("Building agent - connecting to MCP server, loading tools..."):
            import agent as agent_module
            st.session_state.agent = agent_module.build_agent()
            st.session_state._agent_module = agent_module
        st.success("Agent ready with all tools loaded.")

    if st.session_state.agent is None:
        st.warning("Click the button above first - this loads document search, calculator, "
                   "web/YouTube search, and MCP notes tools together.")
        return

    query = st.text_input("Ask the agent anything:", value=st.session_state.agent_query)

    if st.button("▶️ Run agent"):
        st.session_state.agent_query = query
        with st.spinner("Agent is reasoning and calling tools..."):
            result = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 10},
            )
            st.session_state.agent_trace = result["messages"]

    if st.session_state.agent_trace:
        st.subheader("Agent's reasoning trace")
        for msg in st.session_state.agent_trace:
            msg_type = type(msg).__name__
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    st.markdown(f"🔧 **Called tool:** `{tc['name']}({tc['args']})`")
            elif msg_type == "ToolMessage":
                with st.container(border=True):
                    st.caption(f"Result from `{getattr(msg, 'name', '?')}`:")
                    st.write(str(msg.content)[:300])
            elif msg_type == "HumanMessage":
                st.markdown(f"**You asked:** {msg.content}")

        st.subheader("Final answer")
        st.success(st.session_state.agent_trace[-1].content)


# ===========================================================================
# STEP 10: Test case summary
# ===========================================================================
def render_test_summary():
    st.header("Step 10: Test Case Summary")
    st.markdown("""
    Cases validated so far across Phases 1-8 (run manually in terminal so far):
    """)
    cases = [
        ("Retrieval finds correct source doc (8/8 queries)", "✅ PASS"),
        ("Model refuses to hallucinate on out-of-scope questions", "✅ PASS"),
        ("Gemini -> Ollama failover on forced offline mode", "✅ PASS"),
        ("Agent routes math -> calculator, not doc search", "✅ PASS"),
        ("Agent routes doc questions -> document_search", "✅ PASS"),
        ("Agent skips tools for casual conversation", "✅ PASS"),
        ("Agent escalates doc-search-miss -> youtube_search", "✅ PASS"),
        ("Agent escalates doc-search-miss -> web_search", "✅ PASS"),
        ("MCP server: write_note + read_note roundtrip", "🔧 Verify in Step 8 above"),
        ("MCP server: read nonexistent file -> clean error, no crash", "🔧 Verify in Step 8 above"),
    ]
    for case, status in cases:
        st.markdown(f"{status} — {case}")

    st.divider()
    st.info("Phases 9-11 (Memory, final UI, full hardening pass) are not built yet - "
            "this visualizer covers everything built through Phase 8.")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
renderers = [
    render_overview, render_load_chunk, render_embedding_concept, render_faiss_build,
    render_vector_explorer, render_retrieval, render_gemini_deep_dive,
    render_offline_mode, render_mcp, render_agent_demo, render_test_summary,
]
renderers[st.session_state.step]()
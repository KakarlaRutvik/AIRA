"""
Phase 10: ARIA - Final Streamlit App

The real, clickable product: a chat interface wrapping everything built in
Phases 1-9. Zero new logic lives here - this is pure wiring around
agent.py's answer_query(), which already handles retrieval, tool selection,
Gemini/Ollama failover, MCP notes, and conversation memory.

Usage:
    streamlit run src/app.py
"""

import asyncio
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import agent as agent_module

st.set_page_config(page_title="ARIA - Personal AI Assistant", page_icon="🤖", layout="wide")

DOCS_DIR = "data/documents"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "force_offline" not in st.session_state:
    st.session_state.force_offline = False
if "last_source" not in st.session_state:
    st.session_state.last_source = None

# Load/index documents once per session (agent.py keeps them in module-level
# globals used by the document_search tool)
if not st.session_state.docs_loaded:
    with st.spinner("Loading and indexing documents..."):
        try:
            n_chunks, n_files = agent_module.load_document_index()
            st.session_state.docs_loaded = True
            st.session_state.doc_stats = (n_chunks, n_files)
        except Exception as e:
            st.session_state.doc_stats = None
            st.session_state.docs_loaded = "error"
            st.session_state.docs_error = str(e)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🤖 ARIA")
st.sidebar.caption("Your offline-resilient personal AI assistant")

status_placeholder = st.sidebar.empty()
if st.session_state.last_source == "cloud":
    status_placeholder.success("🟢 Online (Gemini)")
elif st.session_state.last_source == "local":
    status_placeholder.warning("🟡 Offline (llama3.2:3b)")
else:
    status_placeholder.info("⚪ Not yet asked anything")

st.sidebar.divider()

st.session_state.force_offline = st.sidebar.toggle(
    "🔌 Force offline mode",
    value=st.session_state.force_offline,
    help="Skip Gemini entirely and use the local llama3.2:3b model, even if internet is available.",
)

st.sidebar.divider()
st.sidebar.subheader("📁 Documents")
if st.session_state.docs_loaded == "error":
    st.sidebar.error(f"Failed to load documents: {st.session_state.docs_error}")
elif st.session_state.doc_stats:
    n_chunks, n_files = st.session_state.doc_stats
    st.sidebar.caption(f"{n_chunks} chunks indexed from {n_files} file(s)")

uploaded = st.sidebar.file_uploader(
    "Add a document",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)
if uploaded:
    os.makedirs(DOCS_DIR, exist_ok=True)
    new_files = []
    for f in uploaded:
        path = os.path.join(DOCS_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        new_files.append(f.name)
    if st.sidebar.button(f"🔄 Re-index with {len(new_files)} new file(s)"):
        with st.spinner("Re-indexing all documents..."):
            n_chunks, n_files = agent_module.load_document_index()
            st.session_state.doc_stats = (n_chunks, n_files)
        st.sidebar.success("Re-indexed!")
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.history = []
    st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("ARIA — Personal AI Assistant")
st.caption(
    "Ask about your documents, do calculations, search the web/YouTube for "
    "topics not in your files, save/read notes, or recall earlier questions."
)

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        badge = "🟢 Gemini" if turn["source"] == "cloud" else "🟡 llama3.2:3b (local)"
        st.caption(f"Answered by: {badge}")
        st.write(turn["answer"])
        if turn.get("trace"):
            with st.expander("See reasoning trace"):
                for line in turn["trace"]:
                    st.text(line)

query = st.chat_input("Ask me anything...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(
                agent_module.answer_query(query, force_offline=st.session_state.force_offline)
            )

        badge = "🟢 Gemini" if result["source"] == "cloud" else "🟡 llama3.2:3b (local)"
        st.caption(f"Answered by: {badge}")
        st.write(result["answer"])
        if result["trace"]:
            with st.expander("See reasoning trace"):
                for line in result["trace"]:
                    st.text(line)

    st.session_state.last_source = result["source"]
    st.session_state.history.append({
        "query": query,
        "answer": result["answer"],
        "source": result["source"],
        "trace": result["trace"],
    })
    st.rerun()
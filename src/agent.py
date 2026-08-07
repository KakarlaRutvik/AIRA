"""
Phase 6: LangChain Agent + Tool Calling

Wraps the Phase 4 RAG retrieval as a Tool, adds a Calculator tool, and builds
an agent that DECIDES which tool to use (or none) based on the query.

Built with Gemini first (better at reliable tool-calling format) - Ollama
support is added afterward once this baseline works, since small local
models are known to struggle with structured tool-calling output.

Usage:
    python src/agent.py "your question here"
    python src/agent.py    # runs a fixed set of test queries
"""

import ast
import asyncio
import operator
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

from document_loader import load_and_chunk_directory
from vector_store import build_index, search as vector_search
from external_tools import web_search, youtube_search
from memory import ConversationMemory

_memory = ConversationMemory()

GEMINI_MODEL = "gemini-flash-lite-latest"  # lite variant: higher free-tier rate limit than gemini-flash-latest


def get_api_key() -> str:
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
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
    return api_key


# ---------------------------------------------------------------------------
# Global state the tools operate on (loaded once, used by every tool call)
# ---------------------------------------------------------------------------
_INDEX = None
_CHUNKS = None


def load_document_index():
    """Load and index the real project documents once, before building the agent."""
    global _INDEX, _CHUNKS
    chunks, sources = load_and_chunk_directory("data/documents")
    if not chunks:
        raise RuntimeError("No documents found in data/documents/. Add files first.")
    _INDEX, _CHUNKS = build_index(chunks)
    return len(chunks), len(set(sources))


# ---------------------------------------------------------------------------
# TOOL 1: Document search (wraps Phase 2's search() function)
# ---------------------------------------------------------------------------
@tool
def document_search(query: str) -> str:
    """Search the user's personal documents (HR policies, API documentation,
    meeting notes) for information relevant to the query. ALWAYS try this
    tool FIRST for any question that might relate to the user's own
    documents, policies, or company information. If this returns 'No
    relevant information found', escalate to web_search (for factual
    questions) or youtube_search (for conceptual/how-to questions) instead
    of giving up. Do NOT use this for general knowledge questions or casual
    conversation - just answer those directly with no tool."""
    if _INDEX is None:
        return "No documents have been indexed yet."
    results = vector_search(_INDEX, _CHUNKS, query, k=3)
    if not results:
        return "No relevant information found in the documents."
    return "\n\n---\n\n".join(results)


@tool
def recall_memory(query: str) -> str:
    """Recall relevant PAST EXCHANGES from earlier in this conversation
    (things the user asked before and what you answered). Use this when the
    user references something said earlier - e.g. 'what did I ask about
    earlier', 'remind me what you said about X', 'what did we decide about Y'.
    Do NOT use this for questions about documents (use document_search) or
    for brand new topics never discussed before."""
    results = _memory.search(query, k=2)
    if not results:
        return "No relevant past exchanges found in memory."
    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# TOOL 2: Calculator (safe expression evaluation, no raw eval())
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a pure math expression, e.g. '384 * 12' or '(45 + 5) / 2'.
    Use this whenever the user asks for a calculation. Do NOT attempt to do
    math yourself - always call this tool for any arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Could not evaluate expression: {e}"


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
async def load_mcp_tools():
    """Connect to mcp_server.py (launched as a subprocess) and load its tools
    (read_note, write_note, list_notes) as LangChain-compatible tool objects.
    """
    client = MultiServerMCPClient({
        "notes": {
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(__file__), "mcp_server.py")],
            "transport": "stdio",
        }
    })
    return await client.get_tools()


OLLAMA_MODEL = "llama3.2:3b"

_SYSTEM_PROMPT = (
    "You are ARIA, a helpful personal assistant. You have access to tools "
    "for searching the user's documents, doing calculations, searching the "
    "web, searching YouTube, and reading/writing/listing saved notes.\n\n"
    "Decide which tool (if any) to use based on the query type:\n"
    "- Pure math/calculation questions -> use calculator DIRECTLY. Do NOT "
    "use document_search, web_search, or youtube_search for these.\n"
    "- Casual conversation or general knowledge you already know -> answer "
    "directly with NO tool at all.\n"
    "- Questions about the user's documents, policies, or company info -> "
    "use document_search FIRST.\n"
    "- Requests to save, remember, write down, or store something -> use "
    "write_note with a sensible filename.\n"
    "- Requests to recall, read back, or check a previously saved note -> "
    "use read_note or list_notes.\n"
    "- References to something asked or discussed EARLIER in this "
    "conversation -> use recall_memory.\n\n"
    "ONLY if document_search returns 'No relevant information found' should "
    "you escalate to a web tool: youtube_search for conceptual/how-to "
    "questions, web_search for factual/informational questions. Never call "
    "more than one of document_search/web_search/youtube_search per "
    "distinct topic - if the first relevant tool gives you an answer, stop "
    "and respond, don't keep searching."
)

_cached_mcp_tools = None
_cached_gemini_agent = None
_cached_ollama_agent = None


async def _get_mcp_tools_async():
    """Async-safe version - use this when already inside a running event
    loop (e.g. called from within answer_query). Awaits directly instead of
    starting a new loop, which is illegal when one is already running."""
    global _cached_mcp_tools
    if _cached_mcp_tools is None:
        _cached_mcp_tools = await load_mcp_tools()
    return _cached_mcp_tools


def _get_mcp_tools_sync():
    """Sync version - safe ONLY when called from plain script code with NO
    event loop already running (e.g. directly from the CLI __main__ block)."""
    global _cached_mcp_tools
    if _cached_mcp_tools is None:
        _cached_mcp_tools = asyncio.run(load_mcp_tools())
    return _cached_mcp_tools


def build_agent(mcp_tools=None):
    """Build the PRIMARY (Gemini-backed) agent. Cached after first build.

    Pass mcp_tools explicitly when calling from async code already inside a
    running event loop (see answer_query). If omitted, falls back to
    synchronously loading them - only safe when no loop is running yet.
    """
    global _cached_gemini_agent
    if _cached_gemini_agent is not None:
        return _cached_gemini_agent

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=get_api_key(), temperature=0)

    if mcp_tools is None:
        print("    Connecting to MCP notes server...")
        mcp_tools = _get_mcp_tools_sync()
        print(f"    Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

    tools = [document_search, calculator, web_search, youtube_search, recall_memory] + mcp_tools
    _cached_gemini_agent = create_agent(model=llm, tools=tools, system_prompt=_SYSTEM_PROMPT)
    return _cached_gemini_agent


def build_ollama_agent(mcp_tools=None):
    """Build the FALLBACK (local Ollama-backed) agent. Cached after first build.
    Same mcp_tools rule as build_agent() above."""
    global _cached_ollama_agent
    if _cached_ollama_agent is not None:
        return _cached_ollama_agent

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    if mcp_tools is None:
        mcp_tools = _get_mcp_tools_sync()
    tools = [document_search, calculator, web_search, youtube_search, recall_memory] + mcp_tools
    _cached_ollama_agent = create_agent(model=llm, tools=tools, system_prompt=_SYSTEM_PROMPT)
    return _cached_ollama_agent


def _extract_text(content) -> str:
    """Message content can be a plain string OR a list of content blocks
    (e.g. Gemini's newer response format: [{'type': 'text', 'text': '...'}]).
    This normalizes either shape into a clean string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


async def answer_query(query: str, force_offline: bool = False) -> dict:
    """The single source of truth for 'answer this query with the agent'.

    Used by BOTH the CLI (run_agent, below) and the Streamlit UI (app.py),
    so there is exactly one place this logic lives - no duplicated behavior
    between the terminal script and the real app.

    Returns a dict: {"answer": str, "source": "cloud"|"local", "trace": [str, ...]}
    """
    if force_offline:
        agent_to_use = build_ollama_agent(mcp_tools=await _get_mcp_tools_async())
        result = await agent_to_use.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 10},
        )
        source = "local"
    else:
        try:
            agent_to_use = build_agent(mcp_tools=await _get_mcp_tools_async())
            result = await agent_to_use.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 10},
            )
            source = "cloud"
        except Exception:
            agent_to_use = build_ollama_agent(mcp_tools=await _get_mcp_tools_async())
            result = await agent_to_use.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 10},
            )
            source = "local"

    messages = result["messages"]
    trace = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                trace.append(f"🔧 called {tc['name']}({tc['args']})")
        elif type(msg).__name__ == "ToolMessage":
            trace.append(f"↩️ {getattr(msg, 'name', '?')} returned: {str(msg.content)[:150]}")

    answer = _extract_text(messages[-1].content)
    _memory.add_exchange(query, answer)  # auto-store every exchange (Phase 9)

    return {"answer": answer, "source": source, "trace": trace}


def run_agent(agent, query: str) -> str:
    """CLI wrapper around answer_query() - prints the trace to the terminal
    the same way it always has, but now delegates the actual logic to the
    shared answer_query() function.
    """
    print("    [invoking agent - this may take a few seconds per tool call...]")
    result = asyncio.run(answer_query(query))

    badge = "🟢 Gemini" if result["source"] == "cloud" else "🟡 llama3.2:3b (local)"
    print(f"\n[Agent trace] (answered by: {badge})")
    for line in result["trace"]:
        print(f"    {line}")

    return result["answer"]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 6: LangChain Agent + Tool Calling")
    print("=" * 70)

    print("\n[1] Loading and indexing documents...")
    n_chunks, n_files = load_document_index()
    print(f"    Indexed {n_chunks} chunks from {n_files} files")

    print("\n[2] Building agent (Gemini + document_search + calculator + web/youtube tools)...")
    agent = build_agent()
    print("    Agent ready.")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        print("-" * 70)
        answer = run_agent(agent, query)
        print(f"\nFinal answer: {answer}")
        sys.exit(0)

    # Test cases covering: doc search, calculator, NO tool, and the NEW
    # fallback chain (doc search -> web/youtube when docs have no answer)
    test_queries = [
        "What's 2384 * 17?",
        "How much is the home office stipend?",
        "Hi, how are you today?",
        "What status code should webhooks return, and what's that number doubled?",
        "How does FAISS similarity search actually work under the hood?",  # not in docs -> should escalate to youtube_search
        "What is the latest version of Python?",  # not in docs -> should escalate to web_search
        "Save a note called reminder.txt that says: follow up with Jordan about the LinkedIn campaign",  # -> write_note
        "What does reminder.txt say?",  # -> read_note
        "What did I ask you earlier about the home office stipend?",  # -> recall_memory
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {q}")
        print("=" * 70)
        answer = run_agent(agent, q)
        print(f"\nFinal answer: {answer}")
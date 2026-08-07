"""
Phase 7: External Fallback Tools — Web Search + YouTube Search

These tools activate when document_search finds NOTHING relevant in the
user's own documents. Instead of just saying "I don't know", the agent
escalates to external sources:
    - youtube_search: for conceptual/how-to questions -> suggests a video
    - web_search: for factual/informational questions -> suggests links

This is the "fallback tool chain" automation pattern: try the primary
source first (your documents), only escalate to a secondary source
(the web) if the primary source comes up empty. The same shape as
Phase 5's Gemini -> Ollama failover, applied to information sources
instead of LLM engines.

No API keys required - uses free, unofficial libraries:
    - ddgs (DuckDuckGo search)
    - youtubesearchpython (YouTube search, no official API needed)

Usage (standalone test, no agent required):
    python src/external_tools.py
"""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information NOT found in the user's documents.
    Use this ONLY after document_search has returned no relevant results,
    and the question is a factual/informational question (not something
    better answered by a video). Returns a list of relevant links with
    short descriptions."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))

        if not results:
            return "No web results found."

        formatted = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            snippet = r.get("body", "")[:150]
            formatted.append(f"- {title}\n  {url}\n  {snippet}...")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def youtube_search(query: str) -> str:
    """Search YouTube for a video explaining a CONCEPT or HOW-TO topic that
    is NOT found in the user's documents. Use this ONLY after document_search
    has returned no relevant results, and the question is asking to understand
    or learn something (e.g. 'how does X work', 'what is Y', 'how do I do Z') -
    not for simple factual lookups, which should use web_search instead.
    Returns a list of relevant video titles and links."""
    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch3:{query}", download=False)
            entries = info.get("entries", []) if info else []

        if not entries:
            return "No YouTube videos found."

        formatted = []
        for e in entries:
            title = e.get("title", "Untitled")
            video_id = e.get("id", "")
            url = f"https://www.youtube.com/watch?v={video_id}" if video_id else e.get("url", "")
            channel = e.get("channel", "unknown channel")
            duration = e.get("duration")
            duration_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "unknown length"
            formatted.append(f"- {title} ({duration_str}, by {channel})\n  {url}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"YouTube search failed: {e}"


# ---------------------------------------------------------------------------
# Standalone test — no agent needed, just calling the tool functions directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 7: External Fallback Tools — Standalone Test")
    print("=" * 70)

    print("\n[1] Testing web_search (factual question)...")
    print("-" * 70)
    result = web_search.invoke("what is the current version of Python")
    print(result)

    print("\n\n[2] Testing youtube_search (conceptual/how-to question)...")
    print("-" * 70)
    result = youtube_search.invoke("how does FAISS similarity search work")
    print(result)

    print("\n" + "=" * 70)
    print("If both returned real links/videos above, the tools work standalone.")
    print("Next step: wire these into agent.py's tool list (Phase 6 dependency).")
    print("=" * 70)
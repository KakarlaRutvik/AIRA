"""
Phase 8: MCP Filesystem Server

A standalone MCP server exposing two tools:
    - read_note(filename): reads a file from the notes/ directory
    - write_note(filename, content): writes a file to the notes/ directory

This runs as its OWN process, separate from your agent. The agent (or any
other MCP client) talks to it over the MCP protocol (stdio transport here),
NOT via a normal Python function call - this is what makes it genuinely
"MCP" rather than just another LangChain @tool function.

This file is NOT meant to be run with `python src/mcp_server.py` directly
for normal use - it's launched automatically as a subprocess by whichever
MCP client connects to it (see test_mcp_client.py and agent.py).
"""

import os

from fastmcp import FastMCP

NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

mcp = FastMCP("notes-server")


def _safe_path(filename: str) -> str:
    """Prevent path traversal - only allow plain filenames, no directory escape."""
    filename = os.path.basename(filename)  # strips any ../ or / components
    return os.path.join(NOTES_DIR, filename)


@mcp.tool()
def read_note(filename: str) -> str:
    """Read the contents of a saved note file by filename."""
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"Error: note '{filename}' does not exist."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading note '{filename}': {e}"


@mcp.tool()
def write_note(filename: str, content: str) -> str:
    """Save content to a note file by filename. Overwrites if it already exists."""
    path = _safe_path(filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved note to '{filename}'."
    except Exception as e:
        return f"Error writing note '{filename}': {e}"


@mcp.tool()
def list_notes() -> str:
    """List all saved note filenames."""
    files = [f for f in os.listdir(NOTES_DIR) if os.path.isfile(os.path.join(NOTES_DIR, f))]
    if not files:
        return "No notes saved yet."
    return "\n".join(files)


if __name__ == "__main__":
    # stdio transport: the server communicates over stdin/stdout with
    # whatever process launched it (the MCP client). It does NOT open a
    # network port - the client starts this script as a subprocess.
    mcp.run(transport="stdio")
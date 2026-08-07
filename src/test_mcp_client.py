"""
Phase 8: Standalone MCP Client Test

This spawns mcp_server.py as a subprocess and talks to it over the MCP
protocol directly - no agent, no LangChain involved. This proves the MCP
server itself works correctly before we ever wire it into the agent.

Usage:
    python src/test_mcp_client.py
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


async def main():
    print("=" * 70)
    print("Phase 8: Standalone MCP Client Test")
    print("=" * 70)

    server_params = StdioServerParameters(
        command=sys.executable,  # use the same python interpreter running this script
        args=[SERVER_SCRIPT],
    )

    print("\n[1] Launching mcp_server.py as a subprocess and connecting...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("    Connected.")

            print("\n[2] Listing available tools on the server...")
            tools_response = await session.list_tools()
            for t in tools_response.tools:
                print(f"    - {t.name}: {t.description}")

            print("\n[3] Calling write_note('test_note.txt', ...)...")
            result = await session.call_tool(
                "write_note",
                arguments={"filename": "test_note.txt", "content": "Hello from the MCP test client!"},
            )
            print(f"    Result: {result.content[0].text}")

            print("\n[4] Calling read_note('test_note.txt')...")
            result = await session.call_tool("read_note", arguments={"filename": "test_note.txt"})
            print(f"    Result: {result.content[0].text}")

            print("\n[5] Calling list_notes()...")
            result = await session.call_tool("list_notes", arguments={})
            print(f"    Result: {result.content[0].text}")

            print("\n[6] Testing error handling: read_note on a file that doesn't exist...")
            result = await session.call_tool("read_note", arguments={"filename": "does_not_exist.txt"})
            print(f"    Result: {result.content[0].text}")

    print("\n" + "=" * 70)
    print("If all steps above completed without a Python traceback, the MCP")
    print("server is working correctly. Check notes/test_note.txt was created.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
"""MCP server that exposes MedScribe's domain tools.

Run it on its own:

    python -m medscribe.mcp_server

Any MCP client can connect (Claude Desktop, an IDE, another agent). MedScribe's own agent
connects to it when MEDSCRIBE_TOOL_BACKEND=mcp, see mcp_backend.py.

Only read-only, stateless tools are exposed. Writing to patient memory is not offered to the
model; the agent does it in code after a note passes verification.
"""

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .tools import REGISTRY


def build_server():
    server = _Server("medscribe-tools")
    for spec in REGISTRY.values():
        server.add_tool(spec.fn, name=spec.name, description=spec.description)
    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()

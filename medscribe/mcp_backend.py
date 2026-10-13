"""Tool backend that talks to the MedScribe MCP server over stdio.

The agent code is synchronous, the MCP client is async. A background thread runs one long-lived
task that owns the connection, and calls are submitted to it from any thread.
"""

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

from . import config
from .tools import ToolError


def _get(obj: Any, *names: str, default=None):
    for n in names:
        if hasattr(obj, n) and getattr(obj, n) is not None:
            return getattr(obj, n)
    return default


class MCPBackend:
    def __init__(self, command: str | None = None, args: list[str] | None = None, timeout: float = 60):
        self.command = command or sys.executable
        self.args = args or ["-m", "medscribe.mcp_server"]
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-client")
        self._thread.start()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session = None
        self._specs: list[dict] = []
        self._error: BaseException | None = None
        asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        if not self._ready.wait(timeout):
            raise RuntimeError("MCP server did not start in time")
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error}")

    async def _main(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._stop = asyncio.Event()
        try:
            params = StdioServerParameters(command=self.command, args=self.args, env=dict(os.environ),
                                           cwd=str(config.ROOT))
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listing = await session.list_tools()
                self._specs = [{
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": _get(t, "inputSchema", "input_schema", default={"type": "object"}),
                    }} for t in listing.tools]
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surface any startup failure to the caller
            self._error = e
            self._ready.set()

    def specs(self) -> list[dict]:
        return self._specs

    def call(self, name: str, arguments: dict[str, Any]) -> dict:
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        res = future.result(self.timeout)
        text = " ".join(getattr(c, "text", "") for c in (res.content or []))
        if _get(res, "isError", "is_error", default=False):
            raise ToolError(text or f"MCP tool {name} failed")
        structured = _get(res, "structuredContent", "structured_content")
        if isinstance(structured, dict):
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"result": text}

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)


def make_backend(kind: str | None = None):
    """Return the tool backend selected by MEDSCRIBE_TOOL_BACKEND ('local' or 'mcp')."""
    from .tools import LocalBackend
    kind = (kind or config.TOOL_BACKEND).lower()
    if kind == "mcp":
        return MCPBackend()
    if kind == "local":
        return LocalBackend()
    raise ValueError(f"Unknown tool backend: {kind}")

"""Stateful MCP proxy — make ANY third-party MCP server stateful, unmodified.

MCP is stateless per call by design: a downstream server keeps no history, and
neither do you. This proxy sits between the client and any downstream MCP
server, mirrors its tools 1:1 (same names, same schemas), and records every
call + result as ``tool_call`` events in the durable log. You get:

- an audit trail of every MCP interaction, per session, replayable
- time-travel ("what had the agent called as of seq N?")
- optional deterministic replay: identical (tool, args) calls return the
  recorded result without hitting the downstream server — for reproducing a
  run exactly, or resuming one without re-executing side effects

Run (stdio, wraps any stdio MCP server command)::

    statefold-mcp-proxy -- npx -y @modelcontextprotocol/server-filesystem C:/data
    STATEFOLD_DSN=postgresql://... STATEFOLD_REPLAY=1 statefold-mcp-proxy -- <cmd...>

Scope comes from env: STATEFOLD_TENANT / STATEFOLD_AGENT / STATEFOLD_SESSION.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

from .store import ConcurrencyError, StateStore
from .types import Event, Scope

_RETRIES = 8


def _args_key(tool: str, args: dict) -> str:
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class ProxyRecorder:
    """Records downstream MCP calls as events; optionally replays cached results.

    Framework-free core so it can be tested without spawning servers: the
    downstream is any ``async (tool_name, args) -> result`` callable.
    """

    def __init__(self, store: StateStore, tenant: str, agent: str,
                 session: str = "mcp", replay: bool = False):
        self.store = store
        self.scope = Scope(tenant, agent, session, "main")
        self.replay = replay

    async def _recorded(self, key: str):
        state = await self.store.get_state(self.scope)
        for call in reversed(state.get("tool_calls", [])):
            if call.get("key") == key:
                return call
        return None

    async def _append(self, event: Event) -> None:
        for _ in range(_RETRIES):
            head = await self.store.head(self.scope)
            try:
                await self.store.append(self.scope, [event], expected_seq=head)
                return
            except ConcurrencyError:
                continue
        raise RuntimeError(f"append retries exhausted on {self.scope.flatten()}")

    async def call(self, tool_name: str, args: dict, downstream):
        """Proxy one tool call through the recorder."""
        key = _args_key(tool_name, args)
        if self.replay:
            hit = await self._recorded(key)
            if hit is not None:
                return hit["result"], True  # (result, was_replayed)

        t0 = time.perf_counter()
        try:
            result = await downstream(tool_name, args)
            error = None
        except Exception as e:
            result, error = None, f"{type(e).__name__}: {e}"
        latency_ms = max(round((time.perf_counter() - t0) * 1000, 3), 0.001)
        await self._append(Event(
            kind="tool_call",
            payload={"name": tool_name, "args": args, "result": result, "key": key,
                     "latency_ms": latency_ms, "error": error},
            actor=f"mcp:{tool_name}",
        ))
        if error is not None:
            raise RuntimeError(f"downstream tool '{tool_name}' failed: {error}")
        return result, False

    async def history(self, as_of: int | None = None) -> list[dict]:
        """Every recorded MCP call for this session, oldest first."""
        state = await self.store.get_state(self.scope, as_of=as_of)
        return state.get("tool_calls", [])


# --- stdio plumbing: mirror a real downstream server ------------------------

async def run_proxy(downstream_cmd: list[str]) -> None:
    import mcp.types as types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    from .memory import InMemoryStore

    dsn = os.getenv("STATEFOLD_DSN")
    if dsn:
        from .postgres import PostgresStore

        store: StateStore = await PostgresStore.connect(dsn)
    else:
        store = InMemoryStore()

    recorder = ProxyRecorder(
        store,
        tenant=os.getenv("STATEFOLD_TENANT", "default"),
        agent=os.getenv("STATEFOLD_AGENT", "default"),
        session=os.getenv("STATEFOLD_SESSION", "mcp"),
        replay=os.getenv("STATEFOLD_REPLAY", "") not in ("", "0", "false"),
    )

    params = StdioServerParameters(command=downstream_cmd[0], args=downstream_cmd[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as downstream:
            await downstream.initialize()
            listed = await downstream.list_tools()

            proxy = Server("statefold-proxy")

            @proxy.list_tools()
            async def list_tools():
                return listed.tools  # downstream schemas, verbatim

            @proxy.call_tool()
            async def call_tool(name: str, arguments: dict):
                async def forward(tool, args):
                    res = await downstream.call_tool(tool, args)
                    return [c.model_dump() for c in res.content]

                content, _replayed = await recorder.call(name, arguments or {}, forward)
                return [types.TextContent(**c) if c.get("type") == "text"
                        else types.TextContent(type="text", text=json.dumps(c))
                        for c in content]

            async with stdio_server() as (srv_read, srv_write):
                await proxy.run(srv_read, srv_write, proxy.create_initialization_options())


def main() -> None:
    import asyncio

    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print("usage: statefold-mcp-proxy -- <downstream mcp server command...>",
              file=sys.stderr)
        raise SystemExit(2)
    asyncio.run(run_proxy(argv))


if __name__ == "__main__":
    main()

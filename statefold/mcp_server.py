"""MCP state server — exposes a StateStore as MCP tools.

Any MCP-capable agent (raw, CrewAI, Agno, a Claude/GPT tool loop) gets durable,
event-sourced state by adding this one server. No framework integration needed.

Run:
    statefold-mcp                      # in-memory (ephemeral)
    STATEFOLD_DSN=postgresql://... statefold-mcp   # durable

Scope: ``tenant``/``agent`` come from env (STATEFOLD_TENANT / STATEFOLD_AGENT)
so a client can't spoof another tenant; ``session``/``thread`` are tool args.

Requires ``mcp`` (``pip install statefold[mcp]``).
"""
from __future__ import annotations

import asyncio
import os

from mcp.server.fastmcp import FastMCP

from .memory import InMemoryStore
from .store import ConcurrencyError
from .types import Event, Scope

mcp = FastMCP("statefold")

_store = None
_store_lock = asyncio.Lock()


async def get_store():
    global _store
    if _store is None:
        async with _store_lock:
            if _store is None:
                dsn = os.getenv("STATEFOLD_DSN")
                if dsn:
                    from .postgres import PostgresStore

                    _store = await PostgresStore.connect(dsn)
                else:
                    _store = InMemoryStore()
    return _store


def _scope(session: str, thread: str) -> Scope:
    return Scope(
        tenant=os.getenv("STATEFOLD_TENANT", "default"),
        agent=os.getenv("STATEFOLD_AGENT", "default"),
        session=session,
        thread=thread,
    )


# --- tool implementations (plain functions, registered below so they stay testable) ---

async def state_head(session: str, thread: str = "main"):
    """Current sequence number of a stream (0 if empty)."""
    store = await get_store()
    return await store.head(_scope(session, thread))


async def state_get(session: str, thread: str = "main", as_of: int | None = None):
    """Current folded state for a session/thread. ``as_of`` bounds the fold (time-travel)."""
    store = await get_store()
    return await store.get_state(_scope(session, thread), as_of=as_of)


async def state_append(
    session: str,
    events: list[dict],
    thread: str = "main",
    expected_seq: int | None = None,
):
    """Append events. Each event: {"kind": str, "payload": dict, "actor"?: str}.

    Pass ``expected_seq`` (from state_head) for optimistic concurrency; omit it
    to append at the current head. Returns {"ok", "head"} or a conflict report.
    """
    store = await get_store()
    scope = _scope(session, thread)
    evs = [
        Event(kind=e["kind"], payload=e.get("payload", {}), actor=e.get("actor", "agent"))
        for e in events
    ]
    head = expected_seq if expected_seq is not None else await store.head(scope)
    try:
        new_head = await store.append(scope, evs, expected_seq=head)
        return {"ok": True, "head": new_head}
    except ConcurrencyError as e:
        return {"ok": False, "error": "conflict", "expected": e.expected, "actual": e.actual}


async def state_checkpoint(session: str, thread: str = "main", label: str | None = None):
    """Persist a snapshot at the current head. Returns a checkpoint ref."""
    store = await get_store()
    return await store.checkpoint(_scope(session, thread), label=label)


async def memory_remember(session: str, text: str, meta: dict | None = None,
                          kind: str = "semantic", level: str = "session"):
    """Write a memory. kind: semantic|episodic|procedural. level: session
    (dies with the session) or agent (long-term, shared across sessions)."""
    store = await get_store()
    return await store.remember(_scope(session, "main"), text, meta=meta,
                                kind=kind, level=level)


async def memory_recall(session: str, query: str, k: int = 5,
                        kind: str | None = None, level: str | None = None):
    """Retrieve relevant memories. Searches session + agent levels by
    default; filter by kind (semantic|episodic|procedural) or level."""
    store = await get_store()
    return await store.recall(_scope(session, "main"), query, k=k,
                              kind=kind, level=level)


async def state_verify(session: str, thread: str = "main"):
    """Verify the tamper-evidence hash chain for a session's event stream."""
    from .integrity import verify_chain

    store = await get_store()
    return await verify_chain(store, _scope(session, thread))


for _fn in (state_head, state_get, state_append, state_checkpoint, state_verify,
            memory_remember, memory_recall):
    mcp.tool(name=_fn.__name__)(_fn)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()

"""Generic façade: a high-level handle to one session's state.

For raw agents, tool loops, or any framework without a purpose-built adapter.
Hides events, seq, and reducers behind an ergonomic API, and handles optimistic
concurrency (retry-on-conflict) internally.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

from ..store import ConcurrencyError, StateStore
from ..types import Event, Scope, new_id

_current_span: ContextVar[str | None] = ContextVar("statefold_current_span",
                                                   default=None)

_RETRIES = 8


class SessionState:
    def __init__(self, store: StateStore, tenant: str, agent: str, session: str,
                 thread: str = "main"):
        self.store = store
        self.scope = Scope(tenant, agent, session, thread)

    async def _append(self, *events: Event) -> int:
        # events recorded inside a `span(...)` block hang under that span
        sid = _current_span.get()
        if sid is not None:
            for ev in events:
                if ev.causation_id is None:
                    ev.causation_id = sid
        for _ in range(_RETRIES):
            head = await self.store.head(self.scope)
            try:
                return await self.store.append(self.scope, list(events), expected_seq=head)
            except ConcurrencyError:
                continue
        raise RuntimeError(f"append conflict retries exhausted on {self.scope.flatten()}")

    # --- conversation ---
    async def add_message(self, role: str, content, **meta) -> None:
        payload = {"role": role, "content": content, **meta}
        await self._append(Event(kind="message", payload=payload, actor=role))

    async def messages(self, as_of: int | None = None) -> list:
        return (await self.state(as_of)).get("messages", [])

    # --- tool calls ---
    async def add_tool_call(self, name: str, args=None, result=None,
                            latency_ms: float | None = None, error: str | None = None) -> None:
        await self._append(Event(
            kind="tool_call",
            payload={"name": name, "args": args or {}, "result": result,
                     "latency_ms": latency_ms, "error": error},
            actor=f"tool:{name}",
        ))

    # --- telemetry / cost ---
    async def add_llm_call(self, model: str, input_tokens: int, output_tokens: int,
                           latency_ms: float | None = None,
                           cost_usd: float | None = None, **meta) -> None:
        await self._append(Event(
            kind="llm_call",
            payload={"model": model, "input_tokens": input_tokens,
                     "output_tokens": output_tokens, "latency_ms": latency_ms,
                     "cost_usd": cost_usd, **meta},
            actor=f"llm:{model}",
        ))

    async def usage(self, as_of: int | None = None) -> dict:
        """Usage/cost summary; ``as_of`` = spend as of step N."""
        from ..telemetry import usage_report

        return await usage_report(self.store, self.scope, as_of=as_of)

    # --- working-state bag ---
    async def set(self, **values) -> None:
        await self._append(Event(kind="state_delta", payload={"set": values}))

    async def unset(self, *keys: str) -> None:
        await self._append(Event(kind="state_delta", payload={"unset": list(keys)}))

    async def get(self, key: str, default=None):
        return (await self.state()).get("data", {}).get(key, default)

    # --- tracing -----------------------------------------------------------

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        """Trace a unit of work as a span in the event log.

        Spans nest automatically (contextvar-based parenting), and any event
        recorded inside the block — llm calls, tool calls, messages — is
        linked to the span via ``causation_id``. Exceptions mark the span
        ``error`` and re-raise. Spans are events, so they are hash-chained,
        replayable, and time-travelable like everything else.

            async with s.span("plan-trip", city="paris"):
                await s.add_llm_call(...)
                async with s.span("book-hotel"):
                    await s.add_tool_call(...)
        """
        sid = new_id()
        parent = _current_span.get()
        await self._append(Event(
            kind="span_start", actor="trace",
            payload={"span_id": sid, "parent_id": parent, "name": name,
                     "attrs": attrs},
        ))
        token = _current_span.set(sid)
        t0 = time.perf_counter()
        status, error = "ok", None
        try:
            yield sid
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            _current_span.reset(token)
            await self._append(Event(
                kind="span_end", actor="trace", causation_id=parent,
                payload={"span_id": sid, "status": status, "error": error,
                         "duration_ms": round((time.perf_counter() - t0) * 1000, 2)},
            ))

    async def trace(self, as_of: int | None = None) -> dict:
        """The span tree for this session: {span_id: {name, parent_id, ...}}."""
        return (await self.state(as_of)).get("spans", {})

    # --- whole state / durability ---
    async def state(self, as_of: int | None = None) -> dict:
        return await self.store.get_state(self.scope, as_of=as_of)

    async def head(self) -> int:
        return await self.store.head(self.scope)

    async def snapshot(self, label: str | None = None) -> str:
        return await self.store.checkpoint(self.scope, label=label)

    # --- memory taxonomy -----------------------------------------------
    # working memory  = the folded state (messages + data bag), free
    # semantic        = facts/preferences (kind="semantic")
    # episodic        = past experiences  (kind="episodic", agent-level)
    # procedural      = learned how-tos   (kind="procedural")
    # level "session" dies with the session; "agent" persists across them.

    async def working_memory(self) -> dict:
        """Short-term memory: the current folded state (messages + data)."""
        state = await self.state()
        return {"messages": state.get("messages", []), "data": state.get("data", {})}

    async def remember(self, text: str, meta: dict | None = None,
                       kind: str = "semantic", level: str = "session") -> str:
        return await self.store.remember(self.scope, text, meta=meta,
                                         kind=kind, level=level)

    async def recall(self, query: str, k: int = 5, kind: str | None = None,
                     level: str | None = None) -> list[dict]:
        return await self.store.recall(self.scope, query, k=k, kind=kind, level=level)

    async def end_episode(self, summary: str, outcome: str = "success", **meta) -> str:
        """Record this session as an episodic memory (agent-level), so future
        sessions can recall 'have I handled something like this before?'."""
        return await self.store.remember(
            self.scope, summary,
            meta={"outcome": outcome, "stream": self.scope.flatten(),
                  "head_seq": await self.head(), **meta},
            kind="episodic", level="agent",
        )

    async def recall_episodes(self, query: str, k: int = 3) -> list[dict]:
        """Relevant past episodes from ANY session of this agent."""
        return await self.store.recall(self.scope, query, k=k,
                                       kind="episodic", level="agent")

    async def learn_procedure(self, text: str, meta: dict | None = None) -> str:
        """Store a learned how-to (agent-level procedural memory)."""
        return await self.store.remember(self.scope, text, meta=meta,
                                         kind="procedural", level="agent")

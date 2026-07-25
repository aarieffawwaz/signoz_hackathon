"""Agno adapter: an Agno v2 ``Db`` whose sessions and user memories are
backed by a StateStore.

Agno's ``BaseDb`` is a very wide interface (sessions, memories, metrics,
evals, traces, schedules, ...). Rather than reimplement all of it, this
adapter subclasses Agno's own ``InMemoryDb`` and *write-throughs* the two
surfaces that constitute agent state — sessions and user memories — to the
durable event log. Everything else (metrics, evals, traces) keeps Agno's
default behavior.

Every session upsert becomes an ``agno_session`` event, so session history is
replayable and time-travelable like any other stream. On construction the
adapter rehydrates Agno's in-memory caches from the log, which is what makes
a restarted process resume where it left off.

Usage::

    from agno.agent import Agent
    from statefold import InMemoryStore  # or PostgresStore
    from statefold.adapters.agno import AgentStateDb

    db = AgentStateDb(store, tenant="acme", agent="assistant")
    agent = Agent(db=db, add_history_to_context=True, ...)

Requires ``agno>=2`` (``pip install statefold[agno]``).
"""
from __future__ import annotations


from agno.db.in_memory import InMemoryDb

from ..reducers import register
from ..store import ConcurrencyError, StateStore
from ..types import Event, Scope

_RETRIES = 8


@register("agno_session")
def _agno_session(state: dict, payload: dict) -> dict:
    """Latest-wins fold of Agno session upserts; a delete clears it."""
    if payload.get("deleted"):
        state.pop("_agno_session", None)
    else:
        state["_agno_session"] = {"session": payload["session"],
                                  "type": payload.get("type", "agent")}
    return state


from ..sync_bridge import SyncBridge as _Bridge  # noqa: E402


class AgentStateDb(InMemoryDb):
    def __init__(self, store: StateStore, tenant: str, agent: str, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self.tenant = tenant
        self.agent = agent
        self._bridge = _Bridge()
        self._rehydrate()

    # --- plumbing ---------------------------------------------------------

    def _scope(self, session_id: str) -> Scope:
        return Scope(self.tenant, self.agent, session_id, "main")

    def _append(self, scope: Scope, event: Event) -> None:
        async def go():
            for _ in range(_RETRIES):
                head = await self.store.head(scope)
                try:
                    await self.store.append(scope, [event], expected_seq=head)
                    return
                except ConcurrencyError:
                    continue
            raise RuntimeError(f"append retries exhausted on {scope.flatten()}")

        self._bridge.run(go())

    def _rehydrate(self) -> None:
        """Rebuild Agno's in-memory caches from the durable log on startup."""

        async def load():
            registry = Scope(self.tenant, self.agent, "_registry", "main")
            state = await self.store.get_state(registry)
            sessions = []
            for session_id in state.get("data", {}).get("sessions", []):
                s = await self.store.get_state(self._scope(session_id))
                if s.get("_agno_session"):
                    sessions.append(s["_agno_session"])
            mem_scope = Scope(self.tenant, self.agent, "_memories", "main")
            mems = await self.store.get_state(mem_scope)
            return sessions, mems.get("data", {})

        sessions, memories = self._bridge.run(load())
        for entry in sessions:
            self._restore_session_dict(entry)
        for mid, raw in memories.items():
            self._restore_memory_dict(mid, raw)

    def _restore_session_dict(self, entry: dict) -> None:
        from agno.session.agent import AgentSession
        from agno.session.team import TeamSession
        from agno.session.workflow import WorkflowSession

        cls = {"agent": AgentSession, "team": TeamSession,
               "workflow": WorkflowSession}.get(entry.get("type", "agent"), AgentSession)
        InMemoryDb.upsert_session(self, cls.from_dict(entry["session"]))

    def _restore_memory_dict(self, mid: str, raw: dict) -> None:
        from agno.db.schemas.memory import UserMemory

        InMemoryDb.upsert_user_memory(self, UserMemory.from_dict(raw))

    def _track_session_id(self, session_id: str) -> None:
        registry = Scope(self.tenant, self.agent, "_registry", "main")

        async def go():
            state = await self.store.get_state(registry)
            known = state.get("data", {}).get("sessions", [])
            if session_id in known:
                return
            for _ in range(_RETRIES):
                head = await self.store.head(registry)
                ev = Event(kind="state_delta",
                           payload={"set": {"sessions": known + [session_id]}},
                           actor="agno")
                try:
                    await self.store.append(registry, [ev], expected_seq=head)
                    return
                except ConcurrencyError:
                    state = await self.store.get_state(registry)
                    known = state.get("data", {}).get("sessions", [])
                    if session_id in known:
                        return

        self._bridge.run(go())

    # --- write-through overrides: sessions --------------------------------

    def upsert_session(self, session, deserialize=True):
        from agno.session.team import TeamSession
        from agno.session.workflow import WorkflowSession

        result = InMemoryDb.upsert_session(self, session, deserialize=deserialize)
        stype = ("team" if isinstance(session, TeamSession)
                 else "workflow" if isinstance(session, WorkflowSession) else "agent")
        self._track_session_id(session.session_id)
        self._append(
            self._scope(session.session_id),
            Event(kind="agno_session",
                  payload={"session": session.to_dict(), "type": stype},
                  actor="agno"),
        )
        return result

    def delete_session(self, session_id: str, user_id=None) -> bool:
        ok = InMemoryDb.delete_session(self, session_id, user_id=user_id)
        if ok:
            self._append(
                self._scope(session_id),
                Event(kind="agno_session", payload={"session": None, "deleted": True},
                      actor="agno"),
            )
        return ok

    # --- write-through overrides: user memories ---------------------------

    def upsert_user_memory(self, memory, deserialize=True):
        result = InMemoryDb.upsert_user_memory(self, memory, deserialize=deserialize)
        mem_scope = Scope(self.tenant, self.agent, "_memories", "main")
        self._append(
            mem_scope,
            Event(kind="state_delta",
                  payload={"set": {memory.memory_id: memory.to_dict()}},
                  actor="agno"),
        )
        # also index into semantic recall so cross-framework recall sees it
        self._bridge.run(self.store.remember(
            mem_scope, str(memory.memory), meta={"memory_id": memory.memory_id}))
        return result

    def delete_user_memory(self, memory_id: str, user_id=None):
        result = InMemoryDb.delete_user_memory(self, memory_id, user_id=user_id)
        mem_scope = Scope(self.tenant, self.agent, "_memories", "main")
        self._append(
            mem_scope,
            Event(kind="state_delta", payload={"unset": [memory_id]}, actor="agno"),
        )
        return result

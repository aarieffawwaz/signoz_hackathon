"""Prove Agno sessions and user memories survive a process restart."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore

pytest.importorskip("agno")

from agno.db.schemas.memory import UserMemory  # noqa: E402
from agno.session.agent import AgentSession  # noqa: E402

from statefold.adapters.agno import AgentStateDb  # noqa: E402


def _mk_session(sid: str) -> AgentSession:
    return AgentSession(session_id=sid, agent_id="assistant", user_id="u1",
                        session_data={"topic": "hotels"})


def test_session_survives_restart():
    store = InMemoryStore()

    db1 = AgentStateDb(store, tenant="acme", agent="assistant")
    db1.upsert_session(_mk_session("s1"))

    # process "dies"; a brand-new Db on the same store rehydrates from the log
    db2 = AgentStateDb(store, tenant="acme", agent="assistant")
    got = db2.get_session("s1")
    assert got is not None
    assert got.session_id == "s1"
    assert got.session_data == {"topic": "hotels"}


def test_deleted_session_stays_deleted_after_restart():
    store = InMemoryStore()
    db1 = AgentStateDb(store, tenant="acme", agent="assistant")
    db1.upsert_session(_mk_session("s1"))
    db1.delete_session("s1")

    db2 = AgentStateDb(store, tenant="acme", agent="assistant")
    assert db2.get_session("s1") is None


def test_user_memories_survive_restart_and_are_recallable():
    store = InMemoryStore()
    db1 = AgentStateDb(store, tenant="acme", agent="assistant")
    db1.upsert_user_memory(UserMemory(memory="user prefers window seats",
                                      memory_id="m1", user_id="u1"))

    db2 = AgentStateDb(store, tenant="acme", agent="assistant")
    mems = db2.get_user_memories(user_id="u1")
    assert [m.memory for m in mems] == ["user prefers window seats"]

    # the same memory is visible through the platform's semantic recall,
    # i.e. shareable with non-Agno agents on the same store
    import asyncio

    from statefold.types import Scope

    hits = asyncio.run(store.recall(
        Scope("acme", "assistant", "_memories", "main"), "window seats", k=1))
    assert hits and "window" in hits[0]["text"]


def test_deleted_memory_stays_deleted_after_restart():
    store = InMemoryStore()
    db1 = AgentStateDb(store, tenant="acme", agent="assistant")
    db1.upsert_user_memory(UserMemory(memory="temp fact", memory_id="m1", user_id="u1"))
    db1.delete_user_memory("m1", user_id="u1")

    db2 = AgentStateDb(store, tenant="acme", agent="assistant")
    assert db2.get_user_memories(user_id="u1") == []


def test_session_history_is_an_event_stream():
    import asyncio

    from statefold.types import Scope

    store = InMemoryStore()
    db = AgentStateDb(store, tenant="acme", agent="assistant")
    s = _mk_session("s1")
    db.upsert_session(s)
    s.session_data = {"topic": "flights"}
    db.upsert_session(s)

    async def kinds():
        return [e.kind async for e in store.read_events(Scope("acme", "assistant", "s1", "main"))]

    assert asyncio.run(kinds()) == ["agno_session", "agno_session"]
    # time-travel: first upsert had the old topic
    state_v1 = asyncio.run(store.get_state(Scope("acme", "assistant", "s1", "main"), as_of=1))
    assert state_v1["_agno_session"]["session"]["session_data"] == {"topic": "hotels"}

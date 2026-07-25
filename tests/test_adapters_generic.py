"""Generic SessionState façade + CrewAI storage adapter."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore
from statefold.adapters.crewai import AgentStateStorage
from statefold.adapters.generic import SessionState


async def test_session_facade_roundtrip():
    s = SessionState(InMemoryStore(), tenant="acme", agent="a", session="s1")
    await s.add_message("user", "book a hotel")
    await s.set(stage="searching", city="paris")
    await s.add_tool_call("search", {"q": "hotel"}, result="3 hits")

    assert await s.get("stage") == "searching"
    assert [m["content"] for m in await s.messages()] == ["book a hotel"]
    state = await s.state()
    assert state["tool_calls"][0]["name"] == "search"


async def test_session_facade_time_travel_and_snapshot():
    s = SessionState(InMemoryStore(), tenant="acme", agent="a", session="s1")
    await s.set(n="a")
    await s.snapshot("cp")
    await s.set(n="b")
    assert (await s.state(as_of=1))["data"]["n"] == "a"
    assert await s.get("n") == "b"


async def test_session_facade_memory():
    s = SessionState(InMemoryStore(), tenant="acme", agent="a", session="s1")
    await s.remember("user prefers window seats", {"tag": "pref"})
    hits = await s.recall("window seats", k=1)
    assert hits and "window" in hits[0]["text"]


def test_crewai_storage_is_sync_and_roundtrips():
    # CrewAI calls storage synchronously; verify save/search/reset without a loop.
    storage = AgentStateStorage(InMemoryStore(), tenant="acme", agent="crew1")
    storage.save("customer is vegetarian", metadata={"kind": "pref"})
    storage.save("meeting is on tuesday", metadata={"kind": "fact"})

    hits = storage.search("vegetarian customer", limit=1)
    assert hits and hits[0]["context"] == "customer is vegetarian"
    assert hits[0]["metadata"] == {"kind": "pref"}

    storage.reset()
    assert storage.search("vegetarian", limit=5) == []

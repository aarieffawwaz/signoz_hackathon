"""Exercise the MCP tool functions against an in-memory store."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from statefold import InMemoryStore  # noqa: E402
from statefold import mcp_server as m  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    m._store = InMemoryStore()
    yield
    m._store = None


async def test_append_get_roundtrip():
    r = await m.state_append("s1", [
        {"kind": "message", "payload": {"role": "user", "content": "hi"}},
        {"kind": "state_delta", "payload": {"set": {"stage": "start"}}},
    ])
    assert r == {"ok": True, "head": 2}
    state = await m.state_get("s1")
    assert state["data"]["stage"] == "start"
    assert await m.state_head("s1") == 2


async def test_optimistic_conflict_is_reported_not_raised():
    await m.state_append("s1", [{"kind": "message", "payload": {}}], expected_seq=0)
    r = await m.state_append("s1", [{"kind": "message", "payload": {}}], expected_seq=0)
    assert r["ok"] is False and r["error"] == "conflict" and r["actual"] == 1


async def test_time_travel_via_as_of():
    await m.state_append("s1", [
        {"kind": "state_delta", "payload": {"set": {"n": "a"}}},
        {"kind": "state_delta", "payload": {"set": {"n": "b"}}},
    ])
    assert (await m.state_get("s1", as_of=1))["data"]["n"] == "a"
    assert (await m.state_get("s1"))["data"]["n"] == "b"


async def test_memory_tools():
    await m.memory_remember("s1", "user likes aisle seats", meta={"tag": "pref"})
    hits = await m.memory_recall("s1", "aisle seats", k=1)
    assert hits and "aisle" in hits[0]["text"]


async def test_tools_are_registered():
    tools = await m.mcp.list_tools()
    names = {t.name for t in tools}
    assert {"state_append", "state_get", "state_head", "memory_recall"} <= names

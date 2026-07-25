"""ProxyRecorder: audit trail, deterministic replay, time-travel over MCP calls."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore
from statefold.mcp_proxy import ProxyRecorder


@pytest.fixture
def store():
    return InMemoryStore()


def make_downstream(counter: dict):
    async def downstream(tool, args):
        counter[tool] = counter.get(tool, 0) + 1
        return {"echo": args, "call_n": counter[tool]}

    return downstream


async def test_calls_are_recorded_as_events(store):
    rec = ProxyRecorder(store, "acme", "bot")
    down = make_downstream({})

    r1, replayed = await rec.call("search", {"q": "hotels"}, down)
    assert r1["echo"] == {"q": "hotels"} and replayed is False

    hist = await rec.history()
    assert len(hist) == 1
    assert hist[0]["name"] == "search"
    assert hist[0]["args"] == {"q": "hotels"}


async def test_replay_mode_returns_cached_without_reexecuting(store):
    counter = {}
    down = make_downstream(counter)
    rec = ProxyRecorder(store, "acme", "bot", replay=True)

    r1, rep1 = await rec.call("fetch", {"url": "x"}, down)
    r2, rep2 = await rec.call("fetch", {"url": "x"}, down)  # identical call

    assert rep1 is False and rep2 is True
    assert r1 == r2                       # deterministic
    assert counter["fetch"] == 1          # downstream hit exactly once


async def test_replay_off_always_reexecutes(store):
    counter = {}
    down = make_downstream(counter)
    rec = ProxyRecorder(store, "acme", "bot", replay=False)

    await rec.call("fetch", {"url": "x"}, down)
    await rec.call("fetch", {"url": "x"}, down)
    assert counter["fetch"] == 2
    assert len(await rec.history()) == 2


async def test_different_args_are_not_cache_hits(store):
    counter = {}
    rec = ProxyRecorder(store, "acme", "bot", replay=True)
    down = make_downstream(counter)

    await rec.call("fetch", {"url": "a"}, down)
    _, replayed = await rec.call("fetch", {"url": "b"}, down)
    assert replayed is False and counter["fetch"] == 2


async def test_time_travel_over_call_history(store):
    rec = ProxyRecorder(store, "acme", "bot")
    down = make_downstream({})
    await rec.call("step1", {}, down)
    await rec.call("step2", {}, down)

    assert [c["name"] for c in await rec.history(as_of=1)] == ["step1"]
    assert [c["name"] for c in await rec.history()] == ["step1", "step2"]


async def test_sessions_are_isolated(store):
    down = make_downstream({})
    a = ProxyRecorder(store, "acme", "bot", session="run-a")
    b = ProxyRecorder(store, "acme", "bot", session="run-b")
    await a.call("t", {"x": 1}, down)

    assert len(await a.history()) == 1
    assert await b.history() == []

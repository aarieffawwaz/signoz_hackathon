"""Core guarantees: append/fold, concurrency, time-travel, checkpoint, fork, memory."""
from __future__ import annotations

import asyncio

import pytest

from statefold import ConcurrencyError, Event, InMemoryStore, Scope


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def scope():
    return Scope(tenant="t", agent="a", session="s")


async def test_append_and_fold(store, scope):
    await store.append(scope, [
        Event(kind="message", payload={"role": "user", "content": "hi"}),
        Event(kind="state_delta", payload={"set": {"stage": "start"}}),
    ], expected_seq=0)
    state = await store.get_state(scope)
    assert state["messages"] == [{"role": "user", "content": "hi"}]
    assert state["data"]["stage"] == "start"
    assert await store.head(scope) == 2


async def test_concurrency_conflict(store, scope):
    await store.append(scope, [Event(kind="message", payload={"c": 1})], expected_seq=0)
    with pytest.raises(ConcurrencyError):
        await store.append(scope, [Event(kind="message", payload={"c": 2})], expected_seq=0)
    # retry on fresh head succeeds
    head = await store.head(scope)
    await store.append(scope, [Event(kind="message", payload={"c": 2})], expected_seq=head)
    assert await store.head(scope) == 2


async def test_parallel_writers_one_wins(store, scope):
    async def writer():
        head = await store.head(scope)
        try:
            await store.append(scope, [Event(kind="message", payload={})], expected_seq=head)
            return True
        except ConcurrencyError:
            return False

    results = await asyncio.gather(*[writer() for _ in range(8)])
    # at least one succeeds; head never skips (gap-free)
    assert any(results)
    assert await store.head(scope) == sum(results)


async def test_time_travel(store, scope):
    await store.append(scope, [
        Event(kind="state_delta", payload={"set": {"stage": "a"}}),
        Event(kind="state_delta", payload={"set": {"stage": "b"}}),
        Event(kind="state_delta", payload={"set": {"stage": "c"}}),
    ], expected_seq=0)
    assert (await store.get_state(scope, as_of=1))["data"]["stage"] == "a"
    assert (await store.get_state(scope, as_of=2))["data"]["stage"] == "b"
    assert (await store.get_state(scope))["data"]["stage"] == "c"


async def test_checkpoint_is_only_an_optimization(store, scope):
    await store.append(scope, [Event(kind="state_delta", payload={"set": {"n": 1}})], expected_seq=0)
    await store.checkpoint(scope)
    await store.append(scope, [Event(kind="state_delta", payload={"set": {"n": 2}})], expected_seq=1)
    with_snap = await store.get_state(scope)
    # wiping snapshots must not change the answer
    store._snaps.clear()
    assert await store.get_state(scope) == with_snap


async def test_fork_isolates_branches(store, scope):
    await store.append(scope, [
        Event(kind="state_delta", payload={"set": {"stage": "shared"}}),
        Event(kind="state_delta", payload={"set": {"stage": "main-only"}}),
    ], expected_seq=0)
    branch = await store.fork(scope, at_seq=1, new_thread="alt")
    await store.append(branch, [Event(kind="state_delta", payload={"set": {"stage": "branch-only"}})],
                       expected_seq=1)
    assert (await store.get_state(scope))["data"]["stage"] == "main-only"
    assert (await store.get_state(branch))["data"]["stage"] == "branch-only"


async def test_replay_stream(store, scope):
    await store.append(scope, [Event(kind="message", payload={"i": i}) for i in range(3)],
                       expected_seq=0)
    seqs = [e.seq async for e in store.read_events(scope)]
    assert seqs == [1, 2, 3]


async def test_long_term_memory(store, scope):
    await store.remember(scope, "user prefers window seats", meta={"tag": "pref"})
    await store.remember(scope, "user is vegetarian")
    hits = await store.recall(scope, "seats window", k=1)
    assert hits and "window" in hits[0]["text"]

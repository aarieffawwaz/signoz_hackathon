"""Tamper-evident hash chain + runtime invariants."""
from __future__ import annotations

import pytest

from statefold import (
    Event,
    GENESIS,
    InMemoryStore,
    InvariantViolation,
    Scope,
    clear_invariants,
    invariant,
    verify_chain,
)


@pytest.fixture(autouse=True)
def no_invariants():
    clear_invariants()
    yield
    clear_invariants()


@pytest.fixture
def scope():
    return Scope("t", "a", "s")


# --- hash chain ---------------------------------------------------------------

async def test_events_are_hash_chained(scope):
    store = InMemoryStore()
    await store.append(scope, [
        Event(kind="message", payload={"content": "one"}),
        Event(kind="message", payload={"content": "two"}),
    ], expected_seq=0)

    evs = [e async for e in store.read_events(scope)]
    assert all(len(e.hash) == 64 for e in evs)
    assert evs[0].hash != evs[1].hash
    assert (await verify_chain(store, scope)) == {"ok": True, "broken_at": None, "events": 2}


async def test_chain_continues_across_appends(scope):
    store = InMemoryStore()
    await store.append(scope, [Event(kind="message", payload={"n": 1})], expected_seq=0)
    await store.append(scope, [Event(kind="message", payload={"n": 2})], expected_seq=1)
    assert (await verify_chain(store, scope))["ok"] is True


async def test_tampering_breaks_the_chain(scope):
    store = InMemoryStore()
    await store.append(scope, [
        Event(kind="llm_call", payload={"model": "m", "input_tokens": 100, "output_tokens": 5}),
        Event(kind="message", payload={"content": "hi"}),
        Event(kind="state_delta", payload={"set": {"x": 1}}),
    ], expected_seq=0)

    # an attacker rewrites history: shrink the recorded token count
    store._events[scope.flatten()][0].payload["input_tokens"] = 1

    result = await verify_chain(store, scope)
    assert result["ok"] is False
    assert result["broken_at"] == 1


async def test_reordering_breaks_the_chain(scope):
    store = InMemoryStore()
    await store.append(scope, [
        Event(kind="message", payload={"n": 1}),
        Event(kind="message", payload={"n": 2}),
    ], expected_seq=0)
    evs = store._events[scope.flatten()]
    evs[0].payload, evs[1].payload = evs[1].payload, evs[0].payload
    assert (await verify_chain(store, scope))["ok"] is False


async def test_fork_chains_stay_valid(scope):
    store = InMemoryStore()
    await store.append(scope, [
        Event(kind="state_delta", payload={"set": {"n": "a"}}),
        Event(kind="state_delta", payload={"set": {"n": "b"}}),
    ], expected_seq=0)
    child = await store.fork(scope, at_seq=1, new_thread="alt")
    await store.append(child, [Event(kind="state_delta", payload={"set": {"n": "c"}})],
                       expected_seq=1)

    assert (await verify_chain(store, scope))["ok"] is True
    assert (await verify_chain(store, child))["ok"] is True


async def test_empty_stream_verifies(scope):
    store = InMemoryStore()
    assert (await verify_chain(store, scope)) == {"ok": True, "broken_at": None, "events": 0}


# --- runtime invariants ---------------------------------------------------------

async def test_invariant_rejects_invalid_write(scope):
    store = InMemoryStore()

    @invariant("state_delta")
    def stage_forward_only(state, event):
        order = ["searching", "booked"]
        old = state.get("data", {}).get("stage")
        new = event.payload.get("set", {}).get("stage")
        if old in order and new in order and order.index(new) < order.index(old):
            raise InvariantViolation(f"stage cannot move {old} -> {new}")

    await store.append(scope, [Event(kind="state_delta", payload={"set": {"stage": "booked"}})],
                       expected_seq=0)
    with pytest.raises(InvariantViolation, match="booked -> searching"):
        await store.append(scope,
                           [Event(kind="state_delta", payload={"set": {"stage": "searching"}})],
                           expected_seq=1)
    # nothing persisted
    assert await store.head(scope) == 1
    assert (await store.get_state(scope))["data"]["stage"] == "booked"


async def test_rejected_batch_is_all_or_nothing(scope):
    store = InMemoryStore()

    @invariant("llm_call")
    def budget_cap(state, event):
        spent = state.get("usage", {}).get("input_tokens", 0)
        if spent + event.payload.get("input_tokens", 0) > 1000:
            raise InvariantViolation("token budget exceeded")

    with pytest.raises(InvariantViolation):
        await store.append(scope, [
            Event(kind="llm_call", payload={"model": "m", "input_tokens": 600, "output_tokens": 0}),
            Event(kind="llm_call", payload={"model": "m", "input_tokens": 600, "output_tokens": 0}),
        ], expected_seq=0)  # second event breaches the cap mid-batch

    assert await store.head(scope) == 0          # first event NOT persisted either
    assert (await verify_chain(store, scope))["events"] == 0


async def test_wildcard_invariant_runs_for_all_kinds(scope):
    store = InMemoryStore()
    seen = []

    @invariant("*")
    def spy(state, event):
        seen.append(event.kind)

    await store.append(scope, [
        Event(kind="message", payload={}),
        Event(kind="tool_call", payload={"name": "t"}),
    ], expected_seq=0)
    assert seen == ["message", "tool_call"]


async def test_no_invariants_means_no_validation_overhead(scope):
    store = InMemoryStore()
    # nothing registered -> append works and never folds state for validation
    await store.append(scope, [Event(kind="message", payload={})], expected_seq=0)
    assert await store.head(scope) == 1

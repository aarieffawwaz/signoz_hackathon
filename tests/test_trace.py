"""Tracing: spans as events — nesting, causation links, errors, time-travel."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore, verify_chain
from statefold.adapters.generic import SessionState


@pytest.fixture
def s():
    return SessionState(InMemoryStore(), tenant="t", agent="a", session="s1")


async def test_span_records_start_end_and_duration(s):
    async with s.span("plan", city="paris"):
        await s.add_message("user", "hi")

    spans = await s.trace()
    assert len(spans) == 1
    sp = next(iter(spans.values()))
    assert sp["name"] == "plan"
    assert sp["status"] == "ok"
    assert sp["duration_ms"] >= 0
    assert sp["attrs"] == {"city": "paris"}
    assert sp["parent_id"] is None


async def test_spans_nest_automatically(s):
    async with s.span("outer") as outer_id:
        async with s.span("inner") as inner_id:
            pass

    spans = await s.trace()
    assert spans[inner_id]["parent_id"] == outer_id
    assert spans[outer_id]["parent_id"] is None


async def test_events_inside_span_are_linked_by_causation(s):
    async with s.span("work") as sid:
        await s.add_llm_call("m", 100, 10)
        await s.add_tool_call("search", {"q": 1}, result="ok")
    await s.add_message("user", "outside")

    events = [e async for e in s.store.read_events(s.scope)]
    by_kind = {e.kind: e for e in events}
    assert by_kind["llm_call"].causation_id == sid
    assert by_kind["tool_call"].causation_id == sid
    assert by_kind["message"].causation_id is None      # outside the span


async def test_exception_marks_span_error_and_propagates(s):
    with pytest.raises(ValueError, match="boom"):
        async with s.span("risky"):
            raise ValueError("boom")

    sp = next(iter((await s.trace()).values()))
    assert sp["status"] == "error"
    assert "boom" in sp["error"]


async def test_spans_are_time_travelable_and_chained(s):
    async with s.span("phase-1"):
        await s.set(stage="one")
    async with s.span("phase-2"):
        await s.set(stage="two")

    # as of seq 3 (start1, delta, end1): only phase-1 exists, completed
    early = await s.trace(as_of=3)
    assert [sp["name"] for sp in early.values()] == ["phase-1"]
    assert list(early.values())[0]["status"] == "ok"

    # mid-span time-travel: phase-2 shows as running at seq 4
    mid = await s.trace(as_of=4)
    assert mid[[k for k, v in mid.items() if v["name"] == "phase-2"][0]]["status"] == "running"

    # span events are part of the tamper-evidence chain like everything else
    assert (await verify_chain(s.store, s.scope))["ok"] is True

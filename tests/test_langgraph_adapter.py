"""Prove a real LangGraph graph survives a crash and resumes from the store."""
from __future__ import annotations

from typing import TypedDict

import pytest

from statefold import InMemoryStore

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from statefold.adapters.langgraph import PlatformCheckpointer  # noqa: E402


class S(TypedDict):
    steps: list


def n1(s):
    return {"steps": s["steps"] + ["n1"]}


def n2(s):
    return {"steps": s["steps"] + ["n2"]}


def _builder():
    b = StateGraph(S)
    b.add_node("n1", n1)
    b.add_node("n2", n2)
    b.add_edge(START, "n1")
    b.add_edge("n1", "n2")
    b.add_edge("n2", END)
    return b


async def test_crash_and_resume_across_saver_instances():
    store = InMemoryStore()
    cfg = {"configurable": {"thread_id": "t1"}}
    builder = _builder()

    # Run until it pauses before n2, then the process "dies".
    g1 = builder.compile(checkpointer=PlatformCheckpointer(store, "acme", "g"),
                         interrupt_before=["n2"])
    await g1.ainvoke({"steps": []}, cfg)

    # Brand-new saver + graph on the SAME store — nothing kept in memory.
    g2 = builder.compile(checkpointer=PlatformCheckpointer(store, "acme", "g"),
                         interrupt_before=["n2"])
    snap = await g2.aget_state(cfg)
    assert snap.values["steps"] == ["n1"]        # state rebuilt from the event log

    result = await g2.ainvoke(None, cfg)         # resume to completion
    assert result["steps"] == ["n1", "n2"]


async def test_history_is_replayable():
    store = InMemoryStore()
    cfg = {"configurable": {"thread_id": "t2"}}
    g = _builder().compile(checkpointer=PlatformCheckpointer(store, "acme", "g"))
    await g.ainvoke({"steps": []}, cfg)

    history = [snap.values.get("steps", []) async for snap in g.aget_state_history(cfg)]
    # newest-first: final state present, and the empty initial state at the tail
    assert ["n1", "n2"] in history
    assert history[-1] == []

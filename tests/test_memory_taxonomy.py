"""Memory taxonomy: kinds (semantic/episodic/procedural) x levels (session/agent)."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore, Scope
from statefold.adapters.generic import SessionState


@pytest.fixture
def store():
    return InMemoryStore()


async def test_session_level_memory_does_not_leak_across_sessions(store):
    s1 = Scope("t", "a", "sess-1")
    s2 = Scope("t", "a", "sess-2")
    await store.remember(s1, "user wants window seat", level="session")

    assert await store.recall(s1, "window seat") != []
    assert await store.recall(s2, "window seat") == []   # other session: invisible


async def test_agent_level_memory_is_long_term_across_sessions(store):
    s1 = Scope("t", "a", "sess-1")
    s2 = Scope("t", "a", "sess-2")
    await store.remember(s1, "user is vegetarian", level="agent")

    hits = await store.recall(s2, "vegetarian")          # different session, same agent
    assert hits and hits[0]["level"] == "agent"


async def test_agent_level_does_not_leak_across_agents(store):
    a = Scope("t", "agent-a", "s")
    b = Scope("t", "agent-b", "s")
    await store.remember(a, "secret preference", level="agent")
    assert await store.recall(b, "secret preference") == []


async def test_recall_searches_both_levels_by_default_and_filters(store):
    s = Scope("t", "a", "s1")
    await store.remember(s, "likes tea in meetings", level="session")
    await store.remember(s, "likes tea at home", level="agent")

    both = await store.recall(s, "likes tea", k=10)
    assert {h["level"] for h in both} == {"session", "agent"}

    only_agent = await store.recall(s, "likes tea", k=10, level="agent")
    assert [h["level"] for h in only_agent] == ["agent"]


async def test_kind_filter(store):
    s = Scope("t", "a", "s1")
    await store.remember(s, "paris hotels are pricey", kind="semantic")
    await store.remember(s, "booked lutetia successfully last time", kind="episodic")

    epi = await store.recall(s, "lutetia booked", kind="episodic")
    assert len(epi) == 1 and epi[0]["kind"] == "episodic"
    sem = await store.recall(s, "paris hotels", kind="semantic")
    assert len(sem) == 1 and sem[0]["kind"] == "semantic"


async def test_list_and_delete_respect_levels(store):
    s = Scope("t", "a", "s1")
    mid = await store.remember(s, "temp note", level="session")
    await store.remember(s, "permanent fact", level="agent")

    assert len(await store.list_memories(s)) == 2
    assert len(await store.list_memories(s, level="agent")) == 1
    assert await store.delete_memory(s, mid) is True
    assert [m["text"] for m in await store.list_memories(s)] == ["permanent fact"]


async def test_forget_by_level(store):
    s = Scope("t", "a", "s1")
    await store.remember(s, "session thing", level="session")
    await store.remember(s, "agent thing", level="agent")
    await store.forget(s, level="session")
    assert [m["level"] for m in await store.list_memories(s)] == ["agent"]


# --- SessionState cognitive-memory helpers -----------------------------------

async def test_working_memory_is_the_folded_state(store):
    s = SessionState(store, "t", "a", "s1")
    await s.add_message("user", "hello")
    await s.set(stage="active")
    wm = await s.working_memory()
    assert wm["messages"][0]["content"] == "hello"
    assert wm["data"]["stage"] == "active"


async def test_episode_lifecycle_across_sessions(store):
    # session 1 does work and records an episode
    run1 = SessionState(store, "acme", "support", "ticket-1")
    await run1.add_message("user", "invoice is wrong")
    await run1.set(stage="resolved")
    await run1.end_episode("resolved duplicate invoice by voiding the copy",
                           outcome="success", tool="crm_lookup")

    # a NEW session later recalls the relevant past experience
    run2 = SessionState(store, "acme", "support", "ticket-2")
    episodes = await run2.recall_episodes("duplicate invoice problem")
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["kind"] == "episodic" and ep["level"] == "agent"
    assert ep["meta"]["outcome"] == "success"
    assert ep["meta"]["stream"] == "acme/support/ticket-1/main"  # provenance


async def test_procedural_memory(store):
    s1 = SessionState(store, "acme", "coder", "run-1")
    await s1.learn_procedure("to fix flaky tests, isolate shared fixtures first")

    s2 = SessionState(store, "acme", "coder", "run-2")
    hits = await s2.recall("flaky tests fix", kind="procedural")
    assert hits and hits[0]["kind"] == "procedural" and hits[0]["level"] == "agent"

"""Export recorded sessions as promptfoo eval suites."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore, register_pricing
from statefold.adapters.generic import SessionState
from statefold.promptfoo import to_promptfoo
from statefold.telemetry import clear_pricing


@pytest.fixture(autouse=True)
def pricing():
    clear_pricing()
    register_pricing("m", 2.0, 10.0)
    yield
    clear_pricing()


@pytest.fixture
def store():
    return InMemoryStore()


async def _seed(store) -> SessionState:
    s = SessionState(store, "acme", "bot", "s1")
    await s.add_message("user", "what is the capital of france?")
    await s.add_llm_call("m", 1000, 200, latency_ms=800)
    await s.add_message("assistant", "The capital of France is Paris.")
    await s.add_message("user", "and germany?")
    await s.add_llm_call("m", 500, 100, latency_ms=400)
    await s.add_message("assistant", "Berlin is the capital of Germany.")
    return s


async def test_exchanges_become_tests_with_similarity_asserts(store):
    s = await _seed(store)
    cfg = await to_promptfoo(store, s.scope)

    assert cfg["prompts"] == ["{{prompt}}"]
    assert len(cfg["tests"]) == 2
    t0 = cfg["tests"][0]
    assert t0["vars"]["prompt"] == "what is the capital of france?"
    sim = t0["assert"][0]
    assert sim["type"] == "similar"
    assert "Paris" in sim["value"]
    assert sim["threshold"] == 0.75


async def test_recorded_latency_and_cost_become_thresholds(store):
    s = await _seed(store)
    cfg = await to_promptfoo(store, s.scope)

    kinds = {a["type"]: a for a in cfg["tests"][0]["assert"]}
    assert kinds["latency"]["threshold"] == 1600            # 800ms x 2.0
    # 1000 in @ $2/M + 200 out @ $10/M = $0.004; x1.5 headroom
    assert kinds["cost"]["threshold"] == pytest.approx(0.006)


async def test_provenance_metadata(store):
    s = await _seed(store)
    cfg = await to_promptfoo(store, s.scope)
    md = cfg["tests"][1]["metadata"]
    assert md["source"] == "statefold"
    assert md["stream"] == "acme/bot/s1/main"
    assert md["seq"] == 6


async def test_unpriced_model_omits_cost_assert(store):
    clear_pricing()
    s = SessionState(store, "acme", "bot", "s2")
    await s.add_message("user", "hi")
    await s.add_llm_call("mystery", 100, 10, latency_ms=100)
    await s.add_message("assistant", "hello!")

    cfg = await to_promptfoo(store, s.scope)
    types = [a["type"] for a in cfg["tests"][0]["assert"]]
    assert "cost" not in types                # never invent a $0 threshold
    assert "latency" in types


async def test_empty_or_uncaptured_stream_exports_no_tests(store):
    s = SessionState(store, "acme", "bot", "s3")
    await s.set(stage="working")              # no messages captured
    cfg = await to_promptfoo(store, s.scope)
    assert cfg["tests"] == []


async def test_custom_factors(store):
    s = await _seed(store)
    cfg = await to_promptfoo(store, s.scope, similar_threshold=0.9,
                             latency_factor=3.0)
    t0 = cfg["tests"][0]
    assert t0["assert"][0]["threshold"] == 0.9
    kinds = {a["type"]: a for a in t0["assert"]}
    assert kinds["latency"]["threshold"] == 2400

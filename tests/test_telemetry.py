"""Telemetry/cost as a fold: accumulation, pricing, time-travel, tool latency."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore, Scope, register_pricing, usage_report
from statefold.adapters.generic import SessionState
from statefold.mcp_proxy import ProxyRecorder
from statefold.telemetry import clear_pricing


@pytest.fixture(autouse=True)
def fresh_pricing():
    clear_pricing()
    yield
    clear_pricing()


def _session() -> SessionState:
    return SessionState(InMemoryStore(), tenant="acme", agent="bot", session="s1")


async def test_cost_accumulates_from_pricing_table():
    register_pricing("model-x", input_per_mtok=1.0, output_per_mtok=10.0)
    s = _session()
    await s.add_llm_call("model-x", input_tokens=1_000_000, output_tokens=100_000)
    await s.add_llm_call("model-x", input_tokens=500_000, output_tokens=0)

    u = await s.usage()
    assert u["llm_calls"] == 2
    assert u["input_tokens"] == 1_500_000
    assert u["cost_usd"] == pytest.approx(1.0 + 1.0 + 0.5)
    assert u["uncosted_calls"] == 0
    assert u["by_model"]["model-x"]["calls"] == 2


async def test_explicit_cost_overrides_table():
    register_pricing("model-x", 1.0, 10.0)
    s = _session()
    await s.add_llm_call("model-x", 1_000_000, 0, cost_usd=99.0)
    assert (await s.usage())["cost_usd"] == pytest.approx(99.0)


async def test_unknown_model_is_counted_not_silently_zero():
    s = _session()
    await s.add_llm_call("mystery-model", 1000, 1000)
    u = await s.usage()
    assert u["cost_usd"] == 0.0
    assert u["uncosted_calls"] == 1   # visible, not hidden


async def test_pricing_registered_later_backfills_on_next_read():
    s = _session()
    await s.add_llm_call("model-x", 1_000_000, 0)
    assert (await s.usage())["uncosted_calls"] == 1

    register_pricing("model-x", 2.0, 0.0)   # price known now; log unchanged
    u = await s.usage()
    assert u["cost_usd"] == pytest.approx(2.0)
    assert u["uncosted_calls"] == 0


async def test_time_travel_over_spend():
    register_pricing("model-x", 1.0, 0.0)
    s = _session()
    await s.add_llm_call("model-x", 1_000_000, 0)   # seq 1: $1
    await s.add_llm_call("model-x", 3_000_000, 0)   # seq 2: +$3

    assert (await s.usage(as_of=1))["cost_usd"] == pytest.approx(1.0)
    assert (await s.usage())["cost_usd"] == pytest.approx(4.0)


async def test_tool_latency_and_errors_aggregate():
    s = _session()
    await s.add_tool_call("search", {"q": "x"}, result="ok", latency_ms=120.0)
    await s.add_tool_call("search", {"q": "y"}, result=None, latency_ms=80.0,
                          error="Timeout")
    u = await s.usage()
    t = u["tools"]["search"]
    assert t["calls"] == 2 and t["errors"] == 1
    assert t["latency_ms_total"] == pytest.approx(200.0)


async def test_proxy_records_latency_automatically():
    store = InMemoryStore()
    rec = ProxyRecorder(store, "acme", "bot")

    async def downstream(tool, args):
        return {"ok": True}

    await rec.call("fetch", {}, downstream)
    u = await usage_report(store, Scope("acme", "bot", "mcp", "main"))
    assert u["tools"]["fetch"]["calls"] == 1
    assert u["tools"]["fetch"]["latency_ms_total"] > 0


async def test_proxy_records_downstream_failures():
    store = InMemoryStore()
    rec = ProxyRecorder(store, "acme", "bot")

    async def bad(tool, args):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await rec.call("fetch", {}, bad)
    u = await usage_report(store, Scope("acme", "bot", "mcp", "main"))
    assert u["tools"]["fetch"]["errors"] == 1


async def test_per_session_isolation_of_cost():
    register_pricing("m", 1.0, 0.0)
    store = InMemoryStore()
    a = SessionState(store, "acme", "bot", "sess-a")
    b = SessionState(store, "acme", "bot", "sess-b")
    await a.add_llm_call("m", 1_000_000, 0)

    assert (await a.usage())["cost_usd"] == pytest.approx(1.0)
    assert (await b.usage())["llm_calls"] == 0

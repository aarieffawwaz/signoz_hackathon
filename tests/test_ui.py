"""UI API endpoints over a seeded in-memory store."""
from __future__ import annotations

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from statefold import InMemoryStore  # noqa: E402
from statefold.adapters.generic import SessionState  # noqa: E402
from statefold.telemetry import clear_pricing, register_pricing  # noqa: E402
from statefold.ui import build_app  # noqa: E402


@pytest.fixture
def client():
    clear_pricing()
    register_pricing("m", 1.0, 10.0)
    store = InMemoryStore()

    async def seed():
        s = SessionState(store, "acme", "bot", "s1")
        await s.add_message("user", "hi")
        await s.add_llm_call("m", 1_000_000, 100_000, latency_ms=500)
        await s.set(stage="done")
        await s.add_tool_call("lookup", {"q": 1}, result="ok", latency_ms=100)
        await s.add_tool_call("lookup", {"q": 2}, result=None, latency_ms=300,
                              error="boom")
        await s.remember("user likes tea")

    import asyncio

    asyncio.run(seed())
    with TestClient(build_app(store)) as c:
        yield c
    clear_pricing()


STREAM = "acme/bot/s1/main"


def test_streams_listed(client):
    streams = client.get("/api/streams").json()
    assert streams == [{"stream": STREAM, "head_seq": 5}]


def test_events_timeline(client):
    evs = client.get("/api/events", params={"stream": STREAM}).json()
    assert [e["kind"] for e in evs] == ["message", "llm_call", "state_delta",
                                        "tool_call", "tool_call"]
    assert [e["seq"] for e in evs] == [1, 2, 3, 4, 5]


def test_state_and_time_travel(client):
    full = client.get("/api/state", params={"stream": STREAM}).json()
    assert full["data"]["stage"] == "done"
    early = client.get("/api/state", params={"stream": STREAM, "as_of": 1}).json()
    assert "data" not in early and len(early["messages"]) == 1


def test_usage_endpoint(client):
    u = client.get("/api/usage", params={"stream": STREAM}).json()
    assert u["llm_calls"] == 1
    assert u["cost_usd"] == pytest.approx(1.0 + 1.0)  # 1M in @ $1 + 0.1M out @ $10

    u0 = client.get("/api/usage", params={"stream": STREAM, "as_of": 1}).json()
    assert u0["llm_calls"] == 0  # spend as of step 1: nothing yet


def test_memories_endpoint(client):
    mems = client.get("/api/memories", params={"stream": STREAM}).json()
    assert len(mems) == 1 and "tea" in mems[0]["text"]


def test_summary_endpoint(client):
    s = client.get("/api/summary").json()
    assert s["totals"]["streams"] == 1
    assert s["totals"]["events"] == 5
    assert s["totals"]["llm_calls"] == 1
    assert s["totals"]["cost_usd"] == pytest.approx(2.0)
    assert s["streams"][0]["stream"] == STREAM
    assert s["streams"][0]["errors"] == 1


def test_series_endpoint_cumulative_cost(client):
    series = client.get("/api/series", params={"stream": STREAM}).json()
    assert [p["seq"] for p in series] == [1, 2, 3, 4, 5]
    assert series[0]["cost_usd"] == 0.0             # before the llm_call
    assert series[1]["cost_usd"] == pytest.approx(2.0)   # after it
    assert series[4]["cost_usd"] == pytest.approx(2.0)   # tool calls cost nothing


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "statefold" in r.text and "time-travel" in r.text
    for nav in ("Console", "Costs", "Tools", "Memories"):
        assert nav in r.text


def test_costs_page_endpoint(client):
    c = client.get("/api/costs").json()
    assert c["totals"]["cost_usd"] == pytest.approx(2.0)
    assert c["by_model"]["m"]["calls"] == 1
    assert "acme/bot" in c["by_agent"]
    assert c["streams"][0]["stream"] == STREAM


def test_tools_page_endpoint(client):
    rows = client.get("/api/tools").json()
    assert len(rows) == 1
    r = rows[0]
    assert r["tool"] == "lookup"
    assert r["calls"] == 2 and r["errors"] == 1
    assert r["err_rate"] == pytest.approx(0.5)
    assert r["avg_ms"] == pytest.approx(200.0)


def test_memories_all_endpoint(client):
    mems = client.get("/api/memories_all").json()
    assert len(mems) == 1
    assert "tea" in mems[0]["text"]
    assert mems[0]["session"] == "acme/bot/s1"


def test_verify_endpoint(client):
    v = client.get("/api/verify", params={"stream": STREAM}).json()
    assert v["ok"] is True and v["events"] == 5


def test_tail_sse_backlog(client):
    import json

    with client.stream("GET", "/api/tail",
                       params={"stream": STREAM, "after": 0, "once": 1}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: ")]
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
    assert all(len(e["hash"]) == 64 for e in events)


def test_tail_after_offset(client):
    import json

    with client.stream("GET", "/api/tail",
                       params={"stream": STREAM, "after": 3, "once": 1}) as r:
        body = "".join(r.iter_text())
    seqs = [json.loads(l[6:])["seq"] for l in body.splitlines() if l.startswith("data: ")]
    assert seqs == [4, 5]

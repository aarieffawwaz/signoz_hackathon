"""Telemetry and cost tracking — a derived view over the event log.

No separate collector, no double-writes: an ``llm_call`` is just another
event, and usage/cost is a pure fold over the stream. Because it's a fold,
everything the core gives you applies to spend too:

- time-travel: ``get_state(scope, as_of=N)["usage"]`` = cost as of step N
- per-tenant / per-session isolation comes from scoping, not extra plumbing
- replayable audit: the events *are* the billing record

Pricing is deliberately not hardcoded (prices change). Register your own:

    from statefold.telemetry import register_pricing
    register_pricing("claude-sonnet-5", input_per_mtok=3.00, output_per_mtok=15.00)

or pass an explicit ``cost_usd`` per call and skip the table entirely.
"""
from __future__ import annotations

from .reducers import REDUCERS, register

# model -> (usd per 1M input tokens, usd per 1M output tokens)
PRICING: dict[str, tuple[float, float]] = {}


def register_pricing(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    PRICING[model] = (float(input_per_mtok), float(output_per_mtok))


def clear_pricing() -> None:
    PRICING.clear()


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _empty_usage() -> dict:
    return {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "uncosted_calls": 0,   # calls with no price info — never silently $0
        "by_model": {},
        "tools": {},
    }


@register("llm_call")
def _llm_call(state: dict, payload: dict) -> dict:
    """payload = {"model", "input_tokens", "output_tokens",
                  "latency_ms"?, "cost_usd"?}

    Cost resolution order: explicit ``cost_usd`` on the event, else the
    pricing table at fold time (so registering prices later back-fills cost
    on the next read — the log itself never needs rewriting).
    """
    u = state.setdefault("usage", _empty_usage())
    model = payload.get("model", "unknown")
    itok = int(payload.get("input_tokens", 0))
    otok = int(payload.get("output_tokens", 0))

    cost = payload.get("cost_usd")
    if cost is None:
        cost = _cost(model, itok, otok)

    u["llm_calls"] += 1
    u["input_tokens"] += itok
    u["output_tokens"] += otok
    if cost is not None:
        u["cost_usd"] = round(u["cost_usd"] + cost, 10)
    else:
        u["uncosted_calls"] += 1

    m = u["by_model"].setdefault(model, {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0.0, "latency_ms_total": 0.0,
    })
    m["calls"] += 1
    m["input_tokens"] += itok
    m["output_tokens"] += otok
    if cost is not None:
        m["cost_usd"] = round(m["cost_usd"] + cost, 10)
    if payload.get("latency_ms") is not None:
        m["latency_ms_total"] += float(payload["latency_ms"])
    return state


def track_tool(state: dict, payload: dict) -> dict:
    """Aggregate tool-call telemetry; composed into the ``tool_call`` reducer."""
    u = state.setdefault("usage", _empty_usage())
    t = u["tools"].setdefault(payload.get("name", "unknown"), {
        "calls": 0, "errors": 0, "latency_ms_total": 0.0,
    })
    t["calls"] += 1
    if payload.get("error"):
        t["errors"] += 1
    if payload.get("latency_ms") is not None:
        t["latency_ms_total"] += float(payload["latency_ms"])
    return state


# Extend the built-in tool_call reducer: keep its list-append behavior,
# then aggregate telemetry. register() overriding built-ins is the documented
# extension mechanism.
_base_tool_call = REDUCERS["tool_call"]


@register("tool_call")
def _tool_call_with_telemetry(state: dict, payload: dict) -> dict:
    state = _base_tool_call(state, payload)
    return track_tool(state, payload)


async def usage_report(store, scope, as_of: int | None = None) -> dict:
    """Usage/cost summary for one session — ``as_of`` gives spend-at-step-N."""
    state = await store.get_state(scope, as_of=as_of)
    return state.get("usage", _empty_usage())

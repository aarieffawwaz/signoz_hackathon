"""Export recorded sessions as promptfoo eval suites.

`promptfoo <https://github.com/promptfoo/promptfoo>`_ is an open-source LLM
eval harness (assertions, model matrices, CI gating). statefold sits on the
other side of the loop: it records what your agent *actually did*. This
module closes the loop — every recorded session becomes a regression suite:

- each captured user→assistant exchange becomes a test case whose ``similar``
  assertion is the answer that worked in production
- recorded latency and cost per exchange become ``latency`` / ``cost``
  assertion thresholds (with headroom factors), so a model/prompt change
  that answers "correctly" but 3x slower or pricier still fails CI

Usage::

    python -m statefold.promptfoo acme/support-bot/ticket-4821/main > promptfooconfig.json
    npx promptfoo eval -c promptfooconfig.json

Content capture must be on for exchanges to exist in the log
(``instrument(client, session, capture_content=True)`` or explicit
``add_message`` calls). Output is JSON — a format promptfoo accepts natively —
so this stays dependency-free.
"""
from __future__ import annotations

import json
import os
import sys

from .telemetry import PRICING
from .types import Scope


def _exchange_cost(llm_calls: list[dict]) -> float | None:
    total, any_cost = 0.0, False
    for p in llm_calls:
        if p.get("cost_usd") is not None:
            total += float(p["cost_usd"])
            any_cost = True
            continue
        price = PRICING.get(p.get("model", ""))
        if price is not None:
            total += (p.get("input_tokens", 0) * price[0]
                      + p.get("output_tokens", 0) * price[1]) / 1_000_000
            any_cost = True
    return total if any_cost else None


async def to_promptfoo(store, scope: Scope, description: str | None = None,
                       similar_threshold: float = 0.75,
                       latency_factor: float = 2.0,
                       cost_factor: float = 1.5) -> dict:
    """Build a promptfoo config dict from one recorded stream.

    Thresholds derive from the recorded run: latency assertions allow
    ``latency_factor`` x the recorded time, cost assertions ``cost_factor`` x
    the recorded spend.
    """
    tests = []
    pending_user: str | None = None
    llm_calls: list[dict] = []

    async for e in store.read_events(scope):
        if e.kind == "llm_call":
            llm_calls.append(e.payload)
        elif e.kind == "message":
            role = e.payload.get("role")
            content = e.payload.get("content")
            if role == "user":
                pending_user, llm_calls = content, []
            elif role == "assistant" and pending_user and content:
                asserts: list[dict] = [{
                    "type": "similar", "value": content,
                    "threshold": similar_threshold,
                }]
                total_ms = sum(p.get("latency_ms") or 0 for p in llm_calls)
                if total_ms:
                    asserts.append({"type": "latency",
                                    "threshold": int(total_ms * latency_factor)})
                cost = _exchange_cost(llm_calls)
                if cost:
                    asserts.append({"type": "cost",
                                    "threshold": round(cost * cost_factor, 6)})
                tests.append({
                    "vars": {"prompt": pending_user},
                    "assert": asserts,
                    "metadata": {"source": "statefold",
                                 "stream": scope.flatten(), "seq": e.seq},
                })
                pending_user = None

    return {
        "description": description or f"statefold replay: {scope.flatten()}",
        "prompts": ["{{prompt}}"],
        "tests": tests,
    }


def main() -> None:
    import asyncio

    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not argv:
        print("usage: python -m statefold.promptfoo <tenant/agent/session/thread>"
              " [> promptfooconfig.json]", file=sys.stderr)
        raise SystemExit(2)
    parts = argv[0].split("/")
    if len(parts) != 4:
        print(f"stream must be tenant/agent/session/thread, got {argv[0]!r}",
              file=sys.stderr)
        raise SystemExit(2)
    scope = Scope(*parts)

    async def run():
        dsn = os.getenv("STATEFOLD_DSN")
        if not dsn:
            print("STATEFOLD_DSN not set (in-memory stores are per-process; "
                  "exporting needs a durable store)", file=sys.stderr)
            raise SystemExit(2)
        from .postgres import PostgresStore

        store = await PostgresStore.connect(dsn)
        cfg = await to_promptfoo(store, scope)
        json.dump(cfg, sys.stdout, indent=2)
        print()

    asyncio.run(run())


if __name__ == "__main__":
    main()

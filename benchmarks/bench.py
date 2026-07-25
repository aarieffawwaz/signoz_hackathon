"""statefold micro-benchmark — honest, reproducible numbers.

    python benchmarks/bench.py                      # in-memory store
    STATEFOLD_DSN=postgresql://... python benchmarks/bench.py   # + Postgres

Measures:
- sequential appends (1 event per append, the worst case: head read + append)
- batched appends (100 events per append)
- fold throughput (get_state over a long stream, no snapshot)
- chain verification throughput

Numbers depend entirely on the machine (and for Postgres, the server), so we
print them rather than hardcoding claims anywhere.
"""
from __future__ import annotations

import asyncio
import os
import time

from statefold import Event, InMemoryStore, Scope, verify_chain

N_SEQ = 2_000
N_BATCH = 10_000
BATCH = 100


def rate(n: int, secs: float) -> str:
    return f"{n / secs:,.0f}/s" if secs > 0 else "inf"


async def bench(store, label: str) -> None:
    print(f"\n== {label} ==")
    scope = Scope("bench", "agent", f"run-{int(time.time())}")

    # sequential single-event appends (durable worst case)
    t0 = time.perf_counter()
    head = 0
    for i in range(N_SEQ):
        head = await store.append(
            scope, [Event(kind="message", payload={"i": i})], expected_seq=head)
    dt = time.perf_counter() - t0
    print(f"sequential appends   {N_SEQ:>7,} events  {dt:6.2f}s  {rate(N_SEQ, dt)}")

    # batched appends
    scope2 = Scope("bench", "agent", f"batch-{int(time.time())}")
    t0 = time.perf_counter()
    head = 0
    for i in range(N_BATCH // BATCH):
        evs = [Event(kind="message", payload={"i": i, "j": j}) for j in range(BATCH)]
        head = await store.append(scope2, evs, expected_seq=head)
    dt = time.perf_counter() - t0
    print(f"batched appends x{BATCH} {N_BATCH:>7,} events  {dt:6.2f}s  {rate(N_BATCH, dt)}")

    # fold (no snapshot) over the batch stream
    t0 = time.perf_counter()
    state = await store.get_state(scope2)
    dt = time.perf_counter() - t0
    assert len(state["messages"]) == N_BATCH
    print(f"fold (get_state)     {N_BATCH:>7,} events  {dt:6.2f}s  {rate(N_BATCH, dt)}")

    # tamper-evidence verification
    t0 = time.perf_counter()
    result = await verify_chain(store, scope2)
    dt = time.perf_counter() - t0
    assert result["ok"]
    print(f"verify_chain         {N_BATCH:>7,} events  {dt:6.2f}s  {rate(N_BATCH, dt)}")


async def main() -> None:
    await bench(InMemoryStore(), "InMemoryStore")

    dsn = os.getenv("STATEFOLD_DSN")
    if dsn:
        from statefold.postgres import PostgresStore

        await bench(await PostgresStore.connect(dsn), f"PostgresStore ({dsn.split('@')[-1]})")
    else:
        print("\n(no STATEFOLD_DSN set — skipping Postgres; run with a DSN for durable numbers)")


if __name__ == "__main__":
    asyncio.run(main())

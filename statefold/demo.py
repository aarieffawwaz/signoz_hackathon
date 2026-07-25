"""The killer demo: crash -> resume, time-travel, and fork — all runnable
with zero external services.

    python -m statefold.demo   (or: statefold-demo)
"""
from __future__ import annotations

import asyncio

from .memory import InMemoryStore
from .store import ConcurrencyError
from .types import Event, Scope


async def main_async() -> None:
    store = InMemoryStore()
    scope = Scope(tenant="acme", agent="researcher", session="sess-1")

    print("== a normal run ==")
    seq = await store.append(scope, [
        Event(kind="message", payload={"role": "user", "content": "find me a hotel"}),
        Event(kind="tool_call", payload={"name": "search", "args": {"q": "hotel"}, "result": "3 hits"}),
        Event(kind="state_delta", payload={"set": {"stage": "searching"}}),
    ], expected_seq=0)
    print("head after run:", seq)
    print("state:", await store.get_state(scope))

    print("\n== crash + resume ==")
    await store.checkpoint(scope, label="before-booking")
    # ...process dies here... a new worker picks up the same session:
    resumed = await store.get_state(scope)
    print("resumed state (rebuilt from log):", resumed)

    print("\n== optimistic concurrency ==")
    try:
        await store.append(scope, [Event(kind="message", payload={"role": "a", "content": "x"})],
                           expected_seq=0)  # stale head on purpose
    except ConcurrencyError as e:
        print("rejected stale write:", e)
    head = await store.head(scope)
    await store.append(scope, [Event(kind="state_delta", payload={"set": {"stage": "booked"}})],
                       expected_seq=head)
    print("state now:", (await store.get_state(scope))["data"])

    print("\n== time-travel ==")
    print("state as of seq 3:", (await store.get_state(scope, as_of=3)).get("data"))
    print("state at head:    ", (await store.get_state(scope)).get("data"))

    print("\n== fork (what-if branch) ==")
    branch = await store.fork(scope, at_seq=3, new_thread="what-if")
    await store.append(branch, [Event(kind="state_delta", payload={"set": {"stage": "cancelled"}})],
                       expected_seq=3)
    print("main branch stage:  ", (await store.get_state(scope))["data"]["stage"])
    print("what-if branch stage:", (await store.get_state(branch))["data"]["stage"])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

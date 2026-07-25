"""Shared sync -> async bridge.

Several integrations (CrewAI storage, Agno Db, SDK instrumentation of sync
clients) are called from synchronous code but must drive the async StateStore.
Each bridge owns one persistent background event loop so the store's async
primitives stay bound to a single loop.
"""
from __future__ import annotations

import asyncio
import threading


class SyncBridge:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

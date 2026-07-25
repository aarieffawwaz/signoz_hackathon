"""The StateStore contract + shared errors.

Every backend (in-memory, Postgres, ...) implements this Protocol. Frameworks
and the MCP server talk to *this*, never to a concrete store.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from .types import Event, Scope


class ConcurrencyError(Exception):
    """Raised when ``append``'s ``expected_seq`` doesn't match the stream head.

    The caller should re-read the head and retry — this is optimistic
    concurrency, not a fatal error.
    """

    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(f"expected head seq {expected}, but stream is at {actual}")


@runtime_checkable
class StateStore(Protocol):
    async def head(self, scope: Scope) -> int:
        """Current sequence number of the stream (0 if empty)."""

    async def append(self, scope: Scope, events: list[Event], expected_seq: int) -> int:
        """Append events iff the stream head equals ``expected_seq``.

        Returns the new head seq. Raises ``ConcurrencyError`` on mismatch.
        """

    async def get_state(self, scope: Scope, as_of: int | None = None) -> dict:
        """Folded state. ``as_of`` bounds the fold for time-travel."""

    def read_events(self, scope: Scope, after: int = 0) -> AsyncIterator[Event]:
        """Stream events with ``seq > after`` in order (for replay/tailing)."""

    async def checkpoint(self, scope: Scope, label: str | None = None) -> str:
        """Persist a snapshot at the current head. Returns a checkpoint ref."""

    async def fork(self, scope: Scope, at_seq: int, new_thread: str) -> Scope:
        """Branch the stream at ``at_seq`` into ``new_thread``. Returns child scope."""

    async def remember(self, scope: Scope, text: str, meta: dict | None = None,
                       embedding: list[float] | None = None, mem_id: str | None = None,
                       kind: str = "semantic", level: str = "session") -> str:
        """Write (or upsert, if ``mem_id`` is reused) a memory.

        ``kind`` classifies it (semantic / episodic / procedural / any string);
        ``level`` scopes it: "session" dies with the session, "agent" is
        long-term and shared across every session of this tenant/agent.
        """

    async def recall(self, scope: Scope, query: str, k: int = 5,
                     kind: str | None = None, level: str | None = None) -> list[dict]:
        """Retrieve up to ``k`` relevant memories for a text query.

        Searches BOTH levels by default (session working set + agent
        long-term); filter with ``kind`` and/or ``level``.
        """

    async def recall_vec(self, scope: Scope, vector: list[float], k: int = 5,
                         kind: str | None = None, level: str | None = None) -> list[dict]:
        """Retrieve memories ranked against a pre-computed query embedding."""

    async def list_memories(self, scope: Scope, limit: int = 200, offset: int = 0,
                            kind: str | None = None, level: str | None = None) -> list[dict]:
        """List memories for a scope (no ranking)."""

    async def delete_memory(self, scope: Scope, mem_id: str,
                            level: str | None = None) -> bool:
        """Delete one memory by id. Returns True if it existed."""

    async def forget(self, scope: Scope, level: str | None = None) -> None:
        """Drop memories for this scope (both levels by default)."""

"""Core value types: identity (Scope) and the append-only unit (Event)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


def new_id() -> str:
    """Lexicographically-sortable unique id (millis prefix + random).

    Not a strict ULID, but sorts by creation time and is collision-safe,
    which is all the core relies on — ordering within a stream comes from
    ``seq``, not from this id.
    """
    return f"{int(time.time() * 1000):013d}-{os.urandom(8).hex()}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Scope:
    """Addresses one piece of state.

    A ``stream`` is the unit of concurrency and ordering: every event in a
    stream has a gap-free ``seq``. ``thread`` lets a session branch (see
    ``StateStore.fork``) so replays and what-if runs don't collide.
    """

    tenant: str
    agent: str
    session: str
    thread: str = "main"

    def flatten(self) -> str:
        return f"{self.tenant}/{self.agent}/{self.session}/{self.thread}"

    def memory_key(self) -> str:
        """Long-term memory is shared across threads of a session."""
        return f"{self.tenant}/{self.agent}/{self.session}"


@dataclass
class Event:
    """The append-only unit. Stored, never mutated (except ``seq`` on append)."""

    kind: str
    payload: dict
    actor: str = "agent"
    causation_id: str | None = None
    id: str = field(default_factory=new_id)
    ts: str = field(default_factory=now_iso)
    seq: int = 0    # assigned by the store on append
    hash: str = ""  # tamper-evidence chain, assigned by the store on append

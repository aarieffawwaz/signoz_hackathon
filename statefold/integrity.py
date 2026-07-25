"""Tamper-evident event chain.

Every appended event carries ``hash = sha256(prev_hash || canonical(event))``,
so the log is a hash chain: editing, reordering, or deleting any historical
event breaks every hash after it. ``verify_chain`` walks a stream and
recomputes the chain from genesis.

The event ``id`` is deliberately excluded from the canonical form — it is a
uniqueness artifact, and forks copy history under fresh ids; content, order,
actor, and time are what the chain protects.

sha256 (stdlib) rather than blake3: identical tamper-evidence, zero deps.
"""
from __future__ import annotations

import hashlib
import json

GENESIS = "0" * 64


def canonical(event) -> bytes:
    return json.dumps(
        {"seq": event.seq, "kind": event.kind, "payload": event.payload,
         "actor": event.actor, "ts": event.ts},
        sort_keys=True, default=str, separators=(",", ":"),
    ).encode()


def compute_hash(prev_hash: str, event) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(canonical(event))
    return h.hexdigest()


async def verify_chain(store, scope) -> dict:
    """Recompute the chain for one stream.

    Returns ``{"ok": bool, "broken_at": seq | None, "events": n}``.
    """
    prev = GENESIS
    n = 0
    async for e in store.read_events(scope):
        n += 1
        if e.hash != compute_hash(prev, e):
            return {"ok": False, "broken_at": e.seq, "events": n}
        prev = e.hash
    return {"ok": True, "broken_at": None, "events": n}

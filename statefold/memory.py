"""In-memory StateStore — for tests, local dev, and the README example.

Same contract as the Postgres store, so anything that works here works there.
No external services required.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from typing import AsyncIterator

from .embedding import Embedder, cosine, embed
from .integrity import GENESIS, compute_hash
from .invariants import has_invariants, validate_batch
from .reducers import fold
from .store import ConcurrencyError
from .types import Event, Scope, new_id


def _similarity(query: str, text: str) -> float:
    """Cheap lexical overlap — a placeholder for a real vector search."""
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


class InMemoryStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._events: dict[str, list[Event]] = defaultdict(list)
        self._snaps: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        self._heads: dict[str, int] = defaultdict(int)
        self._head_hashes: dict[str, str] = {}
        self._mem: dict[str, list[dict]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self.embedder = embedder

    async def head(self, scope: Scope) -> int:
        return self._heads[scope.flatten()]

    async def list_streams(self) -> list[dict]:
        """All streams with their head seq (for inspection/UI)."""
        return [{"stream": s, "head_seq": h} for s, h in sorted(self._heads.items()) if h > 0]

    async def append(self, scope: Scope, events: list[Event], expected_seq: int) -> int:
        async with self._lock:
            s = scope.flatten()
            head = self._heads[s]
            if head != expected_seq:
                raise ConcurrencyError(expected_seq, head)

            # Phase 1: assign seqs, run invariants, compute the hash chain.
            # Nothing is persisted until the whole batch passes.
            seq = head
            prev_hash = self._head_hashes.get(s, GENESIS)
            for ev in events:
                seq += 1
                ev.seq = seq
            if has_invariants():
                validate_batch(await self.get_state(scope), events)
            for ev in events:
                ev.hash = compute_hash(prev_hash, ev)
                prev_hash = ev.hash

            # Phase 2: persist.
            self._events[s].extend(events)
            self._heads[s] = seq
            self._head_hashes[s] = prev_hash
            return seq

    def _base(self, stream: str, as_of: int | None) -> tuple[dict, int]:
        base_state, base_seq = {}, 0
        for upto, st in self._snaps[stream]:
            if (as_of is None or upto <= as_of) and upto >= base_seq:
                base_state, base_seq = st, upto
        return base_state, base_seq

    async def get_state(self, scope: Scope, as_of: int | None = None) -> dict:
        s = scope.flatten()
        base_state, base_seq = self._base(s, as_of)
        items = [
            (e.kind, e.payload)
            for e in self._events[s]
            if e.seq > base_seq and (as_of is None or e.seq <= as_of)
        ]
        return fold(items, base_state)

    async def read_events(self, scope: Scope, after: int = 0) -> AsyncIterator[Event]:
        for e in self._events[scope.flatten()]:
            if e.seq > after:
                yield e

    async def checkpoint(self, scope: Scope, label: str | None = None) -> str:
        s = scope.flatten()
        head = self._heads[s]
        state = await self.get_state(scope)
        self._snaps[s].append((head, state))
        return f"{s}@{head}"

    async def fork(self, scope: Scope, at_seq: int, new_thread: str) -> Scope:
        child = replace(scope, thread=new_thread)
        cs = child.flatten()
        state = await self.get_state(scope, as_of=at_seq)
        # Copy history up to the fork point so the child can replay independently.
        prefix = [e for e in self._events[scope.flatten()] if e.seq <= at_seq]
        self._events[cs] = prefix
        self._heads[cs] = at_seq
        self._head_hashes[cs] = prefix[-1].hash if prefix else GENESIS
        self._snaps[cs].append((at_seq, state))
        return child

    # --- memory taxonomy ---------------------------------------------------
    # kind:  "semantic" (facts/preferences) | "episodic" (past experiences)
    #        | "procedural" (how-tos) — any string is allowed.
    # level: "session" (dies with the session) | "agent" (long-term,
    #        shared across every session of this tenant/agent).

    @staticmethod
    def _mem_key(scope: Scope, level: str) -> str:
        return f"{scope.tenant}/{scope.agent}" if level == "agent" else scope.memory_key()

    def _records(self, scope: Scope, level: str | None, kind: str | None) -> list[dict]:
        levels = [level] if level else ["session", "agent"]
        out: list[dict] = []
        for lv in levels:
            out.extend(self._mem[self._mem_key(scope, lv)])
        if kind is not None:
            out = [r for r in out if r.get("kind", "semantic") == kind]
        return out

    @staticmethod
    def _public(r: dict, score: float | None = None) -> dict:
        d = {"id": r["id"], "text": r["text"], "meta": r["meta"],
             "kind": r.get("kind", "semantic"), "level": r.get("level", "session")}
        if score is not None:
            d["score"] = score
        return d

    async def remember(self, scope: Scope, text: str, meta: dict | None = None,
                       embedding: list[float] | None = None, mem_id: str | None = None,
                       kind: str = "semantic", level: str = "session") -> str:
        rec = {"id": mem_id or new_id(), "text": text, "meta": meta or {},
               "kind": kind, "level": level}
        if embedding is not None:
            rec["embedding"] = list(embedding)
        elif self.embedder is not None:
            rec["embedding"] = await embed(self.embedder, text)
        key = self._mem_key(scope, level)
        self._mem[key] = [r for r in self._mem[key] if r["id"] != rec["id"]] + [rec]
        return rec["id"]

    async def recall_vec(self, scope: Scope, vector: list[float], k: int = 5,
                         kind: str | None = None, level: str | None = None) -> list[dict]:
        """Rank memories against a pre-computed query embedding."""
        records = self._records(scope, level, kind)
        scored = [(cosine(vector, r.get("embedding", [])), r) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._public(r, s) for s, r in scored[:k] if s > 0]

    async def list_memories(self, scope: Scope, limit: int = 200, offset: int = 0,
                            kind: str | None = None, level: str | None = None) -> list[dict]:
        records = self._records(scope, level, kind)[offset:offset + limit]
        return [self._public(r) for r in records]

    async def delete_memory(self, scope: Scope, mem_id: str,
                            level: str | None = None) -> bool:
        deleted = False
        for lv in ([level] if level else ["session", "agent"]):
            key = self._mem_key(scope, lv)
            kept = [r for r in self._mem[key] if r["id"] != mem_id]
            if len(kept) != len(self._mem[key]):
                deleted = True
            self._mem[key] = kept
        return deleted

    async def recall(self, scope: Scope, query: str, k: int = 5,
                     kind: str | None = None, level: str | None = None) -> list[dict]:
        records = self._records(scope, level, kind)
        if self.embedder is not None:
            q = await embed(self.embedder, query)
            scored = [(cosine(q, r.get("embedding", [])), r) for r in records]
        else:
            scored = [(_similarity(query, r["text"]), r) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._public(r) for score, r in scored[:k] if score > 0]

    async def forget(self, scope: Scope, level: str | None = None) -> None:
        """Drop long-term memories for this scope (both levels by default)."""
        for lv in ([level] if level else ["session", "agent"]):
            self._mem.pop(self._mem_key(scope, lv), None)

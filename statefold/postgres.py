"""Postgres StateStore — the durable production backend.

Optional dependency: ``pip install statefold[postgres]``. The append-only
``events`` table with a ``UNIQUE (stream, seq)`` index makes optimistic
concurrency a database guarantee rather than an app-level convention.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from .embedding import Embedder, cosine, embed
from .integrity import GENESIS, compute_hash
from .invariants import has_invariants, validate_batch
from .reducers import fold
from .store import ConcurrencyError
from .types import Event, Scope, new_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    tenant        TEXT NOT NULL,
    stream        TEXT NOT NULL,
    seq           BIGINT NOT NULL,
    kind          TEXT NOT NULL,
    payload       JSONB NOT NULL,
    actor         TEXT NOT NULL,
    causation_id  TEXT,
    hash          TEXT,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stream, seq)
);
ALTER TABLE events ADD COLUMN IF NOT EXISTS hash TEXT;
CREATE INDEX IF NOT EXISTS events_stream_seq ON events (tenant, stream, seq);

CREATE TABLE IF NOT EXISTS snapshots (
    stream    TEXT NOT NULL,
    upto_seq  BIGINT NOT NULL,
    state     JSONB NOT NULL,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream, upto_seq)
);

CREATE TABLE IF NOT EXISTS streams (
    stream       TEXT PRIMARY KEY,
    tenant       TEXT NOT NULL,
    head_seq     BIGINT NOT NULL DEFAULT 0,
    head_hash    TEXT,
    forked_from  TEXT,
    fork_at      BIGINT
);
ALTER TABLE streams ADD COLUMN IF NOT EXISTS head_hash TEXT;

CREATE TABLE IF NOT EXISTS memories (
    id        TEXT PRIMARY KEY,
    mem_key   TEXT NOT NULL,
    text      TEXT NOT NULL,
    meta      JSONB NOT NULL DEFAULT '{}',
    embedding JSONB,
    kind      TEXT NOT NULL DEFAULT 'semantic',
    level     TEXT NOT NULL DEFAULT 'session',
    ts        TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'semantic';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS level TEXT NOT NULL DEFAULT 'session';
CREATE INDEX IF NOT EXISTS memories_key ON memories (mem_key);
"""

# Applied only when pgvector is available; recall then ranks in the database.
PGVECTOR_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_vec vector({dim});
CREATE INDEX IF NOT EXISTS memories_vec ON memories
    USING hnsw (embedding_vec vector_cosine_ops);
"""


class PostgresStore:
    def __init__(self, pool, embedder: Embedder | None = None,
                 embedding_dim: int | None = None) -> None:
        # pool: asyncpg.Pool
        self.pool = pool
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self._pgvector = False

    @classmethod
    async def connect(cls, dsn: str, embedder: Embedder | None = None,
                      embedding_dim: int | None = None) -> "PostgresStore":
        import asyncpg

        pool = await asyncpg.create_pool(dsn)
        self = cls(pool, embedder=embedder, embedding_dim=embedding_dim)
        await self.init_schema()
        return self

    async def init_schema(self) -> None:
        async with self.pool.acquire() as con:
            await con.execute(SCHEMA)
            if self.embedder is not None and self.embedding_dim:
                try:
                    await con.execute(PGVECTOR_SCHEMA.format(dim=self.embedding_dim))
                    self._pgvector = True
                except Exception:
                    # pgvector not installed on this server; recall falls back
                    # to Python-side cosine over the JSONB embeddings.
                    self._pgvector = False

    async def head(self, scope: Scope) -> int:
        async with self.pool.acquire() as con:
            row = await con.fetchval(
                "SELECT head_seq FROM streams WHERE stream=$1", scope.flatten()
            )
            return row or 0

    async def list_streams(self) -> list[dict]:
        """All streams with their head seq (for inspection/UI)."""
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                "SELECT stream, head_seq FROM streams WHERE head_seq > 0 ORDER BY stream"
            )
        return [{"stream": r["stream"], "head_seq": r["head_seq"]} for r in rows]

    async def append(self, scope: Scope, events: list[Event], expected_seq: int) -> int:
        stream = scope.flatten()
        async with self.pool.acquire() as con, con.transaction():
            row = await con.fetchrow(
                "SELECT head_seq, head_hash FROM streams WHERE stream=$1 FOR UPDATE", stream
            )
            head = row["head_seq"] if row else 0
            prev_hash = (row["head_hash"] if row else None) or GENESIS
            if head != expected_seq:
                raise ConcurrencyError(expected_seq, head)

            # Phase 1: seqs, invariants, hash chain — nothing persisted yet.
            # (We hold the stream head lock, so the state read is stable.)
            seq = head
            for ev in events:
                seq += 1
                ev.seq = seq
            if has_invariants():
                validate_batch(await self.get_state(scope), events)
            for ev in events:
                ev.hash = compute_hash(prev_hash, ev)
                prev_hash = ev.hash

            # Phase 2: persist.
            for ev in events:
                await con.execute(
                    "INSERT INTO events (id,tenant,stream,seq,kind,payload,actor,causation_id,hash)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
                    ev.id, scope.tenant, stream, ev.seq, ev.kind,
                    json.dumps(ev.payload), ev.actor, ev.causation_id, ev.hash,
                )
            await con.execute(
                "INSERT INTO streams (stream,tenant,head_seq,head_hash) VALUES ($1,$2,$3,$4)"
                " ON CONFLICT (stream) DO UPDATE SET head_seq=$3, head_hash=$4",
                stream, scope.tenant, seq, prev_hash,
            )
            return seq

    async def get_state(self, scope: Scope, as_of: int | None = None) -> dict:
        stream = scope.flatten()
        bound = as_of if as_of is not None else (1 << 62)
        async with self.pool.acquire() as con:
            snap = await con.fetchrow(
                "SELECT upto_seq, state FROM snapshots WHERE stream=$1 AND upto_seq <= $2"
                " ORDER BY upto_seq DESC LIMIT 1",
                stream, bound,
            )
            base_state = json.loads(snap["state"]) if snap else {}
            base_seq = snap["upto_seq"] if snap else 0
            rows = await con.fetch(
                "SELECT kind, payload FROM events WHERE stream=$1 AND seq > $2 AND seq <= $3"
                " ORDER BY seq",
                stream, base_seq, bound,
            )
        return fold(((r["kind"], json.loads(r["payload"])) for r in rows), base_state)

    async def read_events(self, scope: Scope, after: int = 0) -> AsyncIterator[Event]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                "SELECT id,seq,kind,payload,actor,causation_id,hash,ts::text FROM events"
                " WHERE stream=$1 AND seq > $2 ORDER BY seq",
                scope.flatten(), after,
            )
        for r in rows:
            yield Event(
                kind=r["kind"], payload=json.loads(r["payload"]), actor=r["actor"],
                causation_id=r["causation_id"], id=r["id"], ts=r["ts"], seq=r["seq"],
                hash=r["hash"] or "",
            )

    async def checkpoint(self, scope: Scope, label: str | None = None) -> str:
        stream = scope.flatten()
        state = await self.get_state(scope)
        head = await self.head(scope)
        async with self.pool.acquire() as con:
            await con.execute(
                "INSERT INTO snapshots (stream,upto_seq,state) VALUES ($1,$2,$3)"
                " ON CONFLICT DO NOTHING",
                stream, head, json.dumps(state),
            )
        return f"{stream}@{head}"

    async def fork(self, scope: Scope, at_seq: int, new_thread: str) -> Scope:
        from dataclasses import replace

        child = replace(scope, thread=new_thread)
        cstream = child.flatten()
        state = await self.get_state(scope, as_of=at_seq)
        async with self.pool.acquire() as con, con.transaction():
            # Copy history up to the fork point so the child replays independently.
            await con.execute(
                "INSERT INTO events (id,tenant,stream,seq,kind,payload,actor,causation_id,hash)"
                " SELECT id||':'||$4, tenant, $2, seq, kind, payload, actor, causation_id, hash"
                " FROM events WHERE stream=$1 AND seq <= $3",
                scope.flatten(), cstream, at_seq, new_thread,
            )
            fork_hash = await con.fetchval(
                "SELECT hash FROM events WHERE stream=$1 AND seq=$2",
                scope.flatten(), at_seq,
            )
            await con.execute(
                "INSERT INTO streams (stream,tenant,head_seq,head_hash,forked_from,fork_at)"
                " VALUES ($1,$2,$3,$4,$5,$6)",
                cstream, scope.tenant, at_seq, fork_hash or GENESIS,
                scope.flatten(), at_seq,
            )
            await con.execute(
                "INSERT INTO snapshots (stream,upto_seq,state) VALUES ($1,$2,$3)"
                " ON CONFLICT DO NOTHING",
                cstream, at_seq, json.dumps(state),
            )
        return child

    # --- memory taxonomy: kind (semantic/episodic/procedural...) x level
    # (session = scoped to this session; agent = long-term, cross-session).

    @staticmethod
    def _mem_key(scope: Scope, level: str) -> str:
        return f"{scope.tenant}/{scope.agent}" if level == "agent" else scope.memory_key()

    def _mem_keys(self, scope: Scope, level: str | None) -> list[str]:
        levels = [level] if level else ["session", "agent"]
        return [self._mem_key(scope, lv) for lv in levels]

    async def remember(self, scope: Scope, text: str, meta: dict | None = None,
                       embedding: list[float] | None = None, mem_id: str | None = None,
                       kind: str = "semantic", level: str = "session") -> str:
        mid = mem_id or new_id()
        if embedding is not None:
            vec = list(embedding)
        elif self.embedder is not None:
            vec = await embed(self.embedder, text)
        else:
            vec = None
        async with self.pool.acquire() as con:
            await con.execute(
                "INSERT INTO memories (id,mem_key,text,meta,embedding,kind,level)"
                " VALUES ($1,$2,$3,$4,$5,$6,$7)"
                " ON CONFLICT (id) DO UPDATE SET text=$3, meta=$4, embedding=$5,"
                " kind=$6, level=$7, mem_key=$2",
                mid, self._mem_key(scope, level), text, json.dumps(meta or {}),
                json.dumps(vec) if vec is not None else None, kind, level,
            )
            if vec is not None and self._pgvector:
                await con.execute(
                    "UPDATE memories SET embedding_vec = $2::vector WHERE id = $1",
                    mid, str(vec),
                )
        return mid

    @staticmethod
    def _row(r, score: float | None = None) -> dict:
        d = {"id": r["id"], "text": r["text"], "meta": json.loads(r["meta"]),
             "kind": r["kind"], "level": r["level"]}
        if score is not None:
            d["score"] = score
        return d

    async def recall_vec(self, scope: Scope, vector: list[float], k: int = 5,
                         kind: str | None = None, level: str | None = None) -> list[dict]:
        """Rank memories against a pre-computed query embedding."""
        keys = self._mem_keys(scope, level)
        async with self.pool.acquire() as con:
            if self._pgvector:
                rows = await con.fetch(
                    "SELECT id,text,meta,kind,level,"
                    " 1 - (embedding_vec <=> $2::vector) AS score"
                    " FROM memories WHERE mem_key = ANY($1)"
                    " AND ($4::text IS NULL OR kind=$4)"
                    " AND embedding_vec IS NOT NULL"
                    " ORDER BY embedding_vec <=> $2::vector LIMIT $3",
                    keys, str(list(vector)), k, kind,
                )
                return [self._row(r, r["score"]) for r in rows]
            rows = await con.fetch(
                "SELECT id,text,meta,kind,level,embedding FROM memories"
                " WHERE mem_key = ANY($1) AND ($2::text IS NULL OR kind=$2)"
                " AND embedding IS NOT NULL",
                keys, kind,
            )
        scored = [(cosine(list(vector), json.loads(r["embedding"])), r) for r in rows]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row(r, s) for s, r in scored[:k] if s > 0]

    async def list_memories(self, scope: Scope, limit: int = 200, offset: int = 0,
                            kind: str | None = None, level: str | None = None) -> list[dict]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                "SELECT id,text,meta,kind,level FROM memories"
                " WHERE mem_key = ANY($1) AND ($4::text IS NULL OR kind=$4)"
                " ORDER BY ts LIMIT $2 OFFSET $3",
                self._mem_keys(scope, level), limit, offset, kind,
            )
        return [self._row(r) for r in rows]

    async def delete_memory(self, scope: Scope, mem_id: str,
                            level: str | None = None) -> bool:
        async with self.pool.acquire() as con:
            tag = await con.execute(
                "DELETE FROM memories WHERE mem_key = ANY($1) AND id=$2",
                self._mem_keys(scope, level), mem_id,
            )
        return tag.endswith("1")

    async def recall(self, scope: Scope, query: str, k: int = 5,
                     kind: str | None = None, level: str | None = None) -> list[dict]:
        if self.embedder is None:
            # Lexical fallback when no embedder is configured.
            async with self.pool.acquire() as con:
                rows = await con.fetch(
                    "SELECT id,text,meta,kind,level FROM memories"
                    " WHERE mem_key = ANY($1) AND ($4::text IS NULL OR kind=$4)"
                    " AND text ILIKE '%'||$2||'%' ORDER BY ts DESC LIMIT $3",
                    self._mem_keys(scope, level), query, k, kind,
                )
            return [self._row(r) for r in rows]
        q = await embed(self.embedder, query)
        return await self.recall_vec(scope, q, k=k, kind=kind, level=level)

    async def forget(self, scope: Scope, level: str | None = None) -> None:
        """Drop memories for this scope (both levels by default)."""
        async with self.pool.acquire() as con:
            await con.execute("DELETE FROM memories WHERE mem_key = ANY($1)",
                              self._mem_keys(scope, level))

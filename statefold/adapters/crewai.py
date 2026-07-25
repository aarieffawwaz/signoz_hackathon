"""CrewAI adapter.

Two integrations, matching the two CrewAI generations:

- ``AgentStateBackend`` — CrewAI >= 1.x ``StorageBackend`` (embedding-first).
  This is the real, tested integration: plug it into CrewAI's unified memory
  and every MemoryRecord lives in the durable StateStore, surviving process
  restarts and shareable with non-CrewAI agents on the same store.

- ``AgentStateStorage`` — the legacy (0.x) ``ExternalMemory`` storage shape
  (``save`` / ``search`` / ``reset``), kept for older installs.

Requires ``crewai>=1`` for ``AgentStateBackend``
(``pip install statefold[crewai]``).
"""
from __future__ import annotations

from typing import Any

from ..store import StateStore
from ..types import Scope


from ..sync_bridge import SyncBridge as _Bridge  # noqa: E402


# --------------------------------------------------------------------------
# CrewAI >= 1.x: StorageBackend
# --------------------------------------------------------------------------

def _make_backend_base():
    from crewai.memory.storage.backend import StorageBackend

    return StorageBackend


try:
    _BackendBase: Any = _make_backend_base()
except Exception:  # crewai<1 or not installed; class below still importable
    _BackendBase = object


class AgentStateBackend(_BackendBase):
    """CrewAI 1.x StorageBackend persisting MemoryRecords in a StateStore.

    Records are stored with CrewAI's own embeddings (computed upstream by
    CrewAI); search ranks against the query embedding via the store. The full
    record is kept in ``meta`` so it round-trips losslessly across restarts.
    """

    def __init__(self, store: StateStore, tenant: str, agent: str, session: str = "crew"):
        self.store = store
        self.scope = Scope(tenant, agent, session, "main")
        self._bridge = _Bridge()

    # --- helpers ---

    def _record_cls(self):
        from crewai.memory.types import MemoryRecord

        return MemoryRecord

    def _to_record(self, hit: dict):
        return self._record_cls().model_validate(hit["meta"]["record"])

    def _matches(self, rec, scope_prefix=None, categories=None, metadata_filter=None) -> bool:
        if scope_prefix and not (rec.scope or "").startswith(scope_prefix):
            return False
        if categories and not set(categories) & set(rec.categories or []):
            return False
        if metadata_filter:
            md = rec.metadata or {}
            if any(md.get(k) != v for k, v in metadata_filter.items()):
                return False
        return True

    # --- StorageBackend: writes ---

    def save(self, records) -> None:
        async def go():
            for rec in records:
                raw = rec.model_dump(mode="json")
                await self.store.remember(
                    self.scope, rec.content,
                    meta={"record": raw},
                    embedding=rec.embedding,
                    mem_id=rec.id,
                )

        self._bridge.run(go())

    async def asave(self, records) -> None:
        self.save(records)

    def update(self, record) -> None:
        self.save([record])

    def delete(self, scope_prefix=None, categories=None, record_ids=None,
               older_than=None, metadata_filter=None) -> int:
        async def go():
            deleted = 0
            for hit in await self.store.list_memories(self.scope, limit=100_000):
                rec = self._to_record(hit)
                if record_ids is not None and rec.id not in record_ids:
                    continue
                if not self._matches(rec, scope_prefix, categories, metadata_filter):
                    continue
                if older_than is not None and rec.created_at and rec.created_at >= older_than:
                    continue
                if await self.store.delete_memory(self.scope, hit["id"]):
                    deleted += 1
            return deleted

        return self._bridge.run(go())

    async def adelete(self, **kwargs) -> int:
        return self.delete(**kwargs)

    def reset(self, scope_prefix=None) -> None:
        self.delete(scope_prefix=scope_prefix)

    # --- StorageBackend: reads ---

    def search(self, query_embedding, scope_prefix=None, categories=None,
               metadata_filter=None, limit: int = 10, min_score: float = 0.0):
        async def go():
            # over-fetch, then post-filter on CrewAI's scope/category/metadata
            hits = await self.store.recall_vec(self.scope, query_embedding, k=max(limit * 4, 20))
            out = []
            for h in hits:
                rec = self._to_record(h)
                if not self._matches(rec, scope_prefix, categories, metadata_filter):
                    continue
                if h["score"] < min_score:
                    continue
                out.append((rec, h["score"]))
                if len(out) >= limit:
                    break
            return out

        return self._bridge.run(go())

    async def asearch(self, query_embedding, **kwargs):
        return self.search(query_embedding, **kwargs)

    def get_record(self, record_id: str):
        async def go():
            for hit in await self.store.list_memories(self.scope, limit=100_000):
                if hit["id"] == record_id:
                    return self._to_record(hit)
            return None

        return self._bridge.run(go())

    def list_records(self, scope_prefix=None, limit: int = 200, offset: int = 0):
        async def go():
            recs = [self._to_record(h)
                    for h in await self.store.list_memories(self.scope, limit=100_000)]
            if scope_prefix:
                recs = [r for r in recs if (r.scope or "").startswith(scope_prefix)]
            return recs[offset:offset + limit]

        return self._bridge.run(go())

    def count(self, scope_prefix=None) -> int:
        return len(self.list_records(scope_prefix=scope_prefix, limit=1_000_000))


# --------------------------------------------------------------------------
# CrewAI 0.x (legacy): ExternalMemory storage shape
# --------------------------------------------------------------------------

class AgentStateStorage:
    """Legacy CrewAI (<1.0) external-memory storage: save / search / reset."""

    def __init__(self, store: StateStore, tenant: str, agent: str, session: str = "crew"):
        self.store = store
        self.scope = Scope(tenant, agent, session, "main")
        self._bridge = _Bridge()

    def save(self, value, metadata: dict | None = None) -> None:
        self._bridge.run(self.store.remember(self.scope, str(value), meta=metadata or {}))

    def search(self, query: str, limit: int = 3, filter=None,
               score_threshold: float = 0.35) -> list[dict]:
        hits = self._bridge.run(self.store.recall(self.scope, query, k=limit))
        return [
            {"id": h["id"], "metadata": h.get("meta", {}), "context": h["text"], "score": 1.0}
            for h in hits
        ]

    def reset(self) -> None:
        forget = getattr(self.store, "forget", None)
        if forget is not None:
            self._bridge.run(forget(self.scope))

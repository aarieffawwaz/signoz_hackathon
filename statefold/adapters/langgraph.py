"""LangGraph adapter: a ``BaseCheckpointSaver`` backed by a StateStore.

Every LangGraph checkpoint and pending-write becomes an event in the durable
log, so the event log *is* the graph's memory. Point any existing graph at
this saver and it survives crashes, resumes mid-run, and can be replayed —
with zero changes to the graph itself.

Config mapping: ``thread_id`` -> session, ``checkpoint_ns`` -> thread
(``""`` becomes ``"main"``). ``tenant`` and ``agent`` are fixed per saver.

Requires ``langgraph`` (``pip install statefold[langgraph]``).
"""
from __future__ import annotations

import base64
from collections import defaultdict
from typing import Any, AsyncIterator, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from ..store import ConcurrencyError, StateStore
from ..types import Event, Scope

_MAX_RETRIES = 8


def _enc(serde, obj) -> list:
    """Serialize an arbitrary object to a JSON-safe [type, base64] pair."""
    t, b = serde.dumps_typed(obj)
    return [t, base64.b64encode(b).decode("ascii")]


def _dec(serde, pair):
    t, s = pair
    return serde.loads_typed((t, base64.b64decode(s)))


class PlatformCheckpointer(BaseCheckpointSaver):
    def __init__(self, store: StateStore, tenant: str, agent: str):
        super().__init__()
        self.store = store
        self.tenant = tenant
        self.agent = agent

    # --- helpers ---------------------------------------------------------

    def _scope(self, config) -> Scope:
        c = config["configurable"]
        ns = c.get("checkpoint_ns") or "main"
        return Scope(self.tenant, self.agent, c["thread_id"], ns)

    async def _append(self, scope: Scope, event: Event) -> None:
        # Checkpoint writes within a run are sequential per thread; a conflict
        # just means we read a stale head, so re-read and retry.
        for _ in range(_MAX_RETRIES):
            head = await self.store.head(scope)
            try:
                await self.store.append(scope, [event], expected_seq=head)
                return
            except ConcurrencyError:
                continue
        raise RuntimeError(f"append failed after {_MAX_RETRIES} retries on {scope.flatten()}")

    def _cfg(self, config, checkpoint_id: str) -> dict:
        c = config["configurable"]
        return {
            "configurable": {
                "thread_id": c["thread_id"],
                "checkpoint_ns": c.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_id,
            }
        }

    async def _scan(self, scope: Scope):
        """Return (ordered checkpoint payloads, writes-by-checkpoint-id)."""
        checkpoints: dict[str, dict] = {}
        order: list[str] = []
        writes: dict[str, list] = defaultdict(list)
        async for e in self.store.read_events(scope):
            if e.kind == "lg_checkpoint":
                cid = e.payload["checkpoint_id"]
                if cid not in checkpoints:
                    order.append(cid)
                checkpoints[cid] = e.payload
            elif e.kind == "lg_writes":
                for channel, enc in e.payload["writes"]:
                    writes[e.payload["checkpoint_id"]].append(
                        (e.payload["task_id"], channel, _dec(self.serde, enc))
                    )
        return checkpoints, order, writes

    def _tuple(self, config, scope, payload, writes) -> CheckpointTuple:
        cid = payload["checkpoint_id"]
        parent = payload.get("parent_id")
        return CheckpointTuple(
            config=self._cfg(config, cid),
            checkpoint=_dec(self.serde, payload["checkpoint"]),
            metadata=_dec(self.serde, payload["metadata"]),
            parent_config=self._cfg(config, parent) if parent else None,
            pending_writes=writes.get(cid, []),
        )

    # --- BaseCheckpointSaver (async) -------------------------------------

    async def aput(self, config, checkpoint: Checkpoint, metadata: CheckpointMetadata,
                   new_versions: ChannelVersions):
        scope = self._scope(config)
        ev = Event(
            kind="lg_checkpoint",
            actor="langgraph",
            payload={
                "checkpoint_id": checkpoint["id"],
                "parent_id": config["configurable"].get("checkpoint_id"),
                "checkpoint": _enc(self.serde, checkpoint),
                "metadata": _enc(self.serde, get_checkpoint_metadata(config, metadata)),
            },
        )
        await self._append(scope, ev)
        return self._cfg(config, checkpoint["id"])

    async def aput_writes(self, config, writes: Sequence[tuple[str, Any]], task_id: str,
                          task_path: str = ""):
        scope = self._scope(config)
        ev = Event(
            kind="lg_writes",
            actor="langgraph",
            payload={
                "checkpoint_id": config["configurable"]["checkpoint_id"],
                "task_id": task_id,
                "task_path": task_path,
                "writes": [[channel, _enc(self.serde, val)] for channel, val in writes],
            },
        )
        await self._append(scope, ev)

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        scope = self._scope(config)
        checkpoints, order, writes = await self._scan(scope)
        if not order:
            return None
        cid = get_checkpoint_id(config) or order[-1]
        payload = checkpoints.get(cid)
        return self._tuple(config, scope, payload, writes) if payload else None

    async def alist(self, config, *, filter=None, before=None, limit=None
                    ) -> AsyncIterator[CheckpointTuple]:
        scope = self._scope(config)
        checkpoints, order, writes = await self._scan(scope)
        before_id = get_checkpoint_id(before) if before else None
        count = 0
        for cid in reversed(order):  # newest first
            if before_id and cid >= before_id:
                continue
            yield self._tuple(config, scope, checkpoints[cid], writes)
            count += 1
            if limit and count >= limit:
                break

"""Integration test against REAL crewai 1.x: MemoryRecords in the StateStore."""
from __future__ import annotations

import pytest

from statefold import InMemoryStore

pytest.importorskip("crewai", reason="crewai>=1 not installed")

from crewai.memory.types import MemoryRecord  # noqa: E402

from statefold.adapters.crewai import AgentStateBackend  # noqa: E402


def _rec(rid: str, content: str, emb: list[float], scope="/crew/agent",
         categories=None, metadata=None) -> MemoryRecord:
    return MemoryRecord(
        id=rid, content=content, scope=scope, embedding=emb,
        categories=categories or [], metadata=metadata or {},
    )


@pytest.fixture
def backend():
    return AgentStateBackend(InMemoryStore(), tenant="acme", agent="crew1")


def test_save_and_search_by_embedding(backend):
    backend.save([
        _rec("r1", "customer is vegetarian", [1.0, 0.0, 0.0]),
        _rec("r2", "flight leaves at 9am", [0.0, 1.0, 0.0]),
    ])
    hits = backend.search([0.9, 0.1, 0.0], limit=1)
    assert len(hits) == 1
    rec, score = hits[0]
    assert rec.content == "customer is vegetarian"
    assert score > 0.9


def test_records_survive_restart():
    store = InMemoryStore()
    b1 = AgentStateBackend(store, tenant="acme", agent="crew1")
    b1.save([_rec("r1", "important fact", [1.0, 0.0])])

    b2 = AgentStateBackend(store, tenant="acme", agent="crew1")  # "new process"
    got = b2.get_record("r1")
    assert got is not None and got.content == "important fact"
    assert b2.count() == 1


def test_scope_and_category_filters(backend):
    backend.save([
        _rec("r1", "a", [1.0, 0.0], scope="/crew/researcher", categories=["pref"]),
        _rec("r2", "b", [1.0, 0.0], scope="/crew/writer", categories=["fact"]),
    ])
    hits = backend.search([1.0, 0.0], scope_prefix="/crew/writer")
    assert [r.id for r, _ in hits] == ["r2"]
    hits = backend.search([1.0, 0.0], categories=["pref"])
    assert [r.id for r, _ in hits] == ["r1"]


def test_update_is_upsert(backend):
    backend.save([_rec("r1", "old content", [1.0, 0.0])])
    backend.update(_rec("r1", "new content", [1.0, 0.0]))
    assert backend.count() == 1
    assert backend.get_record("r1").content == "new content"


def test_delete_and_reset(backend):
    backend.save([
        _rec("r1", "a", [1.0, 0.0], metadata={"kind": "tmp"}),
        _rec("r2", "b", [1.0, 0.0], metadata={"kind": "keep"}),
    ])
    assert backend.delete(metadata_filter={"kind": "tmp"}) == 1
    assert backend.count() == 1
    backend.reset()
    assert backend.count() == 0


def test_min_score_threshold(backend):
    backend.save([_rec("r1", "unrelated", [0.0, 1.0])])
    assert backend.search([1.0, 0.0], min_score=0.5) == []

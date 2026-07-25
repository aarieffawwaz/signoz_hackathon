"""Runtime invariants — reject invalid writes before they happen.

An invariant is a pure function ``(state_before, event) -> None`` that raises
``InvariantViolation`` to veto a write. Stores validate the whole batch
against the folded state (advancing it event by event) BEFORE persisting
anything, so a rejected append leaves the stream untouched — all-or-nothing.

    from statefold import invariant, InvariantViolation

    @invariant("state_delta")
    def stage_never_goes_backward(state, event):
        order = ["searching", "reviewing", "booked"]
        old = state.get("data", {}).get("stage")
        new = event.payload.get("set", {}).get("stage")
        if old in order and new in order and order.index(new) < order.index(old):
            raise InvariantViolation(f"stage cannot move {old} -> {new}")

    @invariant("*")                       # runs for every event kind
    def budget_cap(state, event): ...

Validation only runs when at least one invariant is registered, so the
zero-invariant fast path costs nothing.
"""
from __future__ import annotations

import copy
from typing import Callable

from .reducers import REDUCERS

Invariant = Callable[[dict, object], None]

INVARIANTS: dict[str, list[Invariant]] = {}


class InvariantViolation(Exception):
    """Raised by an invariant to reject a write; nothing is persisted."""


def invariant(kind: str = "*") -> Callable[[Invariant], Invariant]:
    """Register an invariant for one event ``kind`` (or ``"*"`` for all)."""

    def deco(fn: Invariant) -> Invariant:
        INVARIANTS.setdefault(kind, []).append(fn)
        return fn

    return deco


def clear_invariants() -> None:
    INVARIANTS.clear()


def has_invariants() -> bool:
    return bool(INVARIANTS)


def validate_batch(state_before: dict, events) -> None:
    """Check every event in an append batch against the evolving state.

    Raises ``InvariantViolation`` on the first failure. The caller must not
    have persisted anything yet.
    """
    state = copy.deepcopy(state_before)
    for ev in events:
        for fn in INVARIANTS.get(ev.kind, []):
            fn(state, ev)
        for fn in INVARIANTS.get("*", []):
            fn(state, ev)
        reducer = REDUCERS.get(ev.kind)
        if reducer is not None:
            state = reducer(state, ev.payload)

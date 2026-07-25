"""Reducers turn a stream of events into current state.

State is a *fold* over the event log — never mutated in place. Reducers are
pure functions ``(state, payload) -> state`` and are the public extension
point: register your own event kinds without touching the store.
"""
from __future__ import annotations

import copy
from typing import Callable, Iterable

Reducer = Callable[[dict, dict], dict]

REDUCERS: dict[str, Reducer] = {}


def register(kind: str) -> Callable[[Reducer], Reducer]:
    """Register a reducer for an event ``kind``. Overrides any built-in."""

    def deco(fn: Reducer) -> Reducer:
        REDUCERS[kind] = fn
        return fn

    return deco


# --- built-in event kinds -------------------------------------------------

@register("message")
def _message(state: dict, payload: dict) -> dict:
    """payload = {"role": ..., "content": ...}"""
    state.setdefault("messages", []).append(payload)
    return state


@register("state_delta")
def _state_delta(state: dict, payload: dict) -> dict:
    """payload = {"set": {k: v}, "unset": [k, ...]} — mutates the "data" bag."""
    data = state.setdefault("data", {})
    data.update(payload.get("set", {}))
    for k in payload.get("unset", []):
        data.pop(k, None)
    return state


@register("tool_call")
def _tool_call(state: dict, payload: dict) -> dict:
    """payload = {"name": ..., "args": ..., "result": ...}"""
    state.setdefault("tool_calls", []).append(payload)
    return state


@register("checkpoint")
def _checkpoint(state: dict, payload: dict) -> dict:
    # A checkpoint event is a marker; it carries no working-state change.
    # (LangGraph's checkpointer stores the graph checkpoint blob in payload.)
    if payload.get("checkpoint") is not None:
        state["_last_checkpoint"] = payload
    return state


@register("span_start")
def _span_start(state: dict, payload: dict) -> dict:
    """payload = {"span_id", "parent_id"?, "name", "attrs"?}"""
    spans = state.setdefault("spans", {})
    spans[payload["span_id"]] = {
        "name": payload.get("name", "span"),
        "parent_id": payload.get("parent_id"),
        "attrs": payload.get("attrs", {}),
        "status": "running",
    }
    return state


@register("span_end")
def _span_end(state: dict, payload: dict) -> dict:
    """payload = {"span_id", "status", "duration_ms"?, "error"?}"""
    sp = state.setdefault("spans", {}).get(payload["span_id"])
    if sp is not None:
        sp["status"] = payload.get("status", "ok")
        sp["duration_ms"] = payload.get("duration_ms")
        if payload.get("error"):
            sp["error"] = payload["error"]
    return state


@register("memory_write")
def _memory_write(state: dict, payload: dict) -> dict:
    # Long-term memory lives on a separate (vector) path; no working-state change.
    return state


def fold(items: Iterable[tuple[str, dict]], base: dict | None = None) -> dict:
    """Apply reducers over ``(kind, payload)`` pairs, starting from ``base``.

    ``base`` is a snapshot (an optimization). Passing ``None`` folds from
    empty — always correct, just slower. Unknown kinds are skipped, so an
    old log stays readable after a reducer is removed.
    """
    state = copy.deepcopy(base) if base else {}
    for kind, payload in items:
        reducer = REDUCERS.get(kind)
        if reducer is not None:
            state = reducer(state, payload)
    return state

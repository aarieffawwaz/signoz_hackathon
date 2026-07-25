"""statefold — a framework-agnostic, event-sourced state platform for agents."""
from .integrity import GENESIS, compute_hash, verify_chain
from .invariants import InvariantViolation, clear_invariants, invariant
from .memory import InMemoryStore
from .reducers import REDUCERS, fold, register
from .store import ConcurrencyError, StateStore
from .telemetry import register_pricing, usage_report
from .types import Event, Scope, new_id, now_iso

__all__ = [
    "Scope",
    "Event",
    "new_id",
    "now_iso",
    "StateStore",
    "ConcurrencyError",
    "InMemoryStore",
    "REDUCERS",
    "fold",
    "register",
    "register_pricing",
    "usage_report",
    "verify_chain",
    "compute_hash",
    "GENESIS",
    "invariant",
    "clear_invariants",
    "InvariantViolation",
]
__version__ = "0.2.0"

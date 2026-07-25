"""Semantic recall with a pluggable embedder (deterministic toy model)."""
from __future__ import annotations

from statefold import InMemoryStore, Scope
from statefold.embedding import cosine

# Toy embedding space: axis 0 = "food", axis 1 = "travel", axis 2 = "work".
_VOCAB = {
    "vegetarian": [1, 0, 0], "vegan": [0.9, 0, 0], "pizza": [0.8, 0, 0],
    "flight": [0, 1, 0], "hotel": [0, 0.9, 0], "paris": [0, 0.8, 0],
    "meeting": [0, 0, 1], "deadline": [0, 0, 0.9],
}


def toy_embedder(text: str) -> list[float]:
    v = [0.0, 0.0, 0.0]
    for w in text.lower().split():
        for i, x in enumerate(_VOCAB.get(w, [0, 0, 0])):
            v[i] += x
    return v


async def test_semantic_recall_beats_lexical():
    store = InMemoryStore(embedder=toy_embedder)
    scope = Scope("t", "a", "s")
    await store.remember(scope, "user is vegetarian")
    await store.remember(scope, "hotel booked in paris")
    await store.remember(scope, "deadline is friday meeting")

    # "vegan pizza" shares NO words with "user is vegetarian" — only meaning.
    hits = await store.recall(scope, "vegan pizza", k=1)
    assert hits and hits[0]["text"] == "user is vegetarian"

    hits = await store.recall(scope, "flight", k=1)
    assert hits[0]["text"] == "hotel booked in paris"


async def test_async_embedder_supported():
    async def aembedder(text: str) -> list[float]:
        return toy_embedder(text)

    store = InMemoryStore(embedder=aembedder)
    scope = Scope("t", "a", "s")
    await store.remember(scope, "user is vegetarian")
    hits = await store.recall(scope, "vegan", k=1)
    assert hits and "vegetarian" in hits[0]["text"]


async def test_no_embedder_still_works_lexically():
    store = InMemoryStore()
    scope = Scope("t", "a", "s")
    await store.remember(scope, "user is vegetarian")
    assert await store.recall(scope, "vegetarian user", k=1)


def test_cosine_edge_cases():
    assert cosine([], []) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0   # dim mismatch
    assert cosine([0, 0], [1, 1]) == 0.0      # zero vector
    assert abs(cosine([1, 1], [1, 1]) - 1.0) < 1e-9

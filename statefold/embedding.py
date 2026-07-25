"""Embedding support for semantic recall.

An embedder is any callable ``(text: str) -> list[float]`` (sync or async).
Bring your own model — OpenAI, sentence-transformers, Ollama, anything:

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    async def embedder(text: str) -> list[float]:
        r = await client.embeddings.create(model="text-embedding-3-small", input=text)
        return r.data[0].embedding

    store = InMemoryStore(embedder=embedder)   # or PostgresStore(pool, embedder=...)

Without an embedder, stores fall back to lexical matching — correct, just
less clever.
"""
from __future__ import annotations

import inspect
import math
from typing import Awaitable, Callable, Union

Embedder = Callable[[str], Union[list[float], Awaitable[list[float]]]]


async def embed(embedder: Embedder, text: str) -> list[float]:
    """Call a sync-or-async embedder uniformly."""
    out = embedder(text)
    if inspect.isawaitable(out):
        out = await out
    return list(out)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

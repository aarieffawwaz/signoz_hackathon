"""SDK instrumentation — auto-record LLM calls from OpenAI, Anthropic, and
Google GenAI clients as ``llm_call`` events.

These SDKs are LLM *clients*, not agent frameworks, so the integration is
instrumentation rather than an adapter: wrap the client once and every
completion call lands in the event log with model, token usage, and latency —
cost, state, and the console light up with no manual ``add_llm_call``.

    from openai import OpenAI
    from statefold.adapters.generic import SessionState
    from statefold.instrument import instrument_openai

    session = SessionState(store, tenant="acme", agent="bot", session="s1")
    client = instrument_openai(OpenAI(), session)      # same client, now recorded
    client.chat.completions.create(model="gpt-5", messages=[...])

Works with sync and async clients (method type is detected). No SDK is
imported here — wrapping is duck-typed on the client object — so none of them
are dependencies.

Caveat: streaming responses only carry usage when the SDK is asked for it
(e.g. OpenAI's ``stream_options={"include_usage": True}``). Calls whose
response has no usable usage block are recorded with zero tokens and
``"no_usage": true`` in the payload rather than being silently dropped.
"""
from __future__ import annotations

import inspect
import time
from typing import Callable

from .sync_bridge import SyncBridge

_bridge: SyncBridge | None = None


def _get_bridge() -> SyncBridge:
    global _bridge
    if _bridge is None:
        _bridge = SyncBridge()
    return _bridge


def _num(obj, *names) -> int | None:
    """First present numeric attribute among ``names`` (dicts also supported)."""
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if isinstance(v, (int, float)):
            return int(v)
    return None


# --- usage extractors: (response, call_kwargs) -> llm_call payload ----------

def _extract_openai(resp, kwargs) -> dict:
    u = getattr(resp, "usage", None)
    itok = _num(u, "prompt_tokens", "input_tokens") if u is not None else None
    otok = _num(u, "completion_tokens", "output_tokens") if u is not None else None
    return {
        "model": getattr(resp, "model", None) or kwargs.get("model", "unknown"),
        "input_tokens": itok or 0,
        "output_tokens": otok or 0,
        "no_usage": itok is None and otok is None,
    }


def _extract_anthropic(resp, kwargs) -> dict:
    u = getattr(resp, "usage", None)
    itok = _num(u, "input_tokens") if u is not None else None
    otok = _num(u, "output_tokens") if u is not None else None
    return {
        "model": getattr(resp, "model", None) or kwargs.get("model", "unknown"),
        "input_tokens": itok or 0,
        "output_tokens": otok or 0,
        "no_usage": itok is None and otok is None,
    }


def _extract_google(resp, kwargs) -> dict:
    u = getattr(resp, "usage_metadata", None)
    itok = _num(u, "prompt_token_count", "input_token_count") if u is not None else None
    otok = _num(u, "candidates_token_count", "output_token_count") if u is not None else None
    return {
        "model": kwargs.get("model") or getattr(resp, "model_version", None) or "unknown",
        "input_tokens": itok or 0,
        "output_tokens": otok or 0,
        "no_usage": itok is None and otok is None,
    }


# --- content extractors: (response, call_kwargs) -> (user_text, assistant_text)

_MAX_CONTENT = 4000  # chars per captured message


def _last_user_message(msgs):
    for m in reversed(msgs or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else str(c)
    return None


def _content_openai(resp, kwargs):
    user = _last_user_message(kwargs.get("messages"))
    if user is None and kwargs.get("input") is not None:      # responses API
        user = str(kwargs["input"])
    text = getattr(resp, "output_text", None)
    choices = getattr(resp, "choices", None)
    if text is None and choices:
        text = getattr(getattr(choices[0], "message", None), "content", None)
    return user, text


def _content_anthropic(resp, kwargs):
    user = _last_user_message(kwargs.get("messages"))
    text = None
    blocks = getattr(resp, "content", None)
    if isinstance(blocks, list) and blocks:
        text = getattr(blocks[0], "text", None)
    return user, text


def _content_google(resp, kwargs):
    contents = kwargs.get("contents")
    user = contents if isinstance(contents, str) else (str(contents) if contents else None)
    return user, getattr(resp, "text", None)


# --- generic method wrapper --------------------------------------------------

def _record(session, payload: dict, latency_ms: float,
            content: tuple | None = None) -> None:
    meta = {"no_usage": True} if payload.pop("no_usage", False) else {}
    coro = session.add_llm_call(
        payload["model"], payload["input_tokens"], payload["output_tokens"],
        latency_ms=round(latency_ms, 1), **meta,
    )
    _get_bridge().run(coro)
    if content is not None:
        user, text = content
        if user:
            _get_bridge().run(session.add_message("user", str(user)[:_MAX_CONTENT],
                                                  captured=True))
        if text:
            _get_bridge().run(session.add_message("assistant", str(text)[:_MAX_CONTENT],
                                                  captured=True))


def _wrap_method(owner, name: str, session, extract: Callable,
                 content_fn: Callable | None = None) -> None:
    orig = getattr(owner, name)
    if getattr(orig, "_statefold_instrumented", False):
        return

    if inspect.iscoroutinefunction(orig):
        async def wrapper(*args, **kwargs):  # type: ignore[misc]
            t0 = time.perf_counter()
            resp = await orig(*args, **kwargs)
            _record(session, extract(resp, kwargs), (time.perf_counter() - t0) * 1000,
                    content_fn(resp, kwargs) if content_fn else None)
            return resp
    else:
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            resp = orig(*args, **kwargs)
            _record(session, extract(resp, kwargs), (time.perf_counter() - t0) * 1000,
                    content_fn(resp, kwargs) if content_fn else None)
            return resp

    wrapper._statefold_instrumented = True  # type: ignore[attr-defined]
    setattr(owner, name, wrapper)


def _wrap_path(client, path: tuple[str, ...], session, extract,
               content_fn: Callable | None = None) -> bool:
    owner = client
    for part in path[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return False
    if not callable(getattr(owner, path[-1], None)):
        return False
    _wrap_method(owner, path[-1], session, extract, content_fn)
    return True


# --- public API ---------------------------------------------------------------

def instrument_openai(client, session, capture_content: bool = False):
    """Record OpenAI SDK calls (chat.completions.create + responses.create).

    ``capture_content=True`` additionally logs the last user message and the
    assistant reply as ``message`` events (opt-in: privacy + payload size).
    """
    cf = _content_openai if capture_content else None
    wrapped = _wrap_path(client, ("chat", "completions", "create"), session,
                         _extract_openai, cf)
    wrapped |= _wrap_path(client, ("responses", "create"), session, _extract_openai, cf)
    if not wrapped:
        raise TypeError("client has neither chat.completions.create nor responses.create")
    return client


def instrument_anthropic(client, session, capture_content: bool = False):
    """Record Anthropic SDK calls (messages.create)."""
    cf = _content_anthropic if capture_content else None
    if not _wrap_path(client, ("messages", "create"), session, _extract_anthropic, cf):
        raise TypeError("client has no messages.create")
    return client


def instrument_google(client, session, capture_content: bool = False):
    """Record Google GenAI SDK calls (models.generate_content)."""
    cf = _content_google if capture_content else None
    wrapped = _wrap_path(client, ("models", "generate_content"), session,
                         _extract_google, cf)
    wrapped |= _wrap_path(client, ("aio", "models", "generate_content"), session,
                          _extract_google, cf)
    if not wrapped:
        raise TypeError("client has no models.generate_content")
    return client


def instrument(client, session, capture_content: bool = False):
    """Auto-detect the SDK by the client's module and instrument it."""
    mod = type(client).__module__ or ""
    if mod.startswith("openai"):
        return instrument_openai(client, session, capture_content)
    if mod.startswith("anthropic"):
        return instrument_anthropic(client, session, capture_content)
    if mod.startswith("google"):
        return instrument_google(client, session, capture_content)
    # fall back to shape detection
    for fn in (instrument_openai, instrument_anthropic, instrument_google):
        try:
            return fn(client, session, capture_content)
        except TypeError:
            continue
    raise TypeError(f"don't know how to instrument {type(client)!r}")

"""SDK instrumentation with duck-typed fake clients (no SDK dependencies)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from statefold import InMemoryStore, register_pricing
from statefold.adapters.generic import SessionState
from statefold.instrument import (
    instrument,
    instrument_anthropic,
    instrument_google,
    instrument_openai,
)
from statefold.telemetry import clear_pricing


@pytest.fixture(autouse=True)
def pricing():
    clear_pricing()
    register_pricing("gpt-5", 1.25, 10.0)
    register_pricing("claude-sonnet-5", 3.0, 15.0)
    register_pricing("gemini-3-pro", 2.0, 12.0)
    yield
    clear_pricing()


@pytest.fixture
def session():
    return SessionState(InMemoryStore(), tenant="t", agent="a", session="s")


def usage(session):
    return asyncio.run(session.usage())


# --- fakes mirroring each SDK's response/attribute shape ---------------------

class FakeOpenAI:
    def __init__(self):
        resp = NS(model="gpt-5", usage=NS(prompt_tokens=1000, completion_tokens=200))
        self.chat = NS(completions=NS(create=lambda **kw: resp))


class FakeAsyncOpenAI:
    def __init__(self):
        async def create(**kw):
            return NS(model="gpt-5", usage=NS(input_tokens=500, output_tokens=100))
        self.responses = NS(create=create)


class FakeAnthropic:
    def __init__(self):
        resp = NS(model="claude-sonnet-5", usage=NS(input_tokens=800, output_tokens=150))
        self.messages = NS(create=lambda **kw: resp)


class FakeGoogle:
    def __init__(self):
        resp = NS(model_version="gemini-3-pro",
                  usage_metadata=NS(prompt_token_count=600, candidates_token_count=90))
        self.models = NS(generate_content=lambda **kw: resp)


# --- tests -------------------------------------------------------------------

def test_openai_sync_recorded(session):
    client = instrument_openai(FakeOpenAI(), session)
    out = client.chat.completions.create(model="gpt-5", messages=[])
    assert out.model == "gpt-5"                       # response passthrough
    u = usage(session)
    assert u["llm_calls"] == 1
    assert u["input_tokens"] == 1000 and u["output_tokens"] == 200
    assert u["cost_usd"] == pytest.approx(1000 * 1.25 / 1e6 + 200 * 10 / 1e6)
    assert u["by_model"]["gpt-5"]["latency_ms_total"] >= 0


def test_openai_async_responses_api(session):
    client = instrument_openai(FakeAsyncOpenAI(), session)

    async def go():
        return await client.responses.create(model="gpt-5", input="hi")

    asyncio.run(go())
    u = usage(session)
    assert u["llm_calls"] == 1 and u["input_tokens"] == 500


def test_anthropic_recorded(session):
    client = instrument_anthropic(FakeAnthropic(), session)
    client.messages.create(model="claude-sonnet-5", max_tokens=100, messages=[])
    u = usage(session)
    assert u["by_model"]["claude-sonnet-5"]["input_tokens"] == 800
    assert u["cost_usd"] > 0


def test_google_recorded(session):
    client = instrument_google(FakeGoogle(), session)
    client.models.generate_content(model="gemini-3-pro", contents="hi")
    u = usage(session)
    assert u["by_model"]["gemini-3-pro"]["output_tokens"] == 90


def test_autodetect_by_shape(session):
    client = instrument(FakeAnthropic(), session)
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])
    assert usage(session)["llm_calls"] == 1


def test_missing_usage_is_flagged_not_dropped(session):
    class NoUsage:
        def __init__(self):
            self.messages = NS(create=lambda **kw: NS(model="claude-sonnet-5", usage=None))

    client = instrument_anthropic(NoUsage(), session)
    client.messages.create(model="claude-sonnet-5", max_tokens=1, messages=[])
    u = usage(session)
    assert u["llm_calls"] == 1 and u["input_tokens"] == 0


def test_double_instrument_is_idempotent(session):
    client = FakeOpenAI()
    instrument_openai(client, session)
    instrument_openai(client, session)   # second wrap must be a no-op
    client.chat.completions.create(model="gpt-5", messages=[])
    assert usage(session)["llm_calls"] == 1


def test_unknown_client_raises():
    with pytest.raises(TypeError):
        instrument(object(), None)


def test_real_openai_sdk_wraps(session):
    """Wrap a real openai client (no network) and confirm interception."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="test-not-used")
    instrument_openai(client, session)
    assert getattr(client.chat.completions.create, "_statefold_instrumented", False)


def test_real_anthropic_sdk_wraps(session):
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="test-not-used")
    instrument(client, session)   # autodetect by module
    assert getattr(client.messages.create, "_statefold_instrumented", False)


def test_real_google_sdk_wraps(session):
    genai = pytest.importorskip("google.genai")
    client = genai.Client(api_key="test-not-used")
    instrument(client, session)   # autodetect by module
    assert getattr(client.models.generate_content, "_statefold_instrumented", False)


class FakeOpenAIWithContent:
    def __init__(self):
        msg = NS(content="Paris is the capital of France.")
        resp = NS(model="gpt-5", usage=NS(prompt_tokens=10, completion_tokens=8),
                  choices=[NS(message=msg)])
        self.chat = NS(completions=NS(create=lambda **kw: resp))


def test_capture_content_opt_in(session):
    client = instrument_openai(FakeOpenAIWithContent(), session, capture_content=True)
    client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": "be brief"},
                  {"role": "user", "content": "capital of france?"}])

    msgs = asyncio.run(session.messages())
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "capital of france?"
    assert "Paris" in msgs[1]["content"]
    assert all(m.get("captured") for m in msgs)
    assert usage(session)["llm_calls"] == 1        # usage still recorded


def test_content_not_captured_by_default(session):
    client = instrument_openai(FakeOpenAIWithContent(), session)
    client.chat.completions.create(model="gpt-5",
                                   messages=[{"role": "user", "content": "hi"}])
    assert asyncio.run(session.messages()) == []
    assert usage(session)["llm_calls"] == 1

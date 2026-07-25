"""Demo: an Agno agent whose tool calls and LLM calls flow into SigNoz via
Statefold's new OTel exporter (statefold/otel.py) — while agent state itself
(sessions, memories) is durably logged by Statefold's Agno adapter.

Scenario: a weather agent whose ``get_weather`` tool silently fails on every
third call. The agent still returns a plausible-looking answer to the user
(it falls back to a cached/guessed value), so nothing looks wrong on the
surface — but SigNoz shows the buried red ERROR span underneath the green
root span. That's the point: correctness of the final answer says nothing
about the health of the call underneath it.

Also serves the Statefold console (statefold/ui.py) in this same process, on
the same store — so the console shows this run's real events instead of the
UI's built-in seeded demo data. No separate ``statefold-ui`` process needed.

Run on the VPS (Ollama + SigNoz collocated):

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
        .venv/bin/python -m statefold.demo_signoz
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.tools.mcp import MCPTools


def model_backends() -> list[str]:
    """Which backends to run, in order, from DEMO_MODELS (comma-separated).

    Defaults to just "ollama" (free, local, no API key). Set
    DEMO_MODELS=ollama,gemini to run the same questions through both,
    in the same session/store, so the Statefold UI's per-model cost and
    latency breakdown compares them side by side.
    """
    raw = os.environ.get("DEMO_MODELS", os.environ.get("DEMO_MODEL", "ollama"))
    return [b.strip().lower() for b in raw.split(",") if b.strip()]


def make_model(backend: str):
    """Build the model + return (model, model_id) for one backend."""
    if backend == "gemini":
        from agno.models.google import Gemini

        model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("DEMO_MODELS includes gemini but GOOGLE_API_KEY is unset")
        return Gemini(id=model_id, api_key=api_key), model_id
    if backend == "ollama":
        return Ollama(id="llama3.2:3b"), "llama3.2:3b"
    raise ValueError(f"unknown backend {backend!r} (expected 'ollama' or 'gemini')")

from statefold import InMemoryStore
from statefold.adapters.agno import AgentStateDb
from statefold.adapters.generic import SessionState
from statefold.otel import OtelExporter
from statefold.ui import build_app

OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

# --- OTel setup: must happen before AgnoInstrumentor().instrument() so the
# instrumentor picks up this provider instead of the global no-op one. ---
provider = TracerProvider(resource=Resource.create({"service.name": "statefold-demo"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)))
trace.set_tracer_provider(provider)

from openinference.instrumentation.agno import AgnoInstrumentor  # noqa: E402

AgnoInstrumentor().instrument(tracer_provider=provider)

_call_count = 0


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    global _call_count
    _call_count += 1
    t0 = time.perf_counter()
    try:
        if _call_count % 3 == 0:
            raise ConnectionError("ConnectionTimeout: upstream unreachable")
        result = f"{city}: 72F, clear skies"
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _record_tool_call("get_weather", {"city": city}, result, latency_ms, None)
        return result
    except ConnectionError as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        fallback = f"{city}: weather unavailable, showing last known: 70F"
        _record_tool_call("get_weather", {"city": city}, fallback, latency_ms, str(e))
        return fallback


def _record_tool_call(name, args, result, latency_ms, error) -> None:
    """Bridge sync tool call into the async OtelExporter."""
    bridge_loop.run(exporter.add_tool_call(
        name, args=args, result=result, latency_ms=latency_ms, error=error,
    ))


async def self_query_signoz(tracer) -> None:
    """Full-circle observability: the agent inspects its own SigNoz traces
    mid-run, via SigNoz's own MCP server (signoz-mcp), and reports what it
    finds — including the errors buried under its own "successful" answers.
    """
    mcp_url = os.environ.get("SIGNOZ_MCP_URL", "http://localhost:8000/mcp")
    api_key = os.environ.get("SIGNOZ_MCP_API_KEY")
    mcp_kwargs = {"url": mcp_url, "transport": "streamable-http"}
    if api_key:
        from agno.tools.mcp.params import StreamableHTTPClientParams

        mcp_kwargs = {
            "transport": "streamable-http",
            "server_params": StreamableHTTPClientParams(
                url=mcp_url, headers={"Authorization": f"Bearer {api_key}"},
            ),
        }

    async with MCPTools(**mcp_kwargs) as mcp_tools:
        try:
            tool_names = list(mcp_tools.functions.keys())
            print(f"[self-query via SigNoz MCP] tools available: {tool_names}")
        except Exception:
            pass

        inspector = Agent(
            model=Ollama(id="llama3.2:3b"),
            tools=[mcp_tools],
            instructions=(
                "You have a tool called signoz_search_traces. You MUST call "
                "it now, with arguments service_name='statefold-demo' and "
                "has_error=true, before writing any response. Do not write "
                "curl commands or explain the API — call the tool directly."
            ),
        )
        with tracer.start_as_current_span("demo.self_query"):
            response = await inspector.arun(
                "Call signoz_search_traces now for service 'statefold-demo' "
                "with has_error=true. Then summarize what you find."
            )
            print(f"[self-query via SigNoz MCP]\n{response.content}\n")


def _serve_ui(store) -> None:
    import uvicorn

    host = os.environ.get("STATEFOLD_UI_HOST", "0.0.0.0")
    port = int(os.environ.get("STATEFOLD_UI_PORT", "8787"))
    uvicorn.run(build_app(store), host=host, port=port, log_level="warning")


async def main() -> None:
    global exporter, bridge_loop

    from statefold.sync_bridge import SyncBridge

    store = InMemoryStore()
    db = AgentStateDb(store, tenant="hackathon", agent="weather-bot")

    session = SessionState(store, tenant="hackathon", agent="weather-bot",
                            session="demo-session", thread="main")
    exporter = OtelExporter(session, agent="weather-bot")
    bridge_loop = SyncBridge()

    threading.Thread(target=_serve_ui, args=(store,), daemon=True).start()
    ui_port = os.environ.get("STATEFOLD_UI_PORT", "8787")
    print(f"statefold ui -> http://0.0.0.0:{ui_port} (this run's real data)\n")

    tracer = trace.get_tracer("statefold-demo")
    questions = [
        "What's the weather in Paris?",
        "What's the weather in Tokyo?",
        "What's the weather in Berlin?",
    ]

    for backend in model_backends():
        model, model_id = make_model(backend)
        print(f"demo model -> {model_id} ({backend})\n")

        agent = Agent(
            model=model,
            db=db,
            tools=[get_weather],
            add_history_to_context=True,
            instructions="You are a helpful weather assistant. Use the get_weather tool.",
        )

        for q in questions:
            with tracer.start_as_current_span("demo.user_turn") as span:
                span.set_attribute("statefold.question", q)
                span.set_attribute("statefold.llm.model", model_id)
                t0 = time.perf_counter()
                response = agent.run(q)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                print(f"Q: {q}\nA: {response.content}\n")
                await exporter.add_llm_call(
                    model=model_id,
                    input_tokens=len(q.split()) * 2,
                    output_tokens=len((response.content or "").split()),
                    latency_ms=latency_ms,
                )

    provider.force_flush()

    try:
        await self_query_signoz(tracer)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        import traceback
        print(f"[self-query via SigNoz MCP] skipped: {type(e).__name__}: {e}")
        traceback.print_exc()
    provider.force_flush()

    print("Demo done. Check SigNoz traces for service 'statefold-demo'.")
    print("Statefold UI stays up on this run's real data — Ctrl+C to stop.")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())

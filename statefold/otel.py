"""OpenTelemetry exporter for Statefold sessions.

Wraps a ``SessionState`` and mirrors every tool/LLM call into OTel: a child
span (nested under whatever span is active on the current context — e.g. an
Agno-instrumented agent run) plus counters/histograms for tokens, cost, and
latency. Purely additive: the event log and hash chain are untouched, this
only emits telemetry alongside the normal write-through.

    from statefold.otel import OtelExporter

    exporter = OtelExporter(session, agent="weather-bot")
    await exporter.add_tool_call("get_weather", args={"city": "nyc"},
                                  result="72F", latency_ms=120.0)
    await exporter.add_llm_call("llama3.2:3b", input_tokens=50,
                                 output_tokens=20, latency_ms=430.0)
"""
from __future__ import annotations

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .adapters.generic import SessionState

_tracer = trace.get_tracer("statefold")
_meter = metrics.get_meter("statefold")

_token_counter = _meter.create_counter(
    "statefold.tokens", unit="1", description="LLM tokens consumed (input+output)"
)
_cost_counter = _meter.create_counter(
    "statefold.cost_usd", unit="USD", description="LLM spend"
)
_latency_hist = _meter.create_histogram(
    "statefold.latency_ms", unit="ms", description="Tool/LLM call latency"
)


class OtelExporter:
    """Drop-in wrapper around SessionState that also emits OTel spans/metrics."""

    def __init__(self, session: SessionState, agent: str):
        self._session = session
        self._agent = agent

    async def add_tool_call(self, name: str, args=None, result=None,
                             latency_ms: float | None = None,
                             error: str | None = None) -> None:
        with _tracer.start_as_current_span(f"tool.{name}") as span:
            span.set_attribute("statefold.agent", self._agent)
            span.set_attribute("statefold.tool.name", name)
            if latency_ms is not None:
                span.set_attribute("statefold.latency_ms", latency_ms)
            if error is not None:
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute("statefold.error", error)
            else:
                span.set_status(Status(StatusCode.OK))

            if latency_ms is not None:
                _latency_hist.record(latency_ms, {
                    "statefold.agent": self._agent,
                    "statefold.call_type": "tool",
                    "statefold.tool.name": name,
                })

            await self._session.add_tool_call(
                name, args=args, result=result,
                latency_ms=latency_ms, error=error,
            )

    async def add_llm_call(self, model: str, input_tokens: int, output_tokens: int,
                            latency_ms: float | None = None,
                            cost_usd: float | None = None, **meta) -> None:
        with _tracer.start_as_current_span(f"llm.{model}") as span:
            span.set_attribute("statefold.agent", self._agent)
            span.set_attribute("statefold.llm.model", model)
            span.set_attribute("statefold.llm.input_tokens", input_tokens)
            span.set_attribute("statefold.llm.output_tokens", output_tokens)
            if latency_ms is not None:
                span.set_attribute("statefold.latency_ms", latency_ms)
            if cost_usd is not None:
                span.set_attribute("statefold.cost_usd", cost_usd)
            span.set_status(Status(StatusCode.OK))

            attrs = {"statefold.agent": self._agent, "statefold.llm.model": model}
            _token_counter.add(input_tokens + output_tokens, attrs)
            if cost_usd is not None:
                _cost_counter.add(cost_usd, attrs)
            if latency_ms is not None:
                _latency_hist.record(latency_ms, {**attrs, "statefold.call_type": "llm"})

            await self._session.add_llm_call(
                model, input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=latency_ms, cost_usd=cost_usd, **meta,
            )

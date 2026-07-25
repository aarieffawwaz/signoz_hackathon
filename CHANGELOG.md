# Changelog

## Unreleased — Agents of SigNoz hackathon

Built for the WeMakeDevs "Agents of SigNoz" hackathon, Track 01 (AI & Agent
Observability). Statefold's README had `- [ ] OpenTelemetry exporter` on its
roadmap — this fills that gap and proves it live against SigNoz.

### Added

- `statefold/otel.py` — `OtelExporter`, a thin wrapper around `SessionState`
  that mirrors `add_tool_call` / `add_llm_call` into OTel child spans, nested
  automatically under whatever span is active on the current context (e.g.
  an auto-instrumented agent framework run). Emits three instruments:
  - `statefold.tokens` (counter)
  - `statefold.cost_usd` (counter)
  - `statefold.latency_ms` (histogram)
  Failed tool/LLM calls set `StatusCode.ERROR` on the span, so a failure
  buried under an otherwise-successful agent run is visible in any
  OTel-native trace UI — not just Statefold's own console.
- `statefold/demo_signoz.py` — a weather agent (Agno v2.8.2 + Ollama
  `llama3.2:3b`, local, free, no API key) whose `get_weather` tool fails
  silently on every third call. The agent still returns a plausible-looking
  fallback answer, so the failure is invisible from the chat transcript
  alone — but shows up as a red child span in SigNoz under a green root
  span. Demonstrates why "the agent answered" and "the agent worked" are
  different claims, and why OTel-level observability matters for
  event-sourced state libraries that otherwise only expose their own
  internal event log.
- `openinference-instrumentation-agno` used for auto-instrumenting the Agno
  agent's own run/step spans; Statefold's spans nest as children underneath.

### Deployment

- SigNoz deployed via Foundry (`casting.yaml` / `casting.yaml.lock`) on a
  standalone VPS, always-on for judging.
- Statefold's existing observability console (`statefold/ui.py`) served
  alongside SigNoz as a second, complementary dashboard — event log,
  time-travel replay, cost breakdown, span tree — showing the same run from
  the event-sourcing side.

### Why this design

- No new dashboard was built from scratch. Statefold already ships a
  production-quality console (`statefold/ui.py`); duplicating that in a
  hackathon timebox would only produce a worse version of what exists.
  SigNoz and the Statefold console show the same underlying event stream
  from two different angles: OTel-native (SigNoz) and event-sourced
  (Statefold), which is the actual point of building an exporter — it
  doesn't replace the event log, it projects a second, standards-based view
  of it.
- `OtelExporter` is additive only. It never touches the event log or hash
  chain — telemetry emission and durable state writes are two independent
  side effects of the same call.

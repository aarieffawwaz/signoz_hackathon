"""statefold UI — a zero-build observability console with time-travel.

One command, no npm, no separate frontend deploy:

    statefold-ui                      # in-memory + demo data
    STATEFOLD_DSN=postgresql://... statefold-ui    # inspect a real store

Arize/Phoenix-style console: KPI cards, a streams table with cost/error
columns, a span waterfall with duration bars — plus the things only an
event-sourced store can do: replay any run step by step (time-travel), see
state diffs between steps, and watch cumulative spend at any point in time.
"""
from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .store import StateStore
from .types import Scope

_store: StateStore | None = None


def _scope_from(stream: str) -> Scope:
    tenant, agent, session, thread = stream.split("/", 3)
    return Scope(tenant, agent, session, thread)


async def api_streams(request):
    return JSONResponse(await _store.list_streams())


async def api_summary(request):
    """Global KPIs + per-stream usage rows for the console header/table."""
    from .telemetry import usage_report

    streams = await _store.list_streams()
    totals = {"streams": len(streams), "events": 0, "llm_calls": 0,
              "tokens": 0, "cost_usd": 0.0, "errors": 0}
    rows = []
    for s in streams:
        u = await usage_report(_store, _scope_from(s["stream"]))
        errors = sum(t.get("errors", 0) for t in u.get("tools", {}).values())
        totals["events"] += s["head_seq"]
        totals["llm_calls"] += u.get("llm_calls", 0)
        totals["tokens"] += u.get("input_tokens", 0) + u.get("output_tokens", 0)
        totals["cost_usd"] += u.get("cost_usd", 0.0)
        totals["errors"] += errors
        rows.append({**s, "cost_usd": u.get("cost_usd", 0.0),
                     "llm_calls": u.get("llm_calls", 0), "errors": errors})
    return JSONResponse({"totals": totals, "streams": rows})


async def api_events(request):
    scope = _scope_from(request.query_params["stream"])
    events = [
        {"seq": e.seq, "kind": e.kind, "actor": e.actor, "ts": e.ts,
         "payload": e.payload, "causation_id": e.causation_id}
        async for e in _store.read_events(scope)
    ]
    return JSONResponse(events)


async def api_state(request):
    scope = _scope_from(request.query_params["stream"])
    as_of = request.query_params.get("as_of")
    state = await _store.get_state(scope, as_of=int(as_of) if as_of else None)
    return JSONResponse(state)


async def api_usage(request):
    from .telemetry import usage_report

    scope = _scope_from(request.query_params["stream"])
    as_of = request.query_params.get("as_of")
    return JSONResponse(await usage_report(_store, scope, as_of=int(as_of) if as_of else None))


async def api_series(request):
    """Cumulative cost at every sequence — one incremental fold, one response."""
    from .reducers import REDUCERS

    scope = _scope_from(request.query_params["stream"])
    state: dict = {}
    out = []
    async for e in _store.read_events(scope):
        reducer = REDUCERS.get(e.kind)
        if reducer is not None:
            state = reducer(state, e.payload)
        u = state.get("usage", {})
        out.append({"seq": e.seq, "kind": e.kind,
                    "cost_usd": u.get("cost_usd", 0.0),
                    "llm_calls": u.get("llm_calls", 0)})
    return JSONResponse(out)


async def api_memories(request):
    scope = _scope_from(request.query_params["stream"])
    return JSONResponse(await _store.list_memories(scope, limit=500))


async def api_costs(request):
    """Spend analytics aggregated across every stream."""
    from .telemetry import usage_report

    streams = await _store.list_streams()
    by_model: dict = {}
    by_agent: dict = {}
    rows = []
    totals = {"cost_usd": 0.0, "uncosted_calls": 0, "llm_calls": 0, "tokens": 0}
    for s in streams:
        scope = _scope_from(s["stream"])
        u = await usage_report(_store, scope)
        for m, d in u.get("by_model", {}).items():
            agg = by_model.setdefault(m, {"calls": 0, "input_tokens": 0,
                                          "output_tokens": 0, "cost_usd": 0.0})
            for k in agg:
                agg[k] += d.get(k, 0)
        agent = f"{scope.tenant}/{scope.agent}"
        ba = by_agent.setdefault(agent, {"cost_usd": 0.0, "llm_calls": 0})
        ba["cost_usd"] += u.get("cost_usd", 0.0)
        ba["llm_calls"] += u.get("llm_calls", 0)
        totals["cost_usd"] += u.get("cost_usd", 0.0)
        totals["uncosted_calls"] += u.get("uncosted_calls", 0)
        totals["llm_calls"] += u.get("llm_calls", 0)
        totals["tokens"] += u.get("input_tokens", 0) + u.get("output_tokens", 0)
        rows.append({"stream": s["stream"], "cost_usd": u.get("cost_usd", 0.0),
                     "llm_calls": u.get("llm_calls", 0),
                     "uncosted": u.get("uncosted_calls", 0)})
    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return JSONResponse({"totals": totals, "by_model": by_model,
                         "by_agent": by_agent, "streams": rows})


async def api_tools(request):
    """Tool performance aggregated across every stream."""
    from .telemetry import usage_report

    streams = await _store.list_streams()
    tools: dict = {}
    for s in streams:
        u = await usage_report(_store, _scope_from(s["stream"]))
        for name, t in u.get("tools", {}).items():
            agg = tools.setdefault(name, {"calls": 0, "errors": 0,
                                          "latency_ms_total": 0.0, "streams": 0})
            agg["calls"] += t.get("calls", 0)
            agg["errors"] += t.get("errors", 0)
            agg["latency_ms_total"] += t.get("latency_ms_total", 0.0)
            agg["streams"] += 1
    rows = [{"tool": n, **d,
             "avg_ms": (d["latency_ms_total"] / d["calls"]) if d["calls"] else 0.0,
             "err_rate": (d["errors"] / d["calls"]) if d["calls"] else 0.0}
            for n, d in tools.items()]
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return JSONResponse(rows)


async def api_memories_all(request):
    """Every memory across all sessions: session-level (deduped per session
    scope) plus agent-level long-term (deduped per tenant/agent)."""
    streams = await _store.list_streams()
    seen_sessions: set = set()
    seen_agents: set = set()
    out = []
    for s in streams:
        scope = _scope_from(s["stream"])
        skey = scope.memory_key()
        if skey not in seen_sessions:
            seen_sessions.add(skey)
            for m in await _store.list_memories(scope, limit=500, level="session"):
                out.append({**m, "session": skey})
        akey = f"{scope.tenant}/{scope.agent}"
        if akey not in seen_agents:
            seen_agents.add(akey)
            for m in await _store.list_memories(scope, limit=500, level="agent"):
                out.append({**m, "session": akey})
    return JSONResponse(out)


async def api_promptfoo(request):
    """Download this stream as a promptfoo eval config (regression suite)."""
    from .promptfoo import to_promptfoo

    scope = _scope_from(request.query_params["stream"])
    cfg = await to_promptfoo(_store, scope)
    return JSONResponse(cfg, headers={
        "Content-Disposition": f'attachment; filename="promptfooconfig.json"'})


async def api_verify(request):
    """Recompute the tamper-evidence hash chain for one stream."""
    from .integrity import verify_chain

    scope = _scope_from(request.query_params["stream"])
    return JSONResponse(await verify_chain(_store, scope))


async def api_tail(request):
    """SSE tail of a stream: emits each event as it lands.

    ``after=N`` starts past seq N; ``once=1`` emits the backlog then closes
    (used by tests and one-shot consumers). Works with plain ``curl -N``.
    """
    import asyncio as _aio
    import json as _json

    from starlette.responses import StreamingResponse

    scope = _scope_from(request.query_params["stream"])
    after = int(request.query_params.get("after", 0))
    once = request.query_params.get("once") in ("1", "true")

    async def gen():
        last = after
        idle = 0
        while True:
            head = await _store.head(scope)
            if head > last:
                async for e in _store.read_events(scope, after=last):
                    data = _json.dumps({"seq": e.seq, "kind": e.kind, "actor": e.actor,
                                        "ts": e.ts, "payload": e.payload, "hash": e.hash})
                    yield f"data: {data}\n\n"
                    last = e.seq
                idle = 0
            if once:
                return
            idle += 1
            if idle % 30 == 0:
                yield ": keepalive\n\n"
            await _aio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


async def index(request):
    return HTMLResponse(PAGE)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>statefold — observability console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
--bg:#0a0a0b;--surface:#111112;--surface2:#18181a;--raise:#1f1f22;
--line:#232326;--line2:#33333a;
--text:#ece9e2;--dim:#9b978c;--faint:#67635a;
--gold:#e3b341;--gold2:#f2d27c;--green:#8ab66d;--amber:#d9955a;--red:#e5534b;--cyan:#7fb4b0;
--mono:'JetBrains Mono',ui-monospace,'Cascadia Code',Consolas,monospace;
--sans:'Inter','Segoe UI Variable Text','Segoe UI',system-ui,sans-serif}
*{box-sizing:border-box;margin:0}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes shimmer{0%{background-position:0% 50%}100%{background-position:200% 50%}}
@keyframes glowpulse{0%,100%{opacity:.45}50%{opacity:1}}
@keyframes flashrow{0%{background:#e3b34129}100%{background:transparent}}
@keyframes dashin{to{stroke-dashoffset:0}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
::-webkit-scrollbar-track{background:transparent}
body{background:var(--bg);color:var(--text);font:13px/1.5 var(--sans);height:100vh;display:flex;flex-direction:column;overflow:hidden}

nav{height:54px;display:flex;align-items:center;gap:12px;padding:0 18px;flex-shrink:0;
 position:relative;z-index:20;background:#111112d9;backdrop-filter:blur(16px) saturate(1.25);
 -webkit-backdrop-filter:blur(16px) saturate(1.25);border-bottom:1px solid var(--line)}
nav::after{content:'';position:absolute;left:0;right:0;bottom:-1px;height:1px;
 background:linear-gradient(90deg,transparent 5%,#e3b34130 50%,transparent 95%)}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--text)}
.brand:hover .logo{transform:rotate(-7deg) scale(1.07)}
.navdiv{width:1px;height:20px;background:var(--line2);margin:0 4px}
#navind{position:absolute;bottom:0;height:2px;border-radius:2px 2px 0 0;
 background:linear-gradient(90deg,var(--gold),var(--gold2));opacity:0;
 transition:left .32s cubic-bezier(.2,.7,.3,1),width .32s cubic-bezier(.2,.7,.3,1),opacity .2s;
 box-shadow:0 0 10px -2px #e3b341aa}
.logo{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--gold),var(--gold2));display:grid;place-items:center;font-weight:700;font-size:14px;color:#0a0a0b}
nav h1{font-size:14.5px;font-weight:600;margin-right:10px}
.navtab{display:inline-flex;align-items:center;gap:7px;color:var(--dim);font-size:13px;
 text-decoration:none;padding:6px 13px;border-radius:9px;transition:color .15s,background .15s}
.navtab svg{width:13px;height:13px;opacity:.7;transition:opacity .15s}
.navtab:hover{color:var(--text);background:var(--surface2)}
.navtab:hover svg{opacity:1}
.navtab.on{color:var(--gold2)}
.navtab.on svg{opacity:1}
nav .crumb{display:flex;align-items:center;gap:7px;margin-left:8px;
 border:1px solid var(--line);background:var(--surface2);border-radius:18px;
 padding:4px 13px;font-size:11.5px;color:var(--faint);white-space:nowrap}
nav .crumb .cdot{width:6px;height:6px;border-radius:50%;background:var(--gold);flex-shrink:0}
nav .crumb b{color:var(--dim);font-weight:500;font-family:var(--mono);font-size:11px}
main.page{display:none}
main.page.on{display:grid;animation:pagein .18s ease}
.pagefull{display:none;flex:1;overflow-y:auto;padding:18px 22px}
.pagefull.on{display:block;animation:pagein .18s ease}
@keyframes pagein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.row.sel{background:var(--surface2);box-shadow:inset 2.5px 0 0 var(--gold)}
.row.new{animation:flashrow 1.6s ease-out}
.playbtn.playing{border-color:var(--gold);box-shadow:0 0 14px -3px #e3b341aa;animation:glowpulse 1.1s ease-in-out infinite}
.spark.sa .line{stroke-dasharray:1;stroke-dashoffset:1;animation:dashin .9s .12s cubic-bezier(.2,.7,.3,1) forwards}
.pagegrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1100px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:15px 16px}
.panel h3{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);font-weight:600;margin-bottom:10px}
.panel.wide{grid-column:1/-1}
.hbar{display:grid;grid-template-columns:170px 1fr 92px;gap:10px;align-items:center;padding:5px 0;font-size:12px}
.hbar .n{color:var(--dim);font-family:var(--mono);font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hbar .v{font-family:var(--mono);text-align:right;color:var(--text);font-size:11.5px}
.htrack{height:9px;border-radius:5px;background:var(--line);overflow:hidden}
.htrack i{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,var(--gold),var(--gold2))}
.htrack i.g{background:linear-gradient(90deg,#5c8a4a,var(--green))}
table.data{width:100%;border-collapse:collapse;font-size:12px}
table.data th{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-weight:600}
table.data td{padding:8px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11.5px;color:var(--dim)}
table.data th:nth-child(n+2),table.data td:nth-child(n+2){text-align:right}
table.data td:first-child{color:var(--text);font-family:var(--sans);font-weight:500}
.pill{font-family:var(--mono);font-size:10px;padding:1px 8px;border-radius:9px}
.pill.ok{background:#8ab66d22;color:var(--green)}
.pill.bad{background:#e5534b22;color:var(--red)}
.search{width:100%;max-width:420px;background:var(--surface2);border:1px solid var(--line2);border-radius:9px;color:var(--text);padding:8px 12px;font:12.5px var(--sans);margin-bottom:14px;outline:none}
.search:focus{border-color:var(--gold)}
.memgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;max-width:1100px}

body.home .kpis{display:none}
body.home .crumb{display:none}
.hero{max-width:880px;margin:0 auto;padding:84px 20px 0;text-align:center;position:relative}
.hero::before{content:'';position:absolute;top:-60px;left:50%;transform:translateX(-50%);
 width:760px;height:460px;border-radius:50%;pointer-events:none;
 background:radial-gradient(ellipse at center,#e3b34116,transparent 65%);
 animation:glowpulse 8s ease-in-out infinite}
.hero>*{animation:rise .6s cubic-bezier(.2,.7,.3,1) both}
.hero>*:nth-child(1){animation-delay:.03s}
.hero>*:nth-child(3){animation-delay:.16s}
.hero>*:nth-child(4){animation-delay:.24s}
.hero>*:nth-child(5){animation-delay:.34s}
.herobadge{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--gold);border:1px solid #e3b34155;border-radius:20px;padding:5px 14px;background:#e3b34110;letter-spacing:.04em}
.hero h2{font-size:52px;font-weight:700;letter-spacing:-.02em;line-height:1.08;margin:26px 0 18px;
 background:linear-gradient(120deg,var(--text) 25%,var(--gold) 45%,var(--gold2) 55%,var(--text) 75%);
 background-size:200% auto;
 -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
 animation:rise .6s .09s cubic-bezier(.2,.7,.3,1) both,shimmer 8s linear 1.4s infinite}
.hero .sub{color:var(--dim);font-size:16.5px;line-height:1.65;max-width:640px;margin:0 auto 30px}
.ctas{display:flex;gap:12px;justify-content:center;margin-bottom:44px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 24px;border-radius:11px;font-size:14px;font-weight:600;text-decoration:none;transition:all .15s;cursor:pointer}
.btn.gold{background:linear-gradient(135deg,var(--gold),#c9962f);color:#0a0a0b;box-shadow:0 4px 18px -4px #e3b34166;position:relative;overflow:hidden}
.btn.gold::before{content:'';position:absolute;top:0;left:-90%;width:55%;height:100%;
 background:linear-gradient(100deg,transparent,#ffffff66,transparent);
 transform:skewX(-20deg);transition:left .55s ease}
.btn.gold:hover::before{left:140%}
.btn.gold:hover{box-shadow:0 6px 24px -4px #e3b34199;transform:translateY(-1px)}
.btn.ghost{border:1px solid var(--line2);color:var(--dim);background:var(--surface)}
.btn.ghost:hover{color:var(--text);border-color:var(--faint)}
.herostats{display:flex;gap:34px;justify-content:center;flex-wrap:wrap;padding:22px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);max-width:720px;margin:0 auto}
.hstat .hv{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--gold)}
.hstat .hl{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin-top:2px}
.reveal{opacity:0;transform:translateY(18px);transition:opacity .55s cubic-bezier(.2,.7,.3,1),transform .55s cubic-bezier(.2,.7,.3,1)}
.reveal.vis{opacity:1;transform:none}
.sectionhead{max-width:980px;margin:110px auto 0;padding:0 20px;text-align:center}
.sectionhead .eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--gold)}
.sectionhead h3{font-size:26px;font-weight:600;letter-spacing:-.01em;margin:10px 0 6px}
.sectionhead p{color:var(--dim);font-size:14px;max-width:520px;margin:0 auto}
.features{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:980px;margin:36px auto 0;padding:0 20px}
.feat{background:var(--surface);border:1px solid var(--line);border-radius:13px;padding:20px;text-align:left;transition:border-color .2s,transform .22s cubic-bezier(.2,.7,.3,1),box-shadow .22s}
.feat:hover{border-color:#e3b34155;transform:translateY(-3px);box-shadow:0 14px 34px -20px #e3b34166}
.feat:hover .fic{transform:rotate(-6deg) scale(1.08)}
.feat .fic{width:34px;height:34px;border-radius:10px;background:#e3b34114;border:1px solid #e3b34133;display:grid;place-items:center;color:var(--gold);font-size:15px;margin-bottom:12px;transition:transform .25s cubic-bezier(.2,.7,.3,1)}
.feat h4{font-size:14px;font-weight:600;margin-bottom:6px}
.feat p{color:var(--dim);font-size:12.5px;line-height:1.55}
.install{max-width:980px;margin:56px auto 90px;padding:0 20px;text-align:center}
.install code{display:inline-block;font-family:var(--mono);font-size:13px;color:var(--gold2);background:var(--surface);border:1px solid var(--line2);border-radius:11px;padding:14px 26px}
.install code b{color:var(--faint);font-weight:400;margin:0 10px}
.live{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--dim);font-size:11px;font-family:var(--mono);border:1px solid var(--line);background:var(--surface2);border-radius:18px;padding:4px 13px}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:pu 2.2s infinite}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.3}}

.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--bg);flex-shrink:0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;position:relative;overflow:hidden;transition:border-color .18s,transform .18s}
.kpi:hover{transform:translateY(-2px)}
.kpi:hover{border-color:var(--line2)}
.kpi::before{content:'';position:absolute;inset:0 0 auto 0;height:2px;background:linear-gradient(90deg,var(--gold),transparent 70%);opacity:.55}
.kpi .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);font-weight:600}
.kpi .val{font-size:21px;font-weight:600;font-family:var(--mono);margin-top:3px}
.kpi .val.money{color:var(--green)}
.kpi .val.err{color:var(--red)}
.kpi .sub{font-size:10.5px;color:var(--faint);margin-top:1px}

main{flex:1;display:grid;grid-template-columns:295px minmax(430px,1fr) 400px;min-height:0}
.col{border-right:1px solid var(--line);min-height:0;display:flex;flex-direction:column}
.col:last-child{border-right:none}
.colhead{padding:12px 16px 8px;font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);font-weight:600;display:flex;align-items:baseline;gap:8px;flex-shrink:0}
.colhead .n{color:var(--gold);font-family:var(--mono)}
.scroll{overflow-y:auto;flex:1;min-height:0}

table.streams{width:100%;border-collapse:collapse;font-size:12.5px}
table.streams th{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);font-weight:600;position:sticky;top:0;background:var(--bg)}
table.streams th:nth-child(n+2),table.streams td:nth-child(n+2){text-align:right}
table.streams td{padding:9px 10px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11.5px;color:var(--dim)}
table.streams tr{cursor:pointer;transition:background .1s}
table.streams tbody tr:hover{background:var(--surface)}
table.streams tr.active{background:var(--surface2)}
table.streams tr.active td:first-child{box-shadow:inset 2.5px 0 0 var(--gold)}
.sname{font-family:var(--sans);color:var(--text);font-weight:600;font-size:12.5px;display:flex;align-items:center;gap:7px}
.avatar{width:8px;height:8px;border-radius:2.5px;flex-shrink:0}
.spath{color:var(--faint);font-size:10.5px;margin:2px 0 0 15px;font-family:var(--mono)}
.errcell{color:var(--red)}

.controls{display:flex;gap:10px;align-items:center;padding:10px 16px 2px;flex-shrink:0}
.playbtn{width:30px;height:30px;border-radius:8px;border:1px solid var(--line2);background:var(--surface2);color:var(--gold);cursor:pointer;font-size:12px;display:grid;place-items:center;transition:all .12s}
.playbtn:hover{border-color:var(--gold);box-shadow:0 0 0 3px #e3b34122}
.playbtn:disabled{opacity:.3;cursor:default;box-shadow:none}
input[type=range]{flex:1;accent-color:var(--gold);height:4px;cursor:pointer}
.seqbadge{font-family:var(--mono);font-size:11.5px;color:var(--gold);min-width:96px;text-align:right}
.hint{color:var(--faint);font-size:10.5px;padding:2px 16px 6px;flex-shrink:0}
kbd{font-family:var(--mono);font-size:9.5px;background:var(--surface2);border:1px solid var(--line2);border-radius:4px;padding:0 4px}
.chips{display:flex;gap:6px;padding:4px 16px 8px;flex-wrap:wrap;flex-shrink:0}
.chip{font-size:10.5px;font-family:var(--mono);padding:2px 9px;border-radius:20px;border:1px solid var(--line2);color:var(--dim);cursor:pointer;background:transparent;transition:all .12s}
.chip.on{background:#e3b3411a;color:var(--gold);border-color:var(--gold)}

.axis{display:flex;justify-content:space-between;padding:2px 16px 4px calc(16px + 200px);font-family:var(--mono);font-size:9.5px;color:var(--faint);flex-shrink:0}
.wf{padding:0 16px 16px}
.row{display:flex;align-items:center;gap:0;padding:3px 0;border-radius:7px;cursor:pointer;transition:background .1s,opacity .2s}
.row:hover{background:var(--surface)}
.row.sel{background:var(--surface2)}
.row.future{opacity:.28}
.rmeta{width:200px;flex-shrink:0;display:flex;align-items:center;gap:7px;padding-left:6px;min-width:0}
.rmeta .seq{font-family:var(--mono);font-size:9.5px;color:var(--faint);width:24px;flex-shrink:0}
.kdot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.rmeta .klabel{font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dim)}
.rmeta .klabel b{color:var(--text);font-weight:500}
.rtrack{flex:1;height:20px;position:relative;border-left:1px solid var(--line);
 background:repeating-linear-gradient(90deg,transparent 0,transparent calc(25% - 1px),var(--line) calc(25% - 1px),var(--line) 25%)}
.rbar{position:absolute;top:4px;height:12px;border-radius:4px;min-width:5px;box-shadow:0 1px 5px rgba(0,0,0,.4);transition:filter .1s}
.row:hover .rbar{filter:brightness(1.18)}
.rdot{position:absolute;top:7px;width:6px;height:6px;border-radius:50%;transform:translateX(-3px);box-shadow:0 0 5px currentColor}
.rdur{width:64px;flex-shrink:0;text-align:right;font-family:var(--mono);font-size:10px;color:var(--faint);padding-right:6px}
.rdur.err{color:var(--red)}

.tabs{display:flex;gap:2px;padding:10px 14px 0;flex-shrink:0}
.tab{padding:6px 13px;border-radius:8px 8px 0 0;font-size:12px;cursor:pointer;color:var(--dim);border:1px solid transparent;border-bottom:none}
.tab:hover{color:var(--text)}
.tab.on{background:var(--surface2);color:var(--text);border-color:var(--line2)}
.pane{background:var(--surface2);border:1px solid var(--line2);border-radius:0 10px 10px 10px;margin:0 14px 14px;padding:14px;overflow:auto;flex:1;min-height:0}
pre{font-family:var(--mono);font-size:11.5px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
.jk{color:var(--cyan)}.js{color:var(--green)}.jn{color:var(--amber)}.jb{color:var(--gold2)}
.chg{background:#e3b34126;display:inline-block;width:100%;border-radius:3px}
.cost{font-family:var(--mono);font-size:25px;color:var(--green);font-weight:600}
.cost small{font-size:11.5px;color:var(--faint);font-weight:400;font-family:var(--sans)}
.stat{display:flex;justify-content:space-between;padding:5px 0;font-size:12px;border-bottom:1px solid var(--line);color:var(--dim)}
.stat b{font-family:var(--mono);font-weight:500;color:var(--text)}
.stat .err{color:var(--red)}
.subhead{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);margin:14px 0 6px;font-weight:600}
.bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden;margin:4px 0 8px}
.bar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,var(--gold),var(--gold2))}
.spark{width:100%;height:62px;margin:8px 0 2px}
.memcard{background:var(--surface);border:1px solid var(--line2);border-radius:9px;padding:10px 12px;margin-bottom:8px}
.memcard .txt{font-size:12.5px}
.memcard .meta{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:4px}
.sprow{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--line);font-size:12px}
.sprow .spname{color:var(--text);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sprow .spdur{margin-left:auto;font-family:var(--mono);font-size:10.5px;color:var(--faint);flex-shrink:0}
.spbar{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--gold),var(--gold2));flex-shrink:0}
.spbar.err{background:var(--red)}
.spbar.run{background:var(--line2)}
.spdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.spev{font-family:var(--mono);font-size:10.5px;color:var(--faint);padding:2px 0 2px 0;border-bottom:1px dashed var(--line)}
.empty{color:var(--faint);padding:26px;text-align:center;font-size:12px}
.warnrow{color:var(--amber)}
.detail-kv{display:grid;grid-template-columns:88px 1fr;gap:4px 12px;font-size:11.5px;margin-bottom:12px}
.detail-kv .k{color:var(--faint)}
.detail-kv .v{font-family:var(--mono);color:var(--dim);word-break:break-all}
</style></head><body>
<nav>
 <a href="#/home" class="brand"><div class="logo">s</div><h1>statefold</h1></a>
 <span class="navdiv"></span>
 <a class="navtab" href="#/home" data-p="home"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 7.5L8 2.5l5.5 5v6h-3.7v-3.6H6.2v3.6H2.5z"/></svg>Home</a>
 <a class="navtab on" href="#/console" data-p="console"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="10" rx="1.6"/><path d="M4.6 6.4L6.8 8l-2.2 1.6M8.6 10.4h2.8"/></svg>Console</a>
 <a class="navtab" href="#/costs" data-p="costs"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3.2v9.6M10.4 5.4c0-1.1-1.1-1.7-2.4-1.7s-2.4.6-2.4 1.7c0 2.4 4.8 1.2 4.8 3.7 0 1.1-1.1 1.7-2.4 1.7s-2.4-.6-2.4-1.7"/></svg>Costs</a>
 <a class="navtab" href="#/tools" data-p="tools"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11"/><circle cx="10.5" cy="4.5" r="1.5" fill="var(--bg)"/><circle cx="5.5" cy="8" r="1.5" fill="var(--bg)"/><circle cx="11" cy="11.5" r="1.5" fill="var(--bg)"/></svg>Tools</a>
 <a class="navtab" href="#/memories" data-p="memories"><svg viewBox="0 0 16 16" fill="currentColor" stroke="none"><path d="M8 1.8l1.4 3.9 3.9 1.4-3.9 1.4L8 12.4 6.6 8.5 2.7 7.1l3.9-1.4z"/></svg>Memories</a>
 <div id="navind"></div>
 <span class="crumb"><span class="cdot"></span><b id="crumb">no stream selected</b></span>
 <div class="live"><div class="pulse"></div><span id="livelabel">live</span></div>
</nav>
<div class="kpis" id="kpis"></div>
<main class="page on" id="page-console">
 <div class="col">
   <div class="colhead">Streams <span class="n" id="scount"></span></div>
   <div class="scroll"><table class="streams">
     <thead><tr><th>session</th><th>ev</th><th>cost</th><th>err</th></tr></thead>
     <tbody id="streams"><tr><td colspan="4"><div class="empty">loading&hellip;</div></td></tr></tbody>
   </table></div>
 </div>
 <div class="col">
   <div class="colhead">Run waterfall <span class="n" id="wtotal"></span><span id="chainbadge" style="margin-left:auto"></span></div>
   <div class="controls">
     <button class="playbtn" id="play" disabled title="replay the run">&#9654;</button>
     <input type="range" id="slider" min="1" max="1" value="1" disabled>
     <span class="seqbadge" id="asof">&mdash;</span>
   </div>
   <div class="hint"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &nbsp; <kbd>space</kbd> play &nbsp; <kbd>end</kbd> head</div>
   <div class="chips" id="chips"></div>
   <div class="axis" id="axis"></div>
   <div class="scroll"><div class="wf" id="events"><div class="empty">select a stream</div></div></div>
 </div>
 <div class="col">
   <div class="tabs">
     <div class="tab on" data-t="detail">Details</div>
     <div class="tab" data-t="state">State</div>
     <div class="tab" data-t="trace">Trace</div>
     <div class="tab" data-t="usage">Usage</div>
     <div class="tab" data-t="memories">Memories</div>
   </div>
   <div class="pane" id="pane"><div class="empty">select a stream</div></div>
 </div>
</main>
<div class="pagefull" id="page-home">
 <div class="hero">
  <span class="herobadge">&#9670; open source &middot; event-sourced &middot; framework-agnostic</span>
  <h2>State for every agent.</h2>
  <p class="sub">Durable resume, time-travel replay, layered memory and cost telemetry —
   one append-only event log underneath LangGraph, CrewAI, Agno, and any MCP tool.
   Crash mid-run, resume exactly where you left off, and audit every step and every dollar.</p>
  <div class="ctas">
   <a class="btn gold" href="#/console">Open console &rarr;</a>
   <a class="btn ghost" href="https://github.com/Joshuaakaspace/statefold" target="_blank" rel="noopener">GitHub</a>
  </div>
  <div class="herostats" id="herostats"></div>
 </div>
 <div class="sectionhead reveal">
  <div class="eyebrow">what's inside</div>
  <h3>One log. Every concern.</h3>
  <p>State, memory, cost and audit are folds over the same append-only
   event log &mdash; not four systems glued together.</p>
 </div>
 <div class="features">
  <div class="feat reveal"><div class="fic">&#9648;</div><h4>Event-sourced core</h4>
   <p>State is a fold over an append-only log — never a mutable blob. Snapshots are an optimization, never the truth.</p></div>
  <div class="feat reveal"><div class="fic">&#8635;</div><h4>Time-travel &amp; replay</h4>
   <p>Rewind any run to any step. See exactly what the agent believed — and had spent — at sequence N.</p></div>
  <div class="feat reveal"><div class="fic">&#9096;</div><h4>Framework adapters</h4>
   <p>LangGraph checkpointer, CrewAI storage backend, Agno v2 Db, and a generic fa&ccedil;ade. Zero changes to your agent code.</p></div>
  <div class="feat reveal"><div class="fic">&#9737;</div><h4>Stateful MCP proxy</h4>
   <p>Wrap any third-party MCP server, unmodified. Flight-recorder audit trail plus deterministic replay of side effects.</p></div>
  <div class="feat reveal"><div class="fic">$</div><h4>Cost telemetry</h4>
   <p>Tokens, spend, per-model breakdowns and tool latency folded from the same log. Prices back-fill without rewriting history.</p></div>
  <div class="feat reveal"><div class="fic">&#10022;</div><h4>Layered memory</h4>
   <p>Working, semantic, episodic and procedural memory &mdash; session-scoped or agent-wide long-term. Saved by one framework, recallable by another.</p></div>
 </div>
 <div class="install reveal"><code>pip install statefold <b>&middot;</b> statefold-ui <b>&middot;</b> STATEFOLD_DSN=postgres://&hellip; for production</code></div>
</div>
<div class="pagefull" id="page-costs"><div class="pagegrid" id="costsbody"><div class="empty">loading&hellip;</div></div></div>
<div class="pagefull" id="page-tools"><div id="toolsbody"><div class="empty">loading&hellip;</div></div></div>
<div class="pagefull" id="page-memories">
  <input class="search" id="memsearch" placeholder="search memories&hellip;">
  <div class="memgrid" id="memsbody"><div class="empty">loading&hellip;</div></div>
</div>
<script>
const $=id=>document.getElementById(id);
const j=async u=>(await fetch(u)).json();
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const KCOL={message:'var(--green)',tool_call:'var(--amber)',llm_call:'var(--gold)',
 span_start:'var(--cyan)',span_end:'var(--cyan)',
 state_delta:'#9a94b8',memory_write:'var(--cyan)',checkpoint:'var(--faint)',
 lg_checkpoint:'var(--faint)',lg_writes:'var(--faint)',agno_session:'var(--cyan)'};
const fmt$=v=>'$'+(v||0).toFixed(4);
const fmtMs=v=>v>=1000?(v/1000).toFixed(1)+'s':Math.round(v)+'ms';

let cur=null, head=1, evs=[], series=[], rows=[], totalMs=1, hidden=new Set(),
    tab='detail', prevLines=null, playing=null, selSeq=null,
    animateNext=false, flashAfter=Infinity, countersDone=false, sparkAnim=false;

function hlLine(line){
  return esc(line).replace(/("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|(-?\d+\.?\d*(?:[eE][+-]?\d+)?)/g,
    (m,str,col,bool,num)=>str?(col?`<span class="jk">${str}</span>${col}`:`<span class="js">${str}</span>`)
      :bool?`<span class="jb">${bool}</span>`:`<span class="jn">${num}</span>`);
}
const hlJson=o=>{const r=JSON.stringify(o,null,2);return r===undefined?'':r.split('\n').map(hlLine).join('\n')};
function hlJsonDiff(obj){
  const lines=JSON.stringify(obj,null,2).split('\n');
  const prev=prevLines; prevLines=new Set(lines);
  return lines.map(l=>{const h=hlLine(l);
    return prev&&!prev.has(l)?`<span class="chg">${h}</span>`:h}).join('\n');
}
function label(e){
  const p=e.payload||{};
  if(e.kind==='message')return `<b>${esc(p.role||'msg')}</b> ${esc(String(p.content||'').slice(0,26))}`;
  if(e.kind==='tool_call')return `<b>${esc(p.name||'tool')}</b>`;
  if(e.kind==='llm_call')return `<b>${esc(p.model||'llm')}</b>`;
  if(e.kind==='state_delta'){const k=Object.keys(p.set||{}).concat(p.unset||[]);return `<b>&Delta;</b> ${esc(k.join(', ').slice(0,26))}`}
  if(e.kind==='span_start')return `<b>&#9656; ${esc(p.name||'span')}</b>`;
  if(e.kind==='span_end')return `<b>&#9666;</b> ${p.status==='error'?'<span style="color:var(--red)">error</span>':'done'}`;
  return `<b>${esc(e.kind)}</b>`;
}
function countUp(el){
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const raw=el.textContent.trim();
  const m=raw.match(/^\$?([\d,]+(?:\.\d+)?)$/);if(!m)return;
  const target=parseFloat(m[1].replace(/,/g,''));if(!isFinite(target))return;
  const dec=(m[1].split('.')[1]||'').length,pre=raw.startsWith('$')?'$':'';
  const commas=m[1].includes(','),t0=performance.now(),D=900;
  const step=now=>{const pr=Math.min((now-t0)/D,1),e=1-Math.pow(1-pr,3);
    let v=(target*e).toFixed(dec);
    if(commas)v=(+v).toLocaleString(undefined,{minimumFractionDigits:dec,maximumFractionDigits:dec});
    el.textContent=pre+v;
    if(pr<1)requestAnimationFrame(step);else el.textContent=raw;};
  requestAnimationFrame(step);
}
function hashColor(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))>>>0;return `hsl(${h%360} 60% 62%)`}

async function loadSummary(){
  const sum=await j('/api/summary');
  const t=sum.totals;
  $('kpis').innerHTML=
    kpi('streams',t.streams,'active sessions')+
    kpi('events',t.events,'in the log')+
    kpi('llm calls',t.llm_calls,'across all streams')+
    kpi('tokens',t.tokens.toLocaleString(),'in + out')+
    kpi('total cost',fmt$(t.cost_usd),'fold-time pricing','money')+
    kpi('tool errors',t.errors,t.errors?'needs attention':'all healthy',t.errors?'err':'');
  $('scount').textContent=t.streams;
  $('herostats').innerHTML=[
    [t.streams,'sessions'],[t.events,'events logged'],
    [t.llm_calls,'llm calls'],[fmt$(t.cost_usd),'tracked spend']
  ].map(([v,l])=>`<div class="hstat"><div class="hv">${v}</div><div class="hl">${l}</div></div>`).join('');
  const tb=$('streams');
  tb.innerHTML=sum.streams.length?sum.streams.map(s=>{
    const[ten,ag,sess,th]=s.stream.split('/');
    return `<tr class="${s.stream===cur?'active':''}" data-s="${esc(s.stream)}"><td>
      <div class="sname"><span class="avatar" style="background:${hashColor(ag+sess)}"></span>${esc(sess)}</div>
      <div class="spath">${esc(ten)}/${esc(ag)} &middot; ${esc(th)}</div></td>
      <td>${s.head_seq}</td><td>${fmt$(s.cost_usd)}</td>
      <td class="${s.errors?'errcell':''}">${s.errors||'&mdash;'}</td></tr>`}).join('')
    :'<tr><td colspan="4"><div class="empty">no streams yet &mdash; run your agent and sessions appear here</div></td></tr>';
  tb.querySelectorAll('tr').forEach(x=>x.onclick=()=>x.dataset.s&&pick(x.dataset.s));
  if(page==='console'&&!cur&&sum.streams.length)pick(sum.streams[0].stream);
  if(!countersDone){countersDone=true;
    document.querySelectorAll('.kpi .val,.hstat .hv').forEach(countUp);}
}
const kpi=(l,v,sub,cls='')=>`<div class="kpi"><div class="lbl">${l}</div><div class="val ${cls}">${v}</div><div class="sub">${sub}</div></div>`;

let es=null;
async function pick(stream){
  cur=stream;selSeq=null;prevLines=null;hidden.clear();stopPlay();animateNext=true;
  if(es){es.close();es=null}
  const[,,sess]=stream.split('/');
  $('crumb').textContent=sess;
  document.querySelectorAll('table.streams tr').forEach(x=>x.classList.toggle('active',x.dataset.s===stream));
  await reloadEvents(true);
  j('/api/verify?stream='+encodeURIComponent(stream)).then(v=>{
    $('chainbadge').innerHTML=v.ok
      ?'<span style="color:var(--green);font-family:var(--mono);font-size:10px">&#10003; chain intact ('+v.events+' ev)</span>'
      :'<span style="color:var(--red);font-family:var(--mono);font-size:10px">&#9888; chain BROKEN @ seq '+v.broken_at+'</span>';
  });
  let pending=false;
  es=new EventSource('/api/tail?stream='+encodeURIComponent(stream)+'&after='+head);
  es.onmessage=()=>{if(pending)return;pending=true;
    setTimeout(async()=>{pending=false;await reloadEvents(false)},150);};
}
async function reloadEvents(jump){
  const prevHead=head;
  [evs,series]=await Promise.all([
    j('/api/events?stream='+encodeURIComponent(cur)),
    j('/api/series?stream='+encodeURIComponent(cur))]);
  head=evs.length?evs[evs.length-1].seq:1;
  if(!jump&&cur&&head>prevHead)flashAfter=prevHead;
  let acc=0;
  rows=evs.map(e=>{
    const d=(e.payload&&e.payload.latency_ms)||0;
    const r={e,start:acc,dur:d};acc+=d;return r});
  totalMs=Math.max(acc,1);
  $('wtotal').textContent=fmtMs(totalMs)+' total';
  $('axis').innerHTML=[0,.25,.5,.75,1].map(f=>`<span>${fmtMs(totalMs*f)}</span>`).join('');
  const sl=$('slider'),wasHead=+sl.value>=+sl.max;
  sl.max=head;sl.disabled=!evs.length;$('play').disabled=!evs.length;
  if(jump||wasHead)sl.value=head;
  renderChips();renderWaterfall();refresh();
}
function renderChips(){
  const kinds=[...new Set(evs.map(e=>e.kind))];
  $('chips').innerHTML=kinds.map(k=>
    `<button class="chip${hidden.has(k)?'':' on'}" data-k="${esc(k)}">${esc(k)}</button>`).join('');
  $('chips').querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
    const k=c.dataset.k;hidden.has(k)?hidden.delete(k):hidden.add(k);
    renderChips();renderWaterfall();});
}
function renderWaterfall(){
  const v=+$('slider').value;
  const list=rows.filter(r=>!hidden.has(r.e.kind));
  $('events').innerHTML=list.length?list.map((r,i)=>{
    const e=r.e,p=e.payload||{},col=p.error?'var(--red)':(KCOL[e.kind]||'var(--faint)');
    const left=(r.start/totalMs*100).toFixed(2);
    const track=r.dur>0
      ?`<div class="rbar" style="left:${left}%;width:${Math.max(r.dur/totalMs*100,1).toFixed(2)}%;background:${col}"></div>`
      :`<div class="rdot" style="left:${left}%;background:${col}"></div>`;
    const tip=`#${e.seq} ${e.kind} · ${e.actor}${r.dur?' · '+fmtMs(r.dur):''}${p.error?' · '+p.error:''}`;
    const entrance=animateNext?` style="animation:rise .42s ${Math.min(i*26,520)}ms cubic-bezier(.2,.7,.3,1) both"`:'';
    return `<div class="row${e.seq>v?' future':''}${e.seq===selSeq?' sel':''}${e.seq>flashAfter?' new':''}" data-seq="${e.seq}" title="${esc(tip)}"${entrance}>
      <div class="rmeta"><span class="seq">#${e.seq}</span>
        <span class="kdot" style="background:${col}"></span>
        <span class="klabel">${label(e)}</span></div>
      <div class="rtrack">${track}</div>
      <div class="rdur${p.error?' err':''}">${p.error?'ERR':r.dur?fmtMs(r.dur):''}</div></div>`}).join('')
    :'<div class="empty">no events (all kinds filtered out?)</div>';
  animateNext=false;flashAfter=Infinity;
  $('events').querySelectorAll('.row').forEach(el=>el.onclick=()=>{
    selSeq=+el.dataset.seq;$('slider').value=selSeq;setTab('detail');refresh();});
}
function sparkline(){
  if(!series.length)return'';
  const W=352,H=54,max=Math.max(...series.map(p=>p.cost_usd),1e-9);
  const x=i=>4+(W-8)*(series.length===1?1:i/(series.length-1));
  const y=c=>H-4-(H-10)*(c/max);
  const pts=series.map((p,i)=>`${x(i).toFixed(1)},${y(p.cost_usd).toFixed(1)}`).join(' ');
  const v=+$('slider').value;
  let ci=-1;series.forEach((p,i)=>{if(p.seq<=v)ci=i});
  const cx=ci>=0?x(ci):4, cy=ci>=0?y(series[ci].cost_usd):H-4;
  return `<svg class="spark${sparkAnim?' sa':''}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#e3b341" stop-opacity=".32"/>
      <stop offset="1" stop-color="#e3b341" stop-opacity="0"/></linearGradient></defs>
    <polyline points="4,${H-4} ${pts} ${x(series.length-1).toFixed(1)},${H-4}" fill="url(#sg)" stroke="none"/>
    <polyline class="line" pathLength="1" points="${pts}" fill="none" stroke="var(--gold)" stroke-width="1.6" stroke-linejoin="round"/>
    <line x1="${cx}" y1="4" x2="${cx}" y2="${H-4}" stroke="var(--line2)" stroke-dasharray="2 3"/>
    <circle cx="${cx}" cy="${cy}" r="3" fill="var(--gold)"/></svg>`;
}
function setTab(t){tab=t;if(t==='usage')sparkAnim=true;
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  refresh();}
document.querySelectorAll('.tab').forEach(x=>x.onclick=()=>setTab(x.dataset.t));

async function refresh(){
  if(!cur)return;
  const v=+$('slider').value, atHead=v>=head;
  $('asof').textContent=atHead?('head · seq '+head):('seq '+v+' / '+head);
  document.querySelectorAll('.row').forEach(el=>el.classList.toggle('future',+el.dataset.seq>v));
  const q='?stream='+encodeURIComponent(cur)+(atHead?'':'&as_of='+v);
  const pane=$('pane');
  if(tab==='detail'){
    const e=evs.find(x=>x.seq===selSeq)||evs.filter(x=>x.seq<=v).slice(-1)[0];
    if(!e){pane.innerHTML='<div class="empty">no event at this position</div>';return}
    const p=e.payload||{};
    pane.innerHTML=
      `<div class="detail-kv">
        <span class="k">event</span><span class="v">#${e.seq} &middot; ${esc(e.kind)}</span>
        <span class="k">actor</span><span class="v">${esc(e.actor)}</span>
        <span class="k">time</span><span class="v">${esc(e.ts)}</span>
        ${p.latency_ms!=null?`<span class="k">latency</span><span class="v">${fmtMs(p.latency_ms)}</span>`:''}
        ${p.error?`<span class="k">error</span><span class="v" style="color:var(--red)">${esc(p.error)}</span>`:''}
      </div><div class="subhead">Payload</div><pre>${hlJson(p)}</pre>`;
  }else if(tab==='state'){
    const state=await j('/api/state'+q);
    pane.innerHTML=`<pre>${hlJsonDiff(state)}</pre>`;
  }else if(tab==='trace'){
    const state=await j('/api/state'+q);
    const spans=state.spans||{};
    const ids=Object.keys(spans);
    if(!ids.length){pane.innerHTML='<div class="empty">no spans in this stream<br><br>wrap work in <span style="font-family:var(--mono)">async with session.span(&quot;name&quot;)</span></div>';return}
    const kids={},roots=[];
    ids.forEach(id=>{const pr=spans[id].parent_id;
      if(pr&&spans[pr]){(kids[pr]=kids[pr]||[]).push(id)}else{roots.push(id)}});
    const evsBySpan={};
    evs.filter(e=>e.causation_id&&e.kind!=='span_end'&&e.seq<=v)
       .forEach(e=>{(evsBySpan[e.causation_id]=evsBySpan[e.causation_id]||[]).push(e)});
    const maxDur=Math.max(...ids.map(i=>spans[i].duration_ms||0),1);
    let html='';
    const draw=(id,depth)=>{
      const sp=spans[id];
      const w=sp.duration_ms?Math.max(sp.duration_ms/maxDur*160,6):0;
      const cls=sp.status==='error'?'err':(sp.status==='running'?'run':'');
      const col=sp.status==='error'?'var(--red)':(sp.status==='running'?'var(--faint)':'var(--green)');
      html+=`<div class="sprow" style="padding-left:${depth*16}px">
        <span class="spdot" style="background:${col}"></span>
        <span class="spname">${esc(sp.name)}</span>
        ${sp.duration_ms?`<span class="spbar ${cls}" style="width:${w.toFixed(0)}px"></span>`:''}
        <span class="spdur">${sp.status==='running'?'running&hellip;':(sp.duration_ms!=null?fmtMs(sp.duration_ms):'')}</span></div>`;
      (evsBySpan[id]||[]).forEach(e=>{
        html+=`<div class="spev" style="padding-left:${depth*16+15}px">#${e.seq} ${esc(e.kind)} &middot; ${esc(e.actor)}</div>`});
      (kids[id]||[]).forEach(c=>draw(c,depth+1));
    };
    roots.forEach(r=>draw(r,0));
    pane.innerHTML=`<div class="subhead">span tree${atHead?'':' @ seq '+v}</div>`+html;
  }else if(tab==='usage'){
    const u=await j('/api/usage'+q)||{};
    const models=Object.entries(u.by_model||{}),tools=Object.entries(u.tools||{});
    const total=u.cost_usd||0;
    pane.innerHTML=
      `<div class="cost">${fmt$(total)}<small> total spend${atHead?'':' @ seq '+v}</small></div>`+
      sparkline()+
      `<div class="stat"><span>LLM calls</span><b>${u.llm_calls||0}</b></div>`+
      `<div class="stat"><span>Tokens in / out</span><b>${(u.input_tokens||0).toLocaleString()} / ${(u.output_tokens||0).toLocaleString()}</b></div>`+
      (u.uncosted_calls?`<div class="stat warnrow"><span>uncosted calls (no pricing registered)</span><b>${u.uncosted_calls}</b></div>`:'')+
      (models.length?`<div class="subhead">By model</div>`+models.map(([m,d])=>
        `<div class="stat"><span>${esc(m)}</span><b>${fmt$(d.cost_usd)} &middot; ${d.calls}&times;</b></div>
         <div class="bar"><i style="width:${total?Math.round(100*(d.cost_usd||0)/total):0}%"></i></div>`).join(''):'')+
      (tools.length?`<div class="subhead">Tools</div>`+tools.map(([n,t])=>
        `<div class="stat"><span>${esc(n)}</span><b>${t.calls}&times; &middot; ${fmtMs(t.latency_ms_total||0)}${t.errors?` <span class="err">&#9888; ${t.errors}</span>`:''}</b></div>`).join(''):'');
    sparkAnim=false;
  }else if(tab==='memories'){
    const mems=await j('/api/memories?stream='+encodeURIComponent(cur));
    pane.innerHTML=mems.length?mems.map(m=>
      `<div class="memcard"><div class="txt">${esc(m.text)}</div>
       <div class="meta"><span class="kind k-memory_write">${esc(m.kind||'semantic')}</span> <span class="kind">${esc(m.level||'session')}</span>${Object.keys(m.meta||{}).length?' &middot; '+esc(JSON.stringify(m.meta)):''}</div></div>`).join('')
      :'<div class="empty">no memories for this session</div>';
  }
}

function stopPlay(){if(playing){clearInterval(playing);playing=null;$('play').innerHTML='&#9654;';$('play').classList.remove('playing')}}
$('play').onclick=()=>{
  if(playing){stopPlay();return}
  const sl=$('slider');
  if(+sl.value>=head)sl.value=1;
  prevLines=null;setTab('state');
  $('play').innerHTML='&#10074;&#10074;';$('play').classList.add('playing');
  playing=setInterval(()=>{
    if(+sl.value>=head){stopPlay();return}
    sl.value=+sl.value+1;refresh();
  },650);
  refresh();
};
$('slider').oninput=()=>{stopPlay();refresh()};
document.addEventListener('keydown',e=>{
  if(!cur)return;
  const sl=$('slider');
  if(e.key==='ArrowLeft'){sl.value=Math.max(1,+sl.value-1);stopPlay();refresh()}
  else if(e.key==='ArrowRight'){sl.value=Math.min(head,+sl.value+1);stopPlay();refresh()}
  else if(e.key==='End'){sl.value=head;stopPlay();refresh()}
  else if(e.key===' '){e.preventDefault();$('play').click()}
});

// ---------- pages ----------
let page='home', allMems=[];
function route(){
  page=(location.hash.replace('#/','')||'home');
  if(!['home','console','costs','tools','memories'].includes(page))page='home';
  document.body.classList.toggle('home',page==='home');
  document.querySelectorAll('.navtab').forEach(t=>t.classList.toggle('on',t.dataset.p===page));
  moveInd(document.querySelector('.navtab.on'));
  $('page-console').classList.toggle('on',page==='console');
  ['home','costs','tools','memories'].forEach(p=>$('page-'+p).classList.toggle('on',page===p));
  if(page==='console'&&!cur){
    const first=document.querySelector('table.streams tbody tr[data-s]');
    if(first)pick(first.dataset.s);
  }
  if(page==='costs')renderCosts();
  if(page==='tools')renderTools();
  if(page==='memories')renderMemsAll();
}
window.addEventListener('hashchange',route);

const navind=$('navind');
function moveInd(el){
  if(!el){navind.style.opacity=0;return}
  navind.style.opacity=1;
  navind.style.left=el.offsetLeft+'px';
  navind.style.width=el.offsetWidth+'px';
}
document.querySelectorAll('.navtab').forEach(tb=>
  tb.addEventListener('mouseenter',()=>moveInd(tb)));
document.querySelector('nav').addEventListener('mouseleave',()=>
  moveInd(document.querySelector('.navtab.on')));
window.addEventListener('resize',()=>moveInd(document.querySelector('.navtab.on')));
window.addEventListener('load',()=>moveInd(document.querySelector('.navtab.on')));

async function renderCosts(){
  const c=await j('/api/costs');
  const t=c.totals,models=Object.entries(c.by_model),agents=Object.entries(c.by_agent);
  const maxM=Math.max(...models.map(([,d])=>d.cost_usd),1e-9);
  const maxA=Math.max(...agents.map(([,d])=>d.cost_usd),1e-9);
  $('costsbody').innerHTML=
    `<div class="panel"><h3>By model</h3>`+(models.length?models.map(([m,d])=>
      `<div class="hbar"><span class="n">${esc(m)}</span>
       <div class="htrack"><i style="width:${Math.max(100*d.cost_usd/maxM,2)}%"></i></div>
       <span class="v">${fmt$(d.cost_usd)}</span></div>
      <div class="hbar" style="padding-top:0"><span class="n" style="color:var(--faint);font-size:10px">${d.calls} calls &middot; ${(d.input_tokens+d.output_tokens).toLocaleString()} tok</span><span></span><span></span></div>`).join('')
      :'<div class="empty">no llm calls yet</div>')+`</div>`+
    `<div class="panel"><h3>By agent</h3>`+(agents.length?agents.map(([a,d])=>
      `<div class="hbar"><span class="n">${esc(a)}</span>
       <div class="htrack"><i class="g" style="width:${Math.max(100*d.cost_usd/maxA,2)}%"></i></div>
       <span class="v">${fmt$(d.cost_usd)}</span></div>`).join('')
      :'<div class="empty">&mdash;</div>')+
      `<div class="subhead" style="margin-top:16px">Totals</div>
       <div class="stat"><span>Total spend</span><b>${fmt$(t.cost_usd)}</b></div>
       <div class="stat"><span>LLM calls</span><b>${t.llm_calls}</b></div>
       <div class="stat"><span>Tokens</span><b>${t.tokens.toLocaleString()}</b></div>`+
      (t.uncosted_calls?`<div class="stat warnrow"><span>uncosted calls</span><b>${t.uncosted_calls}</b></div>`:'')+`</div>`+
    `<div class="panel wide"><h3>By session</h3><table class="data">
      <thead><tr><th>session</th><th>llm calls</th><th>uncosted</th><th>cost</th></tr></thead><tbody>`+
      c.streams.map(s=>`<tr><td>${esc(s.stream)}</td><td>${s.llm_calls}</td>
        <td>${s.uncosted||'&mdash;'}</td><td>${fmt$(s.cost_usd)}</td></tr>`).join('')+
      `</tbody></table></div>`;
}

async function renderTools(){
  const rows=await j('/api/tools');
  $('toolsbody').innerHTML=rows.length?
    `<div class="panel" style="max-width:1100px"><h3>Tool performance &mdash; all streams</h3>
     <table class="data"><thead><tr><th>tool</th><th>calls</th><th>streams</th>
       <th>errors</th><th>error rate</th><th>avg latency</th><th>total latency</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td>${esc(r.tool)}</td><td>${r.calls}</td><td>${r.streams}</td>
      <td>${r.errors||'&mdash;'}</td>
      <td><span class="pill ${r.err_rate>0?'bad':'ok'}">${(r.err_rate*100).toFixed(0)}%</span></td>
      <td>${fmtMs(r.avg_ms)}</td><td>${fmtMs(r.latency_ms_total)}</td></tr>`).join('')+
    `</tbody></table></div>`
    :'<div class="empty">no tool calls recorded yet</div>';
}

async function renderMemsAll(){
  allMems=await j('/api/memories_all');
  drawMems();
}
function drawMems(){
  const q=($('memsearch').value||'').toLowerCase();
  const list=allMems.filter(m=>!q||m.text.toLowerCase().includes(q)||m.session.toLowerCase().includes(q));
  $('memsbody').innerHTML=list.length?list.map(m=>
    `<div class="memcard"><div class="txt">${esc(m.text)}</div>
     <div class="meta"><span class="kind k-memory_write">${esc(m.kind||'semantic')}</span> <span class="kind">${esc(m.level||'session')}</span> &middot; ${esc(m.session)}${Object.keys(m.meta||{}).length?' &middot; '+esc(JSON.stringify(m.meta)):''}</div></div>`).join('')
    :'<div class="empty">no memories'+(q?' matching "'+esc(q)+'"':'')+'</div>';
}
$('memsearch').oninput=drawMems;

const io=new IntersectionObserver(entries=>entries.forEach(x=>{
  if(x.isIntersecting){x.target.classList.add('vis');io.unobserve(x.target)}
}),{threshold:.12});
document.querySelectorAll('.reveal').forEach((el,i)=>{
  el.style.transitionDelay=((i%3)*80)+'ms';io.observe(el)});
// fallbacks: container-scroll check (zero-height-viewport iframes) + a hard
// reveal-all guarantee so content can never stay hidden anywhere
function checkReveals(){
  const home=$('page-home');
  const vh=innerHeight||document.documentElement.clientHeight||home.clientHeight||800;
  document.querySelectorAll('.reveal:not(.vis)').forEach(el=>{
    if(el.getBoundingClientRect().top<vh*.94+home.scrollTop*0)el.classList.add('vis');
    else if(el.getBoundingClientRect().top-home.scrollTop<vh*.94)el.classList.add('vis');
  });
}
$('page-home').addEventListener('scroll',checkReveals,{passive:true});
setTimeout(()=>document.querySelectorAll('.reveal:not(.vis)').forEach(el=>el.classList.add('vis')),3000);

route();
loadSummary();
setInterval(async()=>{
  await loadSummary();
  if(page==='costs')renderCosts();
  if(page==='tools')renderTools();
  $('livelabel').textContent=new Date().toLocaleTimeString();
},4000);
</script></body></html>"""


def build_app(store: StateStore) -> Starlette:
    global _store
    _store = store
    return Starlette(routes=[
        Route("/", index),
        Route("/api/streams", api_streams),
        Route("/api/summary", api_summary),
        Route("/api/events", api_events),
        Route("/api/state", api_state),
        Route("/api/usage", api_usage),
        Route("/api/series", api_series),
        Route("/api/memories", api_memories),
        Route("/api/costs", api_costs),
        Route("/api/tools", api_tools),
        Route("/api/memories_all", api_memories_all),
        Route("/api/promptfoo", api_promptfoo),
        Route("/api/verify", api_verify),
        Route("/api/tail", api_tail),
    ])


async def _seed_demo(store: StateStore) -> None:
    """Seed a realistic multi-agent workload so the console looks like prod.

    Deterministic (seeded RNG): ~10 sessions across 4 agents over the last
    8 hours, mixed models (one deliberately unpriced), flaky tools, and
    long-term memories.
    """
    import random
    from datetime import datetime, timedelta, timezone

    from .telemetry import register_pricing
    from .types import Event, Scope

    register_pricing("claude-sonnet-5", 3.00, 15.00)
    register_pricing("claude-haiku-4-5", 0.80, 4.00)

    rng = random.Random(42)
    now = datetime.now(timezone.utc)

    AGENTS = {
        "researcher": dict(
            sessions=["hotel-booking", "flight-search", "market-analysis"],
            asks=["find me a boutique hotel in paris under 300/night",
                  "compare flights LHR to JFK next tuesday",
                  "summarize the EU AI act changes for our compliance doc"],
            tools=["search_web", "search_hotels", "fetch_page", "book_hotel"],
            replies=["I found 3 strong options; Hotel Lutetia fits best.",
                     "BA 117 at 09:40 is cheapest with one stop.",
                     "Draft summary ready — 6 obligations apply to us."],
            stages=["searching", "reviewing", "drafting", "done"]),
        "support-bot": dict(
            sessions=["ticket-4821", "ticket-4977", "ticket-5104"],
            asks=["my invoice from march is wrong",
                  "cannot reset my password, link expired",
                  "the export to csv times out for large projects"],
            tools=["crm_lookup", "issue_refund", "reset_password", "search_kb"],
            replies=["The duplicate March invoice has been voided.",
                     "A fresh reset link is on its way to you.",
                     "Escalated — engineering confirmed a fix for Friday."],
            stages=["triaging", "investigating", "escalated", "resolved"]),
        "code-assistant": dict(
            sessions=["pr-review-1182", "flaky-test-hunt"],
            asks=["review PR #1182 for the auth refactor",
                  "find why test_checkout is flaky on CI"],
            tools=["read_file", "run_tests", "grep_repo", "post_comment"],
            replies=["Left 4 comments; the token-refresh race is the blocker.",
                     "Root cause: shared fixture mutates a module global."],
            stages=["reading", "testing", "reviewing", "done"]),
        "outreach": dict(
            sessions=["lead-gen-batch7", "followup-wave2"],
            asks=["build a lead list for fintech CTOs in nordics",
                  "send follow-ups to last week's warm replies"],
            tools=["fetch_leads", "enrich_contact", "send_email"],
            replies=["42 qualified leads exported to the CRM.",
                     "18 follow-ups sent; 3 bounced and were flagged."],
            stages=["collecting", "enriching", "sending", "done"]),
    }
    MODELS = [("claude-sonnet-5", 0.55), ("claude-haiku-4-5", 0.35), ("gpt-x-preview", 0.10)]

    def pick_model():
        r, acc = rng.random(), 0.0
        for m, w in MODELS:
            acc += w
            if r <= acc:
                return m
        return MODELS[0][0]

    for agent, cfg in AGENTS.items():
        for si, session in enumerate(cfg["sessions"]):
            t = now - timedelta(hours=rng.uniform(0.4, 8.0))
            events: list[Event] = []

            def emit(kind, payload, actor, ms=0.0, causation=None):
                nonlocal t
                t += timedelta(milliseconds=ms + rng.uniform(800, 9000))
                events.append(Event(kind=kind, payload=payload, actor=actor,
                                    ts=t.isoformat(), causation_id=causation))

            ask = cfg["asks"][si % len(cfg["asks"])]
            from .types import new_id as _nid
            run_span = _nid()
            emit("span_start", {"span_id": run_span, "parent_id": None,
                                "name": f"run:{session}", "attrs": {}}, "trace")
            emit("message", {"role": "user", "content": ask}, "user",
                 causation=run_span)
            rounds = rng.randint(2, 4)
            run_t0 = t
            for r in range(rounds):
                rspan = _nid()
                emit("span_start", {"span_id": rspan, "parent_id": run_span,
                                    "name": f"round-{r + 1}", "attrs": {}}, "trace")
                round_t0 = t
                model = pick_model()
                lat = rng.uniform(400, 2600)
                emit("llm_call", {"model": model,
                                  "input_tokens": rng.randint(700, 5200),
                                  "output_tokens": rng.randint(90, 1100),
                                  "latency_ms": round(lat, 1)},
                     f"llm:{model}", lat, causation=rspan)
                round_err = None
                for _ in range(rng.randint(1, 2)):
                    tool = rng.choice(cfg["tools"])
                    fail = rng.random() < 0.09
                    if fail:
                        round_err = "Timeout after 5s"
                    tlat = rng.uniform(3000, 8000) if fail else rng.uniform(90, 1600)
                    emit("tool_call",
                         {"name": tool, "args": {"q": ask.split()[0], "round": r},
                          "result": None if fail else "ok",
                          "latency_ms": round(tlat, 1),
                          "error": "Timeout after 5s" if fail else None},
                         f"tool:{tool}", tlat, causation=rspan)
                emit("state_delta",
                     {"set": {"stage": cfg["stages"][min(r, len(cfg["stages"]) - 1)]}},
                     "agent", causation=rspan)
                emit("span_end", {"span_id": rspan,
                                  "status": "error" if round_err else "ok",
                                  "error": round_err,
                                  "duration_ms": round((t - round_t0).total_seconds() * 1000, 1)},
                     "trace", causation=run_span)
            reply = cfg["replies"][si % len(cfg["replies"])]
            emit("message", {"role": "assistant", "content": reply}, "agent",
                 causation=run_span)
            emit("state_delta", {"set": {"stage": cfg["stages"][-1]}}, "agent",
                 causation=run_span)
            emit("span_end", {"span_id": run_span, "status": "ok",
                              "duration_ms": round((t - run_t0).total_seconds() * 1000, 1)},
                 "trace")

            scope = Scope("acme", agent, session)
            await store.append(scope, events, expected_seq=0)
            if rng.random() < 0.7:
                await store.remember(scope, f"context from {session}: {ask[:60]}",
                                     {"agent": agent})
            if rng.random() < 0.5:
                await store.remember(
                    scope, f"{session}: {reply[:70]}",
                    {"outcome": "success", "stream": scope.flatten()},
                    kind="episodic", level="agent")
        await store.remember(
            Scope("acme", agent, "_"), f"{agent} playbook: {cfg['asks'][0][:50]}",
            {"source": "learned"}, kind="procedural", level="agent")


def main() -> None:
    import asyncio

    import uvicorn

    dsn = os.getenv("STATEFOLD_DSN")
    if dsn:
        from .postgres import PostgresStore

        store = asyncio.run(PostgresStore.connect(dsn))
    else:
        from .memory import InMemoryStore

        store = InMemoryStore()
        asyncio.run(_seed_demo(store))
        print("no STATEFOLD_DSN set -> in-memory store with demo data")

    host = os.getenv("STATEFOLD_UI_HOST", "127.0.0.1")
    port = int(os.getenv("STATEFOLD_UI_PORT", "8787"))
    print(f"statefold ui -> http://{host}:{port}")
    uvicorn.run(build_app(store), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

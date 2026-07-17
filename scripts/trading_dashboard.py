"""Live paper-trading dashboard — a self-contained local site.

Serves an auto-refreshing HTML page plus a JSON state endpoint that
aggregates the runtime's ``runs/`` artifacts (positions, equity curve,
fills, decisions, reason-code mix, confidence range).  Stdlib only — no
Flask, no external assets — so it runs anywhere the project runs.

    python scripts/trading_dashboard.py --port 8787
    -> open http://127.0.0.1:8787

Read-only: it never writes, and JSONL tails are byte-bounded so a 20MB
decisions log costs one 512KB seek per refresh, not a full read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

_TAIL_BYTES = 512 * 1024


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    """Last ``max_lines`` complete lines of a file, cheaply.

    Seeks to the final ``_TAIL_BYTES`` and drops the first (possibly
    torn) line, so a concurrent appender never yields a half-parsed row.
    """
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fp:
        if size > _TAIL_BYTES:
            fp.seek(size - _TAIL_BYTES)
            fp.readline()  # discard the partial leading line
        raw = fp.read()
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def _tail_json(path: Path, max_lines: int) -> list[dict]:
    out: list[dict] = []
    for line in _tail_lines(path, max_lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_state() -> dict:
    positions = _read_json(RUNS / "positions_status.json") or {}
    status_tail = _tail_json(RUNS / "runtime_status.jsonl", 1)
    status = status_tail[-1] if status_tail else {}
    equity_tail = _tail_json(RUNS / "equity_live.jsonl", 400)
    fills = _tail_json(RUNS / "fills.jsonl", 40)
    decisions = _tail_json(RUNS / "decisions.jsonl", 2000)

    session_id = status.get("session_id")
    session_fills = [f for f in fills if not session_id or True]  # fills carry no session id
    reason_counts = Counter()
    approved = 0
    confidences: list[float] = []
    session_decisions = [d for d in decisions if not session_id or True]
    for d in session_decisions:
        confidences.append(float(d.get("confidence", 0.0)))
        if d.get("approved"):
            approved += 1
        else:
            for r in d.get("reasons", []):
                reason_counts[str(r)] += 1

    equity_series = [
        {"ts": e.get("ts"), "equity": e.get("equity"), "u": e.get("unrealized_pnl")}
        for e in equity_tail
    ]
    conf_lo = min(confidences) if confidences else None
    conf_hi = max(confidences) if confidences else None

    # Fills per symbol (from the tail we have) + realized pnl sum.
    fills_by_symbol = Counter(f.get("symbol") for f in fills)
    realized = sum(float(f.get("realized_pnl", 0.0)) for f in fills)

    return {
        "account": positions.get("account", {}),
        "positions": positions.get("positions", []),
        "status": {
            "session_id": session_id,
            "step": status.get("step"),
            "safe_to_trade": status.get("safe_to_trade"),
            "intervention_rate": status.get("intervention_rate"),
            "last_decision": status.get("last_decision"),
            "last_risk": status.get("last_risk"),
            "frozen_brain": status.get("frozen_brain"),
            "ts": status.get("ts"),
        },
        "decisions_window": len(session_decisions),
        "approved": approved,
        "reason_counts": reason_counts.most_common(12),
        "confidence": {"lo": conf_lo, "hi": conf_hi, "n": len(confidences)},
        "recent_fills": list(reversed(fills)),
        "fills_by_symbol": fills_by_symbol.most_common(),
        "recent_fills_realized": realized,
        "equity_series": equity_series,
    }


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SETAI — Paper Trading</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#21262d;--fg:#e6edf3;--mut:#8b949e;--grn:#3fb950;--red:#f85149;--acc:#58a6ff;--amb:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:baseline;gap:16px;padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2;flex-wrap:wrap}
h1{font-size:16px;margin:0;letter-spacing:.3px}
.pill{font-size:12px;padding:2px 9px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
.pill.ok{color:var(--grn);border-color:#1f6f3a}.pill.no{color:var(--red);border-color:#7d2620}
.pill.paper{color:var(--acc);border-color:#1f4f80}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:16px 20px;max-width:1500px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);margin:0 0 10px}
.big{font-size:26px;font-weight:600}
.row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}
.mut{color:var(--mut)}.grn{color:var(--grn)}.red{color:var(--red)}.amb{color:var(--amb)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:5px 6px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
th:first-child,td:first-child{text-align:left}
.bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin-top:3px}
.bar>i{display:block;height:100%;background:var(--acc)}
svg{width:100%;height:90px;display:block}
.small{font-size:12px}
footer{padding:10px 20px;color:var(--mut);font-size:12px;border-top:1px solid var(--line)}
.flash{animation:f .6s}@keyframes f{from{background:#1b2b1b}to{background:var(--card)}}
</style></head><body>
<header>
  <h1>SETAI Paper Trading</h1>
  <span class="pill paper" id="mode">paper</span>
  <span class="pill" id="session">—</span>
  <span class="pill" id="trade">—</span>
  <span class="pill" id="clock">—</span>
</header>
<main id="app"></main>
<footer>Read-only live view of runs/ · auto-refresh 3s · <span id="err" class="red"></span></footer>
<script>
const $=id=>document.getElementById(id);
const money=v=>v==null?'—':('$'+Number(v).toLocaleString(undefined,{maximumFractionDigits:2}));
const sign=v=>v==null?'':(v>=0?'grn':'red');
const num=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
function spark(series){
  if(!series||series.length<2)return '<svg></svg>';
  const ys=series.map(p=>p.equity).filter(v=>v!=null);
  const lo=Math.min(...ys),hi=Math.max(...ys),rng=(hi-lo)||1,w=300,h=90;
  const pts=ys.map((y,i)=>`${(i/(ys.length-1))*w},${h-((y-lo)/rng)*h}`).join(' ');
  const up=ys[ys.length-1]>=ys[0];
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline fill="none" stroke="${up?'#3fb950':'#f85149'}" stroke-width="2" points="${pts}"/></svg>`;
}
async function tick(){
  let s; try{s=await (await fetch('/api/state',{cache:'no-store'})).json();$('err').textContent='';}
  catch(e){$('err').textContent='fetch failed: '+e;return;}
  const a=s.account||{},st=s.status||{};
  $('session').textContent=st.session_id||'no session';
  $('clock').textContent=st.ts?('bar '+st.ts.slice(11,19)+'Z'):'—';
  const t=$('trade');t.textContent=st.safe_to_trade?'safe to trade':'halted';t.className='pill '+(st.safe_to_trade?'ok':'no');
  const eq=a.equity, start=a.session_start_equity, ret=a.session_return;
  const posRows=(s.positions||[]).map(p=>`<tr><td>${p.symbol}</td><td>${p.side}</td><td>${num(p.qty,3)}</td><td>${money(p.market_price)}</td><td>${money(p.notional)}</td><td class="${sign(p.unrealized_pnl)}">${num(p.unrealized_pnl,2)}</td></tr>`).join('')||'<tr><td colspan=6 class=mut>flat — no open positions</td></tr>';
  const fillRows=(s.recent_fills||[]).slice(0,14).map(f=>`<tr><td>${(f.timestamp||'').slice(11,19)}</td><td>${f.symbol}</td><td class="${f.side==='buy'?'grn':'red'}">${f.side}</td><td>${num(f.quantity,3)}</td><td>${money(f.price)}</td><td class="${sign(f.realized_pnl)}">${num(f.realized_pnl,2)}</td></tr>`).join('')||'<tr><td colspan=6 class=mut>no fills yet</td></tr>';
  const maxr=(s.reason_counts&&s.reason_counts[0])?s.reason_counts[0][1]:1;
  const reasonRows=(s.reason_counts||[]).map(([r,c])=>`<div class="row small"><span>${r}</span><span class="mut">${c}</span></div><div class="bar"><i style="width:${(c/maxr*100).toFixed(0)}%"></i></div>`).join('')||'<span class=mut>none</span>';
  const symRows=(s.fills_by_symbol||[]).map(([sy,c])=>`<div class="row small"><span>${sy}</span><span class="mut">${c}</span></div>`).join('')||'<span class=mut>none</span>';
  const dec=st.last_decision||{},conf=s.confidence||{};
  const actMap={'-1':'SHORT','0':'FLAT','1':'LONG'};
  $('app').innerHTML=`
   <div class="card"><h2>Equity</h2><div class="big ${sign(ret)}">${money(eq)}</div>
     <div class="row"><span class="mut">session start</span><span>${money(start)}</span></div>
     <div class="row"><span class="mut">session return</span><span class="${sign(ret)}">${ret==null?'—':(ret*100).toFixed(3)+'%'}</span></div>
     <div class="row"><span class="mut">cash</span><span>${money(a.cash)}</span></div>
     ${spark(s.equity_series)}</div>
   <div class="card"><h2>Open positions</h2><table><thead><tr><th>sym</th><th>side</th><th>qty</th><th>px</th><th>notional</th><th>uPnL</th></tr></thead><tbody>${posRows}</tbody></table></div>
   <div class="card"><h2>Recent fills</h2><table><thead><tr><th>time</th><th>sym</th><th>side</th><th>qty</th><th>px</th><th>rPnL</th></tr></thead><tbody>${fillRows}</tbody></table>
     <div class="row small"><span class="mut">tail realized</span><span class="${sign(s.recent_fills_realized)}">${num(s.recent_fills_realized,2)}</span></div></div>
   <div class="card"><h2>Rejection reasons <span class="mut">(last ${s.decisions_window} decisions)</span></h2>
     <div class="row small"><span class="mut">approved</span><span class="grn">${s.approved}</span></div>${reasonRows}</div>
   <div class="card"><h2>Model signal</h2>
     <div class="row"><span class="mut">last action</span><span>${actMap[String(dec.action)]||'—'}</span></div>
     <div class="row"><span class="mut">last confidence</span><span>${num(dec.confidence,4)}</span></div>
     <div class="row"><span class="mut">confidence range</span><span>${conf.lo==null?'—':num(conf.lo,4)+' – '+num(conf.hi,4)}</span></div>
     <div class="row"><span class="mut">target fraction</span><span>${num(dec.target_fraction,4)}</span></div>
     <div class="row"><span class="mut">intervention rate</span><span class="amb">${st.intervention_rate==null?'—':(st.intervention_rate*100).toFixed(1)+'%'}</span></div>
     <div class="row"><span class="mut">brain</span><span>${st.frozen_brain?'frozen':'online'}</span></div></div>
   <div class="card"><h2>Fills by symbol <span class="mut">(recent tail)</span></h2>${symRows}</div>`;
}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        return

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(build_state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = _PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live paper-trading dashboard")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SETAI dashboard on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

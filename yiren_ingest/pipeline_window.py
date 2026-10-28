"""
Yiren 流水线 Web 仪表盘 — 浏览器中打开，实时展示两阶段进度。

设计方向: "Precision Instrument" — 高级灰蓝底 + 卡片化布局 + 磷光强调色，
          圆角卡片 + 质感进度条 + 微光 Badge + 低对比度网格纹理。

架构:
  - 本地 HTTP server (localhost 随机端口)，serve_forever 在守护线程
  - SSE (/events) 实时推送进度事件
  - 内嵌 HTML/CSS/JS 前端（零外部依赖）
  - 管线后台线程运行，主线程等待管线完成
"""
import os
import sys
import json
import queue
import threading
import time
import socket
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── task state machine ─────────────────────────────────────────────────
_ST_WAIT = 0
_ST_ASR  = 1
_ST_REF  = 2
_ST_DONE = 3
_ST_FAIL = 4

_STATE_LABEL = {_ST_WAIT: "WAIT", _ST_ASR: "ASR", _ST_REF: "REFINE",
                _ST_DONE: "DONE", _ST_FAIL: "FAIL"}


class _TaskState:
    __slots__ = ("name", "asr_pct", "refine_pct", "status",
                 "done_time", "exit_alpha", "error",
                 "asr_start_time",   # ASR 真实启动时刻（用于平滑插值）
                 "asr_done_time",    # ASR 真实完成时刻（用于统计平均耗时）
                 "refine_start_time")  # 后处理启动时刻（用于平滑插值）
    def __init__(self, name: str):
        self.name = name[:56]
        self.asr_pct = 0.0
        self.refine_pct = 0.0
        self.status = _ST_WAIT
        self.error = ""
        self.refine_start_time = 0.0
        self.done_time = 0.0
        self.exit_alpha = 1.0
        self.asr_start_time = 0.0
        self.asr_done_time = 0.0


def _pick_port() -> int:
    """获取一个可用的 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── HTML template ──────────────────────────────────────────────────────
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yiren Pipeline Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════════════════════════
   DESIGN TOKENS
   ═══════════════════════════════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-root:      #FFFFFF;
  --bg-surface:   #F8F9FA;
  --bg-card:      #FFFFFF;
  --bg-card-hover:#F5F6F8;
  --border:       #E5E6E8;
  --border-light: #EEEEF0;

  --cyan:         #1A8FBA;
  --cyan-bg:      #1A8FBA12;
  --cyan-border:  #1A8FBA33;
  --amber:        #C4922A;
  --amber-bg:     #C4922A12;
  --amber-border: #C4922A33;
  --green:        #4A9E3E;
  --green-bg:     #4A9E3E12;
  --green-border: #4A9E3E33;
  --red:          #D94444;
  --red-bg:       #D9444412;

  --text-primary:   #1A1A1A;
  --text-secondary: #5A5D63;
  --text-dim:       #9A9DA3;
  --text-subtle:    #C5C8CC;

  --font-mono: "JetBrains Mono","Cascadia Code","Fira Code","Consolas",monospace;
  --font-sans: system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;

  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-pill: 999px;
}
html,body{height:100%;overflow:hidden}
body{
  font-family:var(--font-sans);
  background:var(--bg-root);
  color:var(--text-secondary);
  display:flex;flex-direction:column;
  -webkit-font-smoothing:antialiased;
}
/* grid-dot texture — very subtle, barely visible */
body::before{
  content:"";
  position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:radial-gradient(circle,rgba(0,0,0,0.025) 0.5px,transparent 0.5px);
  background-size:18px 18px;
}

/* ═══════════════════════════════════════════════════════════════════════
   HEADER — summary stats bar
   ═══════════════════════════════════════════════════════════════════════ */
.header{
  position:relative;z-index:1;
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;
  background:var(--bg-surface);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.header-left{display:flex;align-items:baseline;gap:14px}
.header-logo{
  font-family:var(--font-mono);font-size:16px;font-weight:600;
  letter-spacing:0.06em;color:var(--text-primary);
}
.header-count{
  font-family:var(--font-mono);font-size:14px;color:var(--text-dim);
}
.header-right{display:flex;align-items:center;gap:18px}

/* stat pills in header */
.stat-group{display:flex;align-items:center;gap:6px}
.stat-dot{
  width:7px;height:7px;border-radius:50%;flex-shrink:0;
}
.stat-dot.cyan{background:var(--cyan);box-shadow:0 0 6px #1A8FBA44}
.stat-dot.amber{background:var(--amber);box-shadow:0 0 6px #C4922A44}
.stat-dot.green{background:var(--green);box-shadow:0 0 6px #4A9E3E44}
.stat-label{
  font-family:var(--font-mono);font-size:12px;color:var(--text-dim);
  letter-spacing:0.04em;
}
.stat-value{
  font-family:var(--font-mono);font-size:16px;font-weight:600;color:var(--text-primary);
}

/* phase badge */
.phase-badge{
  font-family:var(--font-mono);font-size:12px;font-weight:500;
  letter-spacing:0.05em;
  padding:4px 14px;border-radius:var(--radius-pill);
}
.phase-badge.asr{color:var(--cyan);background:var(--cyan-bg);border:1px solid var(--cyan-border)}
.phase-badge.mix{color:var(--amber);background:var(--amber-bg);border:1px solid var(--amber-border)}
.phase-badge.done{color:var(--green);background:var(--green-bg);border:1px solid var(--green-border)}

/* overall mini-bar in header */
.overall-wrap{display:flex;align-items:center;gap:8px}
.overall-track{
  width:140px;height:6px;border-radius:10px;
  background:rgba(0,0,0,0.06);overflow:hidden;
}
.overall-fill{
  height:100%;border-radius:10px;transition:width 0.4s ease;
  background:var(--cyan);
}
.overall-fill.complete{background:var(--green)}
.overall-pct{
  font-family:var(--font-mono);font-size:14px;font-weight:500;
  color:var(--text-secondary);width:42px;text-align:right;
}

/* ═══════════════════════════════════════════════════════════════════════
   TASK CARD LIST
   ═══════════════════════════════════════════════════════════════════════ */
.task-scroll{
  position:relative;z-index:1;
  flex:1;overflow-y:auto;overflow-x:hidden;
  padding:12px 24px;
}
.task-scroll::-webkit-scrollbar{width:4px}
.task-scroll::-webkit-scrollbar-track{background:transparent}
.task-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* card */
.task-card{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:12px 16px;
  margin-bottom:8px;
  transition:opacity 0.5s ease,transform 0.4s ease,max-height 0.5s ease,padding 0.4s ease,margin 0.4s ease,border-color 0.3s ease;
  overflow:hidden;
  max-height:120px;
}
.task-card:hover{border-color:var(--border-light)}
.task-card.fading{
  opacity:0.3;
  transform:translateX(6px);
}
.task-card.removing{
  opacity:0;
  max-height:0;
  padding-top:0;padding-bottom:0;
  margin-bottom:0;
  border-width:0;
}

/* card top row: index + name + status badge */
.card-top{
  display:flex;align-items:center;gap:10px;margin-bottom:8px;
}
.task-index{
  font-family:var(--font-mono);font-size:13px;font-weight:500;
  color:var(--text-dim);width:28px;flex-shrink:0;
}
.task-name{
  font-family:var(--font-sans);font-size:16px;font-weight:600;
  color:var(--text-primary);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.task-name.dim{color:var(--text-dim);font-weight:400}

/* status badge — pill */
.status-badge{
  font-family:var(--font-mono);font-size:11px;font-weight:600;
  letter-spacing:0.06em;
  padding:3px 12px;border-radius:var(--radius-pill);
  flex-shrink:0;text-transform:uppercase;
}
.badge-wait{color:var(--text-dim);background:rgba(0,0,0,0.04)}
.badge-asr{color:var(--cyan);background:var(--cyan-bg)}
.badge-ref{color:var(--amber);background:var(--amber-bg)}
.badge-done{color:var(--green);background:var(--green-bg)}
.badge-fail{color:var(--red);background:var(--red-bg)}

/* card bottom: progress bars */
.card-bars{display:flex;flex-direction:column;gap:6px}

.bar-row{display:flex;align-items:center;gap:8px}
.bar-label{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:0.06em;width:32px;flex-shrink:0;text-align:right;
  color:var(--text-dim);
}
.bar-track{
  flex:1;height:8px;border-radius:10px;
  background:rgba(0,0,0,0.06);
  overflow:hidden;position:relative;
}
.bar-fill{
  height:100%;border-radius:10px;transition:width 0.35s ease;
  position:relative;
}
.bar-fill::after{
  content:"";
  position:absolute;top:1px;left:4px;right:4px;height:2px;
  border-radius:2px;
  background:linear-gradient(90deg,rgba(255,255,255,0.22),rgba(255,255,255,0.04));
}
.bar-fill.asr{background:var(--cyan);box-shadow:0 0 8px #1A8FBA33}
.bar-fill.ref{background:var(--amber);box-shadow:0 0 8px #C4922A33}
.bar-fill.done{background:var(--green);box-shadow:0 0 6px #4A9E3E22}

/* sweep animation for opaque API progress */
.bar-fill.sweep{
  width:25% !important;
  animation:sweep 1.4s ease-in-out infinite;
  border-radius:10px 3px 3px 10px;
}
@keyframes sweep{
  0%{transform:translateX(-100%)}
  100%{transform:translateX(400%)}
}
.bar-pct{
  font-family:var(--font-mono);font-size:12px;width:40px;flex-shrink:0;
  text-align:left;color:var(--text-dim);
}

/* ═══════════════════════════════════════════════════════════════════════
   SPARK BURST — particle effect on task completion
   ═══════════════════════════════════════════════════════════════════════ */
.spark{
  position:fixed;width:4px;height:4px;border-radius:50%;
  pointer-events:none;z-index:200;
  background:var(--green);
  animation:spark-pop 0.7s ease-out forwards;
}
@keyframes spark-pop{
  0%{transform:translate(0,0) scale(1);opacity:1}
  100%{transform:translate(var(--sx),var(--sy)) scale(0);opacity:0}
}

/* ═══════════════════════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════════════════════ */
.footer{
  position:relative;z-index:1;
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 24px;
  background:var(--bg-surface);
  border-top:1px solid var(--border);
  flex-shrink:0;
}
.footer-elapsed{
  font-family:var(--font-mono);font-size:14px;color:var(--text-dim);
}

/* scan-line overlay — very light */
.scan-overlay{
  position:fixed;inset:0;z-index:100;pointer-events:none;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.01) 2px,
    rgba(0,0,0,0.01) 4px
  );
}
</style>
</head>
<body>

<!-- ═══════════ HEADER ═══════════ -->
<div class="header">
  <div class="header-left">
    <span class="header-logo">YIREN PIPELINE</span>
    <span class="header-count" id="headerCount"></span>
    <span class="phase-badge asr" id="phaseLabel">ASR</span>
  </div>
  <div class="header-right">
    <div class="stat-group">
      <span class="stat-dot cyan"></span>
      <span class="stat-label">ASR</span>
      <span class="stat-value" id="statAsr">0</span>
    </div>
    <div class="stat-group">
      <span class="stat-dot amber"></span>
      <span class="stat-label">REFINE</span>
      <span class="stat-value" id="statRef">0</span>
    </div>
    <div class="stat-group">
      <span class="stat-dot green"></span>
      <span class="stat-label">DONE</span>
      <span class="stat-value" id="statDone">0</span>
    </div>
    <div class="overall-wrap">
      <div class="overall-track">
        <div class="overall-fill" id="overallBar" style="width:0%"></div>
      </div>
      <span class="overall-pct" id="overallPct">0%</span>
    </div>
  </div>
</div>

<!-- ═══════════ TASK LIST ═══════════ -->
<div class="task-scroll" id="taskList"></div>

<!-- ═══════════ FOOTER ═══════════ -->
<div class="footer">
  <span class="footer-elapsed" id="elapsedLabel">0s</span>
  <span class="footer-elapsed" id="statTotalLabel"></span>
</div>

<div class="scan-overlay"></div>

<script>
// ═══════════ state ═══════════
const tasks = [];
let nTotal = 0;
let t0 = performance.now();

// ═══════════ DOM refs ═══════════
const $list = document.getElementById("taskList");
const $headerCount = document.getElementById("headerCount");
const $phaseLabel = document.getElementById("phaseLabel");
const $overallPct = document.getElementById("overallPct");
const $overallBar = document.getElementById("overallBar");
const $elapsedLabel = document.getElementById("elapsedLabel");
const $statTotalLabel = document.getElementById("statTotalLabel");
const $statDone = document.getElementById("statDone");
const $statAsr = document.getElementById("statAsr");
const $statRef = document.getElementById("statRef");

const BADGE_MAP = {WAIT:["badge-wait","WAIT"],ASR:["badge-asr","ASR"],REFINE:["badge-ref","REF"],DONE:["badge-done","DONE"],FAIL:["badge-fail","FAIL"]};

// ═══════════ build cards once ═══════════
function initTasks(names) {
  names.forEach((name, i) => {
    const t = { name: name.slice(0,56), asrPct:0, refinePct:0, status:"WAIT", fading:false, removed:false,
                _dom:{} };
    tasks.push(t);

    // ── card shell ──
    const card = document.createElement("div");
    card.className = "task-card";

    // top row
    const top = document.createElement("div"); top.className = "card-top";
    const idxEl = document.createElement("span"); idxEl.className = "task-index";
    const nameEl = document.createElement("span"); nameEl.className = "task-name dim"; nameEl.textContent = t.name;
    const badgeEl = document.createElement("span"); badgeEl.className = "status-badge badge-wait"; badgeEl.textContent = "WAIT";
    top.append(idxEl, nameEl, badgeEl);

    // bars container
    const bars = document.createElement("div"); bars.className = "card-bars";

    // ASR row
    const asrRow = document.createElement("div"); asrRow.className = "bar-row";
    const asrLbl = document.createElement("span"); asrLbl.className = "bar-label"; asrLbl.textContent = "ASR";
    const asrTrack = document.createElement("div"); asrTrack.className = "bar-track";
    const asrFill = document.createElement("div"); asrFill.className = "bar-fill asr"; asrFill.style.width = "0%";
    asrTrack.appendChild(asrFill);
    const asrPctEl = document.createElement("span"); asrPctEl.className = "bar-pct"; asrPctEl.textContent = "0%";
    asrRow.append(asrLbl, asrTrack, asrPctEl);

    // REF row
    const refRow = document.createElement("div"); refRow.className = "bar-row";
    const refLbl = document.createElement("span"); refLbl.className = "bar-label"; refLbl.textContent = "REF";
    const refTrack = document.createElement("div"); refTrack.className = "bar-track";
    const refFill = document.createElement("div"); refFill.className = "bar-fill ref"; refFill.style.width = "0%";
    refTrack.appendChild(refFill);
    const refPctEl = document.createElement("span"); refPctEl.className = "bar-pct"; refPctEl.textContent = "0%";
    refRow.append(refLbl, refTrack, refPctEl);

    bars.append(asrRow, refRow);
    card.append(top, bars);
    $list.appendChild(card);

    // stash DOM refs for in-place patching
    t._dom = {card, idxEl, nameEl, badgeEl, asrFill, asrPctEl, refFill, refPctEl};
  });
  nTotal = names.length;
  $statTotalLabel.textContent = nTotal + " files";
  _reindexCards();
}

// ═══════════ in-place card patch (no DOM destroy/create) ═══════════
function patchCard(t) {
  const d = t._dom;
  if (!d || t.removed) return;

  // ── card-level class (fading / removing) ──
  let cardCls = "task-card";
  if (t.fading) cardCls += " fading";
  if (t.removed) cardCls += " removing";
  d.card.className = cardCls;

  // ── name ──
  const active = t.status !== "DONE" && t.status !== "WAIT";
  d.nameEl.className = "task-name" + (active ? "" : " dim");
  d.nameEl.textContent = t.name;

  // ── badge ──
  const [bc, bn] = BADGE_MAP[t.status] || ["badge-wait", "?"];
  d.badgeEl.className = "status-badge " + bc;
  d.badgeEl.textContent = bn;

  // ── ASR bar ──
  const asrPct = t.asrPct;
  d.asrFill.style.width = (asrPct * 100) + "%";
  if (t.status === "DONE" || (t.status === "ASR" && asrPct >= 1)) {
    d.asrFill.className = "bar-fill done";
  } else {
    d.asrFill.className = "bar-fill asr";
  }
  d.asrPctEl.textContent = Math.round(asrPct * 100) + "%";

  // ── REF bar ──
  const refPct = t.refinePct;
  if (t.status === "REFINE" && refPct <= 0) {
    d.refFill.className = "bar-fill ref sweep";
    d.refFill.style.width = "";  // let CSS animation control it
    d.refPctEl.textContent = "...";
  } else {
    d.refFill.style.width = (refPct * 100) + "%";
    d.refPctEl.textContent = Math.round(refPct * 100) + "%";
    if (t.status === "DONE") {
      d.refFill.className = "bar-fill done";
    } else {
      d.refFill.className = "bar-fill ref";
    }
  }
}

// re-index visible cards (called when tasks get removed)
function _reindexCards() {
  let vi = 0;
  tasks.forEach(t => {
    if (!t.removed) { vi++; t._dom.idxEl.textContent = "#" + vi; }
  });
}

// ═══════════ spark particle burst ═══════════
function sparkAt(card) {
  if (!card) return;
  const rect = card.getBoundingClientRect();
  const cx = rect.right - 20;
  const cy = rect.top + rect.height / 2;
  for (let i = 0; i < 8; i++) {
    const s = document.createElement("div");
    s.className = "spark";
    const angle = (Math.PI * 2 * i) / 8 + Math.random() * 0.4;
    const dist = 18 + Math.random() * 22;
    s.style.cssText = `left:${cx}px;top:${cy}px;--sx:${Math.cos(angle)*dist}px;--sy:${Math.sin(angle)*dist}px;animation-delay:${Math.random()*0.2}s`;
    document.body.appendChild(s);
    setTimeout(() => s.remove(), 800);
  }
}

// ═══════════ elapsed timer (runs on rAF, just a text update) ═══════════
function tick() {
  const elapsed = (performance.now() - t0) / 1000;
  const mm = Math.floor(elapsed / 60);
  const ss = Math.floor(elapsed % 60);
  $elapsedLabel.textContent = mm > 0 ? mm + "m " + ss + "s" : ss + "s";
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ═══════════ SSE ═══════════
const es = new EventSource("/events");

es.addEventListener("init", e => {
  const data = JSON.parse(e.data);
  t0 = performance.now();
  initTasks(data.files);
});

es.addEventListener("tick", e => {
  const d = JSON.parse(e.data);

  // header counts
  $headerCount.textContent = d.n_total + " files";

  // phase badge
  const phase = d.phase_label;
  if (phase.includes("All done") || phase.includes("Done")) {
    $phaseLabel.textContent = "DONE";
    $phaseLabel.className = "phase-badge done";
  } else if (phase.includes("Refine") || phase.includes("ASR +")) {
    $phaseLabel.textContent = "ASR + REFINE";
    $phaseLabel.className = "phase-badge mix";
  } else {
    $phaseLabel.textContent = "ASR";
    $phaseLabel.className = "phase-badge asr";
  }

  // overall progress
  $overallPct.textContent = Math.round(d.overall_pct * 100) + "%";
  $overallBar.style.width = (d.overall_pct * 100) + "%";
  $overallBar.className = d.overall_pct >= 0.99 ? "overall-fill complete" : "overall-fill";

  // ── per-task: in-place patch instead of full re-render ──
  const tstates = d.tasks;
  let nDone = 0, nAsr = 0, nRef = 0;
  let needReindex = false;

  for (let i = 0; i < tasks.length && i < tstates.length; i++) {
    const cur = tasks[i];
    const upd = tstates[i];
    const wasDone = cur.status === "DONE";

    // update state
    cur.status = upd.status;
    cur.asrPct = upd.asr_pct;
    cur.refinePct = upd.refine_pct;
    cur.fading = upd.exit_alpha < 0.5 && upd.status === "DONE";

    if (!wasDone && cur.status === "DONE") {
      sparkAt(cur._dom.card);
    }

    // handle removal transition
    if (upd.exit_alpha < 0.02 && cur.status === "DONE" && !cur.removed) {
      cur.removed = true;
      cur._dom.card.className = "task-card removing";
      needReindex = true;
    }

    // in-place DOM update
    if (!cur.removed) patchCard(cur);

    // tally stats
    if (cur.status === "DONE") nDone++;
    if (cur.status === "ASR") nAsr++;
    if (cur.status === "REFINE") nRef++;
  }

  if (needReindex) _reindexCards();

  $statDone.textContent = nDone;
  $statAsr.textContent = nAsr;
  $statRef.textContent = nRef;
});

es.addEventListener("done", e => {
  $phaseLabel.textContent = "DONE";
  $phaseLabel.className = "phase-badge done";
  $overallPct.textContent = "100%";
  $overallBar.style.width = "100%";
  $overallBar.className = "overall-fill complete";
  tasks.forEach(t => {
    if (t.status !== "DONE" && t.status !== "FAIL") {
      t.status = "DONE"; t.asrPct = 1; t.refinePct = 1;
    }
    if (!t.removed) patchCard(t);
  });
});

es.onerror = () => {};
</script>
</body>
</html>"""


# ── HTTP server ─────────────────────────────────────────────────────────
class _DashboardHandler(BaseHTTPRequestHandler):
    """极简 HTTP handler: / 返回页面, /events SSE 流, 其余 404。"""

    def log_message(self, format, *args):
        pass  # 静默

    def do_GET(self):
        if self.path == "/":
            self._serve_page()
        elif self.path == "/events":
            self._serve_sse()
        else:
            self.send_error(404)

    def _serve_page(self):
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sse_q: queue.Queue = self.server.sse_queue
        init_data = self.server.init_payload
        if init_data:
            self._write_sse("init", init_data)

        while self.server.running:
            try:
                msg = sse_q.get(timeout=1.0)
            except queue.Empty:
                self._write_sse("tick", self.server.snapshot_payload())
                continue
            if msg is None:
                break
            event_name, payload = msg
            self._write_sse(event_name, payload)
            if event_name == "done":
                break

    def _write_sse(self, event: str, data):
        try:
            payload = json.dumps(data, ensure_ascii=False, default=str)
            self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise


# ── dashboard controller ────────────────────────────────────────────────
class PipelineWindow:
    """Web 仪表盘：启动本地 HTTP server + SSE，浏览器自动打开。

    用法:
        win = PipelineWindow(file_names)
        result = win.run(pipeline_fn, series_id=..., upload_paths=..., ...)
    """

    def __init__(self, file_names: list[str]):
        self.task_states = [_TaskState(fn) for fn in file_names]
        self.n_total = len(file_names)
        self.phase_label = "Initializing..."
        self.overall_pct = 0.0
        self.elapsed = 0.0
        self.refine_active = False
        self.phase2_done = False
        self.result = None

        self.queue: queue.Queue = queue.Queue()
        self.sse_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self.t0 = time.time()

        # 平滑插值状态
        self._asr_durations: list[float] = []     # 已完成文件的真实 ASR 耗时（秒）
        self._refine_durations: list[float] = []  # 已完成文件的后处理耗时（秒）
        self._last_smooth = 0.0                    # 上次平滑 tick 时刻

    # ── server callbacks ─────────────────────────────────────────────
    def init_payload(self) -> dict:
        return {"files": [t.name for t in self.task_states]}

    def snapshot_payload(self) -> dict:
        with self._lock:
            tasks_out = []
            for t in self.task_states:
                tasks_out.append({
                    "status": _STATE_LABEL.get(t.status, "?"),
                    "asr_pct": round(t.asr_pct, 3),
                    "refine_pct": round(t.refine_pct, 3),
                    "exit_alpha": round(t.exit_alpha, 3),
                    "error": t.error[:120] if t.error else "",
                })
            return {
                "n_total": self.n_total,
                "phase_label": self.phase_label,
                "overall_pct": round(self.overall_pct, 4),
                "tasks": tasks_out,
            }

    # ── progress callback (pipeline thread → queue) ─────────────────
    def on_progress(self, stage: str, progress: Optional[float], **meta):
        self.queue.put(("progress", stage, progress, meta))

    # ── event dispatch (consumer thread) ─────────────────────────────
    def _dispatch(self, stage: str, progress: Optional[float], meta: dict):
        now = time.time()
        # 并发模式：orchestrator 在 meta 中传入 file_index（0-indexed）精确定位任务
        file_idx = meta.get("file_index")

        with self._lock:
            if progress is not None:
                self.overall_pct = progress

            # ── refine_started：用 file_index 精确定位 ──
            if stage == "refine_started":
                self.refine_active = True
                if file_idx is not None and 0 <= file_idx < self.n_total:
                    t = self.task_states[file_idx]
                    if t.asr_pct >= 0.84 and t.status in (_ST_WAIT, _ST_ASR):
                        t.status = _ST_REF
                        t.asr_pct = 1.0         # ASR 阶段真正结束
                        t.refine_pct = 0.0
                        t.refine_start_time = now
                else:
                    # 回退：遍历查找
                    for t in self.task_states:
                        if t.asr_pct >= 0.84 and t.status in (_ST_WAIT, _ST_ASR):
                            t.status = _ST_REF
                            t.asr_pct = 1.0     # ASR 阶段真正结束
                            t.refine_pct = 0.0
                            t.refine_start_time = now
                            break

            # ── text_cleaning ──
            # 两类事件：
            #   A) 带 file_index：来自 as_completed 循环，该文件精修 future 已 resolve
            #      → 无论 progress 值多少，该文件精修完成
            #   B) 不带 file_index：全局事件（0.90 整体收尾开始）
            #      → 标记所有 REFINE 任务为活跃，触发平滑动画
            if stage == "text_cleaning":
                if file_idx is not None and 0 <= file_idx < self.n_total:
                    # 单个文件精修完成（future 已 resolve）
                    t = self.task_states[file_idx]
                    if t.status == _ST_REF:
                        t.refine_pct = 1.0
                        t.asr_pct = 1.0
                        t.status = _ST_DONE
                        t.done_time = now
                        t.exit_alpha = 1.0
                        if t.refine_start_time > 0:
                            dur = now - t.refine_start_time
                            self._refine_durations.append(dur)
                elif progress is not None and progress < 0.93:
                    # 全局事件 0.90：标记所有 REFINE 任务为活跃
                    for t in self.task_states:
                        if t.status == _ST_REF and t.refine_pct < 0.05:
                            t.refine_pct = 0.05
                if progress is not None and progress >= 0.98:
                    self.phase2_done = True

            # ── per-task ASR 进度 ──
            _ASR_STAGES = {"audio_extracting", "asr_transcribing", "audio_segmenting"}
            if (not self.phase2_done and progress is not None and self.n_total > 0
                    and stage in _ASR_STAGES):

                if file_idx is not None and 0 <= file_idx < self.n_total:
                    # ── 并发模式：直接更新 file_idx 对应的任务 ──
                    t = self.task_states[file_idx]
                    local = progress * self.n_total - file_idx
                    new_pct = min(local, 1.0)
                    if new_pct > t.asr_pct:
                        t.asr_pct = new_pct
                    if t.status != _ST_ASR:
                        t.status = _ST_ASR
                        if t.asr_start_time <= 0:
                            t.asr_start_time = now
                    elif local >= 0.84:
                        if t.asr_start_time > 0 and t.asr_done_time <= 0:
                            t.asr_done_time = now
                            dur = now - t.asr_start_time
                            self._asr_durations.append(dur)
                        t.status = _ST_WAIT
                else:
                    # ── 回退：串行模式，从全局进度推导 ──
                    derived_idx = min(int(progress * self.n_total), self.n_total - 1)
                    local = progress * self.n_total - derived_idx
                    for i, t in enumerate(self.task_states):
                        if i < derived_idx:
                            if t.status == _ST_ASR and t.asr_done_time <= 0:
                                t.asr_done_time = now
                                if t.asr_start_time > 0:
                                    dur = now - t.asr_start_time
                                    self._asr_durations.append(dur)
                                t.status = _ST_WAIT
                            t.asr_pct = 1.0
                        elif i == derived_idx:
                            new_pct = min(local, 1.0)
                            if new_pct > t.asr_pct:
                                t.asr_pct = new_pct
                            if t.status != _ST_ASR:
                                t.status = _ST_ASR
                                if t.asr_start_time <= 0:
                                    t.asr_start_time = now
                            elif local >= 0.84:
                                if t.asr_start_time > 0 and t.asr_done_time <= 0:
                                    t.asr_done_time = now
                                    dur = now - t.asr_start_time
                                    self._asr_durations.append(dur)
                                t.status = _ST_WAIT

            # ── 并发场景下重新计算 overall_pct（聚合所有任务进度） ──
            if file_idx is not None and self.n_total > 0:
                total_progress = 0.0
                for t in self.task_states:
                    if t.status == _ST_DONE:
                        total_progress += 1.0
                    elif t.status == _ST_REF:
                        total_progress += 0.90 + t.refine_pct * 0.10
                    elif t.status == _ST_ASR:
                        total_progress += t.asr_pct
                    # WAIT: 0.0
                self.overall_pct = round(total_progress / self.n_total, 4)

            now2 = time.time()
            for t in self.task_states:
                if t.status == _ST_DONE and t.exit_alpha > 0.0:
                    elapsed_done = now2 - t.done_time
                    if elapsed_done > 0.8:
                        t.exit_alpha = max(0.0, 1.0 - (elapsed_done - 0.8) / 0.6)

            if self.phase2_done:
                self.phase_label = "All done"
            elif self.refine_active:
                self.phase_label = "ASR + Refine/Summary (background)"
            else:
                self.phase_label = "ASR (serial)"

    # ── smooth ASR interpolation ─────────────────────────────────────
    def _smooth_asr(self):
        """基于已完成文件的真实 ASR 耗时，线性插值活跃 ASR 任务的进度。

        每 200ms 调用一次，将进度条从 15% 平滑推进到 82%（留 3% 缓冲，
        真实完成事件到达时进度条会自然跨越最后的间隙）。
        """
        with self._lock:
            now = time.time()
            # 估算 ASR 耗时：历史均值 or 默认 90s
            if self._asr_durations:
                est_asr_sec = sum(self._asr_durations) / len(self._asr_durations)
            else:
                est_asr_sec = 90.0

            # ASR 进度在文件本地窗口中是 [0.15, 0.85]
            ASR_START = 0.15
            ASR_END = 0.82   # 不到 0.85，留给真实事件收尾

            for t in self.task_states:
                if t.status != _ST_ASR or t.asr_start_time <= 0:
                    continue
                elapsed = now - t.asr_start_time
                if elapsed <= 0:
                    continue
                # 线性插值：15% + elapsed/est * (82% - 15%)
                smooth_pct = ASR_START + min(elapsed / est_asr_sec, 1.0) * (ASR_END - ASR_START)
                # 只在平滑值高于真实值时覆盖（不覆盖真实的高进度）
                if smooth_pct > t.asr_pct:
                    t.asr_pct = round(smooth_pct, 4)

    # ── smooth Refine interpolation ─────────────────────────────────
    def _smooth_refine(self):
        """基于精修启动时间，线性插值活跃 REFINE 任务的进度。

        精修走网络 API，耗时波动大（30-180s）。默认估算 90s，
        有过往真实耗时后自动校准。进度范围 0.05→0.95，留首尾给真实事件。
        """
        with self._lock:
            now = time.time()
            # 估算精修耗时：历史均值 or 默认 90s
            if self._refine_durations:
                est_sec = sum(self._refine_durations) / len(self._refine_durations)
            else:
                est_sec = 90.0

            REFINE_START = 0.05
            REFINE_END = 0.92  # 不到 1.0，留给真实完成事件

            for t in self.task_states:
                if t.status != _ST_REF or t.refine_start_time <= 0:
                    continue
                if t.refine_pct >= 0.95:
                    continue  # 已完成或接近完成，不再平滑
                elapsed = now - t.refine_start_time
                if elapsed <= 0:
                    continue
                smooth_pct = REFINE_START + min(elapsed / est_sec, 1.0) * (REFINE_END - REFINE_START)
                if smooth_pct > t.refine_pct:
                    t.refine_pct = round(smooth_pct, 4)

    # ── consumer thread ────────────────────────────────────────────────
    def _consume_events(self):
        """消费原始进度事件 → 更新状态（含平滑插值）。在守护线程中运行。"""
        last_smooth = time.time()
        while self._running:
            try:
                msg = self.queue.get(timeout=0.2)
                if msg is None:
                    break
                kind = msg[0]
                if kind == "progress":
                    _, stage, progress, meta = msg
                    self._dispatch(stage, progress, meta)
                elif kind == "done":
                    _, result = msg
                    self.result = result
                    self.phase2_done = True
                    # 判断管线是否成功（result 为 dict 时检查 status 字段）
                    ok = (isinstance(result, dict) and
                          result.get("status") not in (None, "error"))
                    done_count = sum(1 for t in self.task_states
                                     if t.status in (_ST_DONE, _ST_FAIL))
                    err_msg = ""
                    if isinstance(result, dict):
                        err_msg = result.get("message", "") or str(
                            result.get("stats", "")) or ""
                    with self._lock:
                        for t in self.task_states:
                            if t.status not in (_ST_DONE, _ST_FAIL):
                                t.asr_pct = 1.0
                                t.refine_pct = 1.0
                                t.status = (_ST_DONE if ok else _ST_FAIL)
                                t.done_time = time.time()
                                t.exit_alpha = 1.0
                                if not ok and err_msg:
                                    t.error = err_msg[:120]
                    self.sse_queue.put(("done", {
                        "status": "complete" if ok else "error",
                        "done": done_count,
                        "total": self.n_total,
                        "message": err_msg[:200] if err_msg else "",
                    }))
            except Exception:
                pass
            # 每 200ms 运行平滑插值
            now = time.time()
            if now - last_smooth >= 0.20:
                self._smooth_asr()
                self._smooth_refine()
                last_smooth = now

    def _run_pipeline(self, pipeline_fn, *args, **kwargs):
        """在当前线程中运行管线函数，结果推入队列。"""
        try:
            result = pipeline_fn(*args, **kwargs,
                                 on_progress=self.on_progress)
        except Exception as e:
            result = {"status": "error", "message": str(e)}
        self.queue.put(("done", result))

    def start_pipeline(self, pipeline_fn, *args, **kwargs):
        """启动消费者线程 + 管线线程。返回管线线程引用。"""
        self._running = True
        consumer = threading.Thread(target=self._consume_events, daemon=True)
        consumer.start()
        pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            args=(pipeline_fn,) + args, kwargs=kwargs, daemon=True,
        )
        pipeline_thread.start()
        return pipeline_thread

    def stop(self):
        """发送停止信号给消费者线程。"""
        self._running = False

    # ── run ──────────────────────────────────────────────────────────
    def run(self, pipeline_fn, *args, **kwargs):
        """启动管线 + HTTP server + 浏览器。主线程等待管线完成。"""
        port = _pick_port()
        url = f"http://127.0.0.1:{port}"

        server = HTTPServer(("127.0.0.1", port), _DashboardHandler)
        server.sse_queue = self.sse_queue
        server.init_payload = self.init_payload()
        server.running = True
        server.snapshot_payload = self.snapshot_payload
        server.timeout = 0.5

        # 启动消费者 + 管线
        pipeline_thread = self.start_pipeline(pipeline_fn, *args, **kwargs)

        # HTTP server 守护线程
        def _serve():
            try:
                server.serve_forever(poll_interval=0.5)
            except Exception:
                pass
        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()

        # 打开浏览器 (独立线程，避免阻塞)
        print(f"\n  Yiren Pipeline Monitor  →  {url}\n")
        def _open_browser():
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open_browser, daemon=True).start()

        # 主线程等待管线完成
        try:
            pipeline_thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            server.running = False
            # 等待 SSE handler 退出（1.0s get timeout + 缓冲）
            time.sleep(1.3)
            # 关闭 server socket 终止 serve_forever 和 EventSource 重连循环
            try:
                server.server_close()
            except Exception:
                pass

        return self.result


# ── convenience ──────────────────────────────────────────────────────────
def run_with_window(series_id, source_paths, output_dir, config=None):
    """启动浏览器仪表盘运行 Yiren 流水线。"""
    from yiren_ingest import ingest_series_material

    cfg = dict(config) if config else {}
    win = PipelineWindow(source_paths)

    return win.run(
        ingest_series_material,
        series_id=series_id,
        upload_paths=source_paths,
        output_dir=output_dir,
        config=cfg,
    )

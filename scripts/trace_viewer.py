"""Agent Trace Viewer - Web UI for browsing agent conversation traces.

Usage:
    python scripts/trace_viewer.py [--port 8090] [--outputs-dir outputs]

Serves a chat-style UI that reads .trace.json files from experiment directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Agent Trace Viewer")
OUTPUTS_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--outputs-dir", default="outputs")
    return p.parse_args()


@app.get("/api/experiments")
def list_experiments():
    experiments = []
    for d in sorted(OUTPUTS_DIR.iterdir()):
        trace_dir = d / "traces"
        if trace_dir.is_dir():
            count = len(list(trace_dir.glob("*.trace.json")))
            if count > 0:
                experiments.append({"name": d.name, "count": count})
    return experiments


@app.get("/api/traces/{experiment}")
def list_traces(experiment: str):
    trace_dir = OUTPUTS_DIR / experiment / "traces"
    if not trace_dir.is_dir():
        raise HTTPException(404, f"experiment not found: {experiment}")
    traces = []
    for f in sorted(trace_dir.glob("*.trace.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            traces.append({
                "sample_id": data.get("sample_id", f.stem),
                "steps": len(data.get("steps", [])),
                "elapsed": data.get("elapsed_seconds", 0),
                "errors": len(data.get("errors", [])),
            })
        except Exception:
            pass
    return traces


@app.get("/api/trace/{experiment}/{sample_id}")
def get_trace(experiment: str, sample_id: str):
    trace_path = OUTPUTS_DIR / experiment / "traces" / f"{sample_id}.trace.json"
    if not trace_path.exists():
        raise HTTPException(404, f"trace not found: {sample_id}")
    return json.loads(trace_path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Trace Viewer</title>
<style>
:root {
  --bg: #0f0f0f;
  --surface: #1a1a1a;
  --surface2: #242424;
  --border: #333;
  --text: #e0e0e0;
  --text-dim: #888;
  --user-bg: #1b3a5c;
  --assistant-bg: #2d2d2d;
  --tool-bg: #1a2e1a;
  --tool-fail-bg: #3a1a1a;
  --accent: #5b9bd5;
  --success: #4caf50;
  --error: #ef5350;
  --warning: #ff9800;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; }

/* Sidebar */
.sidebar {
  width: 320px; min-width: 280px; background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}
.sidebar-header {
  padding: 16px; border-bottom: 1px solid var(--border); font-size: 15px; font-weight: 600;
  display: flex; align-items: center; gap: 8px;
}
.sidebar-header svg { width: 20px; height: 20px; }
.experiment-select {
  width: 100%; padding: 8px 12px; background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); border-radius: 6px; font-size: 13px; margin: 12px 16px 8px; width: calc(100% - 32px);
}
.trace-list { flex: 1; overflow-y: auto; padding: 4px 8px; }
.trace-item {
  padding: 10px 12px; margin: 2px 0; border-radius: 6px; cursor: pointer; font-size: 13px;
  display: flex; justify-content: space-between; align-items: center; transition: background 0.15s;
}
.trace-item:hover { background: var(--surface2); }
.trace-item.active { background: var(--accent); color: #fff; }
.trace-item .id { font-weight: 500; font-family: monospace; }
.trace-item .meta { font-size: 11px; color: var(--text-dim); }
.trace-item.active .meta { color: rgba(255,255,255,0.7); }
.trace-item .err-badge {
  background: var(--error); color: #fff; font-size: 10px; padding: 1px 5px;
  border-radius: 8px; margin-left: 6px;
}

/* Main */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header {
  padding: 12px 20px; border-bottom: 1px solid var(--border); background: var(--surface);
  display: flex; align-items: center; gap: 16px; font-size: 14px; flex-shrink: 0;
}
.main-header .sample-id { font-weight: 600; font-family: monospace; font-size: 15px; }
.main-header .elapsed { color: var(--text-dim); font-size: 12px; }
.main-header .step-count { color: var(--text-dim); font-size: 12px; }

.chat-container {
  flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px;
}

/* Messages */
.msg {
  max-width: 90%; padding: 12px 16px; border-radius: 12px; font-size: 14px;
  line-height: 1.6; word-break: break-word; position: relative;
}
.msg-user {
  background: var(--user-bg); align-self: flex-end; border-bottom-right-radius: 4px;
  white-space: pre-wrap;
}
.msg-assistant {
  background: var(--assistant-bg); align-self: flex-start; border-bottom-left-radius: 4px;
  border-left: 3px solid var(--accent);
}
.msg-tool {
  background: var(--tool-bg); align-self: flex-start; border-bottom-left-radius: 4px;
  border-left: 3px solid var(--success); font-size: 13px;
}
.msg-tool.error {
  background: var(--tool-fail-bg); border-left-color: var(--error);
}

.msg-label {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}
.msg-label .step-num {
  background: var(--surface2); padding: 1px 6px; border-radius: 4px; font-size: 10px;
}
.msg-user .msg-label { color: rgba(255,255,255,0.6); }
.msg-assistant .msg-label { color: var(--accent); }
.msg-tool .msg-label { color: var(--success); }
.msg-tool.error .msg-label { color: var(--error); }

.tool-name {
  display: inline-block; background: var(--surface2); padding: 2px 8px; border-radius: 4px;
  font-family: monospace; font-size: 12px; margin: 0 2px;
}
.msg-assistant .tool-name { background: rgba(91,155,213,0.2); }
.msg-tool .tool-name { background: rgba(76,175,80,0.15); }

.reason { color: var(--text-dim); font-size: 13px; margin-top: 4px; }
.args-block {
  background: var(--surface); padding: 6px 10px; border-radius: 6px; margin-top: 6px;
  font-family: monospace; font-size: 12px; color: var(--text-dim); overflow-x: auto;
}
.evidence-block {
  margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;
}
.evidence-item {
  background: var(--surface); padding: 8px 10px; border-radius: 6px; margin: 4px 0;
  font-size: 12px;
}
.evidence-item .ev-source {
  font-weight: 600; color: var(--accent); font-size: 11px; margin-bottom: 4px;
}
.evidence-item .ev-content {
  color: var(--text-dim); max-height: 200px; overflow-y: auto; white-space: pre-wrap;
  line-height: 1.5;
}
.answer-block {
  background: var(--surface); padding: 10px 12px; border-radius: 6px; margin-top: 8px;
  white-space: pre-wrap; font-size: 13px; line-height: 1.6; max-height: 500px; overflow-y: auto;
}
.error-list {
  margin-top: 6px; padding: 6px 10px; background: rgba(239,83,80,0.1); border-radius: 6px;
  font-size: 12px; color: var(--error);
}

.empty-state {
  flex: 1; display: flex; align-items: center; justify-content: center;
  color: var(--text-dim); font-size: 15px;
}

/* Collapsible evidence */
.evidence-toggle {
  cursor: pointer; color: var(--accent); font-size: 12px; margin-top: 4px;
  user-select: none;
}
.evidence-toggle:hover { text-decoration: underline; }
.evidence-content { display: none; }
.evidence-content.open { display: block; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Agent Trace Viewer
  </div>
  <select class="experiment-select" id="experimentSelect">
    <option value="">Loading experiments...</option>
  </select>
  <div class="trace-list" id="traceList"></div>
</div>
<div class="main">
  <div class="main-header" id="mainHeader" style="display:none">
    <span class="sample-id" id="headerSampleId"></span>
    <span class="elapsed" id="headerElapsed"></span>
    <span class="step-count" id="headerSteps"></span>
  </div>
  <div class="chat-container" id="chatContainer">
    <div class="empty-state">Select a trace from the sidebar to view the conversation</div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let currentExperiment = '';
let currentSampleId = '';

async function loadExperiments() {
  const resp = await fetch('/api/experiments');
  const data = await resp.json();
  const sel = $('#experimentSelect');
  sel.innerHTML = data.length
    ? '<option value="">Select experiment...</option>' + data.map(e =>
        `<option value="${e.name}">${e.name} (${e.count})</option>`).join('')
    : '<option value="">No experiments found</option>';
  sel.onchange = () => {
    currentExperiment = sel.value;
    if (currentExperiment) loadTraces(currentExperiment);
    else $('#traceList').innerHTML = '';
  };
  if (data.length) {
    sel.value = data[0].name;
    currentExperiment = data[0].name;
    loadTraces(currentExperiment);
  }
}

async function loadTraces(experiment) {
  const resp = await fetch(`/api/traces/${experiment}`);
  const data = await resp.json();
  const list = $('#traceList');
  list.innerHTML = data.map(t => `
    <div class="trace-item" data-id="${t.sample_id}" onclick="loadTrace('${t.sample_id}')">
      <span class="id">${t.sample_id}</span>
      <span class="meta">${t.steps} steps · ${t.elapsed.toFixed(1)}s${t.errors ? '<span class="err-badge">err</span>' : ''}</span>
    </div>
  `).join('');
}

async function loadTrace(sampleId) {
  currentSampleId = sampleId;
  document.querySelectorAll('.trace-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === sampleId));

  const resp = await fetch(`/api/trace/${currentExperiment}/${sampleId}`);
  const data = await resp.json();

  $('#mainHeader').style.display = 'flex';
  $('#headerSampleId').textContent = sampleId;
  $('#headerElapsed').textContent = `${data.elapsed_seconds.toFixed(1)}s`;
  $('#headerSteps').textContent = `${data.steps.length} steps`;

  const chat = $('#chatContainer');
  chat.innerHTML = '';

  for (const step of data.steps) {
    const role = step.role;
    const content = step.content;

    if (role === 'user') {
      chat.appendChild(createUserMsg(step.step, content));
    } else if (role === 'assistant') {
      chat.appendChild(createAssistantMsg(step.step, content));
    } else if (role === 'tool') {
      chat.appendChild(createToolMsg(step.step, content));
    }
  }

  // Show errors
  if (data.errors && data.errors.length) {
    const errDiv = document.createElement('div');
    errDiv.className = 'error-list';
    errDiv.textContent = 'Errors: ' + data.errors.join('; ');
    chat.appendChild(errDiv);
  }

  chat.scrollTop = chat.scrollHeight;
}

function createUserMsg(step, content) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-label">User</div>${escHtml(content)}`;
  return div;
}

function createAssistantMsg(step, content) {
  const div = document.createElement('div');
  div.className = 'msg msg-assistant';
  const tool = typeof content === 'object' ? content.tool : '';
  const reason = typeof content === 'object' ? content.reason : '';
  const args = typeof content === 'object' ? content.args : null;
  const error = typeof content === 'object' ? content.error : '';

  let html = `<div class="msg-label"><span class="step-num">Step ${step}</span> Assistant</div>`;
  if (error) {
    html += `<div style="color:var(--error)">Error: ${escHtml(error)}</div>`;
  } else {
    html += `Use tool <span class="tool-name">${escHtml(tool)}</span>`;
  }
  if (reason) html += `<div class="reason">${escHtml(reason)}</div>`;
  if (args && Object.keys(args).length) {
    html += `<div class="args-block">${escHtml(JSON.stringify(args, null, 2))}</div>`;
  }
  div.innerHTML = html;
  return div;
}

function createToolMsg(step, content) {
  const div = document.createElement('div');
  const isError = content.error || content.success === false;
  div.className = `msg msg-tool${isError ? ' error' : ''}`;

  const tool = content.tool || '';
  let html = `<div class="msg-label"><span class="step-num">Step ${step}</span> Tool · <span class="tool-name">${escHtml(tool)}</span>`;
  if (content.success === true) html += ` <span style="color:var(--success)">✓</span>`;
  else if (content.success === false) html += ` <span style="color:var(--error)">✗</span>`;
  html += `</div>`;

  // Query
  if (content.query) html += `<div class="args-block">query: ${escHtml(content.query)}</div>`;

  // Error
  if (content.error) html += `<div style="color:var(--error);margin-top:6px">Error: ${escHtml(content.error)}</div>`;

  // Answer (for finish_answer)
  if (content.answer) {
    const id = `ans-${step}-${Math.random().toString(36).slice(2,8)}`;
    html += `<div class="evidence-toggle" onclick="toggleBlock('${id}')">▼ Show Answer</div>`;
    html += `<div class="answer-block evidence-content open" id="${id}">${escHtml(content.answer)}</div>`;
  }

  // Evidence
  if (content.evidence && content.evidence.length) {
    const id = `ev-${step}-${Math.random().toString(36).slice(2,8)}`;
    html += `<div class="evidence-toggle" onclick="toggleBlock('${id}')">▼ Evidence (${content.evidence.length})</div>`;
    html += `<div class="evidence-block evidence-content" id="${id}">`;
    for (const ev of content.evidence) {
      const src = ev.source || 'unknown';
      const evContent = ev.content || '';
      const truncated = evContent.length > 500;
      const display = truncated ? evContent.slice(0, 500) + '...' : evContent;
      html += `<div class="evidence-item">
        <div class="ev-source">${escHtml(src)}</div>
        <div class="ev-content">${escHtml(display)}${truncated ? `<span class="evidence-toggle" onclick="this.parentElement.textContent=\`${escJs(evContent)}\`"> [show full]</span>` : ''}</div>
      </div>`;
    }
    html += `</div>`;
  }

  div.innerHTML = html;
  return div;
}

function toggleBlock(id) {
  const el = document.getElementById(id);
  const toggle = el.previousElementSibling;
  el.classList.toggle('open');
  toggle.textContent = (el.classList.contains('open') ? '▼' : '▶') + toggle.textContent.slice(1);
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escJs(s) {
  return s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$/g, '\\$');
}

loadExperiments();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    args = parse_args()
    OUTPUTS_DIR = Path(args.outputs_dir).resolve()
    print(f"[trace-viewer] serving traces from {OUTPUTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

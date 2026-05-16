"""Agent Trace Viewer - Web UI for browsing agent conversation traces.

Usage:
    python scripts/trace_viewer.py [--port 8090]

Open http://localhost:8090 and enter a trace directory path to load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Agent Trace Viewer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    return p.parse_args()


class DirRequest(BaseModel):
    path: str


class TraceRequest(BaseModel):
    dir_path: str
    sample_id: str


@app.post("/api/traces")
def list_traces(req: DirRequest):
    p = Path(req.path).expanduser().resolve()
    if p.is_file() and p.suffix == ".json":
        # Single trace file: wrap into a one-item list
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            trace = {
                "sample_id": data.get("sample_id", p.stem),
                "steps": len(data.get("steps", [])),
                "elapsed": data.get("elapsed_seconds", 0),
                "errors": len(data.get("errors", [])),
            }
            return {"dir": str(p.parent), "count": 1, "traces": [trace]}
        except Exception as e:
            raise HTTPException(400, f"failed to read trace file: {e}")
    if not p.is_dir():
        raise HTTPException(400, f"directory not found: {p}")
    traces = []
    for f in sorted(p.glob("*.trace.json")):
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
    return {"dir": str(p), "count": len(traces), "traces": traces}


@app.post("/api/trace")
def get_trace(req: TraceRequest):
    trace_path = Path(req.dir_path).expanduser().resolve() / f"{req.sample_id}.trace.json"
    if not trace_path.exists():
        raise HTTPException(404, f"trace not found: {req.sample_id}")
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
  --bg: #f5f5f7;
  --surface: #ffffff;
  --surface2: #f0f0f2;
  --border: #e0e0e4;
  --text: #1d1d1f;
  --text-dim: #6e6e73;
  --text-light: #86868b;
  --user-bg: #007aff;
  --user-text: #ffffff;
  --assistant-bg: #ffffff;
  --tool-bg: #f5f5f7;
  --tool-fail-bg: #fff0f0;
  --accent: #007aff;
  --accent-light: #e8f2ff;
  --success: #34c759;
  --success-light: #e8faee;
  --error: #ff3b30;
  --error-light: #fff0ef;
  --warning: #ff9500;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 2px 8px rgba(0,0,0,0.06);
  --shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
  --radius: 12px;
  --radius-sm: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
}
a { color: var(--accent); text-decoration: none; }

/* Top bar */
.topbar {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; gap: 12px;
  box-shadow: var(--shadow-sm); z-index: 10; flex-shrink: 0;
}
.topbar-title {
  font-size: 16px; font-weight: 600; color: var(--text); white-space: nowrap;
  display: flex; align-items: center; gap: 8px;
}
.topbar-title svg { width: 22px; height: 22px; color: var(--accent); }
.dir-input-wrap {
  flex: 1; display: flex; gap: 8px; max-width: 600px;
}
.dir-input {
  flex: 1; padding: 8px 14px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  font-size: 13px; background: var(--surface2); color: var(--text); outline: none;
  transition: border-color 0.2s, box-shadow 0.2s; font-family: monospace;
}
.dir-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,122,255,0.12); }
.dir-input::placeholder { color: var(--text-light); }
.btn {
  padding: 8px 18px; border: none; border-radius: var(--radius-sm); font-size: 13px;
  font-weight: 500; cursor: pointer; transition: all 0.15s; white-space: nowrap;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: #0066d6; }
.btn-primary:active { transform: scale(0.97); }
.topbar-info { font-size: 12px; color: var(--text-light); white-space: nowrap; }

/* Layout */
.content { flex: 1; display: flex; overflow: hidden; }

/* Sidebar */
.sidebar {
  width: 300px; min-width: 260px; background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
}
.sidebar-header {
  padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600;
  color: var(--text-dim); display: flex; justify-content: space-between; align-items: center;
}
.sidebar-header .count { color: var(--text-light); font-weight: 400; }
.trace-list { flex: 1; overflow-y: auto; padding: 6px 8px; }
.trace-item {
  padding: 10px 12px; margin: 2px 0; border-radius: var(--radius-sm); cursor: pointer; font-size: 13px;
  display: flex; justify-content: space-between; align-items: center; transition: all 0.12s;
  border: 1px solid transparent;
}
.trace-item:hover { background: var(--surface2); }
.trace-item.active {
  background: var(--accent-light); border-color: var(--accent); color: var(--accent);
}
.trace-item .id { font-weight: 500; font-family: 'SF Mono', Menlo, monospace; font-size: 12px; }
.trace-item .meta { font-size: 11px; color: var(--text-light); }
.trace-item.active .meta { color: var(--accent); opacity: 0.7; }
.trace-item .err-badge {
  background: var(--error); color: #fff; font-size: 10px; padding: 1px 6px;
  border-radius: 10px; margin-left: 6px; font-weight: 500;
}

/* Main */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header {
  padding: 12px 24px; border-bottom: 1px solid var(--border); background: var(--surface);
  display: flex; align-items: center; gap: 16px; font-size: 13px; flex-shrink: 0;
  box-shadow: var(--shadow-sm);
}
.main-header .sample-id { font-weight: 600; font-family: 'SF Mono', Menlo, monospace; font-size: 14px; }
.main-header .badge {
  padding: 3px 10px; border-radius: 10px; font-size: 11px; font-weight: 500;
}
.badge-time { background: var(--accent-light); color: var(--accent); }
.badge-steps { background: var(--surface2); color: var(--text-dim); }

.chat-container {
  flex: 1; overflow-y: auto; padding: 24px 20px; display: flex; flex-direction: column; gap: 16px;
}

/* Messages */
.msg {
  max-width: 85%; padding: 12px 16px; font-size: 14px; line-height: 1.65;
  word-break: break-word; position: relative; animation: fadeIn 0.2s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
.msg-user {
  background: var(--user-bg); color: var(--user-text); align-self: flex-end;
  border-radius: var(--radius) var(--radius) 4px var(--radius);
  box-shadow: var(--shadow-sm); white-space: pre-wrap;
}
.msg-assistant {
  background: var(--assistant-bg); align-self: flex-start;
  border-radius: var(--radius) var(--radius) var(--radius) 4px;
  box-shadow: var(--shadow-md); border: 1px solid var(--border);
}
.msg-tool {
  background: var(--tool-bg); align-self: flex-start;
  border-radius: var(--radius) var(--radius) var(--radius) 4px;
  border: 1px solid var(--border); font-size: 13px;
}
.msg-tool.error {
  background: var(--tool-fail-bg); border-color: var(--error);
}

.msg-label {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}
.msg-label .step-num {
  background: var(--surface2); padding: 1px 7px; border-radius: 4px; font-size: 10px;
  font-weight: 500;
}
.msg-user .msg-label { color: rgba(255,255,255,0.7); }
.msg-user .msg-label .step-num { background: rgba(255,255,255,0.2); }
.msg-assistant .msg-label { color: var(--accent); }
.msg-tool .msg-label { color: var(--success); }
.msg-tool.error .msg-label { color: var(--error); }

.tool-name {
  display: inline-block; background: var(--accent-light); padding: 2px 8px; border-radius: 4px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 12px; color: var(--accent); font-weight: 500;
}
.msg-tool .tool-name { background: var(--success-light); color: #1a8a3f; }

.reason { color: var(--text-dim); font-size: 13px; margin-top: 4px; font-style: italic; }
.args-block {
  background: var(--surface2); padding: 8px 10px; border-radius: var(--radius-sm); margin-top: 8px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 12px; color: var(--text-dim); overflow-x: auto;
  border: 1px solid var(--border);
}
.evidence-block {
  margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;
}
.evidence-item {
  background: var(--surface); padding: 8px 10px; border-radius: var(--radius-sm); margin: 4px 0;
  font-size: 12px; border: 1px solid var(--border);
}
.evidence-item .ev-source {
  font-weight: 600; color: var(--accent); font-size: 11px; margin-bottom: 4px;
}
.evidence-item .ev-content {
  color: var(--text-dim); max-height: 200px; overflow-y: auto; white-space: pre-wrap;
  line-height: 1.5;
}
.answer-block {
  background: var(--surface); padding: 12px 14px; border-radius: var(--radius-sm); margin-top: 8px;
  white-space: pre-wrap; font-size: 13px; line-height: 1.7; max-height: 500px; overflow-y: auto;
  border: 1px solid var(--border);
}
.error-list {
  margin-top: 8px; padding: 10px 14px; background: var(--error-light); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--error); border: 1px solid var(--error); align-self: center;
  max-width: 90%;
}

.empty-state {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: var(--text-light); gap: 12px;
}
.empty-state svg { width: 48px; height: 48px; opacity: 0.3; }
.empty-state p { font-size: 14px; }

/* Collapsible evidence */
.evidence-toggle {
  cursor: pointer; color: var(--accent); font-size: 12px; margin-top: 4px;
  user-select: none; display: inline-block;
}
.evidence-toggle:hover { text-decoration: underline; }
.evidence-content { display: none; }
.evidence-content.open { display: block; }

/* Loading spinner */
.loading { display: none; align-items: center; gap: 8px; color: var(--text-light); font-size: 13px; }
.loading.show { display: flex; }
.spinner {
  width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent);
  border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c0c0c0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a0a0a0; }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Agent Trace Viewer
  </div>
  <div class="dir-input-wrap">
    <input class="dir-input" id="dirInput" type="text" placeholder="输入目录或 .trace.json 文件路径" value="">
    <button class="btn btn-primary" id="loadBtn" onclick="loadDir()">加载</button>
  </div>
  <div class="loading" id="loading"><div class="spinner"></div>加载中...</div>
  <div class="topbar-info" id="topbarInfo"></div>
</div>

<div class="content">
  <div class="sidebar" id="sidebar" style="display:none">
    <div class="sidebar-header">
      <span>轨迹列表</span>
      <span class="count" id="traceCount"></span>
    </div>
    <div class="trace-list" id="traceList"></div>
  </div>
  <div class="main">
    <div class="main-header" id="mainHeader" style="display:none">
      <span class="sample-id" id="headerSampleId"></span>
      <span class="badge badge-time" id="headerElapsed"></span>
      <span class="badge badge-steps" id="headerSteps"></span>
    </div>
    <div class="chat-container" id="chatContainer">
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <p>输入轨迹目录路径，点击加载</p>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let currentDirPath = '';
let currentSampleId = '';

// Enter key to load
$('#dirInput').addEventListener('keydown', e => { if (e.key === 'Enter') loadDir(); });

async function loadDir() {
  const dirPath = $('#dirInput').value.trim();
  if (!dirPath) return;

  $('#loading').classList.add('show');
  $('#topbarInfo').textContent = '';

  try {
    const resp = await fetch('/api/traces', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path: dirPath}),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'load failed');
    }
    const data = await resp.json();
    currentDirPath = data.dir;
    $('#topbarInfo').textContent = `${data.count} 条轨迹`;

    const sidebar = $('#sidebar');
    sidebar.style.display = 'flex';
    $('#traceCount').textContent = `${data.count} 条`;

    const list = $('#traceList');
    if (!data.traces.length) {
      list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-light);font-size:13px">目录下没有 .trace.json 文件</div>';
      return;
    }

    list.innerHTML = data.traces.map(t => `
      <div class="trace-item" data-id="${t.sample_id}" onclick="loadTrace('${t.sample_id}')">
        <span class="id">${t.sample_id}</span>
        <span class="meta">${t.steps} 步 · ${t.elapsed.toFixed(1)}s${t.errors ? '<span class="err-badge">err</span>' : ''}</span>
      </div>
    `).join('');

    // Auto-load first trace
    loadTrace(data.traces[0].sample_id);
  } catch (e) {
    $('#topbarInfo').textContent = '';
    alert('加载失败: ' + e.message);
  } finally {
    $('#loading').classList.remove('show');
  }
}

async function loadTrace(sampleId) {
  currentSampleId = sampleId;
  document.querySelectorAll('.trace-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === sampleId));

  const resp = await fetch('/api/trace', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dir_path: currentDirPath, sample_id: sampleId}),
  });
  const data = await resp.json();

  $('#mainHeader').style.display = 'flex';
  $('#headerSampleId').textContent = sampleId;
  $('#headerElapsed').textContent = `${data.elapsed_seconds.toFixed(1)}s`;
  $('#headerSteps').textContent = `${data.steps.length} 步`;

  const chat = $('#chatContainer');
  chat.innerHTML = '';

  for (const step of data.steps) {
    const role = step.role;
    const content = step.content;
    if (role === 'user') chat.appendChild(createUserMsg(step.step, content));
    else if (role === 'assistant') chat.appendChild(createAssistantMsg(step.step, content));
    else if (role === 'tool') chat.appendChild(createToolMsg(step.step, content));
  }

  if (data.errors && data.errors.length) {
    const errDiv = document.createElement('div');
    errDiv.className = 'error-list';
    errDiv.textContent = 'Errors: ' + data.errors.join('; ');
    chat.appendChild(errDiv);
  }

  chat.scrollTop = 0;
}

function createUserMsg(step, content) {
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-label"><span class="step-num">Step ${step}</span> User</div>${escHtml(content)}`;
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
    html += `<div>使用工具 <span class="tool-name">${escHtml(tool)}</span></div>`;
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

  if (content.query) html += `<div class="args-block">query: ${escHtml(content.query)}</div>`;
  if (content.error) html += `<div style="color:var(--error);margin-top:6px">Error: ${escHtml(content.error)}</div>`;

  if (content.answer) {
    const id = `ans-${step}-${Math.random().toString(36).slice(2,8)}`;
    html += `<div class="evidence-toggle" onclick="toggleBlock('${id}')">▼ 答案</div>`;
    html += `<div class="answer-block evidence-content open" id="${id}">${escHtml(content.answer)}</div>`;
  }

  if (content.evidence && content.evidence.length) {
    const id = `ev-${step}-${Math.random().toString(36).slice(2,8)}`;
    html += `<div class="evidence-toggle" onclick="toggleBlock('${id}')">▼ 证据 (${content.evidence.length})</div>`;
    html += `<div class="evidence-block evidence-content" id="${id}">`;
    for (const ev of content.evidence) {
      const src = ev.source || 'unknown';
      const evContent = ev.content || '';
      const truncated = evContent.length > 500;
      const display = truncated ? evContent.slice(0, 500) + '...' : evContent;
      html += `<div class="evidence-item">
        <div class="ev-source">${escHtml(src)}</div>
        <div class="ev-content">${escHtml(display)}${truncated ? `<span class="evidence-toggle" onclick="this.parentElement.textContent=\`${escJs(evContent)}\`"> [展开全文]</span>` : ''}</div>
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
</script>
</body>
</html>
"""

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

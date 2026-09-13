"""Dependency-free static HTML study runner.

Trials are embedded into the HTML so the file works from `file://` without a
server and without `fetch` restrictions. Responses are downloaded as JSON and
can be scored by the Python side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 820px; margin: 0 auto; padding: 24px; color: #111; }
  .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
  .prompt { font-size: 1.05rem; margin-bottom: 12px; }
  audio { width: 100%; margin: 4px 0 10px; }
  button { padding: 8px 14px; margin: 4px 6px 4px 0; border-radius: 8px; border: 1px solid #bbb; background: #fafafa; cursor: pointer; }
  button.active { background: #2b6cb0; color: #fff; border-color: #2b6cb0; }
  .muted { color: #666; font-size: .9rem; }
  .progress { height: 6px; background: #eee; border-radius: 3px; overflow: hidden; margin-bottom: 16px; }
  .progress > div { height: 100%; width: 0; background: #2b6cb0; }
  #done { display: none; }
  .opt { border-left: 3px solid #eee; padding-left: 10px; margin: 10px 0; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="muted" id="instructions">__INSTRUCTIONS__</p>
<div class="card" id="startcard">
  <label>Rater ID: <input id="rater" type="text" placeholder="e.g. R01"></label>
  <div style="margin-top:10px"><button id="startbtn">Start</button></div>
</div>
<div class="progress" id="progress"><div></div></div>
<div id="trial"></div>
<div class="card" id="done">
  <h2>Done</h2>
  <p>Thank you. Download your responses and send the file to the experimenter.</p>
  <button id="download">Download responses</button>
</div>
<script>
const STUDY = __STUDY__;
let i = 0;
let rater = "";
let responses = [];
const el = (id) => document.getElementById(id);

function renderProgress() {
  const pct = STUDY.trials.length ? (i / STUDY.trials.length) * 100 : 0;
  el("progress").firstElementChild.style.width = pct + "%";
}

function audioBlock(label, src) {
  return `<div class="opt"><b>${label}</b><audio controls preload="none" src="${src}"></audio></div>`;
}

function render() {
  renderProgress();
  if (i >= STUDY.trials.length) {
    el("trial").innerHTML = "";
    el("done").style.display = "block";
    return;
  }
  const t = STUDY.trials[i];
  let html = `<div class="card"><div class="progress"><div style="width:${(i/STUDY.trials.length)*100}%"></div></div>`;
  html += `<div class="prompt">${t.prompt}</div>`;
  if (t.anchor) { html += audioBlock(t.anchor.label || "A", t.anchor.audio); }
  if (t.mode === "rating") {
    html += audioBlock(t.options[0].label || "B", t.options[0].audio);
    html += `<div>1 (very different) &nbsp;`;
    for (let s = 1; s <= STUDY.scale; s++) html += `<button class="rate" data-v="${s}">${s}</button>`;
    html += `&nbsp; ${STUDY.scale} (same motion)</div>`;
  } else {
    t.options.forEach((o) => { html += audioBlock(o.label || o.id, o.audio); });
    t.options.forEach((o) => { html += `<button class="choose" data-v="${o.id}">${o.label || o.id}</button>`; });
  }
  html += `<div style="margin-top:14px"><button id="next" disabled>Next</button></div></div>`;
  el("trial").innerHTML = html;

  let answer = null;
  el("trial").querySelectorAll(".rate,.choose").forEach((b) => {
    b.onclick = () => {
      answer = isNaN(Number(b.dataset.v)) ? b.dataset.v : Number(b.dataset.v);
      el("trial").querySelectorAll(".rate,.choose").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      el("next").disabled = false;
    };
  });
  el("next").onclick = () => {
    responses.push({ trial_id: t.trial_id, rater_id: rater, answer: answer });
    i += 1;
    render();
  };
}

el("startbtn").onclick = () => {
  rater = (el("rater").value || "anonymous").trim();
  el("startcard").style.display = "none";
  render();
};

el("download").onclick = () => {
  const blob = new Blob([JSON.stringify({ study_id: STUDY.study_id, responses: responses }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = STUDY.study_id + "_responses.json";
  a.click();
};
render();
</script>
</body>
</html>
"""


def render_study_html(study: dict[str, Any], title: str, instructions: str) -> str:
    return (
        _HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__INSTRUCTIONS__", instructions)
        .replace("__STUDY__", json.dumps(study, ensure_ascii=False))
    )


def write_static_study(
    out_dir: str | Path,
    study_id: str,
    trials: list[dict[str, Any]],
    title: str,
    instructions: str,
    scale: int = 7,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study = {"study_id": study_id, "scale": scale, "trials": trials}
    html = render_study_html(study, title, instructions)
    path = out_dir / f"{study_id}.html"
    path.write_text(html, encoding="utf-8")
    (out_dir / f"{study_id}_trials.json").write_text(
        json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path

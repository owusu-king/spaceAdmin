import { api, chip, errorBanner, esc, fmtDate, icon, modal, pct, skeleton, toast } from "../lib.js";
import { setCrumbs } from "../app.js";

const NAMES = { space_admin: ["Space Administrator Agent", "Finds and ranks suitable space"], outage: ["Outage Agent", "Reviews outages and conflicts"] };
const STRATEGY = { A_rule_based: "Rules", B_model_based: "Model", C_combined: "Rules + model" };

export async function render(main) {
  setCrumbs([{ label: "Agents" }]);
  main.innerHTML = `<div class="page-head"><div><h1>Agents</h1><p>Space and outage agents</p></div></div><div id="body">${skeleton(8)}</div>`;
  let ag, ev, ex, act;
  try { [ag, ev, ex, act] = await Promise.all([api("/agents"), api("/evaluation"), api("/experiments"), api("/activity?limit=200")]); }
  catch (e) { main.querySelector("#body").innerHTML = errorBanner(e); return; }
  const fig = (label, v, cap) => `<div class="kpi"><div class="overline">${label}</div><div class="fig">${v}</div><div class="cap">${cap}</div></div>`;
  const comp = ex?.competition, coop = ex?.cooperation?.summary;
  main.querySelector("#body").innerHTML = `
    <div class="kpis">
      ${fig("Evaluation", ev ? `${ev.passed}/${ev.total}` : "–", ev ? "Checks passed" : "Not run")}
      ${fig("Cooperation", coop ? pct(coop.coop_share_with_options) : "–", coop ? `With options · ${pct(coop.solo_share_with_options)} alone` : "Not run")}
      ${fig("Competition", comp ? comp.results[comp.winner].f1.toFixed(2) : "–", comp ? `Best F1 · ${STRATEGY[comp.winner]}` : "Not run")}
      ${fig("Activity", act.audit.length, "Audit events")}
    </div>
    <div class="cols">${ag.agents.map((a, i) => `
      <section class="card-list"><button class="cl-row" data-agent="${i}" style="padding:18px">
        <span class="cl-main"><b style="font:600 17px/24px var(--serif)">${NAMES[a.name][0]}</b>
          <span class="cl-sub">${NAMES[a.name][1]} · ${a.permissions.length} tools · ${Object.keys(a.locked_actions).length} locked actions</span></span>
        ${icon("chev", "chev")}</button></section>`).join("")}</div>
    <div class="acc">
      ${row("eval", "Evaluation", ev ? `${ev.scenarios.length} scenarios · ${fmtDate(ev.generated)}` : "Not run")}
      ${row("exp", "Experiments", "Cooperation and competition")}
      ${row("comm", "Communication", "Agent to agent")}
      ${row("log", "Activity log", `${act.audit.length} events`)}
    </div>`;
  main.querySelectorAll("[data-agent]").forEach(b => b.onclick = () => agentDialog(ag.agents[+b.dataset.agent]));
  const open = { eval: () => evalDialog(ev, () => render(main)), exp: () => expDialog(ex, () => render(main)), comm: commDialog, log: () => logDialog(act) };
  main.querySelectorAll("[data-row]").forEach(b => b.onclick = () => open[b.dataset.row]());
}

const row = (key, title, meta) => `<button class="acc-row" data-row="${key}"><span class="acc-title">${title}</span><span class="acc-meta">${meta}</span>${icon("chev", "chev")}</button>`;

function agentDialog(a) {
  modal({ title: NAMES[a.name][0], meta: NAMES[a.name][1], size: "lg", body: `
    <div class="grid-2" style="gap:32px">
      <div><div class="overline">Objectives</div><ul class="list">${a.objectives.map(o => `<li class="sec" style="font-size:13px">${esc(o)}</li>`).join("")}</ul>
        <div class="overline" style="margin-top:18px">Reads</div><ul class="list">${a.data_scopes.map(o => `<li class="sec" style="font-size:13px">${esc(o)}</li>`).join("")}</ul></div>
      <div><div class="overline">Tools · ${a.permissions.length}</div><ul class="list">${Object.entries(a.tools).map(([n, doc]) => `<li class="mono" style="font-size:12px" title="${esc(doc)}">${esc(n)}</li>`).join("")}</ul>
        <div class="overline" style="margin-top:18px">Locked</div><ul class="list">${Object.entries(a.locked_actions).map(([n, why]) => `<li title="${esc(why)}">${icon("lock")}<span class="mono" style="font-size:12px">${esc(n)}</span></li>`).join("")}</ul></div>
    </div>`, footer: `<button class="btn" data-close>Close</button>` });
}

function commDialog() {
  modal({ title: "Communication", body: `
    <ol class="list" style="padding-left:0">
      <li><span class="qnum">1</span><span class="grow">Outage Agent asks for temporary space, on request.</span></li>
      <li><span class="qnum">2</span><span class="grow">The coordinator checks it and runs the Space Agent.</span></li>
      <li><span class="qnum">3</span><span class="grow">Options return to the case.</span></li></ol>`, footer: `<button class="btn" data-close>Close</button>` });
}

function evalDialog(res, onDone) {
  const body = r => !r ? `<p class="muted">Not run yet.</p>` : `
    <table class="t plain"><thead><tr><th>Scenario</th><th class="r">Passed</th></tr></thead><tbody>
      ${r.scenarios.map(s => `<tr title="${esc(s.checks.map(c => `${c.passed ? "✓" : "✗"} ${c.check}`).join("\n"))}"><td>${esc(s.title)}${s.passed < s.total ? `<div style="margin-top:4px">${s.checks.filter(c => !c.passed).map(c => chip("conflicting", c.check)).join(" ")}</div>` : ""}</td>
        <td class="r">${s.passed === s.total ? chip("confirmed", `${s.passed}/${s.total}`) : `<span class="mono">${s.passed}/${s.total}</span>`}</td></tr>`).join("")}</tbody></table>`;
  const d = modal({ title: "Evaluation", meta: res ? `${res.passed}/${res.total} checks · ${fmtDate(res.generated, true)}` : "", size: "lg", body: body(res),
    footer: `<button class="btn" data-close>Close</button><button class="btn primary" id="run">${icon("refresh")}Run again</button>` });
  d.el.querySelector("#run").onclick = async e => {
    const b = e.currentTarget, label = b.innerHTML;
    b.disabled = true; b.textContent = "Running…";
    try { const r = await api("/evaluation/run", { method: "POST" }); d.body.innerHTML = body(r); d.el.querySelector(".dlg-meta").textContent = `${r.passed}/${r.total} checks · just now`; toast("Evaluation complete"); onDone(); }
    catch (err) { toast(err.message); }
    b.disabled = false; b.innerHTML = label;
  };
}

function expDialog(res, onDone) {
  const body = r => {
    if (!r) return `<p class="muted">Not run yet.</p>`;
    const s = r.cooperation.summary, c = r.competition;
    return `<div class="overline">Cooperation · temporary space</div>
      <dl class="kv" style="margin-top:8px"><dt>Outages tested</dt><dd>${s.outages_evaluated}</dd>
        <dt>Options found</dt><dd>${pct(s.coop_share_with_options)} together · ${pct(s.solo_share_with_options)} alone</dd>
        <dt>Constraints met</dt><dd>${pct(s.constraint_compliance)}</dd><dt>Added time</dt><dd>${s.mean_added_latency_ms} ms</dd></dl>
      <div class="overline" style="margin-top:24px">Competition · records to verify</div>
      <table class="t plain" style="margin-top:8px"><thead><tr><th>Strategy</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1</th></tr></thead><tbody>
        ${Object.entries(c.results).map(([k, v]) => `<tr ${k === c.winner ? 'class="sel"' : ""}><td>${STRATEGY[k] || k}${k === c.winner ? " " + chip("confirmed", "Best") : ""}</td>
          <td class="r mono">${v.precision.toFixed(3)}</td><td class="r mono">${v.recall.toFixed(3)}</td><td class="r mono">${v.f1.toFixed(3)}</td></tr>`).join("")}</tbody></table>
      <p class="muted" style="font-size:13px;margin:12px 0 0">${c.setup.injected_errors} synthetic errors.</p>`;
  };
  const d = modal({ title: "Experiments", size: "lg", body: body(res),
    footer: `<button class="btn" data-close>Close</button><button class="btn primary" id="run">${icon("refresh")}Run again</button>` });
  d.el.querySelector("#run").onclick = async e => {
    const b = e.currentTarget, label = b.innerHTML;
    b.disabled = true; b.textContent = "Running…";
    try { d.body.innerHTML = body(await api("/experiments/run", { method: "POST" })); toast("Experiments complete"); onDone(); }
    catch (err) { toast(err.message); }
    b.disabled = false; b.innerHTML = label;
  };
}

function logDialog(act) {
  modal({ title: "Activity log", size: "lg", body: act.audit.length
    ? `<table class="t plain"><thead><tr><th>Time</th><th>Action</th><th>Case</th><th>Detail</th></tr></thead><tbody>
      ${act.audit.slice(0, 60).map(a => `<tr title="${esc(a.actor)} · run ${esc(a.run_id)}"><td class="mono" style="font-size:12px;white-space:nowrap">${esc(a.ts.slice(5, 16).replace("T", " "))}</td>
        <td>${esc(a.action.replace(/_/g, " "))}</td><td>${a.case_id ? `<a class="mono" href="#/cases/${esc(a.case_id)}" data-close>${esc(a.case_id)}</a>` : ""}</td>
        <td class="sec" style="font-size:13px">${esc(a.detail)}</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">No activity yet.</p>`, footer: `<button class="btn" data-close>Close</button>` });
}

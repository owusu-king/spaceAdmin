import { api, chip, errorBanner, esc, fmtDate, fmtWindow, icon, modal, openPanel, skeleton, toast } from "../lib.js";
import { setCrumbs } from "../app.js";

let VIEW = "upcoming";

export async function render(main) {
  setCrumbs([{ label: "Outages" }]);
  main.innerHTML = `
    <div class="page-head"><div><h1>Outages</h1><p>Notices, conflicts and history</p></div>
      <div class="actions"><button class="btn primary" id="rev">${icon("plus")}Review outage notice</button></div></div>
    <div id="body">${skeleton(8)}</div>`;
  main.querySelector("#rev").onclick = () => newOutageCase();
  let d, ev, inbox;
  try { [d, ev, inbox] = await Promise.all([api("/outages"), api("/events"), api("/inbox")]); }
  catch (e) { main.querySelector("#body").innerHTML = errorBanner(e); return; }
  const now = new Date();
  const upcoming = d.history.filter(h => h.start && new Date(h.end || h.start) >= now).reverse();
  const past = d.history.filter(h => !h.start || new Date(h.end || h.start) < now);
  const blocked = ev.counts.blocked || 0;
  const evTotal = Object.values(ev.counts).reduce((a, b) => a + b, 0);
  const fact = (label, value, id) => `<button class="fact fact-btn" ${id ? `id="${id}"` : ""}><div class="overline">${label}</div><div class="fact-v">${value}</div></button>`;
  main.querySelector("#body").innerHTML = `
    <div class="facts" style="margin-top:0;grid-template-columns:repeat(3,minmax(0,1fr))">
      ${fact("Calendar", `${evTotal} events${blocked ? ` · ${blocked} blocked` : ""}`, "f-ev")}
      ${fact("Email", inbox.source.status === "ok" ? `${inbox.relevant} outage-related` : chip("unknown", "Not connected"), "f-em")}
      ${fact("Recurring", `${d.forecast.length} estimate${d.forecast.length === 1 ? "" : "s"}${d.forecast_accuracy?.building?.n ? ` <span class="muted">· typical error ${d.forecast_accuracy.building.median_error_days} d</span>` : ""}`, "f-fc")}
    </div>
    <div class="section-head section"><div class="seg inline" role="tablist">
        <button data-view="upcoming" class="${VIEW === "upcoming" ? "on" : ""}">Upcoming · ${upcoming.length}</button>
        <button data-view="history" class="${VIEW === "history" ? "on" : ""}">History · ${past.length}</button></div></div>
    <div id="tbl">${table(VIEW === "upcoming" ? upcoming : past)}</div>`;
  const rows = { upcoming, history: past };
  const bindRows = () => main.querySelectorAll("[data-row]").forEach(tr => {
    const open = () => openOutage(rows[VIEW][+tr.dataset.row]);
    tr.onclick = open; tr.onkeydown = e => e.key === "Enter" && open();
  });
  main.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {
    VIEW = b.dataset.view;
    main.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("on", x === b));
    main.querySelector("#tbl").innerHTML = table(rows[VIEW]);
    bindRows();
  });
  bindRows();
  main.querySelector("#f-ev").onclick = () => openEvents(() => render(main));
  main.querySelector("#f-em").onclick = () => openEmail(() => render(main));
  main.querySelector("#f-fc").onclick = () => openForecast(d.forecast, d.forecast_accuracy);
}

function status(h) {
  return h.kinds.includes("planned_notification") || h.kinds.includes("unscheduled_alert") ? chip("confirmed", "Confirmed") : chip("attention", "Pending approval");
}

function table(rows) {
  if (!rows.length) return `<div class="panel fade-in"><div class="cl-empty">Nothing on record.</div></div>`;
  return `<div class="table-wrap fade-in"><table class="t fixed"><colgroup><col style="width:24%"><col style="width:28%"><col style="width:22%"><col style="width:12%"><col style="width:14%"></colgroup><thead><tr><th>When</th><th>Where</th><th>Systems</th><th>Request</th><th>Status</th></tr></thead><tbody>
    ${rows.map((h, i) => `<tr class="click" data-row="${i}" tabindex="0"><td class="num" style="white-space:nowrap">${fmtWindow(h.start, h.end)}</td>
      <td>${h.buildings.length ? esc(h.buildings.join(", ")) : `<span class="muted">${esc((h.buildings_raw || "Not identified").slice(0, 40))}</span>`}</td>
      <td>${esc(h.systems.slice(0, 2).join(", "))}${h.systems.length > 2 ? ` <span class="muted">+${h.systems.length - 2}</span>` : ""}</td>
      <td class="mono" style="font-size:12px">${esc(h.sr)}</td><td>${status(h)}</td></tr>`).join("")}
    </tbody></table></div>`;
}

/** Outage request detail dialog (also used from the Overview). */
export function openOutage(h) {
  const d = modal({ overline: esc(h.sr), title: esc(h.buildings.join(", ") || "Building not identified"),
    meta: `${status(h)}${h.revised ? " " + chip("historical", "Revised") : ""}${h.buildings_partial ? " " + chip("unknown", "Building list truncated") : ""}`,
    body: `<dl class="kv">
      <dt>When</dt><dd>${fmtWindow(h.start, h.end)}${h.duration_hours ? ` <span class="muted">· ${h.duration_hours} h</span>` : ""}</dd>
      <dt>Systems</dt><dd>${esc(h.systems.join(", "))}</dd>
      <dt>Priority</dt><dd>${esc(h.priority || "")}</dd>
      ${h.reason ? `<dt>Reason</dt><dd>${esc(h.reason)}</dd>` : ""}
      ${h.scope ? `<dt>What to expect</dt><dd>${esc(h.scope)}</dd>` : ""}
      ${h.exclusions?.length ? `<dt>Not affected</dt><dd>${h.exclusions.map(esc).join("<br>")}</dd>` : ""}
      <dt>Notices</dt><dd>${h.messages} message${h.messages === 1 ? "" : "s"} · first ${fmtDate(h.first_notice)}</dd></dl>`,
    footer: `<button class="btn" data-close>Close</button><button class="btn primary" id="rv">Review with Outage Agent</button>` });
  d.el.querySelector("#rv").onclick = e => { e.currentTarget.disabled = true; e.currentTarget.textContent = "Analyzing…"; startFromSr(h.sr).then(d.close); };
}

function openForecast(rows, accAll) {
  const pctTxt = v => `${Math.round(v * 100)}%`;
  const acc = accAll?.building, sys = accAll?.systems;
  const accuracy = acc?.n ? `
    <div class="kpis mini" style="grid-template-columns:repeat(3,1fr);margin-bottom:8px">
      <div class="kpi"><div class="overline">Typical error</div><div class="fig">${acc.median_error_days} d</div><div class="cap">Median · baseline ${acc.baseline_median_error_days ?? "–"} d</div></div>
      <div class="kpi"><div class="overline">Inside window</div><div class="fig">${pctTxt(acc.within_window)}</div><div class="cap">Actual date within ± window</div></div>
      <div class="kpi"><div class="overline">Tested</div><div class="fig">${acc.n}</div><div class="cap">Past outages · ${acc.groups} series</div></div>
    </div>
    <p class="muted" style="font-size:12px;margin:0 0 12px">Tested on the next scheduled outage in each building, each predicted from earlier outages only. Baseline: one shared interval for every building.
      ${sys?.n ? "" : "The same-system estimates below have too few repeats to test yet."} Small sample: indicative only.</p>
    <div class="overline" style="margin-bottom:8px">Tested predictions</div>
      <table class="t plain" style="margin-bottom:24px"><thead><tr><th>Building</th><th>Predicted</th><th>Actual</th><th class="r">Error</th><th class="r">Baseline</th></tr></thead><tbody>
      ${acc.trials.map(t => `<tr><td>${esc(t.building)}</td><td>${fmtDate(t.predicted)} <span class="muted">± ${t.window_days} d</span></td><td>${fmtDate(t.actual)}</td>
        <td class="r mono">${t.error_days} d</td><td class="r mono muted">${t.baseline_error_days ?? "–"} d</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted" style="font-size:13px;margin:0 0 16px">Not enough repeated maintenance on record to measure accuracy.</p>`;
  modal({ overline: "Prediction", title: "Recurring maintenance", size: "lg",
    body: accuracy + (rows.length ? `<div class="overline" style="margin-bottom:8px">Next estimates</div><table class="t plain"><thead><tr><th>Building</th><th>Systems</th><th>Around</th><th class="r">Seen</th></tr></thead><tbody>
      ${rows.map(f => `<tr><td>${esc(f.building)}</td><td>${esc(f.systems.join(", "))}</td><td>${fmtDate(f.estimated_next)} <span class="muted">± ${f.window_days} d</span></td><td class="r">${f.occurrences.length}</td></tr>`).join("")}
      </tbody></table><p class="muted" style="font-size:13px;margin:14px 0 0">Median interval between scheduled outages on record. Not confirmed dates.</p>`
      : `<p class="muted">No estimate available.</p>`),
    footer: `<button class="btn" data-close>Close</button>` });
}

// ------------------------------------------------------------------ calendar (events and blocked times)
const SRC = { correspondence: "Email", ics: "Calendar", manual: "Manual", blocked: "Blocked" };

async function openEvents(onChange) {
  const d = modal({ overline: "Used for conflict checks", title: "Events and blocked times", size: "lg", body: skeleton(4),
    footer: `<label class="btn" style="margin-right:auto">${icon("file")}Import .ics<input type="file" id="ev-ics" accept=".ics,text/calendar" hidden></label><button class="btn" data-close>Close</button>` });
  let changed = false;
  const draw = async () => {
    const ev = await api("/events");
    d.body.innerHTML = `
      <div class="filters" style="margin:0 0 16px">
        <input class="input" id="ev-title" placeholder="Title" style="flex:1;min-width:180px" aria-label="Title">
        <input class="input" id="ev-bldg" placeholder="Building (optional)" aria-label="Building">
        <input class="input" type="datetime-local" id="ev-start" aria-label="Start">
        <input class="input" type="datetime-local" id="ev-end" aria-label="End">
        <select class="input" id="ev-kind" aria-label="Type"><option value="event">Event</option><option value="blocked">Blocked</option></select>
        <button class="btn sm primary" id="ev-add">Add</button></div>
      ${ev.events.length ? `<table class="t plain"><thead><tr><th>When</th><th>Event</th><th>Building</th><th>Source</th><th></th></tr></thead><tbody>
        ${ev.events.map(e => `<tr><td class="num" style="white-space:nowrap">${e.all_day ? fmtDate(e.start) : fmtWindow(e.start, e.end)}</td>
          <td title="${esc(e.note || "")}">${esc(e.title)}</td><td>${esc(e.building || "Campus-wide")}</td>
          <td>${e.source === "blocked" ? chip("conflicting", "Blocked") : `<span class="tag">${esc(SRC[e.source] || e.source)}</span>`}</td>
          <td class="r">${e.source !== "correspondence" ? `<button class="btn ghost sm" data-del="${e.id}" aria-label="Remove">${icon("x")}</button>` : ""}</td></tr>`).join("")}
        </tbody></table>` : `<p class="muted">No upcoming events.</p>`}`;
    d.body.querySelector("#ev-add").onclick = async () => {
      const v = id => d.body.querySelector(id).value.trim();
      if (!v("#ev-title") || !v("#ev-start") || !v("#ev-end")) return toast("Title, start and end are required");
      try { await api("/events", { method: "POST", body: { title: v("#ev-title"), building: v("#ev-bldg") || null, start: v("#ev-start"), end: v("#ev-end"), kind: v("#ev-kind") } }); changed = true; toast("Added"); draw(); }
      catch (e) { toast(e.message); }
    };
    d.body.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
      try { await api(`/events/${b.dataset.del}`, { method: "DELETE" }); changed = true; draw(); } catch (e) { toast(e.message); }
    });
  };
  d.el.querySelector("#ev-ics").onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    try { const r = await api("/events/import", { method: "POST", body: { ics: await f.text() } }); changed = true; toast(`${r.imported} event(s) imported`); draw(); }
    catch (err) { toast(err.message); }
  };
  d.el.querySelector("[data-close]").addEventListener("click", () => changed && onChange && onChange());
  draw();
}

// ------------------------------------------------------------------ email intake (read-only)
async function openEmail(onChange) {
  const d = modal({ overline: "Read-only", title: "Email intake", size: "lg", body: skeleton(4),
    footer: `<button class="btn" data-close>Close</button><button class="btn primary" id="poll">${icon("refresh")}Check for new email</button>` });
  let changed = false;
  const draw = s => {
    const src = s.source;
    const state = src.status === "ok" ? chip("confirmed", src.mode === "imap" ? "Connected" : "Replay of supplied email")
      : chip("unknown", src.status === "not_configured" ? "Not configured" : "Connection error");
    d.el.querySelector(".dlg-meta")?.remove();
    d.body.innerHTML = `<div class="row-between" style="margin-bottom:14px">${state}
        <span class="muted" style="font-size:13px">${s.processed} processed · ${s.relevant} outage-related${src.mode === "replay" ? ` · ${src.remaining} left` : ""}</span></div>
      ${s.items.length ? `<table class="t plain"><thead><tr><th>Sent</th><th>From</th><th>Subject</th><th>Result</th></tr></thead><tbody>
        ${s.items.slice(0, 12).map(r => `<tr><td class="num" style="white-space:nowrap">${fmtDate(r.sent)}</td><td>${esc((r.sender || "").split("<")[0])}</td>
          <td>${esc(r.subject)}</td><td>${r.case_id ? `<a class="mono" href="#/cases/${esc(r.case_id)}" data-close>${esc(r.case_id)}</a>` : `<span class="muted">${r.relevant ? "" : "Not an outage"}</span>`}</td></tr>`).join("")}
        </tbody></table>` : `<p class="muted">No email processed yet.</p>`}`;
    d.el.querySelector("#poll").disabled = src.status !== "ok";
  };
  try { draw(await api("/inbox")); } catch (e) { d.body.innerHTML = errorBanner(e); }
  d.el.querySelector("#poll").onclick = async e => {
    const b = e.currentTarget; b.disabled = true;
    try {
      const r = await api("/inbox/poll?limit=10", { method: "POST" });
      changed = true;
      toast(`${r.fetched} message(s) · ${r.cases_opened.length} opened · ${r.cases_updated.length} updated`);
      draw(r.status);
    } catch (err) { toast(err.message); b.disabled = false; }
  };
  d.el.querySelectorAll("[data-close]").forEach(x => x.addEventListener("click", () => changed && onChange && onChange()));
}

async function startFromSr(sr) {
  const src = await api("/sources");
  const m = src.messages.find(x => x.sr === sr && /Notification|ALERT|Approval/.test(x.subject)) || src.messages.find(x => x.sr === sr);
  if (!m) return toast("No source message on record for this request");
  return createCase({ message_id: m.id });
}

async function createCase(input, btn) {
  if (btn) { btn.disabled = true; btn.textContent = "Analyzing…"; }
  try {
    const c = await api("/cases", { method: "POST", body: { kind: "outage", input } });
    document.querySelector(".scrim")?.remove();
    toast(`${c.id} created · Rev A ready for review`);
    location.hash = `#/cases/${c.id}`;
  } catch (e) {
    toast(e.message);
    if (btn) { btn.disabled = false; btn.textContent = "Run analysis"; }
  }
}

export async function newOutageCase() {
  const p = openPanel(`<div class="dr-head"><div><div class="overline">Outage Agent</div><h2>Review an outage notice</h2></div>
      <button class="btn ghost sm" data-close style="margin-left:auto" aria-label="Close">${icon("x")}</button></div>
    <div class="dr-body">${skeleton(6)}</div>
    <div class="dr-foot"><button class="btn" data-close>Cancel</button><button class="btn primary" id="go">Run analysis</button></div>`, { modal: true });
  let src;
  try { src = await api("/sources"); } catch (e) { p.el.querySelector(".dr-body").innerHTML = errorBanner(e); return; }
  p.el.querySelector(".dr-body").innerHTML = `
    <div class="tabs" style="margin-top:0" role="tablist">
      <button class="on" data-t="msg">Notice on record</button><button data-t="thread">Whole thread</button><button data-t="paste">Paste text</button></div>
    <div data-p="msg"><div class="field"><label for="ms">Message</label><input class="input" id="mf" placeholder="Filter by subject, building or SR" style="margin-bottom:8px">
      <select class="input" id="ms" size="12" style="height:auto">${src.messages.map(m => `<option value="${m.id}">${esc(fmtDate(m.date))} · ${esc(m.subject)}</option>`).join("")}</select></div></div>
    <div data-p="thread" hidden><div class="field"><label for="th">Correspondence thread</label>
      <select class="input" id="th">${src.threads.map(t => `<option>${esc(t)}</option>`).join("")}</select>
</div></div>
    <div data-p="paste" hidden><div class="field"><label for="tx">Outage email or TRIRIGA notice</label>
      <textarea class="input" id="tx" rows="12" placeholder="From: …&#10;Subject: …&#10;Date: …&#10;&#10;Body"></textarea>
</div></div>
`;
  let tab = "msg";
  p.el.querySelectorAll("[data-t]").forEach(b => b.onclick = () => {
    tab = b.dataset.t;
    p.el.querySelectorAll("[data-t]").forEach(x => x.classList.toggle("on", x === b));
    p.el.querySelectorAll("[data-p]").forEach(x => x.hidden = x.dataset.p !== tab);
  });
  const ms = p.el.querySelector("#ms"); ms.selectedIndex = 0;
  p.el.querySelector("#mf").oninput = e => {
    const q = e.target.value.toLowerCase();
    [...ms.options].forEach(o => o.hidden = q && !o.text.toLowerCase().includes(q));
  };
  p.el.querySelector("#go").onclick = e => {
    const input = tab === "msg" ? { message_id: Number(ms.value) } : tab === "thread" ? { thread: p.el.querySelector("#th").value }
      : { text: p.el.querySelector("#tx").value, source: "pasted" };
    if (tab === "paste" && input.text.trim().length < 20) return toast("Paste the notice text first");
    if (tab === "msg" && !ms.value) return toast("Choose a message");
    createCase(input, e.target);
  };
}

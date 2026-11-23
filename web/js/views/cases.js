import { api, chip, empty, errorBanner, esc, fmtDate, fmtNum, fmtWindow, icon, miniBar, modal, notOnRecord, pct, roomKey, skeleton, toast, val } from "../lib.js";
import { refreshCounts, setCrumbs } from "../app.js";
import { newOutageCase } from "./outages.js";
import { newSpaceRequest } from "./spaces.js";

export async function render(main, [id]) {
  return id ? workspace(main, id) : list(main);
}

// ===================================================================== case list
let LIST_VIEW = "open";
const OPEN = ["ReadyForReview", "NeedsRevision"];

async function list(main) {
  setCrumbs([{ label: "Cases" }]);
  main.innerHTML = `<div class="page-head"><div><h1>Cases</h1><p>Agent analyses for your review</p></div>
    <div class="actions"><button class="btn" id="ns">${icon("plus")}Space request</button><button class="btn primary" id="no">${icon("plus")}Outage notice</button></div></div>
    <div id="body">${skeleton(5)}</div>`;
  main.querySelector("#ns").onclick = () => newSpaceRequest();
  main.querySelector("#no").onclick = () => newOutageCase();
  let rows;
  try { rows = await api("/cases"); } catch (e) { main.querySelector("#body").innerHTML = errorBanner(e); return; }
  if (!rows.length) { main.querySelector("#body").innerHTML = `<div class="panel">${empty("No cases yet. Each request or notice an agent analyzes becomes a case.")}</div>`; return; }
  const groups = { open: rows.filter(c => OPEN.includes(c.status)), closed: rows.filter(c => !OPEN.includes(c.status)), all: rows };
  const labels = { open: "To review", closed: "Decided", all: "All" };
  const item = c => `<button class="cl-row" data-case="${esc(c.id)}">
      <span class="cl-main"><b>${esc(c.title)}</b><span class="cl-sub"><span class="mono">${esc(c.id)}</span> · ${esc(c.reference)} · Rev ${esc(c.rev_letter)} · ${fmtDate(c.updated)}${c.open_questions ? ` · ${c.open_questions} question${c.open_questions === 1 ? "" : "s"}` : ""}</span></span>
      ${c.status === "ReadyForReview" ? "" : chip(c.status)}</button>`;
  const draw = () => {
    const list = groups[LIST_VIEW];
    const box = main.querySelector("#list");
    box.classList.remove("fade-in"); void box.offsetWidth; box.classList.add("fade-in");
    box.innerHTML = list.length ? list.map(item).join("") : `<div class="cl-empty">${LIST_VIEW === "open" ? "Nothing waiting for review." : "No cases here."}</div>`;
    main.querySelectorAll("[data-case]").forEach(b => b.onclick = () => location.hash = `#/cases/${b.dataset.case}`);
  };
  main.querySelector("#body").innerHTML = `
    <div class="section-head"><div class="seg inline" role="tablist">
      ${Object.keys(labels).map(k => `<button data-lv="${k}" class="${LIST_VIEW === k ? "on" : ""}">${labels[k]} · ${groups[k].length}</button>`).join("")}</div></div>
    <section class="card-list" id="list"></section>`;
  main.querySelectorAll("[data-lv]").forEach(b => b.onclick = () => {
    LIST_VIEW = b.dataset.lv;
    main.querySelectorAll("[data-lv]").forEach(x => x.classList.toggle("on", x === b));
    draw();
  });
  draw();
}

// ===================================================================== workspace
// Layout: header (title, meta, stepper, actions) → facts strip → needs attention → collapsible sections.
async function workspace(main, id) {
  setCrumbs([{ label: "Cases", href: "#/cases" }, { label: id }]);
  main.innerHTML = skeleton(10);
  let c;
  try { c = await api(`/cases/${encodeURIComponent(id)}`); } catch (e) { main.innerHTML = errorBanner(e); return; }
  setCrumbs([{ label: "Cases", href: "#/cases" }, { label: c.id }]);
  const o = c.output;
  SECTIONS = {};
  CANDIDATES = o.candidates || [];
  SEND = { ...(c.sending || {}), approved: c.status === "Approved", rev: c.rev, revLetter: c.rev_letter, sent: c.sent || [] };
  main.innerHTML = `
    ${header(c)}
    ${c.status === "NeedsRevision" ? `<div class="banner attention" style="margin-top:16px">${icon("attention")}<div><b>Returned:</b> ${esc(c.reviews.at(-1)?.rationale || "")}</div></div>` : ""}
    ${c.kind === "space" ? spaceCase(o) : outageCase(o, c.id)}`;
  wire(main, c);
}


function header(c) {
  const open = c.status === "ReadyForReview";
  const actions = open
    ? `<button class="btn ghost sm" id="rerun">${icon("refresh")}Re-run</button>
       <button class="btn sm danger" id="return">Return / Decline</button>
       <button class="btn sm primary" id="approve">Approve Rev ${esc(c.rev_letter)}</button>`
    : c.status === "NeedsRevision" ? `<button class="btn sm primary" id="rerun">${icon("refresh")}Re-run analysis</button>` : "";
  return `<header class="case-head">
    <div class="case-id">
      <div class="overline">${c.kind === "space" ? "Space request" : "Outage"}</div>
      <h1 class="case-title">${esc(c.title)}</h1>
      <div class="case-meta"><span class="mono">${esc(c.id)}</span><span>Rev ${esc(c.rev_letter)}</span>${chip(c.status)}
        <span>Updated ${fmtDate(c.updated, true)}</span>
        ${c.kind === "outage" && c.output.timeline?.length ? `<button class="linkish" id="timeline">Timeline (${c.output.timeline.length})</button>` : ""}
        ${c.reviews.length ? `<button class="linkish" id="history">History (${c.reviews.length})</button>` : ""}</div>
    </div>
    <div class="case-side">
      <div class="case-actions">${actions}</div>
    </div>
  </header>`;
}

const fact = (label, value, title = "") => `<div class="fact" ${title ? `title="${esc(title)}"` : ""}><div class="overline">${label}</div><div class="fact-v">${value}</div></div>`;

let SECTIONS = {};
let DETAIL_KEYS = [];
let CANDIDATES = [];
let SEND = {};   // send state for the open case: switch, approval, revision, what was already sent

/** Register a section body. Details sections are opened one at a time from the Case details tiles. */
function section(key, title, meta, body, size = "lg") {
  SECTIONS[key] = { title, meta, body, size };
  return "";
}

/** The secondary layer: one card whose tiles open the tabbed details dialog on that tab. */
function detailsCard(keys) {
  DETAIL_KEYS = keys.filter(k => SECTIONS[k]);
  return `<section class="details-card"><header><h2>Case details</h2></header>
    <div class="details-tiles" style="--cols:${DETAIL_KEYS.length <= 4 ? DETAIL_KEYS.length : 3}">${DETAIL_KEYS.map(k => `<button class="d-tile" data-detail="${k}"><span class="d-title">${SECTIONS[k].title}</span><span class="d-meta">${SECTIONS[k].meta}</span></button>`).join("")}</div></section>`;
}

function attention(items) {
  if (!items.length) return "";
  return `<section class="attn"><div class="overline">Needs attention</div><ul>${items.join("")}</ul></section>`;
}

const qItem = q => `<li>${icon("unknown")}<span class="grow">${esc(q)}</span><button class="btn ghost sm" data-copy="${esc(q)}" aria-label="Copy question">${icon("copy")}</button></li>`;

// ------------------------------------------------------------------ outage case
function outageCase(o, caseId) {
  const out = o.outage, s = o.scheduling, imp = o.impacts;
  const when = out.date_only ? `${fmtDate(out.proposed_start)} <span class="muted">· time not stated</span>`
    : out.proposed_start ? fmtWindow(out.proposed_start, out.proposed_end) : notOnRecord("Not stated");
  const verdict = !s.window ? chip("unknown", "No date") : s.verdict === "conflict"
    ? chip("conflicting", `Conflict · ${s.conflicts.length}`) : chip("confirmed", "No known conflict");
  const qs = o.follow_up_questions;
  const att = [
    ...s.conflicts.slice(0, 2).map(c => `<li>${icon("conflicting")}<span class="grow"><b>${esc(c.title.split(" (")[0])}</b> <span class="muted">· ${fmtDate(c.start)}</span></span><button class="btn ghost sm" data-open="schedule">Reschedule</button></li>`),
    ...(out.buildings.length === 0 ? [`<li>${icon("unknown")}<span class="grow">Building not identified</span></li>`] : []),
    ...qs.slice(0, 2).map(qItem),
  ];
  const people = o.affected_people;
  const nsDepts = o.affected_departments.natural_sciences.length;
  const contacts = o.stakeholders.departments.reduce((n, d) => n + d.coordination_contacts.length, 0);
  return `
    <div class="facts" style="grid-template-columns:repeat(3, minmax(0,1fr))">
      ${fact("When", `${when}${out.duration_hours ? ` <span class="muted">· ${out.duration_hours} h</span>` : ""}`)}
      ${fact("Schedule", verdict)}
      ${fact("Request", `${out.request_reference ? `<span class="mono">${esc(out.request_reference)}</span>` : "Email"}${out.priority ? ` <span class="muted">· ${esc(out.priority)}</span>` : ""}`)}
    </div>
    ${attention(att)}
    ${people ? section("people", "Affected people", `${people.natural_sciences} in Natural Sciences · ${people.total} in all departments`, affectedPeople(people, caseId)) : ""}
    ${section("drafts", "Drafts", draftsMeta(o.communication_drafts.length), drafts(o.communication_drafts, caseId), "md")}
    ${notifyCard(o, people, nsDepts)}
    ${section("details", "Notice", "Reason, scope, approvals", outageDetails(o))}
    ${section("schedule", "Schedule", scheduleMeta(s), `<div data-buildings="${esc(JSON.stringify(out.buildings.map(b => b.export_name)))}" data-sr="${esc(out.request_reference || "")}">${scheduling(s, o)}</div>`)}
    ${section("impact", "Impact", impactMeta(imp), impactBody(imp))}
    ${section("contacts", "Departments", `${nsDepts} Natural Sciences · ${contacts} contact${contacts === 1 ? "" : "s"}`, stakeholders(o.stakeholders))}
    ${qs.length ? section("questions", "Questions", `${qs.length} open`, `<ul class="qlist">${qs.map(qItem).join("")}</ul>`) : ""}
    ${o.space_support ? section("space", "Temporary space", `${o.space_support.reply.payload.candidates?.length || 0} options`, support(o.space_support)) : ""}
    ${section("activity", "Evidence", `${o.findings.length} findings · ${o.trace.length} tool calls`, evidenceActivity(o))}
    ${detailsCard(["details", "schedule", "impact", "contacts", "questions", "space", "activity"])}`;
}

/** "2 prepared · not sent" or "2 prepared · 1 sent" (current revision only). */
function draftsMeta(n) {
  const sent = (SEND.sent || []).filter(x => x.rev === SEND.rev).length;
  return `${n} draft${n === 1 ? "" : "s"} prepared · ${sent ? `${sent} sent` : "not sent"}`;
}

/** Primary block of an outage case: who is affected and the messages prepared for them. */
function notifyCard(o, people, nsDepts) {
  const n = o.communication_drafts.length;
  return `<section class="primary-card">
    <div class="pc-main">
      <div class="overline">Notify affected people</div>
      <div class="pc-fig">${people ? `${fmtNum(people.natural_sciences)} <span>people in Natural Sciences</span>` : `<span>${notOnRecord("No people on record")}</span>`}</div>
      <div class="pc-sub">${people ? `${fmtNum(people.total)} in all departments · ` : ""}${nsDepts} department${nsDepts === 1 ? "" : "s"} · ${draftsMeta(n)}</div>
    </div>
    <div class="pc-actions">
      ${people ? `<button class="btn" data-sec="people">View people</button>` : ""}
      <button class="btn primary" data-sec="drafts">Review drafts</button>
    </div>
  </section>`;
}

function scheduleMeta(s) {
  if (!s.window) return "No date yet";
  const v = s.verdict === "conflict" ? `${s.conflicts.length} conflict${s.conflicts.length === 1 ? "" : "s"}` : "No known conflict";
  return s.alternatives?.length ? `${v} · ${s.alternatives.length} other times` : v;
}

function impactMeta(imp) {
  const p = [];
  if (imp.confirmed.length) p.push(`${imp.confirmed.length} confirmed`);
  if (imp.potential.length) p.push(`${imp.potential.length} potential`);
  if (imp.explicitly_unaffected.length) p.push(`${imp.explicitly_unaffected.length} unaffected`);
  if (imp.preliminary_superseded.length) p.push(`${imp.preliminary_superseded.length} preliminary`);
  return p.join(" · ") || "Nothing recorded";
}

function impactBody(imp) {
  const groups = [["confirmed", "Confirmed", "confirmed"], ["potential", "Potential", "inferred"],
    ["explicitly_unaffected", "Not affected", "unaffected"], ["unknown", "Unknown", "unknown"], ["preliminary_superseded", "Preliminary", "historical"]];
  return groups.filter(([k]) => imp[k].length).map(([k, label, ch]) => `<div class="group-head">${chip(ch, label)}</div>
    <ul class="list">${imp[k].map(i => `<li class="${k === "explicitly_unaffected" ? "row-unaffected" : ""}"><span class="grow">${esc(i.statement)}</span>
      ${i.note ? `<span class="muted" style="font-size:12px">${esc(i.note)}</span>` : ""}</li>`).join("")}</ul>`).join("");
}

function outageDetails(o) {
  const out = o.outage;
  return `<dl class="kv">
      <dt>Buildings</dt><dd>${out.buildings.map(b => esc(b.name)).join("<br>") || notOnRecord()}</dd>
      <dt>Reason</dt><dd>${val(out.reason)}</dd>
      <dt>Scope of work</dt><dd>${val(out.scope_of_work)}</dd>
      <dt>Approvals</dt><dd>${o.required_approvals.map(esc).join("<br>")}</dd>
      ${o.maintenance_forecast.length ? `<dt>Recurring</dt><dd>${o.maintenance_forecast.map(f => `${esc(f.systems.join(", "))} around ${fmtDate(f.estimated_next)} ${chip("predicted")}`).join("<br>")}</dd>` : ""}
      ${o.related_outages.similar.length ? `<dt>Similar</dt><dd>${o.related_outages.similar.slice(-3).map(h => `<span class="mono">${esc(h.sr)}</span> · ${fmtDate(h.start)}`).join("<br>")}</dd>` : ""}
    </dl>`;
}

// ------------------------------------------------------------------ space case
function spaceCase(o) {
  const req = Object.fromEntries(o.requirements.map(r => [r.field, r.value]));
  const cands = o.candidates;
  const suitable = cands.filter(x => x.status === "potentially_suitable").length;
  const conditional = cands.filter(x => x.status === "conditional").length;
  const result = o.vacancies ? `${o.vacancies.length} vacancies` : cands.length ? `${suitable} suitable · ${conditional} conditional` : chip("unknown", "Deferred");
  const near = req["Close to"];
  const qs = o.verification_questions;
  return `
    <div class="facts">
      ${near ? fact("Close to", esc(near.split(" (")[0]), near) : fact("Scope", esc((req["Search scope"] || "").split(", ").slice(0, 2).join(", ")) || notOnRecord(), req["Search scope"] || "")}
      ${fact("Best candidate", cands[0] ? `<span class="mono">${esc(cands[0].room_key.split(" · ").slice(1).join(" · "))}</span>${cands[0].proximity ? ` <span class="muted">· ${esc(cands[0].proximity.label)}</span>` : ""}` : notOnRecord("None yet"))}
      ${fact("Result", result)}
      ${fact("Minimum area", req["Minimum area"] ? esc(req["Minimum area"]) : '<span class="muted">Any</span>')}
    </div>
    ${attention(qs.slice(0, 3).map(qItem))}
    ${o.vacancies ? `<div class="block">${vacancyTable(o.vacancies)}</div>` : `<div class="block">${candidateTable(cands)}</div>`}
    ${o.alternative_spaces?.length ? section("alt", "More candidates", `${o.alternative_spaces.length} more`, `<ul class="list">${o.alternative_spaces.map(a => `<li><span class="grow mono">${esc(a.space_id)}</span>${chip(a.status)}</li>`).join("")}</ul>`) : ""}
      ${qs.length ? section("questions", "Questions", `${qs.length} open`, `<ul class="qlist">${qs.map(qItem).join("")}</ul>`) : ""}
      ${section("requirements", "Requirements", `${o.requirements.length} requirements · ${o.required_approvals.length} approvals`,
        `<dl class="kv">${o.requirements.map(r => `<dt>${esc(r.field)}</dt><dd>${esc(r.value)} <span class="tag">${esc(r.source)}</span></dd>`).join("")}
         <dt>Approvals</dt><dd>${o.required_approvals.map(esc).join("<br>")}</dd>
         ${o.implications?.length ? `<dt>Implications</dt><dd>${o.implications.map(esc).join("<br>")}</dd>` : ""}</dl>`)}
      ${section("activity", "Evidence", `${o.findings.length} findings · ${o.trace.length} tool calls`, evidenceActivity(o))}
    ${detailsCard(["requirements", "questions", "alt", "activity"])}`;
}

function candidateTable(cands) {
  if (!cands.length) return `<div class="panel">${empty("No candidates yet. Answer the question above first.")}</div>`;
  const hasProx = cands.some(c => c.proximity);
  return `<div class="table-wrap"><table class="t cand"><thead><tr><th>#</th><th>Space</th>${hasProx ? "<th>Distance</th>" : ""}<th class="r">Area</th><th>Type</th><th>Status</th><th class="r">Score</th></tr></thead><tbody>
    ${cands.map((c, i) => `<tr class="click" data-cand="${i}" tabindex="0"><td class="mono muted">${i + 1}</td><td>${roomKey(c.room_key)}</td>
      ${hasProx ? `<td>${c.proximity ? esc(c.proximity.label) : notOnRecord()}</td>` : ""}
      <td class="r">${c.area == null ? notOnRecord() : fmtNum(c.area)}</td><td>${esc(className(c.space_class))}</td>
      <td>${chip(c.status)}</td><td class="r"><span class="score" title="Evidence coverage ${pct(c.evidence_coverage)}${c.provisional ? " (provisional)" : ""}">${Math.round(c.suitability * 100)}</span></td></tr>`).join("")}
    </tbody></table></div>`;
}

const className = s => (s || "").split(" - ").slice(1).join(" - ") || s || "";

function candidateDetail(c) {
  const labels = { proximity: "Proximity", functional: "Function", area: "Area", occupancy: "Occupancy", department: "Department", location: "Location", planning: "Planning" };
  const issues = [...c.constraints_failed.map(x => [chip("incompatible", "Failed"), x]), ...c.conditions.map(x => [chip("conditional"), x]),
    ...c.missing_information.map(x => [chip("unknown"), x]), ...(c.notes || []).map(x => [chip("recorded", "Note"), x])];
  const m = c.model;
  return `<div class="cand-detail">
    <div><div class="overline">Score</div>${Object.entries(c.criteria).map(([k, v]) =>
      `<div class="gauge" style="grid-template-columns:84px 1fr;margin-top:6px"><span>${labels[k] || k}</span>${miniBar(v)}</div>`).join("")}
      <div class="muted" style="font-size:12px;margin-top:8px">Evidence coverage ${pct(c.evidence_coverage)}</div></div>
    <div><div class="overline">Check before assigning</div><ul class="list">${issues.map(([ch, t]) => `<li>${ch}<span class="grow">${esc(t)}</span></li>`).join("") || "<li class='muted'>Nothing flagged</li>"}</ul></div>
    <div><div class="overline">Record</div><dl class="kv" style="grid-template-columns:84px 1fr;margin-top:6px">
      <dt>Seating</dt><dd>${esc(c.seat_status === "Not Available for Seating" ? "Not a workpoint" : c.seat_status)}</dd>
      <dt>Allocation</dt><dd>${val(c.allocation)}</dd><dt>Standard</dt><dd>${val(c.space_standard)}</dd>
      <dt>Model</dt><dd>${m ? `${esc(m.predicted_group)} ${pct(m.confidence)}${m.verify_flag ? " " + chip("inferred", "Verify use") : ""}` : notOnRecord()}</dd></dl>
</div>
  </div>`;
}

function vacancyTable(rows) {
  return `<div class="table-wrap" style="max-height:560px"><table class="t"><thead><tr><th>Space</th><th class="r">Area</th><th>Type</th><th>Seating</th><th>Allocation</th><th>Flags</th></tr></thead><tbody>
    ${rows.map(r => `<tr><td>${roomKey(r.room_key)}</td><td class="r">${r.area == null ? notOnRecord() : fmtNum(r.area)}</td><td>${esc(className(r.space_class))}</td>
      <td>${esc(r.seat_status)}</td><td>${val(r.allocation)}</td>
      <td>${r.flags.map(f => chip(f.startsWith("Model") ? "inferred" : "conditional", f)).join(" ")}</td></tr>`).join("")}
    </tbody></table></div>`;
}

// ------------------------------------------------------------------ shared section bodies
function evidenceActivity(o) {
  const groups = ["confirmed", "recorded", "inferred", "historical", "conflicting", "unknown"].filter(g => o.findings.some(f => f.classification === g));
  return `<div class="grid-2" style="gap:24px">
    <div><div class="overline">Findings</div>${groups.map(g => {
      const fs = o.findings.filter(f => f.classification === g);
      return `<div class="group-head">${chip(g)}<span class="n">${fs.length}</span></div><ul class="list">${fs.slice(0, 4).map(f => `<li class="sec" style="font-size:13px">${esc(f.claim)}</li>`).join("")}${fs.length > 4 ? `<li class="muted">+${fs.length - 4}</li>` : ""}</ul>`;
    }).join("") || '<p class="muted">None</p>'}</div>
    <div><div class="overline">Tool calls</div><ol class="steps" style="margin-top:6px">${o.trace.map(t => `<li><span class="mono muted">${t.step}</span>
      <span class="mono" style="font-size:12px" title="${esc([...t.warnings, ...t.missing_information].join("; "))}">${esc(t.tool_name)}</span>
      <span class="st ${esc(t.status)}">${esc(t.status)}</span><span class="mono muted" style="font-size:11px">${t.duration_ms} ms</span></li>`).join("")}</ol>
      ${(o.messages || []).length ? `<div class="overline" style="margin-top:16px">Agent messages</div><ul class="list">${o.messages.map(m => `<li class="mono" style="font-size:12px">${esc(m.sender)} → ${esc(m.recipient)} · ${esc(m.intent)}</li>`).join("")}</ul>` : ""}</div>
  </div>`;
}

function timeline(nodes) {
  if (!nodes.length) return notOnRecord("No dated messages");
  const names = { preliminary: "Preliminary", planned: "Planned", confirmed: "Confirmed", revised: "Revised",
    update: "Update", restoration_estimate: "Estimated restoration", restoration_confirmed: "Restored" };
  const shown = nodes;
  return `${nodes.length > shown.length ? `<div class="muted" style="font-size:12px;margin-bottom:8px">+${nodes.length - shown.length} earlier</div>` : ""}
    <div class="timeline">${shown.map(n => `<div class="tl-node ${esc(n.type)} ${n.superseded ? "superseded" : ""}" title="${esc(n.excerpt || "")}">
      <div class="tl-when">${fmtDate(n.date, true)}</div>
      <div class="tl-what"><b>${esc(names[n.type] || n.type)}</b>${n.start ? ` · ${fmtWindow(n.start, n.end)}` : ""}</div>
      <div class="tl-ex">${esc((n.sender || "").split("<")[0].trim() || "Sender not recorded")}${n.excerpt ? ` — ${esc(n.excerpt.length > 140 ? n.excerpt.slice(0, 139) + "…" : n.excerpt)}` : ""}</div></div>`).join("")}</div>`;
}

function stakeholders(s) {
  if (!s.departments.length) return `<p class="muted">No Natural Sciences departments in scope.</p>`;
  return `<table class="t plain"><thead><tr><th>Department</th><th>Contacts</th><th class="r">People</th></tr></thead><tbody>
    ${s.departments.map(d => `<tr><td>${esc(d.department)}</td>
      <td>${d.coordination_contacts.length ? d.coordination_contacts.map(c => `<div title="${esc(c.basis)}">${esc(c.name)} <span class="mono muted" style="font-size:12px">${esc(c.email)}</span></div>`).join("") : chip("unknown", "None")}</td>
      <td class="r">${d.notification_recipients.count}</td></tr>`).join("")}
    </tbody></table><p class="muted" style="font-size:12px;margin:10px 0 0">Formal approvers are in TRIRIGA, not in these records.</p>`;
}

const SOURCE_LABEL = { correspondence: "Email", ics: "Calendar", manual: "Manual", blocked: "Blocked", "TRIRIGA notices": "Outage" };

function conflictList(res) {
  if (!res.conflicts.length) return "";
  return `<ul class="list">${res.conflicts.map(c => `<li>${icon("conflicting")}<span class="grow" title="${esc(c.note || "")}"><b style="font-weight:500">${esc(c.title)}</b> <span class="muted">· ${fmtDate(c.start)} · ${esc(c.building || "")}</span></span>
      <span class="tag">${esc(SOURCE_LABEL[c.source] || c.source)}</span></li>`).join("")}</ul>`;
}

function verdictLine(res) {
  if (!res.window) return chip("unknown", "No date");
  return res.verdict === "conflict" ? chip("conflicting", `Conflict · ${res.conflicts.length}`) : chip("confirmed", "No known conflict");
}

function localInput(iso) { return iso ? iso.slice(0, 16) : ""; }

function scheduling(s, o) {
  const w = s.window;
  const dur = o.outage.duration_hours || 2;
  const start = w ? (w.date_only ? `${w.start.slice(0, 10)}T07:00` : localInput(w.start)) : "";
  const endV = start ? localInput(new Date(new Date(start).getTime() + dur * 36e5 - new Date().getTimezoneOffset() * 6e4).toISOString()) : "";
  return `
    ${conflictList(s)}
    ${s.risks.length ? `<ul class="list">${s.risks.map(r => `<li>${icon("attention")}<span class="grow sec">${esc(r)}</span></li>`).join("")}</ul>` : ""}
    ${s.alternatives?.length ? `<div class="overline" style="margin-top:16px">Suggested times</div>
      <div class="slots">${s.alternatives.map(a => `<button class="slot" data-try="${esc(a.start)}|${esc(a.end)}" title="${esc(a.rationale)}">${fmtWindow(a.start, a.end)}</button>`).join("")}</div>` : ""}
    <div class="overline" style="margin-top:18px">Try a time</div>
    <div class="filters" style="margin:8px 0 0">
      <input class="input" type="datetime-local" id="pt-start" value="${start}" aria-label="Start">
      <input class="input" type="datetime-local" id="pt-end" value="${endV}" aria-label="End">
      <button class="btn sm primary" id="pt-check">Check</button>
      <button class="btn sm" id="pt-block-open">Block…</button>
    </div>
    <div class="filters" id="pt-block-row" hidden style="margin:8px 0 0">
      <input class="input" id="pt-note" placeholder="Reason" style="min-width:240px" aria-label="Reason">
      <button class="btn sm danger" id="pt-block">Block this time</button>
    </div>
    <div id="pt-result" style="margin-top:10px"></div>
    <div class="coverage">${s.coverage.map(c => `<span title="${esc(c.period)}" class="cov ${c.status === "covered" ? "ok" : "gap"}">${esc(c.source.replace(/ \(.*\)/, ""))}</span>`).join("")}
      ${!o.space_support ? `<button class="linkish" id="ask-temp" style="margin-left:auto">Find temporary space</button>` : ""}</div>`;
}

function support(s) {
  const rep = s.reply;
  return `${rep.payload.candidates?.length ? `<table class="t plain"><thead><tr><th>Space</th><th>Type</th><th class="r">Area</th><th>Status</th></tr></thead><tbody>
    ${rep.payload.candidates.map(c => `<tr><td>${roomKey(c.space_id)}</td><td>${esc(className(c.space_class))}</td><td class="r">${c.area == null ? notOnRecord() : fmtNum(c.area)}</td><td>${chip(c.status)}</td></tr>`).join("")}
    </tbody></table>` : `<p class="muted">${esc(rep.payload.summary)}</p>`}`;
}

function affectedPeople(a, caseId) {
  const ns = a.people.filter(p => p.natural_sciences);
  const shown = ns.slice(0, 20);
  const csv = scope => `/api/v1/cases/${encodeURIComponent(caseId)}/recipients.csv?scope=${scope}`;
  return `<div class="row-between"><span class="muted" style="font-size:12px">${esc(a.scope_rule)}</span>
      <span style="display:flex;gap:8px"><a class="btn sm" href="${csv("natural_sciences")}">${icon("file")}Natural Sciences CSV</a><a class="btn ghost sm" href="${csv("all")}">All CSV</a></span></div>
    ${ns.length ? `<table class="t plain" style="margin-top:12px"><thead><tr><th>Name</th><th>Email</th><th>Department</th><th>Room</th></tr></thead><tbody>
      ${shown.map(p => `<tr title="${esc(p.reason)}"><td>${esc(p.name)}</td><td class="mono" style="font-size:12px">${esc(p.email)}</td><td>${esc(p.department)}</td><td class="mono" style="font-size:12px">${esc(p.room.split(" · ").slice(1).join(" · "))}</td></tr>`).join("")}
      </tbody></table>${ns.length > shown.length ? `<p class="muted" style="font-size:12px;margin:8px 0 0">+${ns.length - shown.length} more in the CSV</p>` : ""}`
      : `<p class="muted" style="margin:12px 0 0">No Natural Sciences people recorded in the affected spaces.</p>`}`;
}

function drafts(list, caseId) {
  if (!list.length) return `<p class="muted">No drafts.</p>`;
  return `<div class="seg" role="tablist">${list.map((d, i) => `<button class="${i ? "" : "on"}" data-draft="${i}">${esc(d.kind === "notification" ? "Affected people" : "Departments")}</button>`).join("")}</div>
    ${list.map((d, i) => `<div class="draft" data-draft-p="${i}" ${i ? "hidden" : ""}>
      <div class="draft-to">${d.kind === "notification"
        ? `To ${d.recipients.affected_people} people · <a href="/api/v1/cases/${encodeURIComponent(caseId)}/recipients.csv?scope=natural_sciences">list</a>`
        : `To ${d.recipients.coordination_contacts.length ? d.recipients.coordination_contacts.map(x => esc(x.split(" <")[0])).join(", ") : "contacts to confirm"}`}
        <span class="tag" style="margin-left:auto">Draft</span></div>
      <pre>${esc(d.body)}</pre>
      <div class="bar-actions"><button class="btn sm" data-copy="${esc(d.body)}">${icon("copy")}Copy</button>
        ${sendControl(d)}</div></div>`).join("")}`;
}

/** Send is locked unless the 'Send email' switch is on, and only an approved revision can be sent, once. */
function sendControl(d) {
  const done = SEND.sent.find(x => x.draft === d.kind && x.rev === SEND.rev);
  if (done) return `<span class="draft-sent">${icon("check")}Sent ${fmtDate(done.ts, true)}${done.redirected ? " · test address" : ""}</span>`;
  if (!SEND.enabled) return `<button class="btn sm locked" data-locked="send_email" title="Sending is switched off. Turn on 'Send email' in Settings, or send from your own mailbox.">${icon("lock")}Send</button>`;
  if (!SEND.approved) return `<button class="btn sm" disabled title="Approve Rev ${esc(SEND.revLetter)} before sending">Send</button>`;
  return `<button class="btn sm primary" data-send="${esc(d.kind)}">Send…</button>`;
}

function confirmSend(c, d, main, closeParent) {
  const to = d.kind === "notification"
    ? `${d.recipients.affected_people} affected people (BCC)`
    : `${d.recipients.coordination_contacts.length} department contact${d.recipients.coordination_contacts.length === 1 ? "" : "s"}`;
  const where = SEND.live
    ? `<div class="banner error">${icon("attention")}<div><b>Real recipients.</b> This goes to ${esc(to)}.</div></div>`
    : `<div class="banner info">${icon("shield")}<div><b>Test mode.</b> Goes only to <span class="mono">${esc(SEND.redirect_to || "the test address")}</span>, with the ${esc(to)} listed inside.</div></div>`;
  const subject = (d.body.split("\n")[0] || "").replace(/^Subject:\s*/i, "");
  const m = modal({ overline: `${esc(c.id)} · Rev ${esc(c.rev_letter)}`, title: "Send this draft?",
    body: `${where}<dl class="kv" style="margin-top:16px"><dt>Subject</dt><dd>${esc(subject)}</dd><dt>Draft</dt><dd>${esc(d.title)}</dd></dl>
      <p class="muted" style="font-size:12px;margin:14px 0 0">Sent once per revision and recorded in the activity log.</p>`,
    footer: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="snd-ok">Send now</button>` });
  m.el.querySelector("#snd-ok").onclick = async e => {
    e.target.disabled = true;
    try {
      const r = await api(`/cases/${encodeURIComponent(c.id)}/send`, { method: "POST", body: { draft: d.kind, rev: c.rev, confirm: true } });
      m.close(); closeParent && closeParent();
      toast(r.redirected ? "Sent to the test address" : `Sent to ${r.intended} recipient(s)`);
      workspace(main, c.id);
    } catch (err) { toast(err.message); e.target.disabled = false; }
  };
}

// ------------------------------------------------------------------ review dialogs
function openReturn(c, main) {
  const d = modal({ overline: `${esc(c.id)} · Rev ${esc(c.rev_letter)}`, title: "Return or decline",
    body: `<div class="field"><label for="rat">Reason</label><textarea class="input" id="rat" rows="4" style="font-family:var(--sans);font-size:14px"></textarea></div><div id="rev-err" style="margin-top:10px"></div>`,
    footer: `<button class="btn" data-close>Cancel</button><button class="btn danger" data-dec="decline">Decline</button><button class="btn primary" data-dec="return">Return for revision</button>` });
  d.el.querySelector("#rat").focus();
  d.el.querySelectorAll("[data-dec]").forEach(b => b.onclick = async () => {
    try {
      await api(`/cases/${c.id}/review`, { method: "POST", body: { decision: b.dataset.dec, rationale: d.el.querySelector("#rat").value, rev: c.rev } });
      d.close(); toast("Review recorded"); refreshCounts(); workspace(main, c.id);
    } catch (e) { d.el.querySelector("#rev-err").innerHTML = `<div class="banner error">${icon("attention")}<div>${esc(e.message)}</div></div>`; }
  });
}

function openTimeline(c) {
  const o = c.output, aside = o.incident?.set_aside;
  modal({ overline: esc(c.id), title: "Timeline", size: "md",
    meta: `${o.timeline.length} message${o.timeline.length === 1 ? "" : "s"} in this incident${aside ? ` · ${aside} unrelated set aside` : ""}`,
    body: timeline(o.timeline), footer: `<button class="btn" data-close>Close</button>` });
}

function openHistory(c) {
  modal({ overline: esc(c.id), title: "Review history",
    body: `<ul class="list">${c.reviews.map(r => `<li><span class="grow"><b style="font-weight:500">${esc(r.decision)}</b> · Rev ${esc(r.rev_letter)}
      ${r.invalidated ? chip("historical", "Invalidated") : ""}<div class="muted" style="font-size:12px">${fmtDate(r.ts, true)}</div>
      ${r.rationale ? `<div style="font-size:13px">${esc(r.rationale)}</div>` : ""}</span></li>`).join("")}</ul>`,
    footer: `<button class="btn" data-close>Close</button>` });
}

/** A details tile opens that one section; the tiles on the page are the only navigation. */
function openDetails(key, c, main) {
  const sec = SECTIONS[key];
  if (!sec) return;
  const d = modal({ overline: `${esc(c.id)} · Case details`, title: sec.title, meta: sec.meta, body: sec.body, size: "lg",
    footer: `<button class="btn" data-close>Close</button>` });
  bindContent(d.el, c, main, d.close);
}

function openSection(key, c, main) {
  if (DETAIL_KEYS.includes(key)) return openDetails(key, c, main);
  const sec = SECTIONS[key];
  if (!sec) return;
  const d = modal({ overline: esc(c.id), title: sec.title, meta: sec.meta, body: sec.body, size: sec.size });
  bindContent(d.el, c, main, d.close);
}

function openCandidate(i, c, main) {
  const x = CANDIDATES[i];
  const plan = x.floor_plan ? `<a class="btn" target="_blank" rel="noopener" href="/api/v1/floorplans/${encodeURIComponent(x.floor_plan.code)}/${encodeURIComponent(x.floor_plan.floor)}">${icon("file")}Floor plan</a>` : "";
  modal({ overline: `Candidate ${i + 1} of ${CANDIDATES.length}`, title: `<span class="mono" style="font-size:17px">${esc(x.room_key)}</span>`,
    meta: `${chip(x.status)} <span style="margin-left:8px">Score ${Math.round(x.suitability * 100)}${x.proximity ? ` · ${esc(x.proximity.label)}` : ""}</span>`,
    body: candidateDetail(x), size: "lg", footer: `${plan}<button class="btn" data-close>Close</button>` });
}

function wire(main, c) {
  const $ = sel => main.querySelector(sel);
  $("#rerun") && ($("#rerun").onclick = e => rerunCase(c, main, null, e.currentTarget));
  $("#return") && ($("#return").onclick = () => openReturn(c, main));
  $("#history") && ($("#history").onclick = () => openHistory(c));
  $("#timeline") && ($("#timeline").onclick = () => openTimeline(c));
  $("#approve") && ($("#approve").onclick = async () => {
    try { await api(`/cases/${c.id}/review`, { method: "POST", body: { decision: "approve", rationale: "", rev: c.rev } }); toast(`Rev ${c.rev_letter} approved`); refreshCounts(); workspace(main, c.id); }
    catch (e) { toast(e.message); }
  });
  main.querySelectorAll("[data-sec]").forEach(b => b.onclick = () => openSection(b.dataset.sec, c, main));
  main.querySelectorAll("[data-detail]").forEach(b => b.onclick = () => openDetails(b.dataset.detail, c, main));
  main.querySelectorAll("[data-open]").forEach(b => b.onclick = () => openSection(b.dataset.open, c, main));
  main.querySelectorAll("[data-cand]").forEach(tr => {
    tr.onclick = () => openCandidate(+tr.dataset.cand, c, main);
    tr.onkeydown = e => { if (e.key === "Enter") openCandidate(+tr.dataset.cand, c, main); };
  });
  bindContent(main, c, main);
}

async function rerunCase(c, main, input, btn, close) {
  if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
  try {
    await api(`/cases/${c.id}/rerun`, { method: "POST", body: input ? { input } : {} });
    close && close(); toast("New revision ready"); workspace(main, c.id);
  } catch (e) { toast(e.message); if (btn) btn.disabled = false; }
}

/** Bind interactive content (copy, drafts switch, schedule tools). Works on the page or inside a dialog. */
function bindContent(root, c, main, close) {
  root.querySelectorAll("[data-draft]").forEach(b => b.onclick = () => {
    root.querySelectorAll("[data-draft]").forEach(x => x.classList.toggle("on", x === b));
    root.querySelectorAll("[data-draft-p]").forEach(x => x.hidden = x.dataset.draftP !== b.dataset.draft);
  });
  root.querySelectorAll("[data-copy]").forEach(b => b.onclick = e => {
    e.preventDefault(); e.stopPropagation();
    navigator.clipboard.writeText(b.dataset.copy).then(() => toast("Copied"), () => toast("Copy failed"));
  });
  root.querySelectorAll("[data-locked]").forEach(b => b.onclick = async () => {
    try { await api(`/actions/${b.dataset.locked}`, { method: "POST" }); } catch (e) { toast(e.message); }
  });
  root.querySelectorAll("[data-send]").forEach(b => b.onclick = () =>
    confirmSend(c, c.output.communication_drafts.find(d => d.kind === b.dataset.send), main, close));
  const sec = root.querySelector("[data-buildings]");
  if (!sec) return;
  const buildings = JSON.parse(sec.dataset.buildings), sr = sec.dataset.sr || null;
  const box = sec.querySelector("#pt-result");
  const v = id => sec.querySelector(id).value;
  const check = async (a, z) => {
    if (!a || !z) return toast("Enter a start and end");
    box.innerHTML = skeleton(1);
    try {
      const r = await api("/schedule/check", { method: "POST", body: { start: a, end: z, buildings, sr } });
      box.innerHTML = `<div style="margin-bottom:6px">${verdictLine(r)}</div>${conflictList(r)}`;
    } catch (e) { box.innerHTML = `<div class="banner error">${icon("attention")}<div>${esc(e.message)}</div></div>`; }
  };
  sec.querySelector("#pt-check").onclick = () => check(v("#pt-start"), v("#pt-end"));
  sec.querySelectorAll("[data-try]").forEach(b => b.onclick = () => {
    const [a, z] = b.dataset.try.split("|");
    sec.querySelector("#pt-start").value = a.slice(0, 16); sec.querySelector("#pt-end").value = z.slice(0, 16);
    sec.querySelectorAll(".slot").forEach(x => x.classList.toggle("on", x === b));
    check(a.slice(0, 16), z.slice(0, 16));
  });
  sec.querySelector("#pt-block-open").onclick = () => { const r = sec.querySelector("#pt-block-row"); r.hidden = !r.hidden; if (!r.hidden) sec.querySelector("#pt-note").focus(); };
  sec.querySelector("#pt-block").onclick = async () => {
    const note = v("#pt-note").trim();
    if (!note) return toast("Add a reason");
    try {
      for (const b of buildings.length ? buildings : [null])
        await api("/events", { method: "POST", body: { title: `Not available: ${note}`, start: v("#pt-start"), end: v("#pt-end"), building: b, kind: "blocked", note } });
      toast("Time blocked"); sec.querySelector("#pt-block-row").hidden = true; check(v("#pt-start"), v("#pt-end"));
    } catch (e) { toast(e.message); }
  };
  const t = sec.querySelector("#ask-temp");
  if (t) t.onclick = () => rerunCase(c, main, { temporary_space: true }, t, close);
}

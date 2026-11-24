import { api, chip, empty, errorBanner, esc, fmtNum, icon, modal, notOnRecord, noteCases, openPanel, pct, roomKey, skeleton, toast, val } from "../lib.js";
import { setCrumbs } from "../app.js";
import { Conversation, conversationAvailable } from "../voice.js";

const PAGE = 100;
const BLANK = { building: "", group: "", view: "all", q: "", ns: false, offset: 0 };
const state = { ...BLANK };
const VIEWS = [["all", "All"], ["vacant", "Vacancies"], ["flagged", "Model flags"]];

export async function render(main) {
  setCrumbs([{ label: "Spaces" }]);
  // a link can open the page on a given view, e.g. #/spaces?view=vacant&scope=ns
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  if (q.has("view") || q.has("scope")) {
    Object.assign(state, BLANK, { view: VIEWS.some(([v]) => v === q.get("view")) ? q.get("view") : "all", ns: q.get("scope") === "ns" });
    history.replaceState(null, "", "#/spaces");  // later filter changes are the user's own
  }
  main.innerHTML = `
    <div class="page-head"><div><h1>Spaces</h1><p>Room records</p></div>
      <div class="actions"><button class="btn primary" id="req">${icon("plus")}New space request</button></div></div>
    <div class="filters" id="filters"></div>
    <div id="table">${skeleton(8)}</div>`;
  main.querySelector("#req").onclick = () => newSpaceRequest();
  try {
    const b = await api("/buildings");
    state.others = b.other_buildings; state.crosswalk = b.crosswalk;
  } catch (e) { main.querySelector("#filters").innerHTML = errorBanner(e); }
  load(main, true);
}

async function load(main, rebuildFilters = false) {
  const qs = new URLSearchParams({ limit: PAGE, offset: state.offset });
  ["building", "group", "q"].forEach(k => state[k] && qs.set(k, state[k]));
  if (state.view === "vacant") qs.set("vacancy", "true");
  if (state.ns) qs.set("ns", "true");
  if (state.view === "flagged") qs.set("flagged", "true");
  const box = main.querySelector("#table");
  let d;
  box.classList.add("is-loading");
  try { d = await api(`/spaces?${qs}`); } catch (e) { box.innerHTML = errorBanner(e); return; } finally { box.classList.remove("is-loading"); }
  if (rebuildFilters) filters(main, d);
  main.querySelector("#f-x").hidden = !(state.building || state.group || state.q || state.ns || state.view !== "all");
  if (!d.rows.length) { box.innerHTML = `<div class="panel">${empty("No matching space record.")}</div>`; return; }
  box.innerHTML = `<div class="table-wrap" style="max-height:640px"><table class="t fixed"><colgroup><col style="width:21%"><col style="width:21%"><col style="width:9%"><col style="width:11%"><col style="width:21%"><col style="width:17%"></colgroup><thead><tr>
      <th>Room</th><th>Space class</th><th class="r">Area sf</th><th>Seating</th><th>Allocation</th><th>Model</th></tr></thead><tbody>
      ${d.rows.map(r => `<tr class="click" data-id="${esc(r.space_id)}">
        <td>${roomKey(r.space_id)}${r.plan_reservation ? ' <span class="tag" title="Planning reservation">Plan</span>' : ""}</td>
        <td>${val(r.space_class)}</td>
        <td class="r">${r.area == null ? notOnRecord() : fmtNum(r.area)}</td>
        <td>${r.seat_status === "Not Available for Seating" ? '<span class="muted">—</span>' : esc(r.seat_status || "")}</td>
        <td>${val(r.allocation)}</td>
        <td>${r.verify_flag ? chip("inferred", `${r.predicted_group} · ${pct(r.confidence)}`) : ""}</td></tr>`).join("")}
      </tbody></table>
      <div class="table-foot"><span>${fmtNum(d.total)} records</span>
        <span style="margin-left:auto">${state.offset + 1}–${Math.min(state.offset + PAGE, d.total)}</span>
        <button class="btn sm" id="prev" ${state.offset === 0 ? "disabled" : ""}>Previous</button>
        <button class="btn sm" id="next" ${state.offset + PAGE >= d.total ? "disabled" : ""}>Next</button></div></div>`;
  box.querySelector("#prev").onclick = () => { state.offset = Math.max(0, state.offset - PAGE); load(main); };
  box.querySelector("#next").onclick = () => { state.offset += PAGE; load(main); };
  box.querySelectorAll("tr[data-id]").forEach(tr => tr.onclick = () => inspect(tr.dataset.id));
}

function filters(main, d) {
  const opts = (list, cur, all) => `<option value="">${all}</option>` + list.map(x => `<option ${x === cur ? "selected" : ""}>${esc(x)}</option>`).join("");
  const names = [...(state.crosswalk || []).map(b => b.export_name), ...(state.others || [])];
  const f = main.querySelector("#filters");
  f.innerHTML = `
    <div class="seg inline" role="tablist">${VIEWS.map(([v, l]) => `<button data-v="${v}" class="${state.view === v ? "on" : ""}">${l}</button>`).join("")}</div>
    <input class="input" id="f-q" placeholder="Search" value="${esc(state.q)}" aria-label="Search" style="flex:1;min-width:220px">
    <select class="input" id="f-b" aria-label="Building">${opts(names, state.building, "All buildings")}</select>
    <select class="input" id="f-g" aria-label="Function">${opts(d.groups, state.group, "All functions")}</select>
    ${state.ns ? `<button class="tag" id="f-ns" title="Remove filter" style="cursor:pointer;border:0">Natural Sciences ✕</button>` : ""}
    <button class="btn ghost sm" id="f-x">Clear</button>`;
  const apply = () => {
    Object.assign(state, { building: f.querySelector("#f-b").value, group: f.querySelector("#f-g").value, q: f.querySelector("#f-q").value.trim(), offset: 0 });
    load(main);
  };
  f.querySelectorAll("select").forEach(el => el.onchange = apply);
  let t; f.querySelector("#f-q").oninput = () => { clearTimeout(t); t = setTimeout(apply, 300); };
  f.querySelectorAll("[data-v]").forEach(b => b.onclick = () => {
    state.view = b.dataset.v; state.offset = 0;
    f.querySelectorAll("[data-v]").forEach(x => x.classList.toggle("on", x === b));
    load(main);
  });
  f.querySelector("#f-ns")?.addEventListener("click", () => { state.ns = false; state.offset = 0; load(main, true); });
  f.querySelector("#f-x").onclick = () => { Object.assign(state, BLANK); load(main, true); };
}

async function inspect(id) {
  const d = modal({ overline: "Space record", title: `<span class="mono" style="font-size:17px">${esc(id)}</span>`, body: skeleton(8), size: "lg",
    footer: `<span id="plan" style="margin-right:auto"></span><button class="btn" data-close>Close</button>` });
  let r;
  try { r = await api(`/spaces/${encodeURIComponent(id)}`); } catch (e) { d.body.innerHTML = errorBanner(e); return; }
  const s = r.space, m = r.model;
  const seat = s.seat_status === "Not Available for Seating" ? "Not a workpoint" : s.seat_status;
  d.el.querySelector(".dlg-head h2").insertAdjacentHTML("afterend", `<div class="dlg-meta">${esc(r.building.name)}${s.plan_reservation ? " · " + chip("attention", "Planning reservation") : ""}${s.duplicate_key ? " · " + chip("conflicting", "Duplicate key") : ""}</div>`);
  d.body.innerHTML = `<div class="grid-2" style="gap:32px">
    <div><div class="overline" style="margin-bottom:8px">Record</div>
      <dl class="kv" style="grid-template-columns:120px 1fr">
        <dt>Class</dt><dd>${val(s.space_class)}</dd>
        <dt>Standard</dt><dd>${val(s.space_standard)}</dd>
        <dt>Area</dt><dd>${s.area == null ? notOnRecord() : `${fmtNum(s.area, 0)} sf`}</dd>
        <dt>Seating</dt><dd>${val(seat)}</dd>
        <dt>Allocation</dt><dd>${val(s.allocation)}</dd>
        <dt>Alignment</dt><dd>${val(s.ssa)}</dd>
      </dl></div>
    <div>
      <div class="overline" style="margin-bottom:8px">Model</div>
      ${m.status !== "success" ? notOnRecord("No prediction") : `
        <div style="margin-bottom:10px">${m.verify_flag ? chip("inferred", `Looks like ${m.predicted_group}`) : chip("confirmed", "Matches record")}</div>
        ${m.top3.map(t => `<div class="gauge" style="grid-template-columns:130px 1fr 40px;margin-top:6px"><span>${esc(t.group)}</span><div class="bar"><i style="width:${t.p * 100}%"></i></div><span class="v">${t.p.toFixed(2)}</span></div>`).join("")}`}
      <div class="overline" style="margin:22px 0 4px">Occupants · ${r.occupants.length}</div>
      ${r.occupants.length ? `<ul class="list">${r.occupants.slice(0, 6).map(o => `<li><span class="grow">${esc(o.name)}<div class="muted" style="font-size:12px">${esc(o.role)} · ${esc(o.department)}</div></span></li>`).join("")}
        ${r.occupants.length > 6 ? `<li class="muted">+${r.occupants.length - 6} more</li>` : ""}</ul>` : `<p class="muted" style="margin:6px 0 0">None</p>`}
    </div></div>`;
  d.el.querySelector("#plan").innerHTML = r.floor_plan
    ? `<a class="btn" href="${r.floor_plan.url}" target="_blank" rel="noopener" title="${esc(r.floor_plan.snapshot)} snapshot">${icon("file")}Floor plan</a>` : "";
}

// ------------------------------------------------------------------ new space request (shared with overview)
/** New space request. The conversation with the Space agent (talk or type) is the main way in; the form is the
 *  alternative. Switching to the form carries over what was said. Both end in a case for review. */
export async function newSpaceRequest(prefill = {}) {
  let conv = null;
  // Voice assistant switched off in Settings: the dialog is just the form, with no Conversation/Form toggle.
  const voiceOn = await api("/voice/status").then(r => r.enabled).catch(() => false);
  const head = `<div class="dr-head sr-head"><div><h2>New space request</h2></div>
      ${voiceOn ? `<div class="seg inline sr-mode" role="tablist" aria-label="How to make the request">
        <button type="button" role="tab" data-mode="talk" aria-selected="true" class="on">${icon("mic")}Conversation</button>
        <button type="button" role="tab" data-mode="form" aria-selected="false">Form</button></div>` : ""}
      <button type="button" class="sr-close" data-close aria-label="Close" title="Close">${icon("x")}</button></div>`;
  const p = openPanel(voiceOn ? `${head}
    <div class="sr-pane" data-pane="talk"><div class="sr-talk"><div class="vx-loading">${icon("mic")}<span>Starting…</span></div></div>
      <div class="dr-foot"><button class="btn primary" id="sr-create" disabled>Create case</button></div></div>
    <div class="sr-pane" data-pane="form" hidden>${requestForm(prefill)}</div>`
    : `${head}<div class="sr-pane" data-pane="form">${requestForm(prefill)}</div>`,
    { modal: true, dismissible: false, onClose: () => conv && conv.close() });
  p.el.classList.add("sr-dialog");
  if (!voiceOn) p.el.classList.add("sr-form-only");
  const el = p.el;

  const setMode = mode => {
    el.querySelectorAll("[data-mode]").forEach(b => { const on = b.dataset.mode === mode; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
    el.querySelectorAll("[data-pane]").forEach(x => x.hidden = x.dataset.pane !== mode);
    if (mode === "form" && conv) fillForm(el, conv.slots);
    if (conv) conv.setActive(mode === "talk");
  };
  el.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  bindForm(el, p);
  if (!voiceOn) return;

  const create = el.querySelector("#sr-create");
  create.onclick = () => { create.disabled = true; create.textContent = "Creating…"; conv.save(); };

  const avail = await conversationAvailable();
  const talk = el.querySelector(".sr-talk");
  if (!avail.ok) {
    // switched on, but the local models are not ready (e.g. Ollama not running): say so, keep the form one click away
    talk.innerHTML = `<div class="sr-off">${icon("attention")}<div><b>Voice not ready.</b> ${esc(avail.why)}
      <div class="sr-off-actions"><button class="btn sm primary" type="button" id="sr-useform">Use the form</button></div></div></div>`;
    talk.querySelector("#sr-useform").onclick = () => setMode("form");
    return;
  }
  conv = new Conversation(talk, avail.opts, {
    onChange: ({ canSave }) => { create.disabled = !canSave; },
    onError: () => { if (create.textContent === "Creating…") { create.textContent = "Create case"; create.disabled = false; } },
    onSaved: (id, via) => {
      noteCases(id);
      if (via === "voice") {                       // saved by voice: keep talking; the footer opens the case
        create.disabled = false; create.textContent = `Open ${id}`;
        create.onclick = () => { p.close(); location.hash = `#/cases/${id}`; };
        return;
      }
      p.close(); toast(`${id} created`); location.hash = `#/cases/${id}`;
    },
  });
  conv.start();
}

function requestForm(prefill) {
  const fns = ["Office", "Workstation", "Research Lab", "Teaching Lab", "Meeting", "Support & Storage"];
  const types = [["new_office", "New office"], ["department_function", "Department function"],
    ["change_assignment", "Assignment change"], ["vacancy_review", "Vacancy review"]];
  return `<form class="dr-body stack" id="sf">
      <div class="field"><label for="rt">Request type</label><select class="input" id="rt">${types.map(([v, l]) => `<option value="${v}" ${prefill.request_type === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <div class="field"><label for="dp">Department</label><input class="input" id="dp" placeholder="e.g. Chemistry" value="${esc(prefill.department || "")}"></div>
      <div class="field"><label for="nr">Near person</label><input class="input" id="nr" placeholder="Name or NetID" value="${esc(prefill.near || "")}"></div>
      <div class="field" data-for="change_assignment"><label for="ni">Person NetID</label><input class="input" id="ni" placeholder="e.g. ljd1"></div>
      <div class="grid-2" style="gap:12px">
        <div class="field"><label for="fn">Function</label><select class="input" id="fn">${fns.map(f => `<option ${prefill.function === f ? "selected" : ""}>${f}</option>`).join("")}</select></div>
        <div class="field"><label for="ma">Minimum area (sf)</label><input class="input" id="ma" type="number" min="0" step="10" value="${esc(prefill.min_area || "")}"></div>
        <div class="field"><label for="se">Seats</label><input class="input" id="se" type="number" min="1" value="1"></div>
        <div class="field"><label for="pb">Buildings</label><input class="input" id="pb" placeholder="e.g. BRC, Keck"></div>
      </div>
      <div class="field"><label for="no">Notes</label><textarea class="input" id="no" rows="3" placeholder="Optional"></textarea></div>
      <div id="err"></div>
    </form>
    <div class="dr-foot"><button class="btn primary" id="go">Run analysis</button></div>`;
}

/** Carry the conversation's request over to the form (only fields that were established). */
function fillForm(el, slots = {}) {
  const set = (id, v) => { if (v !== undefined && v !== null && v !== "") el.querySelector(id).value = v; };
  set("#rt", slots.request_type); set("#dp", slots.department); set("#nr", slots.near_name || slots.near);
  set("#fn", slots.function); set("#ma", slots.min_area); set("#se", slots.seats); set("#ni", slots.netid);
  if (slots.preferred_buildings && slots.preferred_buildings.length) set("#pb", slots.preferred_buildings.join(", "));
  el.querySelector("#rt").dispatchEvent(new Event("change"));
}

function bindForm(f, p) {
  const sync = () => f.querySelector('[data-for="change_assignment"]').style.display = f.querySelector("#rt").value === "change_assignment" ? "" : "none";
  f.querySelector("#rt").onchange = sync; sync();
  f.querySelector("#go").onclick = async () => {
    const num = id => f.querySelector(id).value ? Number(f.querySelector(id).value) : null;
    const input = {
      request_type: f.querySelector("#rt").value, department: f.querySelector("#dp").value.trim() || null,
      function: f.querySelector("#fn").value, min_area: num("#ma"), seats: num("#se") || 1,
      preferred_buildings: f.querySelector("#pb").value.split(",").map(s => s.trim()).filter(Boolean),
      netid: f.querySelector("#ni").value.trim() || null, notes: f.querySelector("#no").value,
      near: f.querySelector("#nr").value.trim() || null,
    };
    const btn = f.querySelector("#go"); btn.disabled = true; btn.textContent = "Analyzing…";
    try {
      const c = await api("/cases", { method: "POST", body: { kind: "space", input } });
      noteCases(c.id); p.close(); toast(`${c.id} created`); location.hash = `#/cases/${c.id}`;
    } catch (e) { f.querySelector("#err").innerHTML = `<div class="banner error">${icon("attention")}<div>${esc(e.message)}</div></div>`; btn.disabled = false; btn.textContent = "Run analysis"; }
  };
}

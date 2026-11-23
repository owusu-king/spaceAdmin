// Settings: one quiet line per switch. Off by default; every change is logged. Full reasons live in tooltips.
import { api, errorBanner, esc, fmtDate, icon, modal, skeleton, toast } from "../lib.js";
import { setCrumbs } from "../app.js";

// An array keeps the order: object keys that look like numbers would be sorted before "".
const AUTO = [["", "Until off"], ["15", "15 min"], ["60", "1 hour"], ["480", "8 hours"]];

/** Short tag for why a switch cannot be turned on (the full sentence is the tooltip). */
const blocked = r => /FAI_ALLOW_UNLOCK/.test(r) ? "Locked" : /not built/.test(r) ? "Not available"
  : /FAI_SMTP|not configured/i.test(r) ? "Not configured" : /first/.test(r) ? "Needs Send email"
  : /Ollama|model|missing|install|Kokoro|detector/i.test(r) ? "Models missing" : "Unavailable";

export async function render(main) {
  setCrumbs([{ label: "Settings" }]);
  main.innerHTML = `<div class="page-head"><div><h1>Settings</h1><p>Off by default. Every change is logged.</p></div></div>
    <div id="body" class="set-page">${skeleton(6)}</div>`;
  let s;
  try { s = await api("/settings"); } catch (e) {
    // 404 here means the server process predates the Settings feature: the page files are new, the API is not.
    if (e.status === 404) e.message = "The server is running older code without Settings. Restart the server and reload this page.";
    main.querySelector("#body").innerHTML = errorBanner(e); return;
  }
  draw(main, s);
}

function draw(main, s) {
  const groups = [...new Set(s.switches.map(x => x.group))];
  const e = s.email;
  const email = e.configured
    ? `Email via <span class="mono">${esc(e.host)}</span> · from <span class="mono">${esc(e.from)}</span>${e.redirect_to ? ` · test <span class="mono">${esc(e.redirect_to)}</span>` : ""}`
    : `Email not configured`;
  main.querySelector("#body").innerHTML = groups.map(g => `
    <section class="set-sec">
      <div class="set-sec-head"><h2>${esc(g)}</h2>
        ${g === "Actions" && !s.unlock_allowed ? `<span class="set-lock" title="Set FAI_ALLOW_UNLOCK=true in .env and restart the server to allow these switches.">${icon("lock")}Locked on this machine</span>` : ""}</div>
      <div class="set-card">
        ${s.switches.filter(x => x.group === g).map(row).join("")}
        ${g === "Voice" ? `<div class="vpick" id="vpick"></div>` : ""}
      </div>
      ${g === "Actions" ? `<p class="set-foot" title="Configured in .env (FAI_SMTP_*, FAI_EMAIL_REDIRECT_TO)">${email}</p>` : ""}
    </section>`).join("");
  main.querySelectorAll("[data-switch]").forEach(b => b.onclick = () => flip(main, s, b.dataset.switch));
  if (main.querySelector("#vpick")) voicePickers(main);
}

function row(x) {
  const tag = x.on
    ? `<span class="set-tag on">${x.expires ? `On · until ${esc(fmtDate(x.expires, true).split(" ").pop())}` : "On"}</span>`
    : x.available ? "" : `<span class="set-tag" title="${esc(x.reason)}">${esc(blocked(x.reason))}</span>`;
  const timer = !x.on && x.available && x.group === "Actions"
    ? `<select class="input sm set-timer" data-minutes="${esc(x.key)}" aria-label="How long ${esc(x.label)} stays on">
        ${AUTO.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>` : "";
  return `<div class="set-row ${x.on ? "is-on" : ""} ${!x.on && !x.available ? "is-off" : ""}"
      title="${x.updated ? `Changed ${esc(fmtDate(x.updated, true))} by ${esc(x.updated_by || "")}` : ""}">
    <div class="set-main"><div class="set-title"><b>${esc(x.label)}</b>${tag}</div><div class="set-desc">${esc(x.description)}</div></div>
    <div class="set-ctl">${timer}
      <button class="switch" role="switch" aria-checked="${x.on}" aria-label="${esc(x.label)}" data-switch="${esc(x.key)}"
        ${!x.on && !x.available ? `disabled title="${esc(x.reason)}"` : ""}><i></i></button></div>
  </div>`;
}

/** Model pickers: only models installed on this computer can be chosen. */
async function voicePickers(main) {
  const box = main.querySelector("#vpick");
  box.innerHTML = `<span class="muted">Looking for installed models…</span>`;
  let o;
  try { o = await api("/voice/options"); } catch (e) { box.innerHTML = `<span class="set-err">${esc(e.message)}</span>`; return; }
  const sel = o.selected;
  const pick = (key, label, items) => `<label class="vp"><span>${label}</span>
    <select class="input sm" data-choice="${key}" ${items.length ? "" : "disabled"}>
      ${items.length ? items.map(i => `<option value="${esc(i.value)}" ${i.value === sel[key] ? "selected" : ""} ${i.disabled ? "disabled" : ""}>${esc(i.label)}</option>`).join("")
        : `<option>None installed</option>`}</select></label>`;
  box.innerHTML = `
    ${pick("voice_llm", "Model", o.llms.map(m => ({ value: m.name, disabled: !m.tools, label: m.tools ? m.name : `${m.name} (not suitable)` })))}
    ${pick("voice_stt", "Speech", o.stt.map(m => ({ value: m.id, label: m.label.replace(/ tdt.*$/, "").replace(/^\w/, c => c.toUpperCase()) })))}
    ${pick("voice_voice", "Voice", o.voices.map(v => ({ value: v.id, label: v.label })))}
    ${o.problems.length ? `<span class="set-err" title="${esc(o.problems.join(" "))}">${icon("attention")}${esc(o.problems[0])}</span>` : ""}`;
  box.querySelectorAll("[data-choice]").forEach(el => el.onchange = async () => {
    try {
      await api("/voice/options", { method: "PUT", body: { [el.dataset.choice]: el.value } });
      toast("Saved");
    } catch (e) { toast(e.message); }
  });
}

async function flip(main, s, key) {
  const x = s.switches.find(v => v.key === key);
  const on = !x.on;
  const sel = main.querySelector(`[data-minutes="${key}"]`);
  const minutes = sel && sel.value ? +sel.value : null;
  if (on && key === "email_live_recipients" && !await confirmDialog("Send to real recipients?",
    `<p style="margin:0">Sent drafts will go to the actual contacts and affected people instead of the test address.</p>`, "Turn on")) return;
  try {
    draw(main, await api(`/settings/${key}`, { method: "PUT", body: { on, minutes } }));
    toast(`${x.label} ${on ? "on" : "off"}`);
  } catch (e) { toast(e.message); }
}

function confirmDialog(title, body, label) {
  return new Promise(resolve => {
    let done = false;
    const d = modal({ title, body, footer: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="cf-ok">${esc(label)}</button>`,
      onClose: () => { if (!done) resolve(false); } });
    d.el.querySelector("#cf-ok").onclick = () => { done = true; d.close(); resolve(true); };
  });
}

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
  main.innerHTML = `<div class="page-head"><div><h1>Settings</h1><p>Off by default · changes are logged</p></div></div>
    <div id="body" class="set-page">${skeleton(6)}</div>`;
  let s;
  try { s = await api("/settings"); } catch (e) {
    // 404 here means the server process predates the Settings feature: the page files are new, the API is not.
    if (e.status === 404) e.message = "Restart the server, then reload.";
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
  const sec = g => `
    <section class="set-sec">
      <div class="set-sec-head"><h2>${esc(g)}</h2>
        ${g === "Actions" && !s.unlock_allowed ? `<span class="set-lock" title="Set FAI_ALLOW_UNLOCK=true in .env">${icon("lock")}Locked</span>` : ""}</div>
      <div class="set-card">
        ${s.switches.filter(x => x.group === g).map(row).join("")}
        ${g === "Voice" ? `<div class="vpick" id="vpick"></div>` : ""}
      </div>
      ${g === "Actions" ? `<p class="set-foot" title="Configured in .env (FAI_SMTP_*, FAI_EMAIL_REDIRECT_TO)">${email}</p>` : ""}
    </section>`;
  const connections = `<section class="set-sec"><div class="set-sec-head"><h2>Connections</h2></div>
      <div class="set-card" id="conn"><div class="set-row"><span class="muted">Checking…</span></div></div></section>`;
  // Desktop: two columns. Left: what James uses daily (voice, live data); right: actions that need unlocking.
  const left = groups.filter(g => g !== "Actions").map(sec).join("") + connections;
  const right = groups.filter(g => g === "Actions").map(sec).join("");
  main.querySelector("#body").innerHTML = `<div class="set-cols"><div class="set-col">${left}</div><div class="set-col">${right}</div></div>`;
  main.querySelectorAll("[data-switch]").forEach(b => b.onclick = () => flip(main, s, b.dataset.switch));
  if (main.querySelector("#vpick")) voicePickers(main);
  connectionsSection(main);
}

// ------------------------------------------------------------------ connections: live data in, read-only
const SRC = { replay: "Replay", graph: "Microsoft 365", imap: "IMAP" };
let signInTimer = null;

async function connectionsSection(main) {
  const box = main.querySelector("#conn");
  if (!box) return;
  let c;
  try { c = await api("/connections"); } catch (e) { box.innerHTML = `<div class="set-row"><span class="set-err">${esc(e.message)}</span></div>`; return; }
  const ms = c.microsoft, em = c.email, rec = c.records;
  const msTag = { connected: `<span class="set-tag on">Connected</span>`, signing_in: `<span class="set-tag">Signing in</span>`,
    not_connected: `<span class="set-tag">Not connected</span>`, not_configured: `<span class="set-tag" title="${esc(ms.detail || "")}">Not configured</span>` }[ms.status] || "";
  const msDesc = ms.status === "connected" ? `${esc(ms.account || "Signed in")} · read-only`
    : ms.status === "not_configured" ? "Set FAI_GRAPH_* in .env"
    : ms.error ? esc(ms.error) : "Read-only mail and files";
  const code = ms.status === "signing_in" ? `<div class="ms-code">Go to <a href="${esc(ms.verification_uri)}" target="_blank" rel="noopener">${esc(ms.verification_uri.replace("https://", ""))}</a>
      and enter <b class="mono">${esc(ms.user_code)}</b>.</div>` : "";
  const emDesc = `${SRC[em.mode] || em.mode}${em.mode === "graph" && em.folder ? ` · ${esc(em.mailbox)} · ${esc(em.folder)}` : ""}
    · ${em.poll_seconds ? `every ${em.poll_seconds} s` : "manual"}${em.last_sync ? ` · last ${esc(fmtDate(em.last_sync, true))}` : ""}`;
  const d = rec.last_diff;
  const recDesc = `${rec.source.kind === "microsoft" ? "SharePoint / OneDrive" : rec.source.kind === "upload" ? "Uploaded" : `<span title="${esc(rec.source.where)}">Folder ${esc(String(rec.source.where).split(/[\\/]/).filter(Boolean).pop() || "")}</span>`}
    · ${rec.records.spaces.toLocaleString()} spaces, ${rec.records.people.toLocaleString()} people
    · ${rec.refresh_minutes ? `every ${rec.refresh_minutes} min` : "manual"}${rec.last_refresh ? ` · updated ${esc(fmtDate(rec.last_refresh, true))}` : ""}`;
  const recChange = d && !d.unchanged ? `<div class="set-desc">Rooms +${d.spaces.added} −${d.spaces.removed} ~${d.spaces.changed} · People +${d.people.added} −${d.people.removed} moved ${d.people.moved}</div>` : "";
  box.innerHTML = `
    <div class="set-row"><div class="set-main"><div class="set-title"><b>Microsoft 365</b>${msTag}</div><div class="set-desc">${msDesc}</div>${code}</div>
      <div class="set-ctl">${ms.status === "connected" ? `<button class="btn sm" id="ms-off">Disconnect</button>`
        : ms.status === "not_configured" ? "" : ms.status === "signing_in" ? "" : `<button class="btn sm primary" id="ms-on">Connect</button>`}</div></div>
    <div class="set-row"><div class="set-main"><div class="set-title"><b>Outage email</b>${em.status === "ok" ? `<span class="set-tag on">Live</span>` : `<span class="set-tag">${esc(em.status)}</span>`}</div>
      <div class="set-desc">${emDesc}</div></div></div>
    <div class="set-row"><div class="set-main"><div class="set-title"><b>Space and people records</b>${rec.last_error ? `<span class="set-tag" title="${esc(rec.last_error)}">Check failed</span>` : ""}</div>
      <div class="set-desc">${recDesc}</div>${recChange}</div>
      <div class="set-ctl"><button class="btn sm" id="rec-setup">Set up</button><button class="btn sm" id="rec-now">Check now</button></div></div>`;
  const on = box.querySelector("#ms-on"), off = box.querySelector("#ms-off");
  if (on) on.onclick = async () => {
    on.disabled = true;
    try { await api("/connections/microsoft/connect", { method: "POST" }); connectionsSection(main); } catch (e) { toast(e.message); on.disabled = false; }
  };
  if (off) off.onclick = async () => { await api("/connections/microsoft/disconnect", { method: "POST" }); toast("Disconnected"); connectionsSection(main); };
  box.querySelector("#rec-setup").onclick = () => recordsDialog(main);
  box.querySelector("#rec-now").onclick = async e => {
    const b = e.currentTarget; b.disabled = true; b.textContent = "Checking…";
    try {
      const r = await api("/records/refresh", { method: "POST" });
      toast(r.changed ? "Records updated" : r.error ? r.error : "No new version");
      connectionsSection(main);
    } catch (err) { toast(err.message); b.disabled = false; b.textContent = "Check now"; }
  };
  clearTimeout(signInTimer);                              // while James signs in at Microsoft, watch for completion
  if (ms.status === "signing_in") signInTimer = setTimeout(() => document.body.contains(box) && connectionsSection(main), 3000);
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
  box.innerHTML = `<span class="muted">Loading…</span>`;
  let o;
  try { o = await api("/voice/options"); } catch (e) { box.innerHTML = `<span class="set-err">${esc(e.message)}</span>`; return; }
  const sel = o.selected;
  const pick = (key, label, items) => `<label class="vp"><span>${label}</span>
    <select class="input sm" data-choice="${key}" ${items.length ? "" : "disabled"}>
      ${items.length ? items.map(i => `<option value="${esc(i.value)}" ${i.value === sel[key] ? "selected" : ""} ${i.disabled ? "disabled" : ""}>${esc(i.label)}</option>`).join("")
        : `<option>None installed</option>`}</select></label>`;
  box.innerHTML = `
    ${pick("voice_llm", "Model", o.llms.map(m => ({ value: m.name, disabled: !m.tools, label: m.tools ? m.name : `${m.name} (no tools)` })))}
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
    `<p style="margin:0">Drafts go to real contacts, not the test address.</p>`, "Turn on")) return;
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

// ------------------------------------------------------------------ records: James connects the space and people reports
const EVERY = [["0", "Manually"], ["60", "Every hour"], ["240", "Every 4 hours"], ["1440", "Once a day"]];

async function recordsDialog(main) {
  let cfg;
  try { cfg = await api("/records/config"); } catch (e) { toast(e.message); return; }
  const tab = cfg.source === "microsoft" ? "microsoft" : cfg.source === "upload" ? "upload" : "folder";
  const every = `<label class="field"><span>Check</span><select class="input" id="rc-every">
      ${EVERY.map(([v, l]) => `<option value="${v}" ${String(cfg.refresh_minutes) === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>`;
  const d = modal({ title: "Space and people records", size: "md", body: `
    <div class="seg" role="tablist">
      <button type="button" data-rt="microsoft" class="${tab === "microsoft" ? "on" : ""}">SharePoint / OneDrive</button>
      <button type="button" data-rt="folder" class="${tab === "folder" ? "on" : ""}">Folder</button>
      <button type="button" data-rt="upload" class="${tab === "upload" ? "on" : ""}">Upload</button></div>
    <div class="stack rc-pane" data-rp="microsoft" ${tab === "microsoft" ? "" : "hidden"}>
      <p class="muted rc-note">Paste each report's share link.${cfg.microsoft_connected ? "" : " <b>Connect Microsoft 365 first.</b>"}</p>
      <label class="field"><span>Space records</span><input class="input" id="rc-space" value="${esc(cfg.space_url || "")}" placeholder="https://rice.sharepoint.com/…"></label>
      <label class="field"><span>People records</span><input class="input" id="rc-people" value="${esc(cfg.people_url || "")}" placeholder="https://rice.sharepoint.com/…"></label>
      ${every}</div>
    <div class="stack rc-pane" data-rp="folder" ${tab === "folder" ? "" : "hidden"}>
      <p class="muted rc-note">Folder with SpaceRecords.xlsx and Report_PeopleRecords.xlsx.</p>
      <label class="field"><span>Folder</span><input class="input" id="rc-dir" value="${esc(cfg.dir || "")}"></label>
      ${every.replace('id="rc-every"', 'id="rc-every2"')}</div>
    <div class="stack rc-pane" data-rp="upload" ${tab === "upload" ? "" : "hidden"}>
      <p class="muted rc-note">Invalid files are rejected.</p>
      <label class="field"><span>Space records</span><input class="input" type="file" id="rc-fspace" accept=".xlsx"></label>
      <label class="field"><span>People records</span><input class="input" type="file" id="rc-fpeople" accept=".xlsx"></label></div>
    <div id="rc-result"></div>`,
    footer: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="rc-go">Save and check</button>` });
  let mode = tab;
  const setTab = t => {
    mode = t;
    d.el.querySelectorAll("[data-rt]").forEach(b => b.classList.toggle("on", b.dataset.rt === t));
    d.el.querySelectorAll("[data-rp]").forEach(x => x.hidden = x.dataset.rp !== t);
    d.el.querySelector("#rc-go").textContent = t === "upload" ? "Upload and check" : "Save and check";
    d.el.querySelector("#rc-result").innerHTML = "";
  };
  d.el.querySelectorAll("[data-rt]").forEach(b => b.onclick = () => setTab(b.dataset.rt));
  const b64 = f => new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1]); r.onerror = bad; r.readAsDataURL(f); });
  const show = r => {
    const df = r.diff;
    d.el.querySelector("#rc-result").innerHTML = r.error ? `<div class="banner error" style="margin-top:14px">${icon("attention")}<div>${esc(r.error)}</div></div>`
      : r.changed ? `<div class="banner info" style="margin-top:14px">${icon("check")}<div><b>Updated.</b> Rooms +${df.spaces.added} −${df.spaces.removed} ~${df.spaces.changed} · People +${df.people.added} −${df.people.removed} moved ${df.people.moved}</div></div>`
      : `<div class="banner info" style="margin-top:14px">${icon("check")}<div>No new version.</div></div>`;
  };
  d.el.querySelector("#rc-go").onclick = async e => {
    const b = e.currentTarget; b.disabled = true; const label = b.textContent; b.textContent = "Checking…";
    try {
      if (mode === "upload") {
        const fs = d.el.querySelector("#rc-fspace").files[0], fp = d.el.querySelector("#rc-fpeople").files[0];
        if (!fs || !fp) throw new Error("Choose both files");
        show(await api("/records/upload", { method: "POST", body: { space_b64: await b64(fs), people_b64: await b64(fp) } }));
      } else {
        const v = id => d.el.querySelector(id).value.trim();
        await api("/records/config", { method: "PUT", body: mode === "microsoft"
          ? { source: "microsoft", space_url: v("#rc-space"), people_url: v("#rc-people"), refresh_minutes: +v("#rc-every") }
          : { source: "folder", dir: v("#rc-dir"), refresh_minutes: +v("#rc-every2") } });
        show(await api("/records/refresh", { method: "POST" }));
      }
      connectionsSection(main);
    } catch (err) { d.el.querySelector("#rc-result").innerHTML = `<div class="banner error" style="margin-top:14px">${icon("attention")}<div>${esc(err.message)}</div></div>`; }
    b.disabled = false; b.textContent = label;
  };
}

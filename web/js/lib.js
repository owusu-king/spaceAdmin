// Shared client library: API access, escaping, small components and SVG charts.

export async function api(path, opts = {}) {
  const res = await fetch(`/api/v1${path}`, {
    headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    const msg = Array.isArray(d) ? d.map(e => `${(e.loc || []).slice(-1)[0]}: ${e.msg}`).join("; ") : (d || res.statusText);
    throw Object.assign(new Error(msg), { status: res.status });
  }
  return data;
}

export const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const fmtNum = (v, d = 0) => v == null ? "" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
export const pct = v => v == null ? "" : `${Math.round(v * 100)}%`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function fmtDate(iso, withTime = false) {
  if (!iso) return "";
  // a bare date ("2026-08-31") is a calendar day, not UTC midnight; read it as local time
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso}T00:00:00` : iso);
  if (isNaN(d)) return esc(iso);
  const base = `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
  return withTime ? `${base} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}` : base;
}
export function fmtWindow(start, end) {
  if (!start) return notOnRecord("Timing not stated");
  const s = new Date(start), e = end ? new Date(end) : null;
  const t = d => `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  const day = `${["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][s.getDay()]} ${MONTHS[s.getMonth()]} ${s.getDate()}`;
  if (!e) return `${day} ${t(s)}`;
  return s.toDateString() === e.toDateString() ? `${day} · ${t(s)}–${t(e)}` : `${day} ${t(s)} → ${fmtDate(end, true)}`;
}

// ---------------------------------------------------------------- icons (Lucide-style, 1.5 stroke)
const P = {
  overview: '<rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/>',
  spaces: '<path d="M3 21h18M5 21V5l7-2 7 2v16"/><path d="M9 9h1M14 9h1M9 13h1M14 13h1M9 17h1M14 17h1"/>',
  outages: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  cases: '<rect x="4" y="3" width="16" height="18" rx="1"/><path d="M8 8h8M8 12h8M8 16h5"/>',
  model: '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6.5 16 11M8 17.5l8-4.5"/>',
  agents: '<circle cx="8" cy="8" r="3"/><circle cx="17" cy="15" r="3"/><path d="M10.5 10 14.5 13"/><path d="M3 21c0-3 2-5 5-5M21 21c0-1.5-.5-2.5-1.5-3.3"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  send: '<path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/>', stop: '<rect x="7.5" y="7.5" width="9" height="9" rx="1.5"/>',
  micOff: '<path d="M3 3l18 18"/><path d="M9 9v2a3 3 0 0 0 5.1 2.1M15 10V6a3 3 0 0 0-5.7-1.3"/><path d="M5 11a7 7 0 0 0 11.5 5.3M19 11a7 7 0 0 1-.6 2.8M12 18v3M9 21h6"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6"/>',
  check: '<path d="M20 6 9 17l-5-5"/>', confirmed: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  recorded: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>', inferred: '<path d="M12 3 21 12 12 21 3 12z"/>',
  historical: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>', conflicting: '<path d="M7 4v6l-4 4M17 4v6l4 4M3 20h18"/>',
  unknown: '<path d="M9 9a3 3 0 1 1 4 2.8c-.6.3-1 .9-1 1.6V14M12 18h.01"/>', attention: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
  shield: '<path d="M12 3 5 6v6c0 4 3 7 7 9 4-2 7-5 7-9V6z"/><path d="m9 12 2 2 4-4"/>', lock: '<rect x="5" y="11" width="14" height="10" rx="1"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>', x: '<path d="M6 6l12 12M18 6 6 18"/>', copy: '<rect x="8" y="8" width="12" height="12" rx="1"/><path d="M4 16V4h12"/>',
  file: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>', refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>', chev: '<path d="m9 6 6 6-6 6"/>', dot: '<circle cx="12" cy="12" r="3"/>',
};
export const icon = (name, cls = "") => `<svg viewBox="0 0 24 24" class="${cls}" aria-hidden="true">${P[name] || ""}</svg>`;

// ---------------------------------------------------------------- chips & small parts
const CHIP = {
  confirmed: ["confirmed", "Confirmed"], recorded: ["recorded", "Recorded"], inferred: ["inferred", "Inference"],
  historical: ["historical", "Historical"], conflicting: ["conflicting", "Conflicting"], unknown: ["unknown", "Unknown"],
  potentially_suitable: ["confirmed", "Potentially suitable"], conditional: ["attention", "Conditional"],
  incompatible: ["conflicting", "Incompatible"], insufficient_information: ["unknown", "Insufficient information"],
  ReadyForReview: ["recorded", "Ready for review"], NeedsRevision: ["attention", "Needs revision"],
  Approved: ["confirmed", "Approved"], Declined: ["conflicting", "Declined"], predicted: ["inferred", "Prediction"],
  recorded_vacant: ["recorded", "Recorded vacant"], unaffected: ["shield", "Explicitly unaffected"],
};
export function chip(kind, label) {
  const [ic, text] = CHIP[kind] || ["dot", kind];
  return `<span class="chip ${esc(kind)}">${icon(ic)}${esc(label || text)}</span>`;
}
export const notOnRecord = (t = "Not on record") => `<span class="not-on-record">${esc(t)}</span>`;
export const val = (v, fallback) => (v === null || v === undefined || v === "") ? notOnRecord(fallback) : esc(v);
export const roomKey = k => `<span class="mono">${esc(k)}</span>`;

export function gauges(suit, cov, threshold = 0.6) {
  const prov = cov < threshold;
  return `<div class="gauges" title="Suitability and evidence coverage are reported separately">
    <div class="gauge"><span>Suitability</span><div class="bar ${prov ? "prov" : ""}"><i style="width:${Math.round(suit * 100)}%"></i></div><span class="v">${Math.round(suit * 100)}</span></div>
    <div class="gauge"><span>Coverage</span><div class="bar cov"><i style="width:${Math.round(cov * 100)}%"></i></div><span class="v">${pct(cov)}</span></div>
    ${prov ? '<div class="muted" style="font-size:11px">Provisional: evidence coverage below threshold</div>' : ""}</div>`;
}
export function miniBar(v, color = "var(--navy-700)") {
  if (v == null) return notOnRecord("Unknown");
  return `<div class="gauge" style="grid-template-columns:1fr 38px"><div class="bar"><i style="width:${Math.round(v * 100)}%;background:${color}"></i></div><span class="v">${v.toFixed(2)}</span></div>`;
}
export const skeleton = (rows = 4) => `<div class="stack">${Array.from({ length: rows }, (_, i) => `<div class="skel" style="width:${90 - i * 12}%"></div>`).join("")}</div>`;
export const empty = (text, action = "") => `<div class="empty">
  <svg viewBox="0 0 56 40"><rect x="2" y="2" width="52" height="36"/><path d="M2 20h20v18M22 2v10h32M38 20v18"/></svg>
  <div>${esc(text)}</div>${action ? `<div style="margin-top:12px">${action}</div>` : ""}</div>`;
export const errorBanner = (err, retry = "") => `<div class="banner error">${icon("attention")}<div><b>Could not load.</b> ${esc(err.message || err)}
  <div class="mono muted" style="margin-top:4px">ref ${Date.now().toString(36)}</div></div>${retry}</div>`;

/** One toast at a time, top right: a new message replaces the one showing (no stacked or overlapping toasts). */
let toastEl = null, toastTimer = 0;
export function toast(msg, ms = 4000) {
  if (!toastEl || !toastEl.isConnected) {
    toastEl = document.createElement("div");
    toastEl.className = "toast"; toastEl.setAttribute("role", "status"); toastEl.setAttribute("aria-live", "polite");
    document.body.appendChild(toastEl);
  } else {                                                 // replay the entrance so the change is noticed
    toastEl.style.animation = "none"; void toastEl.offsetWidth; toastEl.style.animation = "";
  }
  toastEl.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl && toastEl.remove(); toastEl = null; }, ms);
}

/** Cases this page created or opened. The live "new case" notice is only for cases that arrive on their own
 *  (outage email), never for one James just made. */
const knownCases = new Set();
export function noteCases(...ids) { ids.flat().forEach(id => id && knownCases.add(id)); }
export const isKnownCase = id => knownCases.has(id);

// ---------------------------------------------------------------- drawer / modal
/** dismissible: false keeps the dialog open on backdrop clicks and Escape; only [data-close] elements close it
 *  (for dialogs where closing by accident would lose work, such as the space request conversation). */
export function openPanel(html, { modal = false, onClose, dismissible = true } = {}) {
  const scrim = document.createElement("div");
  scrim.className = "scrim" + (modal ? " modal-wrap" : "");
  scrim.innerHTML = `<div class="${modal ? "modal" : "drawer"}" role="dialog" aria-modal="true">${html}</div>`;
  const close = () => {
    scrim.remove(); document.removeEventListener("keydown", key);
    if (!document.querySelector(".scrim")) document.documentElement.classList.remove("no-scroll");
    onClose && onClose();
  };
  const nudge = () => scrim.firstElementChild.animate(       // a gentle "use the close button" cue
    [{ transform: "scale(1)" }, { transform: "scale(1.012)" }, { transform: "scale(1)" }], { duration: 180, easing: "ease-out" });
  const key = e => { if (e.key === "Escape") dismissible ? close() : nudge(); };
  scrim.addEventListener("click", e => {
    if (e.target.closest("[data-close]")) close();
    else if (e.target === scrim) dismissible ? close() : nudge();
  });
  document.addEventListener("keydown", key);
  document.body.appendChild(scrim);
  if (!document.documentElement.classList.contains("no-scroll")) document.documentElement.style.setProperty("--sbw", `${window.innerWidth - document.documentElement.clientWidth}px`);
  document.documentElement.classList.add("no-scroll");
  const f = scrim.querySelector("input, select, textarea, button"); f && f.focus();
  return { el: scrim.firstElementChild, close };
}

/** Classic centered dialog: overline + title + close, scrolling body, optional footer.
 *  size: "md" (640px) | "lg" (920px). Returns { el, body, close }. */
export function modal({ overline = "", title, meta = "", body, footer = "", size = "md", onClose } = {}) {
  const p = openPanel(`
    <div class="dlg-head"><div>${overline ? `<div class="overline">${overline}</div>` : ""}<h2>${title}</h2>${meta ? `<div class="dlg-meta">${meta}</div>` : ""}</div>
      <button class="btn ghost sm" data-close aria-label="Close">${icon("x")}</button></div>
    <div class="dlg-body">${body}</div>${footer ? `<div class="dlg-foot">${footer}</div>` : ""}`, { modal: true, onClose });
  p.el.classList.add(`dlg-${size}`);
  p.el.querySelector("[data-close]").blur();
  return { el: p.el, body: p.el.querySelector(".dlg-body"), close: p.close };
}

// ---------------------------------------------------------------- charts (hand-built SVG)
let tipEl;
function tip(e, html) {
  tipEl = tipEl || Object.assign(document.createElement("div"), { className: "tip" });
  if (!tipEl.isConnected) document.body.appendChild(tipEl);
  tipEl.innerHTML = html; tipEl.style.display = "block";
  tipEl.style.left = `${e.clientX + 12}px`; tipEl.style.top = `${e.clientY + 12}px`;
}
const hideTip = () => tipEl && (tipEl.style.display = "none");

/** Line chart. series: [{name, color, values:[y...]}]; x = 1..n. One y-axis only. */
export function lineChart(el, series, { yLabel = "", height = 220, width = 560, yMin, ticks: nt = 5 } = {}) {
  const W = width, H = height, m = { l: 36, r: 8, t: 8, b: 24 };
  const n = Math.max(...series.map(s => s.values.length));
  const all = series.flatMap(s => s.values);
  const lo = yMin ?? Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 0.1;
  const y0 = lo - (yMin != null ? 0 : pad), y1 = hi + pad;
  const X = i => m.l + (i / (n - 1)) * (W - m.l - m.r), Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ticks = Array.from({ length: nt }, (_, i) => y0 + (i * (y1 - y0)) / (nt - 1));
  const xt = [0, Math.floor((n - 1) / 2), n - 1];
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} by epoch">
    <g class="grid">${ticks.map(t => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join("")}</g>
    <g class="axis">${ticks.map(t => `<text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`).join("")}
      ${xt.map((i, k) => `<text x="${X(i)}" y="${H - 8}" text-anchor="${k === 0 ? "start" : k === 2 ? "end" : "middle"}">${k === 2 ? `epoch ${i + 1}` : i + 1}</text>`).join("")}</g>
    ${series.map(s => `<path d="${s.values.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"/>`).join("")}
    <line class="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--line-strong)" style="display:none"/>
    <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>
  </svg>`;
  const svg = el.querySelector("svg"), xh = svg.querySelector(".xh"), hit = svg.querySelector("rect");
  hit.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
    xh.setAttribute("x1", X(i)); xh.setAttribute("x2", X(i)); xh.style.display = "";
    tip(e, `<b>Epoch ${i + 1}</b><br>${series.map(s => `<span style="color:${s.color}">■</span> ${esc(s.name)}: <span class="mono">${s.values[i]?.toFixed(4) ?? "–"}</span>`).join("<br>")}`);
  });
  hit.addEventListener("mouseleave", () => { xh.style.display = "none"; hideTip(); });
}

/** Horizontal bar list: [{label, value, note}] values in 0..1. */
export function barList(el, rows, { color = "var(--navy-700)", fmt = v => v.toFixed(2), label = 150, bar = 10, gap = 6, rowH } = {}) {
  el.innerHTML = `<div class="stack" style="--gap:${gap}px">${rows.map(r => `
    <div class="gauge" style="grid-template-columns:${label}px 1fr 40px;${rowH ? `height:${rowH}px;` : ""}" data-tip="${esc(r.note || "")}">
      <span style="color:var(--text-2);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(r.label)}</span>
      <div class="bar" style="height:${bar}px"><i style="width:${Math.max(1, r.value * 100)}%;background:${r.color || color}"></i></div>
      <span class="v">${fmt(r.value)}</span></div>`).join("")}</div>`;
  el.querySelectorAll("[data-tip]").forEach(n => {
    n.addEventListener("mousemove", e => n.dataset.tip && tip(e, n.dataset.tip));
    n.addEventListener("mouseleave", hideTip);
  });
}

/** Confusion matrix as a single-hue sequential heatmap (row-normalized). */
export function heatmap(el, matrix, labels, { cell = 30, left = 150, bottom = 120 } = {}) {
  const n = labels.length, top = 4;
  const W = left + n * cell + 8, H = top + n * cell + bottom;
  const rows = matrix.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s); });
  const short = l => l.length > 16 ? l.slice(0, 15) + "…" : l;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Confusion matrix, rows are recorded class">
    ${rows.map((r, i) => r.map((v, j) => `<rect x="${left + j * cell + 1}" y="${top + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="2"
      fill="${v === 0 ? "var(--sunken)" : `rgba(19,44,82,${(0.08 + 0.92 * v).toFixed(3)})`}" data-i="${i}" data-j="${j}"/>`).join("")).join("")}
    <g class="axis">${labels.map((l, i) => `<text x="${left - 8}" y="${top + i * cell + cell / 2 + 4}" text-anchor="end">${esc(short(l))}</text>`).join("")}
    ${labels.map((l, j) => `<text transform="translate(${left + j * cell + cell / 2 + 4},${top + n * cell + 8}) rotate(-55)" text-anchor="end">${esc(short(l))}</text>`).join("")}</g>
  </svg>`;
  el.querySelectorAll("rect").forEach(r => {
    r.addEventListener("mousemove", e => {
      const i = +r.dataset.i, j = +r.dataset.j;
      tip(e, `Recorded <b>${esc(labels[i])}</b><br>Predicted <b>${esc(labels[j])}</b><br><span class="mono">${matrix[i][j]}</span> records · ${pct(rows[i][j])} of row`);
    });
    r.addEventListener("mouseleave", hideTip);
  });
}

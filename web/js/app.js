// Shell and hash router. Each view module exports render(main, params) and sets the breadcrumb.
import { api, esc, icon } from "./lib.js";
import * as overview from "./views/overview.js";
import * as spaces from "./views/spaces.js";
import * as outages from "./views/outages.js";
import * as cases from "./views/cases.js";
import * as model from "./views/model.js";
import * as agents from "./views/agents.js";
import * as settings from "./views/settings.js";

const NAV = [
  { group: "Operations" },
  { id: "overview", label: "Overview", view: overview },
  { id: "spaces", label: "Spaces", view: spaces },
  { id: "outages", label: "Outages", view: outages },
  { id: "cases", label: "Cases", view: cases, count: true },
  { group: "Analysis" },
  { id: "model", label: "Model", view: model },
  { id: "agents", label: "Agents", view: agents },
  { group: "System" },
  { id: "settings", label: "Settings", view: settings },
];

export function setCrumbs(parts) {
  document.getElementById("crumbs").innerHTML = parts.map((p, i) =>
    i === parts.length - 1 ? `<b>${esc(p.label)}</b>` : `<a href="${p.href}">${esc(p.label)}</a>`).join(" &nbsp;›&nbsp; ");
}

export async function refreshCounts() {
  try {
    const list = await api("/cases");
    const n = list.filter(c => c.status === "ReadyForReview" || c.status === "NeedsRevision").length;
    const el = document.querySelector('[data-count="cases"]');
    if (el) el.textContent = n || "";
  } catch { /* counts are decorative; ignore */ }
}

function renderNav(active) {
  document.getElementById("nav").innerHTML = NAV.map(n => n.group
    ? `<div class="nav-label">${n.group}</div>`
    : `<a href="#/${n.id}" class="${n.id === active ? "active" : ""}" ${n.id === active ? 'aria-current="page"' : ""}>
         ${icon(n.id)}<span>${n.label}</span>${n.count ? `<span class="count" data-count="${n.id}"></span>` : ""}</a>`).join("");
}

async function route() {
  const [, id = "overview", ...rest] = (location.hash || "#/overview").split("?")[0].split("/");
  const entry = NAV.find(n => n.id === id) || NAV[1];
  renderNav(entry.id);
  const main = document.getElementById("main");
  main.innerHTML = "";
  window.scrollTo(0, 0);
  await entry.view.render(main, rest.map(decodeURIComponent));
  refreshCounts();
}

window.addEventListener("hashchange", route);
route();

import { api, chip, errorBanner, esc, fmtDate, fmtNum, icon, skeleton } from "../lib.js";
import { setCrumbs } from "../app.js";
import { newSpaceRequest } from "./spaces.js";

// Overview: key figures, what needs James now, and the state of the space portfolio.
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const localDate = s => new Date(/^\d{4}-\d{2}-\d{2}$/.test(s) ? `${s}T00:00:00` : s);

export async function render(main) {
  setCrumbs([{ label: "Overview" }]);
  const h = new Date().getHours();
  main.innerHTML = `
    <div class="page-head"><div><h1>Good ${h < 12 ? "morning" : h < 18 ? "afternoon" : "evening"}, James</h1><p id="asof"></p></div>
      <div class="actions"><button class="btn primary" id="new-space">${icon("plus")}Space request</button></div></div>
    <div id="body">${skeleton(8)}</div>`;
  main.querySelector("#new-space").onclick = () => newSpaceRequest();
  let d;
  try { d = await api("/overview"); } catch (e) { main.querySelector("#body").innerHTML = errorBanner(e); return; }
  const k = d.kpis, review = d.attention;
  const horizon = d.horizon.slice().sort((a, b) => a.start.localeCompare(b.start));
  const onRecord = horizon.filter(x => x.kind !== "predicted").length;
  const estimates = horizon.length - onRecord;
  const now = new Date(d.as_of);
  main.querySelector("#asof").textContent = `${DAY[now.getDay()]}, ${fmtDate(d.as_of)}`;

  main.querySelector("#body").innerHTML = `
    <div class="ov-kpis">
      ${kpi("#/cases", "To review", k.awaiting_review, review.filter(c => c.status === "NeedsRevision").length ? `${review.filter(c => c.status === "NeedsRevision").length} returned` : "Cases", true)}
      ${kpi("#/outages", "Outages · next 45 days", onRecord, estimates ? `+ ${estimates} estimated` : "Scheduled")}
      ${kpi("#/spaces?view=vacant&scope=ns", "Vacancies", k.ns_recorded_vacancies, "Vacant or under capacity")}
      ${kpi("#/spaces?view=flagged", "Records to verify", fmtNum(k.flagged_records), "Model disagrees")}
    </div>

    <div class="ov-grid">
      <section class="review-card">
        <header><h2>Needs your review</h2><a class="ov-link" href="#/cases">All cases</a></header>
        ${review.length ? review.slice(0, 3).map(c => `<a class="cl-row" href="#/cases/${esc(c.id)}">
            <span class="kind-dot ${c.kind}" title="${c.kind === "space" ? "Space request" : "Outage"}"></span>
            <span class="cl-main"><b>${esc(c.title)}</b><span class="cl-sub"><span class="mono">${esc(c.id)}</span> · ${esc(c.reference)}${c.open_questions ? ` · ${c.open_questions} question${c.open_questions === 1 ? "" : "s"}` : ""}</span></span>
            ${c.status === "ReadyForReview" ? "" : chip(c.status)}${icon("chev", "chev")}</a>`).join("")
          : `<div class="cl-empty">Nothing waiting for review.</div>`}
        ${k.awaiting_review > 3 ? `<a class="cl-more" href="#/cases">+${k.awaiting_review - 3} more</a>` : ""}
      </section>

      <section class="panel-card">
        <header><h2>Next 30 days</h2><a class="ov-link" href="#/outages">Outages</a></header>
        ${d.agenda.length ? `<ul class="agenda">${d.agenda.map(a => {
          const dt = localDate(a.start);
          return `<li class="ag ${a.kind}"><span class="ag-date"><b>${dt.getDate()}</b><span>${MON[dt.getMonth()]}</span></span>
            <span class="ag-main"><span class="ag-title">${esc(a.title)}</span>
              <span class="ag-sub">${a.kind === "outage" ? "Outage" : a.kind === "blocked" ? "Blocked time" : "Event"}${a.where ? ` · ${esc(a.where)}` : " · Campus-wide"}</span></span></li>`;
        }).join("")}</ul>` : `<div class="cl-empty">Nothing scheduled in the next 30 days.</div>`}
      </section>
    </div>

    <div class="ov-grid even">
      <a class="panel-card link-card" href="#/spaces?view=vacant&scope=ns">
        <header><h2>Natural Sciences seating</h2>${icon("chev", "chev")}</header>
        <div class="pad">${seating(d.seating)}</div>
      </a>
      <a class="panel-card link-card" href="#/spaces?view=flagged">
        <header><h2>Model disagreements</h2>${icon("chev", "chev")}</header>
        <div class="pad">${disagreements(d.flag_pairs, d.flagged_total)}</div>
      </a>
    </div>`;
}

function kpi(href, label, value, cap, accent = false) {
  return `<a class="ov-kpi${accent ? " accent" : ""}" href="${href}"><div class="overline">${label}</div>
    <div class="fig">${value}</div><div class="cap">${cap}</div></a>`;
}

// Ordinal states as one navy ramp (fullest = darkest); vacancies, the thing to act on, in brass.
const SEATS = [["Over Capacity", "Over capacity", "var(--navy-900)"], ["At Capacity", "At capacity", "var(--navy-600)"],
  ["Under Capacity", "Under capacity", "var(--navy-300)"], ["Vacant", "Vacant", "var(--brass)"]];

function seating(counts) {
  const rows = SEATS.map(([key, label, color]) => ({ label, color, n: counts[key] || 0 }));
  const total = rows.reduce((a, r) => a + r.n, 0) || 1;
  return `<div class="stack-bar" role="img" aria-label="${rows.map(r => `${r.label} ${r.n}`).join(", ")}">
      ${rows.filter(r => r.n).map(r => `<i style="flex:${r.n};background:${r.color}" title="${r.label}: ${r.n}"></i>`).join("")}</div>
    <ul class="stack-legend">${rows.map(r => `<li><i style="background:${r.color}"></i><span>${r.label}</span><b>${fmtNum(r.n)}</b><span class="muted">${Math.round(r.n / total * 100)}%</span></li>`).join("")}</ul>
    <p class="foot-note">${fmtNum(total)} workpoint rooms</p>`;
}

function disagreements(pairs, total) {
  if (!pairs.length) return `<p class="muted">No disagreements flagged.</p>`;
  const max = pairs[0].n;
  return `<ul class="pairs">${pairs.map(p => `<li><span class="pr-l">${esc(p.recorded)} <span class="muted">→</span> ${esc(p.predicted)}</span>
      <span class="pr-bar"><i style="width:${(p.n / max) * 100}%"></i></span><b>${p.n}</b></li>`).join("")}</ul>
    <p class="foot-note">Recorded → predicted · top 5 of ${fmtNum(total)}</p>`;
}

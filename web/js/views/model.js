import { api, barList, chip, errorBanner, esc, fmtNum, heatmap, icon, lineChart, modal, roomKey, skeleton } from "../lib.js";
import { setCrumbs } from "../app.js";

export async function render(main) {
  setCrumbs([{ label: "Model" }]);
  main.innerHTML = `<div class="page-head"><div><h1>Model</h1><p>SpaceFunctionNet · infers a room's function from its profile</p></div></div><div id="body">${skeleton(8)}</div>`;
  let d, abl;
  try { [d, abl] = await Promise.all([api("/model"), api("/ablation")]); } catch (e) { main.querySelector("#body").innerHTML = errorBanner(e); return; }
  const m = d.metrics, t = m.test, b = m.baselines;
  const fig = (label, v, cap) => `<div class="kpi"><div class="overline">${label}</div><div class="fig">${v}</div><div class="cap">${cap}</div></div>`;
  const ours = t.macro_f1;
  main.querySelector("#body").innerHTML = `
    <div class="kpis" style="grid-template-columns:repeat(3,1fr)">
      ${fig("Macro-F1", ours.toFixed(3), "Test split · all categories equal")}
      ${fig("Accuracy", t.accuracy.toFixed(3), `${fmtNum(t.n)} test records`)}
      ${fig("Unseen buildings", m.building_holdout.macro_f1.toFixed(3), "Macro-F1 · buildings held out")}
    </div>
    <div class="chart-grid g3">
      ${card("Loss", `<span class="legend"><span><i style="background:var(--series-1)"></i>Train</span><span><i style="background:var(--series-2)"></i>Val</span></span>`, "c-loss")}
      ${card("Validation macro-F1", `${m.history.length} epochs`, "c-f1")}
      ${card("Against baselines", "Test macro-F1", "c-base")}
    </div>
    <div class="chart-grid g2">
      ${card("F1 by category", "Test split", "c-class")}
      ${card("Confusion matrix", "Recorded × predicted", "c-cm")}
    </div>`;
  main.querySelector(".page-head").insertAdjacentHTML("beforeend", `<div class="actions">
      <button class="btn ghost" data-row="design">Design</button>
      <button class="btn ghost" data-row="flags">Records to verify</button>
      <button class="btn" data-row="ablation">Ablation study</button></div>`);
  // line charts are drawn at their real pixel width so text keeps its size at any layout width
  const lines = () => {
    const opt = el => ({ width: Math.max(240, el.clientWidth), height: 150, ticks: 4 });
    const loss = main.querySelector("#c-loss"), f1 = main.querySelector("#c-f1");
    if (!loss) return window.removeEventListener("resize", onResize);
    lineChart(loss, [
      { name: "Training loss", color: "var(--series-1)", values: m.history.map(e => e.train_loss) },
      { name: "Validation loss", color: "var(--series-2)", values: m.history.map(e => e.val_loss) }], { yLabel: "Loss", ...opt(loss) });
    lineChart(f1, [{ name: "Validation macro-F1", color: "var(--series-1)", values: m.history.map(e => e.val_macro_f1) }], { yLabel: "Macro-F1", ...opt(f1) });
  };
  let rt; const onResize = () => { clearTimeout(rt); rt = setTimeout(lines, 120); };
  window.addEventListener("resize", onResize);
  lines();
  barList(main.querySelector("#c-base"), [
    { label: "SpaceFunctionNet", value: ours, note: "Deployed model · test split" },
    { label: "Logistic regression", value: b.logistic_regression.macro_f1, color: "var(--line-strong)", note: "Same inputs · test split" },
    { label: "Majority class", value: b.majority_class.macro_f1, color: "var(--line-strong)", note: "Always predicts the most common class" },
  ], { label: 120, bar: 8, gap: 22, fmt: v => v.toFixed(3) });
  const order = m.classes.slice().sort((x, y) => t.per_class[y]["f1-score"] - t.per_class[x]["f1-score"]);
  barList(main.querySelector("#c-class"), order.map(c => ({ label: c, value: t.per_class[c]["f1-score"],
    note: `${esc(c)} · precision ${t.per_class[c].precision.toFixed(2)} · recall ${t.per_class[c].recall.toFixed(2)} · n=${t.per_class[c].support}` })),
    { label: 130, bar: 6, gap: 4, rowH: 20 });
  heatmap(main.querySelector("#c-cm"), t.confusion, m.classes, { cell: 22, left: 118, bottom: 92 });
  const open = {
    ablation: () => modal({ overline: "Model", title: "Ablation study", size: "lg", meta: "One change per variant · 3 seeds · chosen on validation", body: ablationTable(abl) }),
    flags: () => modal({ overline: "Model", title: "Records to verify", meta: "Strongest disagreements with the recorded class", size: "lg",
      body: `<table class="t plain"><thead><tr><th>Space</th><th>Recorded</th><th>Model</th><th class="r">Confidence</th></tr></thead><tbody>
        ${d.flagged_examples.map(f => `<tr><td>${roomKey(f.space_id)}</td><td>${esc(f.recorded)}</td><td>${chip("inferred", f.predicted)}</td><td class="r mono">${f.confidence.toFixed(2)}</td></tr>`).join("")}</tbody></table>`,
      footer: `<a class="btn" href="#/spaces?view=flagged" data-close>Open in Spaces</a><button class="btn" data-close>Close</button>` }),
    design: () => modal({ overline: "Model", title: "Design", body: `<dl class="kv">
        <dt>Inputs</dt><dd>${m.features.categorical.length} categorical (${m.config.use_embeddings === false ? "one-hot" : "embeddings"}) · ${m.features.numeric.length} numeric</dd>
        <dt>Network</dt><dd>${m.config.hidden.join(" → ")} · BatchNorm · ReLU · dropout ${m.config.dropout} → ${m.classes.length} classes</dd>
        <dt>Training</dt><dd>${m.config.class_weighting === false ? "Cross-entropy" : "Weighted cross-entropy"} · AdamW · lr ${m.config.lr}</dd>
        <dt>Data split</dt><dd>${fmtNum(m.split.train)} / ${fmtNum(m.split.val)} / ${fmtNum(m.split.test)} · seed ${m.config.seed}</dd>
        <dt>Leakage</dt><dd>Label-derived fields excluded · encoder fit on training only · agents read out-of-fold predictions</dd>
        <dt>Used by</dt><dd>Space agent (verify actual use) · Outage agent (lab-like rooms)</dd></dl>` }),
  };
  main.querySelectorAll("[data-row]").forEach(b2 => b2.onclick = () => open[b2.dataset.row]());
}

const card = (title, note, id) => `<section class="chart-card"><header><h3>${title}</h3><span>${note}</span></header><div class="chart" id="${id}"></div></section>`;

function ablationTable(a) {
  if (!a) return `<p class="muted">Run <span class="mono">python -m app.ml.ablation</span>.</p>`;
  const rows = a.variants.slice().sort((x, y) => y.val_macro_f1 - x.val_macro_f1);
  return `<table class="t plain"><thead><tr><th>Variant</th><th class="r">Validation</th><th class="r">Test</th><th class="r">Parameters</th></tr></thead><tbody>
    ${rows.map(r => `<tr ${r.variant === a.selected_by_validation ? 'class="sel"' : ""}><td>${esc(r.label.replace(" (selected, deployed)", ""))}${r.variant === a.selected_by_validation ? " " + chip("confirmed", "Deployed") : ""}</td>
      <td class="r mono">${r.val_macro_f1.toFixed(3)} <span class="muted">±${r.val_std.toFixed(3)}</span></td>
      <td class="r mono">${r.macro_f1.toFixed(3)}</td><td class="r">${fmtNum(r.parameters)}</td></tr>`).join("")}</tbody></table>
    <p class="muted" style="font-size:13px;margin:12px 0 0">Data matters most: departmental allocation is worth about 0.1 macro-F1. Test scores are reported, never used to choose.</p>`;
}

"""Generate the two project notebooks from source so they stay in sync with the app.

Run from the project root:  python notebooks/build_notebooks.py
Then execute:               python -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = '''import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import logging, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 70)
from app.logging_setup import setup_logging
setup_logging(logging.WARNING)          # keep notebook output focused on results
NAVY, RUST = "#25508F", "#C0652B"       # validated two-series palette used by the app'''


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


nb1 = [
    md("""
# 01 · Deep Learning: SpaceFunctionNet

**What we build.** A neural network that infers the *functional category* of a room (Office, Research Lab, Classroom, …)
from its physical, administrative and occupant profile, using the real `SpaceRecords.xlsx` and `Report_PeopleRecords.xlsx`.

**Why.** James explained that Space Class "can reflect the original intended use, so actual current use still needs
verification". A model that has learned what each kind of room *looks like* can flag records whose recorded class disagrees
with their profile. The agents use it to (1) ask for actual-use verification on space candidates and (2) catch rooms that are
recorded as non-lab but look lab-like when an exhaust, steam or air outage is coming.

**Why not outage prediction?** The correspondence yields only 40 distinct outage requests: too few for an honest neural model.
Recurrence estimates for outages stay statistical and are labelled as predictions (see notebook 02).

Every function used here lives in `app/` and is the same code the application runs.
"""),
    code(SETUP + '''
from app.data.loader import load_dataset
ds = load_dataset()
spaces = ds.spaces
print(f"{len(spaces):,} space records · {len(ds.people):,} people records · {spaces.building.nunique()} buildings")
print({k: v for k, v in ds.report.items() if k in ("duplicate_room_keys", "spaces_missing_area", "people_with_location", "people_location_unmatched")})
dist = spaces["class_group"].value_counts().rename("records").to_frame()
dist["share"] = (dist.records / dist.records.sum()).round(3)
dist'''),
    md("""
## 1 · Preprocessing and leakage control

* **Target** `class_group`: the 90 Space Class codes grouped into 12 functional categories (`app/config.py → CLASS_GROUPS`).
* **Inputs** (`app/ml/features.py`): building, floor, departmental allocation, room-number pattern and suffix (embedded
  categoricals) plus log-area, allocation count and recorded occupants by role (joined from people records).
* **Excluded on purpose:** Seat Stat, SSA, Avble Seats, W/S Area and spaceStandard. They are *derived from* the class in the
  export, so using them would leak the answer.
* **Split:** stratified 70 / 15 / 15 with a fixed seed. The encoder (vocabularies, means, standard deviations) is fitted on the
  training split only; unseen categories map to `<UNK>`.
"""),
    code('''
from app.config import TRAIN
from app.ml.features import CATEGORICAL, NUMERIC, derive_columns
from app.ml.train import split_indices
X = derive_columns(spaces)
y = spaces["class_group"].astype(str)
train_idx, val_idx, test_idx = split_indices(y, TRAIN)
print(f"train {len(train_idx):,} · validation {len(val_idx):,} · test {len(test_idx):,}")
print("categorical:", CATEGORICAL); print("numeric:", NUMERIC)
X.iloc[train_idx].head(4)'''),
    md("""
## 2 · Model architecture

A multilayer perceptron (MLP). Each categorical column (building, floor, department, room-number pattern, suffix) enters
either as **learned entity embeddings** (a small trainable vector per value) or as **one-hot vectors**. We designed the
network with embeddings, then let the ablation study in section 5 decide: one-hot inputs scored higher on validation,
so the deployed model uses them (`TRAIN.use_embeddings = False`).

`[categoricals ‖ numerics] → Linear(128) → BatchNorm → ReLU → Dropout → Linear(64) → BatchNorm → ReLU → Dropout → Linear(12)`
"""),
    code('''
import torch
from app.ml.features import FeatureEncoder
from app.ml.model import SpaceFunctionNet
enc_preview = FeatureEncoder().fit(X.iloc[train_idx], y.iloc[train_idx], TRAIN.min_category_count)
net = SpaceFunctionNet(enc_preview.cardinalities, len(NUMERIC), len(enc_preview.classes), TRAIN.hidden, TRAIN.dropout,
                       TRAIN.use_embeddings)
print(net)
print(f"trainable parameters: {sum(p.numel() for p in net.parameters()):,}")'''),
    md("""
## 3 · Training

`train_model` (in `app/ml/train.py`) uses **cross-entropy** (class weighting for the imbalance is available but the
ablation showed it lowers validation macro-F1 here), **AdamW** (lr 2e-3, weight decay 1e-4), mini-batches of 256, a learning-rate scheduler and **early stopping on validation
macro-F1** (patience 8). The best epoch's weights are restored. Seeds are fixed, so the run is reproducible.
"""),
    code('''
from app.ml.train import train_model
model, enc, history = train_model(X, y, train_idx, val_idx, TRAIN, verbose=False)
h = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(h.epoch, h.train_loss, color=NAVY, lw=2, label="training loss")
ax[0].plot(h.epoch, h.val_loss, color=RUST, lw=2, label="validation loss")
ax[0].set_xlabel("epoch"); ax[0].set_title("Loss"); ax[0].legend(frameon=False)
ax[1].plot(h.epoch, h.val_macro_f1, color=NAVY, lw=2); ax[1].set_xlabel("epoch"); ax[1].set_title("Validation macro-F1")
for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#D9DFE8")
plt.tight_layout(); plt.show()
best = h.loc[h.val_macro_f1.idxmax()]
print(f"stopped after {len(h)} epochs · best epoch {int(best.epoch)} · val macro-F1 {best.val_macro_f1:.3f}")'''),
    md("""
## 4 · Evaluation on the untouched test split

Macro-F1 is the headline metric because it weights the small categories (Meeting, Classroom, Teaching Lab) as much as the
large ones. Two baselines run on the identical split: majority class and multinomial logistic regression.
"""),
    code('''
from app.ml.train import baselines, evaluate
test = evaluate(model, enc, X, y, test_idx)
base = baselines(X, y, train_idx, test_idx, TRAIN)
summary = pd.DataFrame({"SpaceFunctionNet": {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"]},
                        **{k: v for k, v in base.items()}}).T.round(3)
display(summary)
per_class = pd.DataFrame(test["per_class"]).T.loc[enc.classes, ["precision", "recall", "f1-score", "support"]].round(2)
cm = np.array(test["confusion"]); cm_norm = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.imshow(cm_norm, cmap="Blues"); ax.set_xticks(range(12), enc.classes, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(12), enc.classes, fontsize=8); ax.set_xlabel("predicted"); ax.set_ylabel("recorded")
ax.set_title("Confusion matrix (row share)"); plt.tight_layout(); plt.show()
per_class.sort_values("f1-score", ascending=False)'''),
    md("""
**Interpretation.** The network beats logistic regression on macro-F1 and is far above the majority baseline. It is strong on
categories with a distinctive profile (Workstation, Residential, Building Service, Research Lab) and weak on Meeting and Office
Service, which share buildings, sizes and departments with offices. `python -m app.ml.train` also reports a harder check,
holding out entire buildings: macro-F1 falls below 0.5, so the model should only be used on buildings it has seen
(which is how the agents use it).
Those numbers are in `models/metrics.json` and on the app's **Model** page.
"""),
    code('''
from app.ml.train import predict_proba
proba = predict_proba(model, enc, X.iloc[test_idx])
pred = np.array(enc.classes)[proba.argmax(1)]
examples = pd.DataFrame({"room": spaces.space_id.iloc[test_idx].values, "recorded": y.iloc[test_idx].values,
                         "predicted": pred, "confidence": proba.max(1).round(3)})
disagree = examples[(examples.recorded != examples.predicted) & (examples.confidence >= 0.7)]
print(f"{len(disagree)} of {len(examples):,} test records disagree with ≥70% confidence → candidates for actual-use verification")
pd.concat([examples.sample(4, random_state=3), disagree.sort_values("confidence", ascending=False).head(6)])'''),
    md("""
## 5 · Ablation study: which design choices matter?

`python -m app.ml.ablation` retrains the network with **one change at a time**, three seeds each, on the same split. We choose
the configuration by **validation** macro-F1 only; test scores are reported but never used to choose (that would leak the test
set into model selection). Results are saved in `models/ablation.json`; the run takes about 8 minutes, so we load them here.
"""),
    code('''
import json
abl = json.loads((ROOT / "models" / "ablation.json").read_text())
t = pd.DataFrame(abl["variants"]).set_index("label")[["val_macro_f1", "val_std", "macro_f1", "test_std", "parameters"]]
t.columns = ["val macro-F1", "val ±", "test macro-F1", "test ±", "parameters"]
display(t.sort_values("val macro-F1", ascending=False).round(4))
order = t.sort_values("val macro-F1")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(order.index, order["val macro-F1"], xerr=order["val ±"], color=NAVY, height=0.55)
ax.set_xlim(0.6, 0.78); ax.set_xlabel("validation macro-F1 (mean ± std over 3 seeds)")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()
print("selected by validation:", abl["selected_by_validation"], "| deployed config: use_embeddings =", TRAIN.use_embeddings,
      "class_weighting =", TRAIN.class_weighting)'''),
    md("""
**What the ablation teaches us.** The *data* matters more than the architecture: removing departmental allocation costs about
0.1 macro-F1 and removing room-number features about 0.03, while depth changes almost nothing. Learned embeddings did not beat
one-hot inputs on this tabular data (each category has enough examples for one-hot weights to be learned directly), and class
weighting hurt slightly. Both findings changed the deployed model.
"""),
    md("""
## 6 · Saving, loading and integration

`python -m app.ml.train` saves the deployed artifacts to `models/` (weights, encoder, metrics) and also writes **out-of-fold
predictions** from 5-fold cross-fitting, so every record shown to the agents was scored by a model that never saw its label.

At start-up the application loads them through `SpacePredictor` (`app/ml/inference.py`). Below we save this notebook's model
to a scratch folder, reload the production model, and confirm both give the same predictions (same code, same seed).
"""),
    code('''
from app.config import MODEL_DIR
from app.ml.inference import SpacePredictor
scratch = ROOT / "models" / "notebook_run"; scratch.mkdir(exist_ok=True)
torch.save(model.state_dict(), scratch / "space_function_net.pt")
predictor = SpacePredictor(spaces)                         # what the app loads at start-up
deployed = predictor.predict_frame(spaces.iloc[test_idx[:500]]).to_numpy()
same = np.allclose(deployed, proba[:500], atol=1e-5)
print("production artifacts:", sorted(p.name for p in MODEL_DIR.glob("*.*")))
print("notebook model reproduces deployed predictions:", same)
sid = disagree.room.iloc[0]
print("\\nWhat an agent tool receives for", sid); predictor.assessment(sid)'''),
]

nb2 = [
    md("""
# 02 · Intelligent Agents: Space Administrator and Outage

| | Space Administrator Agent | Outage Agent |
|---|---|---|
| Purpose | Find and assess suitable space; support assignment review | Interpret outages, assess impact, coordinate stakeholders and timing |
| Never does | Book, reserve or assign space | Send messages, approve outages, edit TRIRIGA |
| Uses the model for | Functional fit and "verify actual use" conditions | Rooms recorded as non-lab that look lab-like |

Both agents are deterministic, tool-using workflows (no external LLM: personnel data stays local). Each tool call is
permission-checked, bounded, timed and logged. The agents talk only through the coordinator. Everything below imports
the application modules; nothing is re-implemented.
"""),
    code(SETUP + '''
from datetime import datetime
from app.agents.coordinator import build_system
system = build_system(as_of=datetime(2026, 11, 22))      # fixed reference date → reproducible output
for agent in (system.space, system.outage):
    d = agent.describe()
    print(f"{d['name']:<12} tools={len(d['tools'])}  locked actions={list(d['locked_actions'])}")
    for o in d["objectives"]: print("   •", o)'''),
    md("""
## 1 · How the agents access the deep learning model

The model is one tool among many. Its output is evidence class **Inference**; the agent decides what to do with it.
"""),
    code('''
sid = system.predictor.table.query("verify_flag and recorded_group == 'Office'").index[0]
r = system.space.call("model_space_assessment", space_ids=[sid], function="Office")
print("status:", r.status, "| duration:", r.duration_ms, "ms")
print("model output:", r.data[sid]["predicted_group"], r.data[sid]["confidence"], "| verify flag:", r.data[sid]["verify_flag"])
print("evidence:", r.evidence[0].classification, "→", r.evidence[0].claim)'''),
    md("""
## 2 · Space Administrator Agent: an office close to a professor

James's scenario: *"I need a new room for Professor X's new hire."* The agent finds Professor X in the people records, takes
the recorded office as the reference, and ranks candidates by **proximity** (same floor and nearby room numbers first, then
the same building), together with function, area, occupancy, department and planning criteria. The agent resolves the department, finds compatible records,
applies hard constraints (area, recorded capacity), scores six weighted criteria and reports **suitability** and **evidence
coverage** separately. Recorded vacancy is a note, never permission.
"""),
    code('''
from app.agents.schemas import SpaceRequest
req = SpaceRequest(request_type="new_office", near="Bonnie Bartel", function="Office")
space = system.run_space(req)
print(space["summary"])
print([q["value"] for q in space["requirements"] if q["field"] == "Close to"])
cand = pd.DataFrame(space["candidates"])[["room_key", "area", "status", "suitability", "evidence_coverage"]]
cand.insert(1, "distance", [c["proximity"]["label"] for c in space["candidates"]])
cand["model"] = [c["model"]["predicted_group"] if c["model"] else None for c in space["candidates"]]
display(cand)
print("Verification questions:", *space["verification_questions"], sep="\\n  - ")
print("Required approvals:", *space["required_approvals"], sep="\\n  - ")'''),
    md("""
## 3 · Outage Agent: an emergency steam outage (real ABL thread)

The input is the whole ABL correspondence file, which mixes several incidents. The agent isolates the current incident,
reconstructs the timeline, and keeps the Oct 22 *preliminary* concerns (heating, hot water) apart from the confirmed scope
(autoclaves only), exactly as James described.
"""),
    code('''
from app.agents.schemas import OutageRequest
from app.scenarios import resolve_payload, scenarios
abl = next(s for s in scenarios() if s.id == "OUT-ABL-STEAM")
out = system.run_outage(OutageRequest(**resolve_payload(abl, system.corpus)))
print(out["summary"]); print("messages set aside as other incidents:", out["incident"]["set_aside"])
display(pd.DataFrame(out["timeline"])[["date", "type", "sender", "superseded"]])
rows = [(group, i["statement"][:90], i["classification"], i.get("note", "")) for group, items in out["impacts"].items() for i in items]
pd.DataFrame(rows, columns=["register", "statement", "evidence", "note"])'''),
    md("""
## 4 · Does the outage conflict with anything? (real BRK thread)

James's most important check: *will this outage time conflict with an event?* The BRK emails propose elevator work for
"Wednesday, November 25" (a date in prose, no time). A different email says that date starts the Thanksgiving recess. The
agent reads the date, checks it against the events calendar (dated events from the correspondence, imported calendars,
manual entries and times James has blocked), reports the conflict and suggests conflict-free times. If James says "not
then", blocking a time makes every later check flag it.
"""),
    code('''
brk = next(s for s in scenarios() if s.id == "OUT-BRK-ELEVATOR")
o2 = system.run_outage(OutageRequest(**resolve_payload(brk, system.corpus)))
sch = o2["scheduling"]
print("proposed:", o2["outage"]["proposed_start"][:10], "| date only:", o2["outage"]["date_only"], "| duration:", o2["outage"]["duration_hours"], "h")
print("verdict:", sch["verdict"]); display(pd.DataFrame(sch["conflicts"])[["title", "building", "source"]])
display(pd.DataFrame(sch["alternatives"])[["start", "end", "rationale"]])
blocked = system.outage.events.add("Symposium (James: no outage)", datetime(2026, 11, 28, 0), datetime(2026, 11, 29, 0),
                                   "Brockman Hall", "blocked", "blocked")
again = system.outage.review_window(["Brockman Hall"], datetime(2026, 11, 28, 6), datetime(2026, 11, 28, 14))
print("James blocks Nov 28 → re-check verdict:", again["verdict"], "| next suggestion:", again["alternatives"][0]["start"])
system.outage.events.delete(blocked["id"])'''),
    md("""
## 5 · Agent interaction: the two agents exchange information

Temporary space is rarely needed, so it is **on request only** (`cooperate=True`, or a button in the app). The Outage
Agent sends a `temporary_space_request` through the coordinator; the Space Administrator Agent runs its own workflow and
replies with candidates outside the affected building. Neither agent can call the other's tools.
"""),
    code('''
out = system.run_outage(OutageRequest(**resolve_payload(abl, system.corpus)), cooperate=True)
for m in out["messages"]:
    print(f"{m['sender']} → {m['recipient']} · {m['intent']}")
reply = out["space_support"]["reply"]["payload"]
display(pd.DataFrame(reply["candidates"]))
print(reply["note"])'''),
    md("""
## 6 · Results: affected people, stakeholders and the drafts for approval
"""),
    code('''
st = pd.DataFrame([{"department": d["department"], "formal approver": d["formal_approver"]["status"],
                    "coordination contacts": ", ".join(c["name"] for c in d["coordination_contacts"]) or "none on record",
                    "recipients": d["notification_recipients"]["count"]} for d in out["stakeholders"]["departments"]])
display(st)
people = out["affected_people"]
print(people["scope_rule"]); print(f"{people['total']} recorded people affected, {people['natural_sciences']} in Natural Sciences")
display(pd.DataFrame(people["people"])[["name", "email", "department", "room", "reason"]].head(6))
for d in out["communication_drafts"]:
    print(f"\\n=== {d['title']} · {d['status']}"); print(d["body"][:700])
print("\\nAttempt to send →", system.outage.call("send_email").status)'''),
    md("""
## 7 · Evaluation, cooperation and competition

* **Scenario evaluation** (`app/evaluation.py`): binary checks written from James's documented examples plus governance and
  safety checks (locked actions, permission boundaries, an injected instruction inside a notice).
* **Cooperation** (`app/experiments.py`): every recorded ≥4-hour lab/occupancy outage, run with and without the Space agent.
* **Competition**: three strategies for choosing records to verify (rules, model, combined) on 1,000 injected class errors.
"""),
    code('''
from app.evaluation import evaluate_agents
from app.experiments import competition_experiment, cooperation_experiment
ev = evaluate_agents(system)
print(f"Scenario checks passed: {ev['passed']}/{ev['total']}")
display(pd.DataFrame([{"scenario": s["title"], "passed": f"{s['passed']}/{s['total']}"} for s in ev["scenarios"]]))
coop = cooperation_experiment(system)["summary"]
display(pd.Series(coop, name="cooperation").to_frame())
comp = competition_experiment(system)
display(pd.DataFrame(comp["results"]).T[["precision", "recall", "f1", "flags"]])
print("best strategy:", comp["winner"])'''),
    md("""
## 8 · Limitations

* No approver roster, class schedule or maintenance calendar is in the supplied data, and the events calendar only holds what
  the emails mention plus what the team imports or adds. Gaps are reported as gaps; the agents never assume "no conflict".
* Department contacts come only from what the emails show and are marked *candidate*.
* Dates in prose are read as whole days when no time is given; the agent asks for the hours.
* Proximity is estimated from building, floor and room numbering, not measured walking distance (floor plans are images).
* The model generalizes poorly to buildings it has never seen; competition errors are synthetic and easier than real drift.
* Agents are rule-based workflows; an LLM could later draft richer text behind the same tool and permission boundaries.
"""),
]

for name, cells in [("01_deep_learning.ipynb", nb1), ("02_intelligent_agents.ipynb", nb2)]:
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, HERE / name)
    print("wrote", name, len(cells), "cells")

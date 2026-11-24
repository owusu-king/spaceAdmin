# Campus Facilities AI — How It Works

A guide for the team: what the app does, how the parts connect, what to understand about the deep learning model and
the agents, and how we measured all of it. For installation see [SETUP.md](SETUP.md).

---

## 1. The problem in one paragraph

James, a facilities administrator for Rice's Natural Sciences buildings, does two jobs by hand. **Space:** when someone
needs an office or a lab, he cross-checks 22,408 room records and 16,161 people records, and remembers that a room
recorded as "vacant" is not permission to assign it. **Outages:** when a TRIRIGA notice or an email about a steam,
power or elevator outage arrives, he works out which buildings, departments and people are affected, whether the
time conflicts with anything, and drafts the messages. The app does this review work and prepares the result as a
**case**. James always decides.

## 2. The big picture

```
 Browser (web/)                        FastAPI server (app/api.py)                         Live data (read-only)
 ─────────────                         ───────────────────────────                         ─────────────────────
 Overview · Spaces · Outages      ┌──► Coordinator ──► Space Administrator Agent ◄──┐      Microsoft 365 mail
 Cases · Model · Agents · Settings│    (agents/)       Outage Agent                  │       (every 60 s)
       │  REST /api/v1            │          │  tools: records, model, rules         │      SharePoint reports
       │  WebSocket (voice)       │          ▼                                       │       (on a schedule)
       └──────────────────────────┤    Cases (SQLite): revisions, reviews, audit     │
                                  │                                                  │
                                  ├──► Voice conversation (LangGraph) ───────────────┘
                                  ├──► Email triage agent (LangGraph) ──► Outage Agent ──► case
                                  └──► Records pipeline ──► validated snapshot ──► agents switch to it
```

One Python process serves the web app and the API. Data lives in Excel exports, RTF email threads and two SQLite
files. Every model runs on the same computer: the space-function network (PyTorch), the language model (Ollama),
speech-to-text (Parakeet), text-to-speech (Kokoro) and speech detection (Silero).

## 3. How the files connect

| Folder / file | Role | Talks to |
|---|---|---|
| `app/config.py` | Paths, building crosswalk, departments, weights, thresholds; loads `.env` | everything |
| `app/data/loader.py` | Reads and validates the space and people exports; joins people to rooms | agents, model, records pipeline |
| `app/data/buildings.py` | Building names, codes and aliases ("Brockman" is ambiguous on purpose) | agents, triage, voice |
| `app/data/notices.py` | RTF threads → messages → TRIRIGA fields → outage history | Outage Agent, email replay |
| `app/data/records.py` | **Records pipeline**: watch a source, validate, compare, snapshot, switch | `api.py` |
| `app/ml/` | `features.py` → `model.py` → `train.py` / `ablation.py` → `inference.py` | Space and Outage agents |
| `app/agents/base.py` | Agent framework: tools, permissions, 40-call limit, trace, locked actions | both agents |
| `app/agents/space_agent.py` | Finds and ranks rooms | coordinator, voice |
| `app/agents/outage_agent.py` | Reads notices, assesses impact, finds people, checks the calendar, drafts messages | coordinator, email intake |
| `app/agents/coordinator.py` | Builds the system; routes requests; relays the one allowed agent-to-agent message | `api.py` |
| `app/services/` | Scheduling, events calendar, stakeholders, **cases**, settings switches, outbox | `api.py`, agents |
| `app/voice/` | Local voice conversation: engines, Ollama client, **LangGraph** graph, WebSocket session | `api.py` |
| `app/integrations/email.py` | Email intake: replay, IMAP or Microsoft 365 → triage → cases | `triage.py`, Outage Agent |
| `app/integrations/triage.py` | **Email triage agent (LangGraph)**: rules → model → decide → case / ignore / review | email intake |
| `app/integrations/msgraph.py` | Microsoft 365 sign-in and read-only reads (mail, SharePoint files) | email intake, records |
| `app/integrations/mailer.py` | Sends an approved draft (only when switched on, only by a person) | `outbox.py` |
| `app/evaluation.py`, `app/experiments.py`, `app/triage_eval.py` | Scenario checks, cooperation/competition, triage evaluation | results in `data/*.json` |
| `web/js/views/*.js` | One file per page; `voice.js` is the conversation | the API |
| `notebooks/` | `01` model, `02` agents; regenerate with `build_notebooks.py` | `app/` |
| `tests/` | 105 tests (`pytest -q`) | everything |

**Where to change things:** thresholds and weights in `app/config.py`; a new agent tool is a `@tool` method plus its
name in the agent's `permissions`; a new page is a module in `web/js/views/` registered in `web/js/app.js`.

## 4. Ideas you must be able to explain

1. **Agents are workflows over permitted tools, not chatbots.** Each agent may call only the tools in its permission
   list, at most 40 per run, and every call is logged in a trace. Sending email, writing to TRIRIGA, assigning space and
   approving outages are **locked actions**: no agent can ever perform them.
2. **Evidence has a certainty class.** Every finding is *Confirmed, Recorded, Inference, Historical, Conflicting* or
   *Unknown*. Model output is always *Inference*. "Not on record" is different from zero.
3. **James's rules are code.** Vacancy ≠ availability; allocation ≠ affiliation; an occupant is not an approver; a
   missing schedule is a *coverage gap*, never "no conflict"; a preliminary concern is not a confirmed impact.
4. **Cases are revisioned and audited.** A re-run creates Rev B and invalidates earlier approvals. Approval binds to one
   revision. Every decision, switch change, deletion and send is in the audit log.
5. **Suitability and evidence coverage are reported separately.** A room can look ideal on the little that is known;
   coverage below 60 % marks its score *Provisional*.
6. **Language models only read and talk; rules and people decide.** The local LLM turns speech or email into fields,
   and summarizes reports. Every value it extracts is checked against what was actually said and against the records.

## 5. The deep learning model: SpaceFunctionNet

**Task.** Predict a room's *functional category* (12 classes: Office, Research Lab, Classroom, …) from its profile.
Why: James says the recorded Space Class "can reflect the original intended use, so actual current use still needs
verification". A model that disagrees confidently with the record flags it for checking.

**Inputs.** Building, floor, department allocation, room-number pattern and suffix (categorical); log area, missing-area
flag, number of allocations, room number mod 100, and recorded occupants by role (numeric).

**Leakage control.** Fields derived from the label (Seat Stat, SSA, available seats, W/S area, space standard) are
excluded. The encoder is fit on the training split only. The agents read **5-fold out-of-fold predictions**, so no room
is judged by a model that saw its label.

**Architecture and training.** One-hot categoricals + numerics → Linear 128 → BatchNorm → ReLU → Dropout 0.25 →
Linear 64 → … → 12-way softmax; **49,484 parameters**. Stratified 70/15/15 split (15,684 / 3,362 / 3,362), seed 42,
cross-entropy, AdamW (lr 2e-3), batch 256, early stopping on validation macro-F1 (stopped at epoch 26, best 18).

**Results on the untouched test split (3,362 rooms).**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **SpaceFunctionNet (deployed)** | **0.906** | **0.744** |
| Logistic regression, same features | 0.890 | 0.704 |
| Majority class | 0.325 | 0.041 |
| SpaceFunctionNet on 20 buildings never seen in training | 0.714 | 0.464 |

**Why macro-F1 is the headline.** The classes are imbalanced (Building Service is a third of the rows). Macro-F1 gives
small classes such as Meeting or Teaching Lab the same weight as large ones, so it cannot be inflated by the big classes.

**F1 by class:** Building Service 1.00 · Workstation 1.00 · Residential 0.97 · Research Lab 0.88 · Office 0.85 ·
Classroom 0.74 · Support & Storage 0.74 · Special & General Use 0.72 · Teaching Lab 0.69 · Study & Library 0.58 ·
Office Service 0.45 · Meeting 0.32. Meeting and Office Service share buildings, sizes and departments with offices,
so they are the hardest.

**Ablation study** (3 seeds per variant; the design was **chosen on validation macro-F1 only**):

| Variant | Validation macro-F1 |
|---|---|
| Original design: embeddings + class weights | 0.742 |
| One-hot instead of embeddings | 0.755 |
| **One-hot, no class weighting (deployed)** | **0.761** |
| Without room-number features | 0.714 |
| Without departmental allocation | 0.635 |

Lesson: the *data* matters more than the architecture. Department allocation is worth about 0.1 macro-F1; depth and
embeddings barely matter on this tabular data.

**In use.** 938 records (4.2 %) are flagged for verification at ≥ 70 % confidence. The Space agent uses the model as
one criterion ("functional fit") and adds "verify actual use"; the Outage agent uses it to find rooms that look like
labs but are not recorded as labs (they may depend on steam or exhaust). Rooms added later by the records pipeline get
predictions from the deployed model; retrain when many records change.

**Trade-off to remember.** The deployed configuration is better on the buildings we have, but generalizes worse to
unseen buildings (0.464 vs 0.515 for the original design).

## 6. The two agents, step by step

**Space Administrator Agent.** Resolve the department, buildings and (for "near Professor X") the person's office →
find rooms with a compatible recorded class → hard constraints (minimum area, recorded capacity) → weighted score
`S = Σ wᵢ·sᵢ` (functional 0.30, area 0.20, occupancy 0.20, department 0.15, location 0.10, planning 0.05; with a
reference person, proximity 0.35 replaces location) → status (*Potentially suitable / Conditional / Incompatible /
Insufficient information*) → verification questions and required approvals. Labs are always conditional, because
ventilation, utilities and safety data are not on record.

**Outage Agent.** Split the thread into messages and cut quoted history → pick the current incident → rebuild the
timeline (planned, revised, confirmed, preliminary, restoration) → affected buildings and systems from the notice
*contents* → impact register (confirmed, potential, explicitly unaffected, unknown) → departments, contacts and affected
people → conflicts with the events calendar and blocked times, with alternative windows → drafts marked
"DRAFT — NOT SENT".

**Cooperation.** The Outage Agent may send exactly one kind of message to the coordinator, `temporary_space_request`
(on request, for outages of 4 h or more on lab or occupancy-limiting systems). The coordinator runs the Space agent
with the affected buildings excluded. Any other message is rejected.

## 7. The two LangGraph agents

**Voice conversation** (`app/voice/conversation.py`). Graph: *understand → resolve → ask | search → respond*, plus
*save* ("save it as a case") and *refuse* ("approve it" stays with James). Speech is detected by Silero, transcribed by
Parakeet, understood by `qwen3:8b` (JSON schema output), answered through the **unchanged** Space agent, and spoken by
Kokoro, sentence by sentence. Talking over the assistant interrupts it; muting the mic gives text-only replies.
Typical latency: first agent step about 1 s, spoken answer 1–3 s.

**Email triage** (`app/integrations/triage.py`). Graph: *rules → model → decide → case | ignore | review*. Rules settle
TRIRIGA notices, replies in threads that already have a case, and James's own decisions. The model judges the rest;
only confident answers that do not contradict the keyword rule are filed automatically, everything else waits for James.

## 8. Live data

- **Outage email:** Microsoft 365 mailbox, read-only, checked every 60 s with a *delta query* (only new mail).
- **Space and people records:** the two reports are watched in SharePoint/OneDrive or a folder; each new version is
  validated, compared ("12 rooms changed, 3 people moved"), kept as a snapshot, and the agents switch to it without a
  restart. A broken export never replaces good data.
- **Browser:** new cases appear in the navigation within 30 s.

## 9. Evaluations and what they show

| Evaluation | Result | Where |
|---|---|---|
| Model test split | Accuracy 0.906, macro-F1 0.744 (logistic regression 0.704) | `models/metrics.json`, notebook 01 |
| Unseen buildings | Macro-F1 0.464 | `models/metrics.json` |
| Ablation | Department data +0.13 macro-F1; one-hot beats embeddings | `models/ablation.json` |
| Agent scenarios | **67/67** checks across 12 scenarios built from the real materials | `data/evaluation.json`, Agents page |
| Cooperation | Temporary-space options for **84 %** of 19 long outages (0 % alone); 100 % respect constraints; ≈180 ms added | `data/experiments.json` |
| Competition: finding records to verify | Rules F1 0.53 · model F1 0.91 · **combined F1 0.94** (1,000 injected errors) | `data/experiments.json` |
| Email triage | Keyword rule F1 0.954, 29 outages missed · model alone 0.932, 41 missed · **agent 0.981, 10 missed**, 7.7 % held for James | `data/triage_eval.json`, Email dialog |
| Software | **105 tests** pass | `pytest -q` |

How to read them: the scenario checks are written from James's examples, not from the agents' output. The
competition uses injected errors, which are easier than real record drift, so compare the strategies, not the absolute
numbers. The triage labels are provisional (written by the team, to be confirmed by James).

## 10. Limitations to state honestly

- No approver roster, class schedule or maintenance calendar in the data; unknown events are reported as coverage gaps.
- Floor plans are images: proximity is estimated from room numbering, not measured walking distance.
- The model is weak on Meeting and Office Service rooms and on unseen buildings.
- Ranking weights are engineering defaults that James has not validated.
- The Microsoft 365 connection is built and tested against a simulated Graph service; it needs Rice IT's app
  registration before it runs against real mail. Push notifications and a direct TRIRIGA connection are future work.

## 11. Glossary

| Term | Meaning |
|---|---|
| Case | One analysed request or outage, with revisions, reviews and an audit trail |
| Coverage | How much of the ranking evidence is actually on record |
| Out-of-fold prediction | A prediction for a record made by a model trained without that record |
| Macro-F1 | The average F1 over classes, each class weighted equally |
| Delta query | Microsoft Graph request that returns only what changed since the last call |
| Locked action | An institutional action no agent can perform (send, write TRIRIGA, assign, approve) |
| LangGraph | A library for building agents as graphs of steps with shared state |

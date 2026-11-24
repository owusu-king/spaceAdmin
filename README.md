# Campus Facilities AI — Space Administration & Outage Intelligence

A working decision-support application for Rice University facilities administration (Natural Sciences group), built for our
Deep Learning course. Two intelligent agents, a trained neural network and a professional web interface help a facilities
administrator (James) **find and assess space** and **coordinate infrastructure outages**, using the real records and
correspondence he supplied. The agents prepare analyses and drafts; a human always reviews and approves.

**Start here:** [docs/SETUP.md](docs/SETUP.md) (install and connect to Rice) · [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)
(how it works, metrics and evaluation) · [docs/Campus_Facilities_AI_Presentation.pptx](docs/Campus_Facilities_AI_Presentation.pptx) (15 slides).

---

## 1. Project Overview

**The operational problem** (from `datares/Space and Outage Agents_ Proposed Scope and Workflows.docx`):

* *Space:* a new faculty member needs an office or lab, a department needs space for a function, someone needs an assignment
  change, leadership reviews vacancies. Today this means combining space and people records by hand, reconciling them with actual
  use, and remembering that a "vacant" record is not permission to assign.
* *Outages:* a TRIRIGA notice or an email arrives. James must work out what is really affected, which Natural Sciences
  departments and contacts are involved, whether the timing conflicts with activity, and draft follow-ups, while separating
  preliminary concerns from confirmed scope and keeping stated exclusions (e.g. "fume hoods will not be affected").

**Objectives**

1. Automate the repetitive review work: record joins, impact assessment, stakeholder lookup, conflict and coverage checks, drafting.
2. Make certainty visible: every finding is classified Confirmed / Recorded / Inference / Historical / Conflicting / Unknown.
3. Keep authority with people: the agents never send, book, assign or approve anything. A person may send an approved
   draft from the app only after explicitly switching sending on in **Settings** (off by default, audited).
4. Demonstrate a genuine deep learning component that the agents actually use.

---

## 2. System Architecture

A lightweight **modular monolith**: one Python backend, a static web client, SQLite for cases, files for data and model artifacts.

```
 Browser (web/)  ──REST /api/v1──►  FastAPI (app/api.py)
                                        │
                     ┌──────────────────┼────────────────────────┐
                     ▼                  ▼                        ▼
             Coordinator  ◄──messages──►  agents          CaseStore (SQLite)
          (app/agents/coordinator.py)                     cases · revisions · reviews · audit
              │                 │
   Space Administrator Agent   Outage Agent
              │                 │   uses services: scheduling, stakeholders
              └──────┬──────────┘
                     ▼
     Data layer (app/data)            Intelligence (app/ml)
     space & people exports,          SpaceFunctionNet (PyTorch) + out-of-fold predictions
     building crosswalk, floor plans,
     RTF correspondence → outage history
```

**How components interact:** the API receives a request, the coordinator routes it to the responsible agent, the agent calls its
permitted tools (data queries, model inference, scheduling rules), returns a structured report, and the CaseStore saves it as
Revision A for review. Re-running creates Revision B and invalidates earlier approvals.

**Deliberate deviations from the architecture document** (documented, not accidental): SQLite + in-memory pandas instead of
PostgreSQL/pgvector, plain Python workflows for the two agents (LangGraph is used only for the voice conversation, §8a),
vanilla JS instead of React, and **no external or cloud LLM**: the only language model is a local one (Ollama) used to talk with
James, never to make decisions. These keep installation simple, keep personnel data on the machine, and are enough for the
data volume (22k records).
Module boundaries match the documented ones so each can be swapped later.

---

## 3. The Two Agents

Both inherit `Agent` (`app/agents/base.py`): a tool registry, an explicit **permission set**, a **40-call execution limit**, timing,
structured logging, and a trace of every tool call. Actions that would change institutional state (`send_email`,
`write_tririga`, `assign_space`, `approve_outage`) are **locked** and always refused for agents, whatever the Settings
switches say (see §8, *Settings*).

### Space Administrator Agent (`app/agents/space_agent.py`)

| | |
|---|---|
| Responsibilities | New office, **office close to a person** ("a room for Professor X's new hire"), department function, assignment change, leadership vacancy review, temporary relocation (on request) |
| Tools (13) | `resolve_building`, `get_department`, `get_person`, `find_person`, `measure_proximity`, `search_spaces`, `get_space_details`, `get_space_occupancy`, `check_planning_reservations`, `get_floor_plan`, `model_space_assessment`, `evaluate_space_constraints`, `rank_candidate_spaces` |
| Proximity | Given a person (name or NetID), the agent finds their recorded office and ranks candidates by closeness: same floor and nearby room numbers first (wing letters respected, e.g. `W300` vs `E300`), then same building by floor difference, then other buildings. With a reference office, proximity replaces the generic location criterion and carries weight 0.35. If the name matches several people it **asks which one before ranking** ("Professor" prefers the faculty match). Distance is *estimated from room numbering*; floor plans are images, so walking distance is not measured. |
| Decision logic | Hard constraints first (minimum area, recorded capacity), then weighted criteria `S = Σ wᵢsᵢ` (functional 0.30, area 0.20, occupancy 0.20, department 0.15, location 0.10, planning 0.05; with a reference person, proximity 0.35 instead of location, weights renormalized). **Suitability and evidence coverage are reported separately**; coverage < 60 % makes a score *Provisional*. Status: Potentially suitable / Conditional / Incompatible / Insufficient information |
| Boundaries | Never books or assigns. Recorded vacancy is a note, never permission. Labs are always conditional because ventilation, utilities and safety data are not on record. |
| Output | Requirements (stated vs inferred), ranked candidates, alternatives, verification questions, required approvals, implications, evidence |

### Outage Agent (`app/agents/outage_agent.py`)

| | |
|---|---|
| Responsibilities | Interpret TRIRIGA notices or ordinary emails (pasted, chosen, or arriving through email intake), isolate the current incident in long threads, reconstruct the timeline, assess impact, **find the affected people** and Natural Sciences departments and contacts, review schedule conflicts and coverage, forecast recurring maintenance, **prepare the messages**: a coordination email to department contacts and a notice to the affected people |
| Tools (13) | `parse_outage_notice`, `isolate_incident`, `reconstruct_event_timeline`, `find_related_outages`, `resolve_outage_scope`, `assess_outage_impact`, `find_affected_departments`, `find_affected_people`, `get_stakeholders`, `check_schedule_conflicts`, `forecast_recurring_maintenance`, `request_space_support`, `draft_communication` |
| Schedule conflicts | The proposed window (from TRIRIGA, or a date written in prose such as "Wednesday, November 25th", checked as a whole day when no time is given) is checked against the **events calendar** and other outages in the same buildings. The calendar holds dated events found in the correspondence (closures, drills, holidays, recess dates mentioned in emails), imported `.ics` calendars (e.g. the Rice events calendar), events added by the team, and **times James has blocked**. A conflict gives a red verdict, the plan says "propose another time", and the agent suggests conflict-free windows (early weekday if it can finish before 08:00, otherwise Saturdays). James can check any other proposed time in the case and block times that cannot work; later checks flag them. Example: the BRK elevator work proposed for Nov 25 is flagged against the Thanksgiving recess that Jose's email mentions. |
| Affected people | From the people records, everyone recorded in the affected spaces, each with a reason. Lab-only systems (exhaust, steam, compressed air, …) → occupants of recorded labs and of rooms the model infers are lab-like. Building-wide systems (power, AC, elevator, water) → all recorded occupants of the building. Emails are NetID@rice.edu. The notice draft goes to the Natural Sciences people; the full list (all departments) exports as CSV. People without a recorded location cannot be found from the records. |
| Decision logic | Building from notice *contents* (not filename); quoted replies are not new events; planned / revised / confirmed timing and estimated vs confirmed restoration are distinct; impact register = Confirmed, Potential, Explicitly unaffected, Unknown, Preliminary (marked "later confirmed" or "not in confirmed scope"); three stakeholder roles verified separately (occupants are never approvers); missing schedules are **coverage gaps, never "no conflicts"**; weekends never imply availability |
| Boundaries | Drafts carry "DRAFT — NOT SENT"; the agent cannot send, approve or edit TRIRIGA. Only James can send an approved draft, when *Send email* is on in Settings |

### How they communicate

The agents cannot call each other's tools. The Outage Agent may send one permitted intent, `temporary_space_request`, to the
coordinator. James rarely needs temporary space, so this happens **only on request** (*Temporary space* in an
outage case's Schedule section, or `cooperate=True`), and only for outages of ≥ 4 h on lab or occupancy-limiting systems. The coordinator validates
it, runs the Space Administrator Agent with the affected buildings excluded, and returns `temporary_space_options`. Any other
intent is rejected. All messages carry the run id and appear in the case's *Evidence* section.

---

## 4. Deep Learning Implementation

| | |
|---|---|
| Problem | Infer a room's **functional category** (12 classes) from its profile, to flag records whose recorded Space Class may not reflect actual use, and to find lab-like rooms recorded as non-labs |
| Why this task | James: Space Class "can reflect the original intended use, so actual current use still needs verification". The 22,408 labelled space records support honest supervised learning; the 40 outage requests in the emails do not (outage recurrence stays a clearly labelled statistical estimate) |
| Data | `SpaceRecords.xlsx` joined with `Report_PeopleRecords.xlsx` (4,168 located people, 100 % matched to rooms) |
| Features | Categoricals: building, floor, primary allocation, room-number pattern, room suffix. Numerics: log area, missing-area flag, allocation count, room number mod 100, recorded occupants by role (faculty, postdoc, grad, undergrad, staff, visitor) |
| Leakage control | Seat Stat, SSA, Avble Seats, W/S Area and spaceStandard are excluded (derived from the label). Encoders fit on the training split only. Agents read **5-fold out-of-fold** predictions, so no record is judged by a model that saw its label |
| Architecture | `SpaceFunctionNet` multilayer perceptron: one-hot categoricals + numerics → Linear 128 → BatchNorm → ReLU → Dropout 0.25 → Linear 64 → … → 12-way softmax · 49,484 parameters. The code also supports learned entity embeddings; the ablation study chose one-hot (see below). |
| Training | Stratified 70/15/15 split, seed 42 · cross-entropy · AdamW (lr 2e-3, wd 1e-4) · batch 256 · ReduceLROnPlateau · early stopping on validation macro-F1 (stopped at epoch 26, best 18) |

**Measured results** (test split, 3,362 records; `models/metrics.json`):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **SpaceFunctionNet (deployed)** | **0.906** | **0.744** |
| Original design (embeddings + class weights) | 0.894 | 0.728 |
| Logistic regression (same features, one-hot) | 0.890 | 0.704 |
| Majority class | 0.325 | 0.041 |
| SpaceFunctionNet on 20 **unseen buildings** | 0.714 | 0.464 |

Strong classes: Building Service 1.00, Workstation 1.00, Residential 0.97, Research Lab 0.88, Office 0.85 F1. Weak: Meeting 0.32,
Office Service 0.45 (they share buildings, sizes and departments with offices). 938 records (4.2 %) are flagged for verification
at ≥ 70 % confidence. **Trade-off:** the selected configuration is better on the buildings we have (the agents' use case) but
generalizes worse to buildings never seen in training (0.464 vs 0.515 for the original design).

**Ablation study** (`python -m app.ml.ablation`, results in `models/ablation.json`, also on the Model page and in notebook 01).
Each variant changes one design choice and is trained with 3 seeds on the same split. The configuration is **chosen on validation
macro-F1 only**; test scores are reported, never used to choose.

| Variant | Validation macro-F1 | Test macro-F1 |
|---|---|---|
| Original design: embeddings + class weights | 0.742 ± 0.004 | 0.730 ± 0.005 |
| One-hot instead of embeddings | 0.755 ± 0.002 | 0.749 ± 0.007 |
| Without class weighting | 0.755 ± 0.003 | 0.743 ± 0.006 |
| **One-hot and no class weighting (selected, deployed)** | **0.761 ± 0.004** | 0.747 ± 0.004 |
| One hidden layer (64) | 0.745 ± 0.003 | 0.738 ± 0.005 |
| Three hidden layers (256-128-64) | 0.746 ± 0.001 | 0.734 ± 0.008 |
| Without occupant features | 0.741 ± 0.002 | 0.714 ± 0.005 |
| Without room-number features | 0.714 ± 0.006 | 0.706 ± 0.005 |
| Without departmental allocation | 0.635 ± 0.002 | 0.639 ± 0.008 |

What it shows: the *data* matters more than the architecture (departmental allocation is worth about 0.1 macro-F1, room-number
features about 0.03; depth barely matters). Learned embeddings did not help on this tabular data, and class weighting hurt
slightly, so the deployed model changed accordingly. (The single deployed training run scores 0.744; the 3-seed mean is 0.747.)

**How the application uses it:** `SpacePredictor` (`app/ml/inference.py`) loads the artifacts at start-up. The Space agent's
`model_space_assessment` tool feeds the functional-fit criterion and adds a "verify actual use" condition; the Outage agent adds
model-inferred lab-like rooms as *Potential impacts (Inference)* for exhaust, steam, air and similar outages. Model outputs are
always labelled as predictions and never make a decision alone.

---

## 5. Project Structure

```
FacilitiesAI/
├── app/
│   ├── config.py            Paths, building crosswalk, Natural Sciences departments, contacts, hyperparameters, thresholds
│   ├── logging_setup.py     Structured logging with run ids and durations
│   ├── data/
│   │   ├── loader.py        Load + validate + normalize space and people exports; quality report; cache
│   │   ├── buildings.py     Building resolver (codes, names, aliases, ambiguity) and floor-plan index
│   │   └── notices.py       RTF → messages (quoted history handled) → TRIRIGA fields → deduplicated outage history
│   ├── ml/
│   │   ├── features.py      Feature engineering and train-only encoder
│   │   ├── model.py         SpaceFunctionNet (PyTorch)
│   │   ├── train.py         Training/evaluation pipeline, baselines, building hold-out, cross-fitting  (python -m app.ml.train)
│   │   ├── ablation.py      Ablation study, 3 seeds, validation-based selection  (python -m app.ml.ablation)
│   │   └── inference.py     SpacePredictor used by agents and API
│   ├── agents/
│   │   ├── base.py          Agent, tools, permissions, evidence, locked actions, execution limit
│   │   ├── schemas.py       Validated request models (pydantic)
│   │   ├── space_agent.py   Space Administrator Agent
│   │   ├── outage_agent.py  Outage Agent
│   │   └── coordinator.py   System construction, routing, agent-to-agent relay
│   ├── services/
│   │   ├── scheduling.py    Conflicts, coverage gaps, alternative windows, recurrence estimate
│   │   ├── events.py        Events calendar: correspondence events, .ics import, manual and blocked times; prose dates
│   │   ├── stakeholders.py  Approver / coordination contact / notification recipient resolution
│   │   ├── cases.py         SQLite case store: revisions, reviews, audit log
│   │   ├── settings.py      Settings switches: off by default, two keys to unlock, auto-off timer, audited
│   │   └── outbox.py        Sending an approved draft: guards, recipients, test-address redirect, sent log
│   ├── scenarios.py         Scenario library from the real materials + expectations
│   ├── evaluation.py        Scenario and safety evaluation
│   ├── experiments.py       Cooperation and competition experiments  (python -m app.experiments)
│   ├── integrations/email.py  Email intake: replay or read-only IMAP, threading, automatic cases
│   ├── integrations/triage.py Email triage agent (LangGraph): rules, local model, review by James
│   ├── triage_eval.py       Triage evaluation: keyword rule vs model vs agent  (python -m app.triage_eval)
│   ├── integrations/mailer.py Outgoing email over SMTP (used only by the send endpoint, never by agents)
│   ├── voice/               Local voice conversation for space requests (§8a)
│   │   ├── engines.py       Installed speech models: Parakeet speech-to-text, Kokoro voice, Silero speech detection
│   │   ├── llm.py           Local Ollama client: installed models, schema-bound JSON, streamed replies
│   │   ├── conversation.py  LangGraph graph: understand → resolve → ask / search → respond
│   │   ├── session.py       Live WebSocket session: listening, transcription, speech, interruption
│   │   └── service.py       Model choices (installed only), readiness, preloading
│   └── api.py               FastAPI app + static web client
├── web/                     index.html, css/app.css (design system), js/ (router, lib, one module per view)
├── models/                  space_function_net.pt, encoder.json, metrics.json, oof_predictions.csv, ablation.json
├── data/                    evaluation.json, experiments.json (+ generated cache, facilities.db, events.db)
├── notebooks/               01_deep_learning.ipynb, 02_intelligent_agents.ipynb, build_notebooks.py
├── tests/                   pytest suite (105 tests)
├── datares/                Original project materials (read-only, never modified, not in Git)
├── docs/                    SETUP.md, HOW-IT-WORKS.md, the presentation, and the original design documents
├── .env.example             Every configuration variable, documented (copy to .env, which is git-ignored)
└── requirements.txt
```

---

## 6. Installation and Execution

**Requirements:** Windows, macOS or Linux · Python 3.12 · ~2 GB free disk (PyTorch) · a modern browser.
On this machine Python 3.12 was installed with `winget install Python.Python.3.12`.

From the project folder (PowerShell shown; on macOS/Linux use `.venv/bin/python`):

```powershell
python -m venv .venv
```
```powershell
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```
```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

**Model artifacts are included** in `models/`. To retrain from scratch (about 1.5 minutes on a laptop CPU, including cross-fitting):

```powershell
.venv\Scripts\python -m app.ml.train
```

**Start the application:**

```powershell
.venv\Scripts\python -m uvicorn app.api:app --port 8000
```

Then open **http://localhost:8000**. The console shows structured logs (data load, model load, agent initialization, every
workflow, tool call, message and evaluation with run ids and durations).

**Configuration and secrets (`.env`).** All deployment variables and secrets live in a `.env` file in the project folder,
loaded once at start-up by `app/config.py`. `.env.example` lists and explains every variable; create your own with:

```powershell
Copy-Item .env.example .env
```

`.env` is git-ignored: never commit it or put credentials in code. A variable set in the terminal overrides the file.
Restart the server after editing `.env`.

**Email intake (Outage Agent).** By default (`FAI_EMAIL_SOURCE=replay`) the app *replays* the supplied correspondence as if it
were arriving (Outages → Email intake → *Check now*). To read a real mailbox, set in `.env`:
`FAI_EMAIL_SOURCE=imap`, `FAI_IMAP_HOST`, `FAI_IMAP_USER`, `FAI_IMAP_PASSWORD` (an app password).
Optional: `FAI_IMAP_FOLDER` (default `INBOX`), `FAI_IMAP_SINCE_DAYS` (default 14), `FAI_EMAIL_POLL_SECONDS` (e.g. `300` for
automatic checks; default 0 = manual), `FAI_REPLAY_SINCE` (replay start date, default `2026-10-01`). The adapter is **read-only by construction**: it opens the folder with `readonly=True` and fetches with
`BODY.PEEK[]`, so messages are not marked read, moved or deleted, and it has no send capability. Outage-related messages
open a case; replies on the same thread (same SR number or subject) add a new revision; an informal heads-up and the later
TRIRIGA notice for the same building and system are merged into one incident. For non-outage mail only sender, subject and
date are stored.

> **Before connecting James's mailbox:** get James's consent and Rice IT approval. The least-privilege route is a dedicated
> project mailbox that James forwards outage mail to (or an auto-forward rule for TRIRIGA/facilities senders), not his main
> inbox. If Rice requires Microsoft 365 OAuth instead of IMAP passwords, a Graph adapter with `Mail.Read` can implement the
> same `health_check` / `fetch` interface in `app/integrations/email.py`. The live IMAP path is tested against a simulated
> server only; it has not been run against a real Rice mailbox.

**Run the notebooks** (in order):

```powershell
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_deep_learning.ipynb notebooks/02_intelligent_agents.ipynb
```

or open them in VS Code / Jupyter with the `.venv` interpreter as kernel.

---

## 7. Understanding the Two Notebooks

| | `01_deep_learning.ipynb` (17 cells) | `02_intelligent_agents.ipynb` (17 cells) |
|---|---|---|
| Purpose | How the model is built, trained, evaluated and integrated | How the agents use data, model and tools to make operational decisions |
| Demonstrates | Dataset summary, leakage-safe preprocessing, architecture, training curves, test metrics vs baselines, confusion matrix, example predictions and disagreements, **ablation study and the validation-based choice**, reload of production artifacts | Agent initialization, a model tool call, an office near a professor (proximity), the real ABL steam thread (timeline, impact register, preliminary vs confirmed), the BRK elevator date conflicting with the Thanksgiving recess and a time blocked by James, agent-to-agent messages, affected people and both drafts, locked send, evaluation, cooperation and competition |
| Key outputs | 0.906 accuracy / 0.744 macro-F1; ablation table; "notebook model reproduces deployed predictions: True" | 67/67 scenario checks; conflict verdict and alternatives; cooperation 84 % vs 0 %; competition F1 0.53 / 0.91 / 0.94 |
| Connection to app | Uses `app.data`, `app.ml` exactly as `python -m app.ml.train` does | Imports `build_system` and runs the same agents the API serves |

Run 01 first, then 02. Notebook 02 does not retrain; it loads the saved model. `notebooks/build_notebooks.py` regenerates both
notebooks from source if the code changes.

---

## 8. Application Usage

The interface follows one rule on every page: **the few things that need James come first; everything else is one click
away in a dialog.** Dialogs keep the page behind them still, and page width never shifts when content changes.

| View | What it is for |
|---|---|
| **Overview** | Greeting and date; four linked figures (cases awaiting review, outages in the next 45 days, recorded Natural Sciences vacancies, records to verify); **Needs your review** (top 3, most recently updated first) beside a **Next 30 days** agenda of outages, calendar events and blocked times; a Natural Sciences seating bar (over / at / under capacity, vacant) and the model's most common disagreements. Everything links to its page; nothing opens a dialog |
| **Spaces** | Tabs *All · Vacancies · Model flags*, one search box (room, standard or department), building and function filters. Click a row for the space record dialog: record, model view, occupants and the floor-plan PDF |
| **Outages** | Three facts that open dialogs: **Calendar** (events and blocked times; add, block, import `.ics`), **Email** (intake status, check for new email) and **Recurring** (estimates and their measured accuracy). Below, *Upcoming · History* tabs; a row opens the request, with "Analyze" |
| **Cases** | *To review · Decided · All* tabs. A case page has three levels: (1) header with case no., revision, status, Re-run, **Return / Decline** and Approve, plus *Timeline* and *History* links; key facts; **Needs attention** (conflicts, top questions). (2) **One main card**: for an outage, *Notify affected people* (count, View people, **Review drafts**); for a space request, the ranked candidate table (a row opens the candidate). (3) **Case details** tiles, each opening that section alone: outage — Notice, Schedule, Impact, Departments, Questions, Evidence (and Temporary space when requested); space — Requirements, Questions, More candidates, Evidence |
| **Model** | Three key figures (macro-F1, accuracy, unseen buildings), then charts on the page: loss, validation macro-F1, against baselines, F1 by category, confusion matrix. Top-right buttons open Design, Records to verify and the Ablation study |
| **Agents** | Four figures (evaluation, cooperation, competition, activity), the two agent profiles, and dialogs for Evaluation (run again), Experiments (run again), Communication and the Activity log |
| **Settings** | Switches for actions the agents never take: *Send email*, *Deliver to real recipients*, and (greyed out until an integration exists) *Write to TRIRIGA*, *Assign space*, *Approve outages*. All off by default. A switch can be turned on only when `.env` has `FAI_ALLOW_UNLOCK=true` **and** its integration is configured; it can stay on until turned off or for 15 min / 1 h / 8 h. Turning *Send email* off also turns live delivery off. Every change is in the activity log |

Case titles follow one pattern, *what · where*: "Elevator · BRK, DBH, HRZ +1", "New office · BioSciences". The source (SR
number, Email or Space request) is shown under the title.

**Workflows**

* *Space request:* **New space request** (Overview, Spaces or Cases) opens on **Conversation**: say or type what is
  needed ("a room for Professor Bartel's new hire, at least 120 square feet"); the agent asks for anything missing,
  answers aloud with the best candidates, and **Create case** saves the result for review. The **Form** tab is the
  alternative (it is pre-filled with what was said): type, department, function, area; for a new hire enter the
  professor in **Near person** → **Run analysis**. The candidate table shows the distance to the reference office;
  click a candidate for its criteria, conditions, unknowns, the model's view and the floor plan.
* *Outage review:* **Review outage notice** (Outages) or **Outage notice** (Cases) → pick a notice on record, a whole thread, or paste an email →
  **Run analysis** (or let email intake open the case). On the case, **Review drafts** gives the coordination email and the
  notice to affected people (Copy; **Send…** only when *Send email* is on and the case is approved: each draft is
  sent once per revision after a confirmation dialog, to the test address unless *Deliver to real recipients* is on;
  the people notice goes as BCC); **View people** lists who is affected, with CSV export.
* *"Not at that time":* **Reschedule** in Needs attention (or Case details → Schedule) → pick a suggested time or enter the
  time Facilities or a department proposes → **Check**, or **Block…** with a reason. Blocked times are flagged in every
  later check. *Temporary space* asks the Space agent, on request only.
* *Delete:* the trash icon on a case removes it (after a confirmation), with its revisions and reviews; the activity
  log keeps a record and the case number is never reused.
* *Review:* **Approve Rev X**, or **Return / Decline** with a reason. **Re-run** creates the next revision and marks
  earlier approvals invalidated; *History* shows every decision.

### 8a. Voice conversation (local models only)

The main way to make a space request is a conversation with the Space Administrator Agent, inside the *New space request*
dialog. James talks (or types); the assistant answers aloud; talking while it speaks interrupts it. Everything runs on this
computer: no audio or text is sent anywhere, and no audio is stored (only the text of each turn appears in the server log).

| Part | Model (installed locally) | Where it runs |
|---|---|---|
| Speech detection, interruption | Silero VAD | CPU, server |
| Speech to text | NVIDIA Parakeet TDT 0.6B v2 (English, int8) via `onnx-asr` | CPU, server |
| Understanding and replies | `qwen3:8b` via Ollama (any installed model with tool calling can be chosen) | GPU |
| Voice | Kokoro-82M via `kokoro-onnx` (54 voices; English ones offered) | CPU, server |

**How a turn works** (`app/voice/conversation.py`, a LangGraph graph with per-conversation memory):
1. *understand*: the LLM turns the sentence into request fields with a fixed JSON schema. Small models fill fields eagerly,
   so a number, building, department or person is kept **only if it was actually said** (`_grounded`); a rule also catches
   "Professor X" / "near X" if the model misses it.
2. *resolve*: checks against the records, without the LLM. Speech-to-text misspells names ("Bartle"), so names are
   matched by spelling and sound (exact first, faculty preferred after "Professor"); a correction is said aloud
   ("I took Bartle to mean Bonnie Bartel"). Unclear people, departments or buildings (e.g. "Brockman") lead to a question.
3. *ask* a short question, or *search*: the **existing Space Administrator Agent runs unchanged** (same rules, evidence,
   limits).
4. *respond*: the LLM summarizes the agent's report in 2-3 spoken sentences using only the report; sentences are spoken as
   soon as each is ready.

Follow-ups refine the same request ("make it at least 150 square feet", "in Keck instead"). **Create case** saves the latest
agent result as a normal space case (Rev A, ready for review); nothing is assigned. The **Form** tab takes over what was said.

**What James can say, and what he sees while the agent works:**
* *"Do you have any available space in Anderson Hall?"* runs the agent's vacancy review for that building (rules back up the
  model: a building after in/at/on, and words like available/vacant, always lead to a search, never to small talk).
* While working, the dialog shows the process live: *Understood: available space · Anderson Hall*, a spoken lead-in
  (*"Checking recorded vacancies in Anderson Hall."*), then each agent step as it completes (*Matched building*,
  *Checked actual use with the model*, *Ranked 8 rooms*…). The spoken answer says briefly how the result was reached.
* *"Okay, save it as a case"* saves the rooms it found as a case, ready for review (once per result; the footer turns
  into *Open SA-…*). *"Approve it"* is declined: approval stays a click on the case page.

**Listening, muting and noise.** The microphone listens hands-free from the moment the conversation opens; talking over
the assistant interrupts it. The mic icon in the message field mutes: while muted (or with no microphone) the assistant
**answers in text only**. Background sound: the speech detector (Silero) scores speech rather than loudness, so most
noise never starts an utterance; a cough or a door cannot interrupt the assistant (that needs about a quarter second of
actual speech, and your words are still captured from the start); and transcripts with no words in them ("uh", "hmm")
are dropped. Short answers such as "yes" or "no" always get a reply; "yes" after the offer to save saves the case.
With the voice assistant switched off, *New space request* shows only the form.

**Start-up time.** The models are warmed in the background when the app is opened (and when the switch is turned on);
the language model stays loaded for two hours. Every request uses the same model options, because a different context
size makes Ollama reload the model. Measured cold start (model not in memory, dialog opened immediately): conversation
ready in 2 s, first agent step at about 3 s, full answer at about 5 s.

**Settings → Voice assistant**: one switch (off by default; it does not need `FAI_ALLOW_UNLOCK` because nothing leaves the
machine) and three pickers that list **only models installed on this computer**: language model (from Ollama; models without
tool calling are shown but cannot be chosen), speech-to-text model (folders in `models/voice/stt`) and voice. When the switch is
on, the models are preloaded at start-up so the dialog is ready in about 2 seconds. With the switch off, the dialog opens on
the form.

**Setup** (once; about 1 GB plus the 5.2 GB language model):

```powershell
ollama pull qwen3:8b
```
```powershell
.venv\Scripts\python -c "import onnx_asr; onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2', 'models/voice/stt/parakeet-tdt-0.6b-v2', quantization='int8'); onnx_asr.load_vad('silero', 'models/voice/vad/silero')"
```
```powershell
Invoke-WebRequest https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -OutFile models/voice/tts/kokoro-v1.0.onnx
```
```powershell
Invoke-WebRequest https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -OutFile models/voice/tts/voices-v1.0.bin
```

`models/voice/` is git-ignored. `.env` sets `FAI_OLLAMA_URL` and turns LangSmith tracing off (`LANGSMITH_TRACING=false`).

**Measured on the development laptop (RTX 4070 laptop GPU, 8 GB):** Parakeet transcribes a 7-second sentence in about
0.25 s; the first spoken sentence of a reply starts about 1-4 s after James stops talking (longer when the agent searches);
Kokoro needs about 1.6 s per 7 s of speech on the CPU, so replies are spoken sentence by sentence.

### 8b. Email triage agent (LangGraph)

Every incoming email passes through a small LangGraph agent (`app/integrations/triage.py`) that decides whether it is
about a building outage before anything else happens:

```
rules ─┬─ decided ─────────────────────────────┐
       └─ model ─► decide ─┬─ outage ─► case   │  Outage Agent opens or updates the case, as before
                           ├─ not ────► ignore │  only sender, subject and date are kept
                           └─ unsure ─► hold   │  "Needs your review" in Outages → Email
```

* **Rules** settle the certain cases without a model: TRIRIGA notices, replies in a thread that already has a case,
  and James's own decisions.
* **Model**: the local language model chosen in Settings (Ollama) reads sender, subject and the start of the body and
  returns outage or not, a confidence and a one-line reason. Without Ollama the keyword rule is used, so intake never stops.
* **Decide** files an email automatically only when the model is confident (≥ 0.75) and does not contradict the keyword
  rule; everything else is **held for James**. A real outage wrongly ignored is the costly mistake, so disagreement always
  goes to a person.
* **Review**: held emails appear first in *Outages → Email* with the agent's reason and two buttons, *Outage* / *Not an
  outage*. James's choice runs the same graph again (his label is accepted by the rules node) and files the email: an
  outage opens or updates a case; anything else is ignored and its text deleted. Every row shows who decided
  (Rule, Model with confidence, Keywords, You).

**Evaluation** (`python -m app.triage_eval`, results in `data/triage_eval.json`, summary in the Email dialog).
379 labelled emails: 361 from the supplied correspondence (labelled per thread in
`data/triage_labels.json`; 13 borderline threads left out) and 18 clearly marked synthetic ones
(ordinary inbox mail, and informal outage reports without keywords), 342 outage and
37 not. Model: qwen3:8b.

| Strategy | Precision | Recall | F1 | Outages missed | Held for James |
|---|---|---|---|---|---|
| A · Keyword rule (before) | 0.997 | 0.915 | 0.954 | 29 | – |
| B · Local model alone | 0.990 | 0.880 | 0.932 | 41 | – |
| **C · Triage agent, after James reviews** | **0.991** | **0.971** | **0.981** | **10** | 29 (8%) |
| C · on the emails it filed itself | 0.990 | 0.968 | 0.979 | – | – |

The evaluation scores each email on its own, so it cannot use the thread rule; in live intake, most of the agent's
remaining misses (replies such as "Re: Follow up" deep in a thread) are filed by that rule once the thread has a case.
A few disagreements may be label errors (e.g. a thread labelled as a vendor proposal discusses a compressed-air outage).
Caveats: the labels are provisional (written by the development assistant, to be confirmed by James), the supplied mail
comes mostly from outage threads, the synthetic set is small, and the model's confidence is its own estimate. About
2.71 s per email on the development laptop.

### 8c. Live connections

The app reads live data three ways, all **read-only**, all visible in *Settings → Connections*.

| What | How it stays current | Code |
|---|---|---|
| **Outage email** | Microsoft 365 (Graph). The chosen mailbox folder is checked every 60 s with a *delta query* (only new mail is fetched); each email goes through the triage agent; outages open or update cases, and the navigation shows "N new cases to review" within 30 s | `app/integrations/msgraph.py` |
| **Space and people records** | The two Excel reports are watched where they already live: SharePoint/OneDrive (through the same Microsoft 365 connection) or a folder that a scheduled export or OneDrive sync fills. When a file changes it is validated, compared (rooms added / removed / changed, people added / removed / moved), kept as a dated snapshot, and the agents switch to it without a restart. A broken export never replaces good data | `app/data/records.py` |
| **Case list** | The browser checks for new cases every 30 s | `web/js/app.js` |

**Connecting Microsoft 365 (one-time, done with Rice IT).**
1. Rice IT registers an app (Azure portal → *App registrations* → *New*), single tenant, and turns on *Allow public client flows*.
2. API permissions, Microsoft Graph, **delegated**: `Mail.Read`, `Mail.Read.Shared` (for a shared mailbox), `Files.Read.All`, `offline_access`, `User.Read`. Admin consent if Rice requires it.
   For an unattended service instead: **application** permission `Mail.Read` plus an *application access policy* limiting it to the one mailbox, and a client secret (`FAI_GRAPH_AUTH=app`).
3. In `.env`: `FAI_GRAPH_TENANT_ID`, `FAI_GRAPH_CLIENT_ID`, `FAI_GRAPH_MAILBOX` (`me` or e.g. a shared `facilities-outages@rice.edu`), `FAI_GRAPH_FOLDER`, and `FAI_EMAIL_SOURCE=graph`. Restart.
4. *Settings → Connections → Connect*: the app shows a code; James enters it at microsoft.com/devicelogin and signs in himself. The app never sees his password; the token is stored in `data/graph_token.json` (git-ignored) and refreshed automatically. *Disconnect* deletes it.

The reader can only make GET requests: messages are never marked read, moved, or deleted, and nothing can be sent.
Recommended set-up: a dedicated mailbox (or an Outlook rule that forwards TRIRIGA and facilities mail to one), so the
app sees only facilities mail.

**Records without moving files.** `FAI_RECORDS_SOURCE=microsoft` with the reports' SharePoint/OneDrive links in
`FAI_RECORDS_SPACE_URL` / `FAI_RECORDS_PEOPLE_URL`, or `FAI_RECORDS_SOURCE=folder` with `FAI_RECORDS_DIR` pointing to the
folder the exports land in. `FAI_RECORDS_REFRESH_MINUTES=60` checks hourly (only file metadata is read until a file
changes); *Check now* in Settings checks immediately. Rooms the model has not seen get predictions from the deployed
model; when many records change, retrain (`python -m app.ml.train`).

**Not yet (needs Rice IT):** instant push instead of 60-second checks (Graph change notifications need a public HTTPS
endpoint); reading space and people data straight from TRIRIGA (its REST/OSLC API with a read-only service account)
instead of its exported reports; and the people directory from Rice's identity system.

---

## 9. Cooperation and Competition

Both run from **Agents → Experiments → Run experiments** or `python -m app.experiments` (results saved to `data/experiments.json`).

**Cooperation (shared problem: long outages make space unusable).** All 19 recorded outages of ≥ 4 h on lab or
occupancy-limiting systems are run by the Outage Agent alone and with Space agent support. In daily use James rarely needs
temporary space, so the app only asks for it on request; the experiment measures the capability.

| Metric | Alone | Cooperative |
|---|---|---|
| Outages with temporary-space options | 0 % | **84 %** |
| Options outside affected buildings and not incompatible | – | **100 %** |
| Mean suitability / evidence coverage of options | – | 0.94 / 0.83 |
| Added latency per case | – | about 180–420 ms (varies by run) |

**Competition (alternative strategies for one decision: which records to send for actual-use verification).** 1,000 class errors
are injected into records scored out-of-fold and mixed with 1,000 clean records (seed 42).

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| A · Rule-based field consistency (workpoint class vs seating data) | 0.978 | 0.363 | 0.530 |
| B · Model-based (SpaceFunctionNet disagreement ≥ 70 %) | 0.956 | 0.876 | 0.914 |
| **C · Combined (A or B)** | 0.952 | **0.935** | **0.943** |

The model finds errors the rules cannot see; the rules add some the model misses. Caveat: injected errors are random and easier
than real record drift, so these are relative comparisons, not field performance.

---

## 10. Testing and Validation

```powershell
.venv\Scripts\python -m pytest -q
```

**Current status: 84 passed** (about 35 s). Coverage by file:

* `test_data.py` — normalization, identifier preservation, schema validation, building ambiguity, floor-plan index, date formats,
  TRIRIGA field extraction and exclusions, quoted-history handling, outage deduplication.
* `test_model.py` — artifacts and metrics, probability outputs, out-of-fold assessment, no label-derived features, `<UNK>`
  handling, disjoint stratified split, **bit-for-bit training reproducibility**.
* `test_agents.py` — distinct permissions, permission denial, all locked actions, execution limit, contained tool errors, space
  scenarios, outage exclusions, preliminary vs confirmed, model use, agent messaging, rejected intents, input validation.
* `test_experiments_and_api.py` — cooperation, reproducible competition, full scenario evaluation, all read endpoints, floor-plan
  path safety, complete case lifecycle (revision invalidation, rationale rules), invalid requests, locked action (403), inbox
  and recipients CSV.
* `test_email_and_people.py` — RFC 822 parsing (plain and HTML), **IMAP read-only by construction** (fake server asserts
  `readonly=True` and `BODY.PEEK[]`, never modifies flags), triage and threading, intake opens/updates cases and skips
  duplicates, no body stored for non-outage mail, affected people (lab scope vs building-wide), notice draft, one-hot model.
* `test_voice.py` — unsaid details dropped, misheard names matched and announced, missing department asked first,
  follow-ups refine the same request, unknown buildings questioned, interruption stops the reply, the voice switch needs no
  unlock key but needs installed models, only installed tool-capable models can be chosen, the WebSocket refuses when off,
  a typed turn speaks and saves a case, and (with the real local models) speech is segmented and interrupts the assistant.
* `test_triage.py` — the decide rule (confident agreement filed, disagreement or low confidence held, keyword fallback
  without a model), rules settle TRIRIGA notices, known threads and James's decisions without calling the model, intake
  files / ignores (no body kept) / holds, and James's decision files a held email once.
* `test_settings_and_send.py` — switches off by default and locked without `FAI_ALLOW_UNLOCK`, mail configuration
  required, unbuilt integrations stay locked, every send guard (switch, approval, confirmation, revision, once per revision),
  test-address redirect, live delivery depends on *Send email* and uses BCC, auto-off timer, agents still locked.
* `test_scheduling_and_proximity.py` — prose dates and durations, `.ics` import and building resolution, BRK date conflicting
  with the recess, blocked time flagged and avoided, no timing means no claim, proximity ranking, ambiguous person defers ranking, outage-forecast backtest uses only earlier outages.

**Scenario evaluation:** 67/67 checks pass across 12 scenarios (Agents → Evaluation, or notebook 02). The UI was also exercised in a browser: case
creation, return/re-run/approve/decline, every case-details section, candidate and space dialogs, schedule checks and
blocked times, tab switches without layout shift, and the model and agents pages.

---

## 11. What Our Team Needs to Understand

* **Data flow.** Excel exports → `loader.py` (validated, normalized, original identifiers kept, room keys `BUILDING · FLOOR · ROOM`)
  → joined occupant counts. RTF threads → `notices.py` → messages (quoted replies cut) → TRIRIGA fields → 40 deduplicated outage
  requests. Both feed the agents; the space table also feeds the model.
* **The model.** It answers "what does this room look like it is used for?" An MLP learns it from building, floor,
  department, room-number shape, area and who sits there. Macro-F1 is the right headline because classes are imbalanced.
  The ablation study shows *why* the design is what it is, and that choices were made on validation, not test. Cross-fitting
  gives every record an honest prediction.
* **Agent decisions.** Agents are explicit workflows over permitted tools, not chatbots. Rules encode James's governance rules
  (vacancy ≠ availability, allocation ≠ affiliation, occupant ≠ approver, missing schedule ≠ no conflict, preliminary ≠ confirmed).
  The model contributes evidence labelled *Inference*; rules and humans decide.
* **Evidence and certainty.** Every finding carries a classification and source; "Not on record" (hatched) differs from zero.
* **Human control.** Cases are revisioned; approvals bind to one revision; agents can never execute. Sending is the one
  human action that can be unlocked (Settings), and only for an approved revision.
* **Where to change things.** Thresholds, weights, departments, contacts: `app/config.py`. New tool: add a `@tool` method and
  list it in the agent's `permissions`. New view: add a module in `web/js/views/` and register it in `web/js/app.js`.

---

## 12. Limitations and Future Improvements

**Implemented and verified:** data ingestion and validation, building resolution, outage reconstruction from correspondence,
SpaceFunctionNet with evaluation and cross-fitting, both agents with tools, permissions and limits, agent messaging, case
review lifecycle with audit, recurrence estimate, proximity ranking, events calendar with conflict detection and blocked
times, prose-date reading, affected people and notice drafts, email intake (replay; IMAP against a simulated server), ablation
study, the full web application, notebooks, tests, evaluation and experiments.

**Simulated or approximate:**
* The **reviewer is fixed** as James (no authentication or roles yet).
* **Department contacts** are only the two people the emails identify (marked *candidate*); others appear as unverified
  management staff from people records. The Natural Sciences department list is our inference and needs James's confirmation.
* **Recurring-maintenance estimates** are a median-interval rule on an incomplete history (low confidence), not a trained model. A rolling-origin backtest (each past outage predicted from earlier ones only) is shown on the Outages page. The same-building-and-system rule has too few repeats to test (no series has 3+ scheduled outages); on "next scheduled outage in the building" it gives a median error of 47 days over 6 tests (33% inside the ± window) versus 72 days for one shared interval. Six tests is indicative only.
* **Ranking weights** (including proximity 0.35) are engineering defaults, not validated by James.
* **Proximity** is estimated from building, floor and room numbering, not measured on the floor plans.

**Not available / incomplete:**
* No approver roster, class schedule or maintenance calendar in the data, and the events calendar only knows what the emails
  mention plus what is imported or added: unlisted events cannot be detected, and gaps are reported as gaps.
* Floor-plan PDFs contain no extractable text; rooms are not located on plans (link only, "verify visually").
* Dates written in prose without times are checked as whole days; the hours must be confirmed.
* The live IMAP connection has not been tested against a real Rice mailbox.
* The model is weak on Meeting and Office Service and on buildings it has never seen (macro-F1 0.46 on held-out buildings).
* Outage drafts are still templates; the local LLM is used only for the space-request conversation.
* Voice: English only; the browser needs microphone permission (works on `localhost`); use headphones in noisy rooms, since
  echo cancellation is the browser's. The small LLM can still misunderstand; every value is checked against what was said
  and the records, and nothing is decided by the LLM.

**Recommended next steps:** confirm the department list, contacts and ranking weights with James; import the Rice events
calendar and departmental calendars; connect the project mailbox after approval; add authentication and roles; add a local or approved LLM behind the existing tool/permission boundary for richer drafting and request
intake; read-only TRIRIGA and scheduling adapters (integration Level 2); usability testing with James (SUS) as the UI
specification proposes; collect real verification outcomes to evaluate the model on genuine record drift.

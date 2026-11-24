# Campus Facilities AI — Setup

Follow the steps in order. Commands are for **PowerShell** in the project folder (`FacilitiesAI`).
Steps 1–5 get the app running on sample data. Steps 6–9 connect it to Rice.

---

## 1. Requirements

| | |
|---|---|
| Computer | Windows 10/11 (macOS/Linux work with `.venv/bin/python`) |
| Disk / memory | 15 GB free, 16 GB RAM |
| GPU | Optional. An NVIDIA GPU with 8 GB makes the voice assistant fast (tested on an RTX 4070 laptop) |
| Software | Python 3.12 or newer (tested with 3.14), Git, Google Chrome or Microsoft Edge, [Ollama](https://ollama.com) |

## 2. Get the project and the data

```powershell
git clone <repository-url> FacilitiesAI
```
```powershell
cd FacilitiesAI
```

Copy James's materials into `datares\` (they are **not** in the repository because they contain personnel data):
`SpaceRecords.xlsx`, `Report_PeopleRecords.xlsx`, the floor-plan folders (`May 2026\`, `November 2026\`) and the
`*.rtf` correspondence. Keep this folder on the project computer only.

## 3. Install

```powershell
python -m venv .venv
```
```powershell
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```
```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```
```powershell
Copy-Item .env.example .env
```

The trained model is already in `models\`. To retrain it (about 2 minutes):

```powershell
.venv\Scripts\python -m app.ml.train
```

## 4. Install the local voice and language models (optional, recommended)

All of these run on the computer; nothing is sent to the internet.

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

Without these the app still works: the voice assistant stays off and email triage uses keyword rules.

## 5. Run and check

```powershell
.venv\Scripts\python -m uvicorn app.api:app --port 8000
```

Open **http://localhost:8000**. Then:

1. **Settings → Voice assistant**: switch on (the pickers show the installed models).
2. **Spaces → New space request**: allow the microphone and say *"An office near Professor Bartel, at least 120 square feet."*
3. **Outages → Email → Check for new email**: replays the supplied correspondence into cases.

Run the tests (about 1 minute; expect *105 passed*):

```powershell
.venv\Scripts\python -m pytest -q
```

Restart the server after any change to `.env`.

---

## 6. Rice: access you need first

| Who | What | Why |
|---|---|---|
| **James** | Consent to connect his facilities mail and the record reports | The app reads his outage email and space/people data |
| **Rice IT (Microsoft 365 admin)** | One app registration (step 7) | Read-only access to mail and SharePoint files |
| **Space Management / TRIRIGA owner** | A scheduled export of the space and people reports to SharePoint (step 8) | Records stay current without moving files |
| **Data owner / privacy office** | Approval to keep personnel data on the project computer | `datares\` and `data\` hold names, NetIDs and locations |

Recommended: a **dedicated mailbox** (for example `facilities-outages@rice.edu`), or an Outlook rule in James's mailbox
that forwards TRIRIGA and facilities messages to it. The app then never sees unrelated mail.

## 7. Rice IT: register the app (Microsoft 365, read-only)

In the **Azure portal → Microsoft Entra ID → App registrations → New registration**:

1. Name `Campus Facilities AI`; *Accounts in this organizational directory only*; no redirect URI.
2. **Authentication → Allow public client flows: Yes** (this enables sign-in with a code).
3. **API permissions → Microsoft Graph → Delegated**: `Mail.Read`, `Mail.Read.Shared`, `Files.Read.All`, `User.Read`,
   `offline_access`. Grant admin consent if Rice requires it.
4. Copy the **Directory (tenant) ID** and the **Application (client) ID**.

*Unattended alternative (no sign-in needed):* **Application** permission `Mail.Read` plus an **application access
policy** that limits the app to the one mailbox, and a client secret. Then set `FAI_GRAPH_AUTH=app` and
`FAI_GRAPH_CLIENT_SECRET` in `.env`.

Put the values in `.env`:

```
FAI_GRAPH_TENANT_ID=<tenant id>
FAI_GRAPH_CLIENT_ID=<client id>
FAI_GRAPH_MAILBOX=facilities-outages@rice.edu      # or "me" for the signed-in mailbox
FAI_GRAPH_FOLDER=Inbox                             # or the folder name, e.g. Outages
FAI_EMAIL_SOURCE=graph
```

Restart, then **Settings → Connections → Microsoft 365 → Connect**. The app shows a code: James opens
*microsoft.com/devicelogin*, enters it and signs in with his Rice account. The app never sees his password; the token
is stored in `data\graph_token.json` (not in Git). Outage email is now checked **every 60 seconds**.

## 8. Space and people records (no manual file moves)

**Settings → Connections → Space and people records → Set up**, then choose one:

| Option | When to use | What to enter |
|---|---|---|
| **SharePoint / OneDrive** (recommended) | The TRIRIGA reports are saved to SharePoint on a schedule | The link to each report (in SharePoint: *Share → Copy link*). Needs step 7 |
| **Folder** | The reports land in a synced OneDrive folder or an export folder on this computer | The folder path |
| **Upload** | No automation yet | The two `.xlsx` files |

Pick how often to check (hourly is a good default). Each new version is validated before use; a broken export is
rejected and the current records stay. **Check now** checks immediately and shows what changed.

## 9. Optional switches (`.env`)

| Setting | Purpose |
|---|---|
| `FAI_ALLOW_UNLOCK=true` | Lets Settings switch on locked actions such as *Send email*. Leave `false` unless needed |
| `FAI_SMTP_HOST`, `FAI_SMTP_FROM`, `FAI_EMAIL_REDIRECT_TO` | Outgoing mail for approved drafts. Keep a test address until *Real recipients* is approved |
| `FAI_EMAIL_POLL_SECONDS` | How often to check mail (default 60 with Microsoft 365) |
| `FAI_RECORDS_REFRESH_MINUTES` | Default records check interval (Settings overrides it) |
| `FAI_OLLAMA_URL` | If Ollama runs on another computer |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| A page says the server runs older code | Stop the server (`Ctrl+C`) and start it again |
| Voice assistant shows *Models missing* | Repeat step 4; check `ollama list` shows `qwen3:8b` |
| Microphone does not work | Allow it in the browser; it only works on `http://localhost` or HTTPS |
| *Microsoft 365 not configured* | Step 7 values are missing from `.env`, or the server was not restarted |
| *Rejected the new files* | The export is missing columns or is not the right report; the previous records are still in use |
| First answer is slow | The language model is loading (about 10 s after a long idle); later answers take 1–3 s |

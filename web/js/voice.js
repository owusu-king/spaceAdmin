// Conversation with the Space agent: the main way to make a space request (the form is the alternative).
// Hands-free: the microphone listens from the moment the conversation opens; just talk, and talking while the
// assistant speaks interrupts it. The mic icon mutes and unmutes. A small avatar shows who is talking.
// Everything runs on this computer (app/voice/).
import { api, esc, fmtNum, icon } from "./lib.js";

const SAY = {
  idle: "Microphone off · type, or turn the mic on", connecting: "Getting ready…", loading: "Getting ready…",
  listening: "Listening · just speak", hearing: "Hearing you…", transcribing: "Got it", thinking: "Thinking…",
  searching: "Searching the records…", speaking: "Speaking · talk to interrupt", paused: "Paused",
  nomic: "Microphone unavailable · type below", closed: "Disconnected · reopen to start again",
};
export const SLOT_LABEL = { request_type: "Type", function: "Function", department: "Department", near_name: "Close to",
  min_area: "Min. area", seats: "Seats", preferred_buildings: "Buildings", exclude_buildings: "Not in" };

const AVATAR = `<svg viewBox="0 0 120 120" aria-hidden="true">
  <circle class="av-halo" cx="60" cy="60" r="58"/>
  <circle class="av-face" cx="60" cy="60" r="46"/>
  <g class="av-eyes"><ellipse cx="45" cy="54" rx="4.5" ry="6"/><ellipse cx="75" cy="54" rx="4.5" ry="6"/></g>
  <rect class="av-mouth" x="47" y="72" width="26" height="4" rx="2"/>
</svg>`;

/** Is the conversation available? Returns { ok, why, opts }. */
export async function conversationAvailable() {
  try {
    const opts = await api("/voice/options");
    if (!opts.enabled) return { ok: false, why: "The voice assistant is switched off in Settings.", opts };
    if (!opts.ready) return { ok: false, why: opts.problems[0] || "The local models are not ready.", opts };
    return { ok: true, opts };
  } catch (e) { return { ok: false, why: e.message }; }
}

/** Mounts into `root`. Callbacks: onChange({slots, canSave}) after each reply, onSaved(caseId). */
export class Conversation {
  constructor(root, opts, { onChange, onSaved } = {}) {
    Object.assign(this, { root, opts, onChange, onSaved });
    this.minTurn = 0; this.sources = new Set(); this.next = 0;
    this.listening = false; this.paused = false; this.state = "connecting";
    this.slots = {}; this.canSave = false;
  }

  start() {
    this.root.innerHTML = `
      <div class="vx-stage">
        <div class="vx-avatar" data-state="idle">${AVATAR}</div>
        <div class="vx-say"><b></b><span>${esc(this.opts.selected.voice_llm)} · runs on this computer</span></div>
      </div>
      <div class="vx-scroll">
        <div class="vx-log"><p class="vx-empty">Try: “An office near Professor Bartel, at least 120 square feet.”</p></div>
        <div class="vx-result"></div>
      </div>
      <form class="vx-composer"><div class="vx-field">
        <input placeholder="Describe the space you need…" autocomplete="off" aria-label="Message to the Space agent">
        <button type="button" class="vx-icon vx-mic" aria-pressed="false" aria-label="Tap to speak" title="Speak">${icon("mic")}</button>
        <button type="button" class="vx-icon vx-send" aria-label="Send" title="Send" disabled>${icon("send")}</button>
      </div></form>`;
    this.$ = sel => this.root.querySelector(sel);
    this.show("connecting");
    this.$(".vx-mic").onclick = () => this.toggleMic();
    const input = this.$(".vx-field input");
    input.oninput = () => this.updateAction();
    this.$(".vx-composer").onsubmit = e => {
      e.preventDefault();
      if (!input.value.trim()) return;
      this.stopAudio();
      this.send({ type: "text", text: input.value.trim() }); input.value = "";
      this.updateAction();
    };
    this.$(".vx-send").onclick = () => {                  // one icon: send what was typed, or stop the assistant
      if (this.$(".vx-send").dataset.mode === "stop") { this.send({ type: "stop" }); this.stopAudio(); this.show("listening"); }
      else this.$(".vx-composer").requestSubmit();
    };
    this.connect();
    this.animate();
  }

  /** Pause (e.g. while the form tab is shown) or resume. */
  setActive(active) {
    this.paused = !active;
    if (this.paused) { this.send({ type: "stop" }); this.stopAudio(); }
    else if (this.listening) this.send({ type: "mic_on" });          // start a fresh utterance on return
    this.show(this.paused ? "paused" : "listening");
  }

  save() { this.send({ type: "save" }); }

  // -------------------------------------------------------------------- connection
  connect() {
    this.ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/v1/voice/ws`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = e => typeof e.data === "string" ? this.onJson(JSON.parse(e.data)) : this.onAudio(e.data);
    this.ws.onclose = () => { if (!this.closed) this.show("closed"); };
    this.ws.onopen = () => this.startMic();                           // hands-free from the start
  }
  send(obj) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }

  onJson(m) {
    if (m.type === "status") {
      const st = (m.state === "listening" || m.state === "idle") ? "listening" : m.state;
      if (!this.paused) this.show(st);
    }
    else if (m.type === "transcript") this.bubble("me", m.text);
    else if (m.type === "say") this.sayText(m.turn, m.text);
    else if (m.type === "reply") this.reply(m);
    else if (m.type === "interrupt") { this.minTurn = m.turn; this.stopAudio(); this.cur = null; }
    else if (m.type === "step") this.step(m.turn, m.text);
    else if (m.type === "saved") this.saved(m.case_id, m.via);
    else if (m.type === "error") this.bubble("err", m.message);
  }

  // -------------------------------------------------------------------- microphone: hands-free, the icon mutes
  async toggleMic() {
    if (this.closed) return;
    if (this.listening) {                                            // mute
      this.listening = false;
      this.stream && this.stream.getAudioTracks().forEach(t => t.enabled = false);
      this.micButton(); this.show("listening");
      return;
    }
    if (!this.stream && !(await this.startMic())) return;           // unmute (asks permission the first time)
    this.stream.getAudioTracks().forEach(t => t.enabled = true);
    this.listening = true;
    this.send({ type: "mic_on" });                                   // fresh utterance; also stops the assistant
    this.micButton(); this.show("listening");
  }
  async startMic() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
      if (this.closed) { this.stream.getTracks().forEach(t => t.stop()); return false; }
      this.micCtx = new AudioContext();
      await this.micCtx.audioWorklet.addModule("/js/voice-worklet.js");
      const src = this.micCtx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.micCtx, "mic-processor");
      this.node.port.onmessage = e => { if (this.listening && !this.paused && this.ws.readyState === 1) this.ws.send(e.data); };
      this.micMeter = this.micCtx.createAnalyser(); this.micMeter.fftSize = 512;
      src.connect(this.node); src.connect(this.micMeter);
      this.listening = true;
      this.micButton();
      if (!this.paused) this.show("listening");
      return true;
    } catch (e) {
      this.stream = null; this.listening = false;
      this.show("nomic");
      this.$(".vx-mic").disabled = true;
      return false;
    }
  }
  micButton() {
    const b = this.$(".vx-mic");
    b.setAttribute("aria-pressed", String(this.listening));
    b.setAttribute("aria-label", this.listening ? "Mute microphone" : "Turn microphone on");
    b.title = this.listening ? "Mute" : "Unmute";
    b.innerHTML = icon(this.listening ? "mic" : "micOff");
    b.classList.toggle("on", this.listening);
  }

  // -------------------------------------------------------------------- playback (24 kHz PCM, one sentence per message)
  onAudio(buf) {
    const turn = new DataView(buf).getUint32(0, true);
    if (turn < this.minTurn || this.paused) return;                 // interrupted or paused: drop stale speech
    const pcm = new Int16Array(buf, 4);
    if (!this.outCtx) {
      this.outCtx = new AudioContext();
      this.outMeter = this.outCtx.createAnalyser(); this.outMeter.fftSize = 512;
      this.outMeter.connect(this.outCtx.destination);
    }
    const ab = this.outCtx.createBuffer(1, pcm.length, 24000);
    const ch = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const s = this.outCtx.createBufferSource();
    s.buffer = ab; s.connect(this.outMeter);
    this.next = Math.max(this.next, this.outCtx.currentTime + 0.05);
    s.start(this.next); this.next += ab.duration;
    this.sources.add(s);
    s.onended = () => { this.sources.delete(s); if (!this.sources.size) this.send({ type: "played" }); };
  }
  stopAudio() {
    this.sources.forEach(s => { s.onended = null; try { s.stop(); } catch { /* already stopped */ } });
    if (this.sources.size) this.send({ type: "played" });
    this.sources.clear(); this.next = 0;
    this.updateAction();
  }

  // -------------------------------------------------------------------- avatar: mouth follows the voice, halo follows the mic
  animate() {
    const buf = new Float32Array(512);
    const level = meter => { if (!meter) return 0; meter.getFloatTimeDomainData(buf); let s = 0; for (const v of buf) s += v * v; return Math.sqrt(s / buf.length); };
    let mouth = 4, halo = 1;
    const tick = () => {
      if (this.closed) return;
      const talking = this.sources.size > 0;
      const out = talking ? Math.min(1, level(this.outMeter) * 5) : 0;
      const live = this.listening && !this.paused;
      const mic = live ? Math.min(1, level(this.micMeter) * 6) : 0;
      mouth += ((talking ? 4 + out * 16 : 4) - mouth) * 0.45;
      halo += ((live ? 1 + mic * 0.18 : 1) - halo) * 0.3;
      const m = this.$(".av-mouth"), h = this.$(".av-halo");
      if (m) { m.setAttribute("height", mouth.toFixed(1)); m.setAttribute("y", (74 - mouth / 2).toFixed(1)); }
      if (h) h.style.transform = `scale(${halo.toFixed(3)})`;
      const av = this.$(".vx-avatar");
      if (av) av.dataset.state = talking ? "speaking" : ["thinking", "searching", "transcribing", "connecting", "loading"].includes(this.state)
        ? "thinking" : live ? "listening" : "idle";
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // -------------------------------------------------------------------- view
  show(state) {
    if (state === "listening" && this.state === "nomic") state = "nomic";                             // no microphone
    else if (state === "listening" && !this.listening && this.state !== "connecting") state = "idle";   // muted
    this.state = state;
    this.$(".vx-say b").textContent = SAY[state] || state;
    this.updateAction();
  }
  /** The icon at the end of the field: send (when something is typed) or stop (while the assistant works or talks). */
  updateAction() {
    const b = this.$(".vx-send");
    if (!b) return;
    const typed = !!this.$(".vx-field input").value.trim();
    const busy = this.sources.size > 0 || ["transcribing", "thinking", "searching", "speaking"].includes(this.state);
    const mode = typed ? "send" : busy ? "stop" : "idle";
    if (b.dataset.mode === mode) return;
    b.dataset.mode = mode;
    b.innerHTML = icon(mode === "stop" ? "stop" : "send");
    b.disabled = mode === "idle";
    b.setAttribute("aria-label", mode === "stop" ? "Stop" : "Send");
    b.title = mode === "stop" ? "Stop" : "Send";
  }
  scroll() { const s = this.$(".vx-scroll"); s.scrollTop = s.scrollHeight; }
  bubble(who, text) {
    const empty = this.$(".vx-empty"); if (empty) empty.remove();
    const b = document.createElement("div");
    b.className = `vx-msg ${who}`; b.textContent = text;
    this.$(".vx-log").appendChild(b); this.scroll();
    return b;
  }
  sayText(turn, text) {
    if (turn < this.minTurn) return;
    if (!this.cur || this.curTurn !== turn) { this.cur = this.bubble("bot", ""); this.curTurn = turn; }
    this.cur.textContent = (this.cur.textContent + " " + text).trim();
    this.scroll();
  }
  /** The agent's steps, live: lead-in sentence, then this checklist, then the answer below it. */
  step(turn, text) {
    if (turn < this.minTurn) return;
    if (!this.steps || this.stepsTurn !== turn || this.$(".vx-log").lastElementChild !== this.steps) {   // keep time order
      const empty = this.$(".vx-empty"); if (empty) empty.remove();
      this.steps = document.createElement("ul");
      this.steps.className = "vx-steps"; this.stepsTurn = turn;
      this.$(".vx-log").appendChild(this.steps);
      this.cur = null;                                           // the answer starts a new bubble below the steps
    }
    const li = document.createElement("li");
    li.innerHTML = `${icon("check")}<span>${esc(text)}</span>`;
    this.steps.appendChild(li);
    this.scroll();
  }
  saved(id, via) {
    if (via === "voice") {
      const note = document.createElement("div");
      note.className = "vx-saved";
      note.innerHTML = `${icon("check")}<span>Saved as <b class="mono">${esc(id)}</b>, ready for review</span><a href="#/cases/${esc(id)}" data-close>Open case</a>`;
      this.$(".vx-log").appendChild(note); this.scroll();
    }
    this.onSaved && this.onSaved(id, via);
  }

  reply(m) {
    if (!this.cur || this.curTurn !== m.turn) this.bubble("bot", m.text);
    this.cur = null;
    this.slots = m.slots || {}; this.canSave = !!m.can_save;
    const slots = Object.entries(this.slots).filter(([k, v]) => SLOT_LABEL[k] && v !== null && !(Array.isArray(v) && !v.length));
    const chips = slots.map(([k, v]) => `<span class="vx-chip"><b>${SLOT_LABEL[k]}</b> ${esc(Array.isArray(v) ? v.join(", ")
      : k === "min_area" ? `${fmtNum(v)} sf` : String(v).replace(/_/g, " "))}</span>`).join("");
    const r = m.report;
    const cands = r && r.candidates && r.candidates.length ? `<ol class="vx-cands">${r.candidates.map(c => `
        <li><b class="mono">${esc(c.building)} · ${esc(c.floor)} · ${esc(c.room)}</b>
          <span>${c.area_sf ? `${fmtNum(c.area_sf)} sf · ` : ""}${esc(c.status)}${c.closeness ? ` · ${esc(c.closeness)}` : ""}</span></li>`).join("")}</ol>` : "";
    this.$(".vx-result").innerHTML = slots.length || r ? `<section class="vx-card">
        <div class="vx-chips">${chips}</div>
        ${r ? `<p class="vx-sum">${esc(r.summary)}</p>${cands}` : ""}
      </section>` : "";
    this.scroll();
    this.onChange && this.onChange({ slots: this.slots, canSave: this.canSave });
  }

  close() {
    this.closed = true;
    this.stopAudio();
    try { this.ws && this.ws.close(); } catch { /* ignore */ }
    this.stream && this.stream.getTracks().forEach(t => t.stop());
    this.micCtx && this.micCtx.close(); this.outCtx && this.outCtx.close();
  }
}

// Microphone capture in the audio thread: resample to 16 kHz mono and post 16-bit PCM in 512-sample frames
// (32 ms each, the frame size the speech detector expects).
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;   // sampleRate is the context rate, usually 48000
    this.pos = 0;
    this.buf = new Int16Array(512);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (; this.pos < ch.length; this.pos += this.ratio) {
      // average the samples this output sample covers (a simple low-pass before decimation)
      const a = Math.floor(this.pos), b = Math.min(ch.length, Math.floor(this.pos + this.ratio));
      let s = 0;
      for (let i = a; i < Math.max(a + 1, b); i++) s += ch[i];
      s /= Math.max(1, b - a);
      this.buf[this.n++] = Math.max(-1, Math.min(1, s)) * 32767;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer.slice(0)); this.n = 0; }
    }
    this.pos -= ch.length;
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);

// 3D talking avatar for the conversation: a realistic head and shoulders (Microsoft Rocketbox, MIT), drawn with
// Three.js on this computer. No network, no cloud service.
// Lip sync follows the assistant's actual voice: loudness opens the mouth, the spectrum picks the mouth shape
// (low, dark sounds -> "o"/"u"; bright vowels -> "e"/"i"; hiss -> "s"/"f"; short gaps -> lips together "p/b/m").
// Around that: natural blinks, small eye and head movements, attentive brows while listening, a glance up while thinking.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DIR = "/vendor/avatar/";
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (cur, target, k) => cur + (target - cur) * k;

/** Builds the avatar inside `host` (a sized element). Resolves to { update(frame), dispose() }; rejects without WebGL. */
export async function createAvatar(host) {
  const canvas = document.createElement("canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;

  const tex = new THREE.TextureLoader();
  const load = (file, color = true) => tex.loadAsync(DIR + file).then(t => {
    if (color) t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return t;
  });
  let gltf, maps;
  try {
    [gltf, ...maps] = await Promise.all([new GLTFLoader().loadAsync(DIR + "avatar.glb"),
      load("head_color.jpg"), load("head_normal.jpg", false), load("body_color.jpg"), load("opacity_color.png")]);
  } catch (e) { renderer.dispose(); throw e; }
  const [headMap, headNormal, bodyMap, hairMap] = maps;

  const scene = new THREE.Scene();
  const root = gltf.scene;
  scene.add(root);

  // materials by name: skin, clothes, and the alpha-blended hair, lashes and brows.
  // The loader splits the model into one mesh per material; each carries the same face shapes.
  const parts = [];
  root.traverse(o => {
    if (!o.isMesh) return;
    o.frustumCulled = false;
    if (o.morphTargetDictionary && "jawOpen" in o.morphTargetDictionary) parts.push(o);
    const fix = m => {
      if (m.name.endsWith("head")) Object.assign(m, { map: headMap, normalMap: headNormal, roughness: 0.62, metalness: 0 });
      else if (m.name.endsWith("body")) Object.assign(m, { map: bodyMap, roughness: 0.8, metalness: 0 });
      else if (m.name.endsWith("opacity")) {
        Object.assign(m, { map: hairMap, transparent: true, alphaTest: 0.04, depthWrite: false, side: THREE.DoubleSide, roughness: 0.55, metalness: 0 });
      }
      m.needsUpdate = true;
    };
    [].concat(o.material).forEach(fix);
  });
  if (!parts.length) { renderer.dispose(); throw new Error("Avatar model has no face shapes"); }

  // lighting: soft key from the upper left, cool fill, warm rim for depth
  scene.add(new THREE.HemisphereLight(0xf2f4fa, 0x3c4048, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-60, 220, 160); scene.add(key);
  const fill = new THREE.DirectionalLight(0xcfdcff, 0.9); fill.position.set(120, 160, 120); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffe4c4, 1.6); rim.position.set(80, 220, -150); scene.add(rim);

  const set = (name, v) => { for (const o of parts) { const i = o.morphTargetDictionary[name]; if (i !== undefined) o.morphTargetInfluences[i] = v; } };
  const bone = n => root.getObjectByName(n);
  const headBone = bone("Bip01_Head"), neck = bone("Bip01_Neck");
  const eyes = [bone("Bip01_LEye"), bone("Bip01_REye")].filter(Boolean);
  const rest = new Map([headBone, neck, ...eyes].filter(Boolean).map(b => [b, b.quaternion.clone()]));

  // portrait framing: from the top of the hair down to the shoulders
  root.updateMatrixWorld(true);
  const headPos = headBone ? headBone.getWorldPosition(new THREE.Vector3()) : new THREE.Vector3(0, 152, 0);
  const box = new THREE.Box3().setFromObject(root);
  const top = box.max.y, viewH = (top - headPos.y) * 1.8;     // head, neck and the top of the shoulders
  const camera = new THREE.PerspectiveCamera(18, 1, 1, 2000);
  const dist = viewH / 2 / Math.tan(THREE.MathUtils.degToRad(9));
  const aim = new THREE.Vector3(headPos.x, top - viewH * 0.5 + 1, headPos.z + 4);
  camera.position.set(aim.x, aim.y + viewH * 0.04, aim.z + dist);
  camera.lookAt(aim);

  host.appendChild(canvas);
  const resize = () => {
    const w = host.clientWidth || 1, h = host.clientHeight || 1;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize); ro.observe(host);

  // ---------------------------------------------------------------------------------------------- motion state
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const spec = new Uint8Array(256), wave = new Float32Array(512);
  const m = { open: 0, round: 0, spread: 0, hiss: 0, lips: 0, brow: 0, smile: 0.1, lookUp: 0, blink: 0, attent: 0 };
  let nextBlink = performance.now() + 1500, blinkStart = -1, gap = 0;
  const gaze = { x: 0, y: 0 }; let gazeTarget = { x: 0, y: 0 }, nextSaccade = 0;
  let last = performance.now(), peak = 0.02, drawn = 0;
  const t0 = last, q = new THREE.Quaternion(), e = new THREE.Euler();

  /** Voice features from the playback analyser: loudness, and the balance of low, mid and high energy. */
  function voice(an) {
    if (!an) return null;
    an.getFloatTimeDomainData(wave);
    let s = 0; for (const x of wave) s += x * x;
    const rms = Math.sqrt(s / wave.length);
    peak = Math.max(rms, peak * 0.995, 0.02);                     // follows the speaker's loudness
    an.getByteFrequencyData(spec);
    const hz = an.context.sampleRate / an.fftSize, bin = f => Math.min(spec.length - 1, Math.round(f / hz));
    const band = (a, b) => { let x = 0; for (let i = bin(a); i <= bin(b); i++) x += spec[i]; return x / ((bin(b) - bin(a) + 1) * 255); };
    const low = band(150, 800), mid = band(800, 2500), high = band(3500, 8000), tot = low + mid + high + 1e-3;
    return { level: clamp(rms / peak), low: low / tot, mid: mid / tot, high: high / tot };
  }

  /** Called every animation frame by the conversation. */
  function update({ state = "idle", speech = null, talking = false, mic = 0, pose = null } = {}) {
    const now = performance.now(), dt = Math.min(0.05, (now - last) / 1000); last = now;
    const t = (now - t0) / 1000, k = x => 1 - Math.pow(1 - x, dt * 60);    // frame-rate independent easing

    // ---- mouth
    const v = talking ? voice(speech) : null;
    const voiced = !!v && v.level > 0.14;
    gap = talking && !voiced ? gap + dt : 0;
    const hissy = voiced && v.high > 0.38 && v.low < 0.4;
    m.open = ease(m.open, voiced && !hissy ? 0.15 + v.level * 0.6 : hissy ? 0.08 : 0, k(voiced ? 0.5 : 0.3));
    m.round = ease(m.round, voiced ? clamp((v.low - 0.5) * 2.5) : 0, k(0.3));
    m.spread = ease(m.spread, voiced ? clamp((v.mid - 0.3) * 2.5) * (1 - m.round) : 0, k(0.3));
    m.hiss = ease(m.hiss, hissy ? 0.7 : 0, k(0.4));
    m.lips = ease(m.lips, gap > 0.03 && gap < 0.25 ? 0.7 : 0, k(0.45));       // brief closure between syllables
    set("viseme_aa", m.open * (1 - m.round * 0.6) * 0.9);
    set("viseme_o", m.open * m.round * 0.9); set("viseme_u", m.round * (1 - m.open) * 0.5);
    set("viseme_e", m.open * m.spread * 0.6); set("viseme_i", m.spread * (1 - m.open) * 0.5);
    set("viseme_ss", m.hiss * 0.8); set("viseme_ff", m.hiss * 0.25);
    set("viseme_pp", m.lips);
    set("jawOpen", m.open * 0.18);

    // ---- expression by state
    const thinking = state === "thinking", listening = state === "listening";
    m.lookUp = ease(m.lookUp, thinking ? 1 : 0, k(0.06));
    m.attent = ease(m.attent, listening ? clamp(0.25 + mic * 1.5) : 0, k(0.08));
    m.brow = ease(m.brow, talking ? (v ? v.level : 0) * 0.3 : m.attent * 0.25 + m.lookUp * 0.3, k(0.12));
    m.smile = ease(m.smile, talking ? 0.12 : listening ? 0.22 : 0.12, k(0.05));
    set("browInnerUp", m.brow); set("browOuterUp_L", m.brow * 0.5); set("browOuterUp_R", m.brow * 0.5);
    set("mouthSmile_L", m.smile * (1 - m.round)); set("mouthSmile_R", m.smile * (1 - m.round));
    set("eyeWide_L", m.attent * 0.1); set("eyeWide_R", m.attent * 0.1);
    set("eyeLookUp_L", m.lookUp * 0.25); set("eyeLookUp_R", m.lookUp * 0.25);

    // ---- blinks every 2–6 s, sometimes a double blink
    if (blinkStart < 0 && now > nextBlink) blinkStart = now;
    m.blink = 0;
    if (blinkStart >= 0) {
      const p = (now - blinkStart) / 180;
      m.blink = p < 0.45 ? p / 0.45 : p < 1 ? (1 - p) / 0.55 : 0;
      if (p >= 1) { blinkStart = -1; nextBlink = now + (Math.random() < 0.15 ? 200 : 2000 + Math.random() * 4000); }
    }
    set("eyeBlink_L", m.blink); set("eyeBlink_R", m.blink);

    // ---- gaze: at the viewer with small saccades; up and aside while thinking
    if (now > nextSaccade) {
      const r = listening ? 0.035 : 0.08;
      gazeTarget = { x: (Math.random() - 0.5) * r * 2, y: (Math.random() - 0.5) * r };
      nextSaccade = now + 700 + Math.random() * 2200;
    }
    gaze.x = ease(gaze.x, gazeTarget.x + m.lookUp * 0.2, k(0.35));
    gaze.y = ease(gaze.y, gazeTarget.y + m.lookUp * 0.14, k(0.35));
    const turn = (b, x, y, z) => { if (!b) return; e.set(x, y, z); b.quaternion.copy(rest.get(b)).multiply(q.setFromEuler(e)); };
    // bone axes (Biped rig, checked visually): eyes y = left/right, z = up (+); head x = turn, y = tilt, z = nod (+ down)
    eyes.forEach(b => turn(b, 0, gaze.x, gaze.y));

    // ---- head: breathing sway, a small nod with speech, a slight tilt while listening
    const s = reduced ? 0 : 1;
    turn(headBone, s * (Math.sin(t * 0.53) * 0.035 + m.lookUp * 0.06), s * (Math.sin(t * 0.41) * 0.015 + m.attent * 0.04),
      s * (Math.sin(t * 0.9) * 0.012 + m.open * 0.03 - m.lookUp * 0.05));
    turn(neck, 0, s * Math.sin(t * 0.31) * 0.012, s * Math.sin(t * 0.7) * 0.008);

    if (pose) pose(root);                                            // tests: adjust bones before drawing
    // draw at most 30 times a second, and not at all while hidden: the GPU is shared with the language model
    if (now - drawn >= 32 && host.isConnected && host.getClientRects().length) { drawn = now; renderer.render(scene, camera); }
  }

  function dispose() {
    ro.disconnect();
    root.traverse(o => { if (o.geometry) o.geometry.dispose(); [].concat(o.material || []).forEach(x => x.dispose()); });
    maps.forEach(x => x.dispose());
    renderer.dispose(); renderer.forceContextLoss(); canvas.remove();
  }

  update();
  return { update, dispose };
}

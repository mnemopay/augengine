/**
 * AugEngine detective — browser shell for every registered case.
 *
 * State flows one way: UI action → `act()` (rule engine) → new GameState →
 * dialogue (LLM restates the rule gist, guardrailed; offline shows the gist)
 * → memory recorded → UI re-rendered. The renderer only reads the scene.
 */
import { Scene, Tag } from "@mnemopay/augengine";
import { assetUrl } from "./asset-url.js";
import { createLiveRenderer, lookAtTransform, type WebGpuLiveRenderer } from "@mnemopay/augengine/runtime/live";
import {
  CASES,
  DEFAULT_CASE_ID,
  act,
  blankMemory,
  caseById,
  caseStats,
  isCaseId,
  newGame,
  publicView,
  recordAccusation,
  recordAsk,
  recordPresent,
  rememberLine,
  startSession,
  validateLine,
  moodFor,
  portraitPath,
  pick,
  rotate,
  type CaseDef,
  type PickBox,
  type V3,
  type Q4,
  type Mood,
  type Action,
  type DialogueContext,
  type EvidenceId,
  type GameState,
  type MethodId,
  type MotiveId,
  type PlayerMemory,
  type SuspectId,
  type Utterance,
} from "@mnemopay/augengine-detective";
import { BrowserMemoryStore, clearProgress, loadProgress, playerId, saveProgress } from "./memory-store.js";
import { detectMode, speak, type DialogueMode } from "./llm-client.js";
import {
  BUST_HALF,
  BUST_ROW_HALF,
  CLOSEUP_CAMERA,
  buildCaseSpec,
  bustRowCenter,
  bustUrl,
  cameraId,
  headId,
  headPosition,
  objectEntityId,
  suspectEntityIds,
} from "./scene.js";
import { loadBust } from "./busts.js";
import { drawShareCard, shareText, verdictLabel } from "./share.js";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
const FORCE_OFFLINE = params.has("offline");
const LAST_CASE_KEY = "augengine-detective:last-case";

// ── state ──────────────────────────────────────────────────────────────
let game: CaseDef = caseById(DEFAULT_CASE_ID);
let state: GameState = newGame(game.id);
let memory: PlayerMemory = blankMemory("anon", game.id);
let store = new BrowserMemoryStore(game.id);
let mode: DialogueMode = "offline";
let sessionId: string = crypto.randomUUID();
/** One session per case per page load (a visit); "Investigate again" starts another. */
const sessionByCase = new Map<string, string>();
let talkingTo: SuspectId | null = null;
let busy = false;
let started = false;
const pid = playerId();
/** Suspects already greeted with their memory line this session. */
const greeted = new Set<SuspectId>();
const transcript: { who: string; text: string; source: string }[] = [];

// ── renderer ───────────────────────────────────────────────────────────
const canvas = $<HTMLCanvasElement>("game");
let renderer: WebGpuLiveRenderer | null = null;
let scene = new Scene(buildCaseSpec(game));
/** Current portrait mood per suspect, set only from the rule engine's utterances. */
let moods: Record<SuspectId, Mood> = {};
const frameMs: number[] = [];
const bustsRequested = new Set<string>();

function resetMoods(): void {
  moods = Object.fromEntries(game.suspectOrder.map((s) => [s, "calm" as Mood]));
}
resetMoods();

const suspect = (sid: SuspectId) => game.suspects[sid]!;
const evidence = (e: EvidenceId) => game.evidence[e]!;
const firstName = (sid: SuspectId) => suspect(sid).name.split(" ")[0]!;
const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** Record a suspect's mood; the dialogue panel shows the matching portrait while they are questioned. */
function setMood(sid: SuspectId, mood: Mood): void {
  moods[sid] = mood;
  if (talkingTo === sid) showPortrait(sid, mood);
}

function showPortrait(sid: SuspectId | null, mood: Mood = "calm"): void {
  const img = $<HTMLImageElement>("portrait");
  $("dialogue").classList.toggle("has-portrait", !!sid);
  if (!sid) {
    img.classList.remove("show");
    return;
  }
  const src = assetUrl(portraitPath(sid, mood, game.id));
  img.alt = `${suspect(sid).name}, ${mood}`;
  img.dataset.mood = mood;
  img.dataset.testid = "portrait";
  if (img.getAttribute("src") === src) {
    img.classList.add("show");
    return;
  }
  img.classList.remove("show");
  window.setTimeout(() => {
    img.src = src;
    img.onload = () => img.classList.add("show");
  }, 120);
}

function sizeCanvas(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
}

/**
 * Frame the row of busts inside the part of the screen the panels leave
 * free, so no suspect hides behind a panel at any window size.
 */
let lastFit: { distance: number; window: [number, number, number, number] } | null = null;
function fitInterviewCamera(): void {
  const camId = cameraId(game.interviewRoom);
  const cam = scene.get(camId as never)?.components[Tag.Camera];
  const W = canvas.clientWidth || window.innerWidth;
  const H = canvas.clientHeight || window.innerHeight;
  if (!cam || W < 10 || H < 10) return;
  const visible = (id: string) => {
    const el = $(id);
    return el.offsetParent !== null && !el.classList.contains("collapsed") ? el.getBoundingClientRect() : null;
  };
  const rooms = visible("rooms");
  const notebook = visible("notebook");
  const roomsBar = $("rooms").getBoundingClientRect();
  const notebookBar = $("notebook").getBoundingClientRect();
  const dialogue = $("dialogue").getBoundingClientRect();
  let L = 0;
  let R = W;
  let T = 0;
  const side = window.innerWidth > 900;
  if (side) {
    if (rooms) L = rooms.right;
    if (notebook) R = notebook.left;
  } else {
    T = Math.max(roomsBar.bottom, notebookBar.bottom);
  }
  let B = dialogue.top > 0 ? dialogue.top : H;
  if (R - L < W * 0.3) [L, R] = [0, W];
  if (B - T < H * 0.18) [T, B] = [0, H];
  const t = Math.tan(cam.fovY / 2);
  const aspect = W / H;
  const nl = (2 * L) / W - 1 + 0.03;
  const nr = (2 * R) / W - 1 - 0.03;
  const nyc = 1 - (T + B) / H;
  const S = BUST_ROW_HALF + 0.1;
  let k = (2 * S) / (nr - nl);
  let d = k / (t * aspect);
  d = Math.min(9, Math.max(2.8, d));
  k = d * t * aspect;
  const offX = (-(nl + nr) / 2) * k;
  const [rx, ry, rz] = bustRowCenter(game);
  const eyeY = 1.6;
  const alpha = Math.atan((eyeY - ry) / d);
  const beta = alpha + Math.atan(nyc * t);
  const eye: V3 = [rx + offX, eyeY, rz + d];
  const target: V3 = [rx + offX, eyeY - d * Math.tan(beta), rz];
  const tr = lookAtTransform(eye, target, [0, 1, 0]);
  scene.setLocalTransform(camId as never, tr.position, tr.rotation, [1, 1, 1]);
  lastFit = { distance: d, window: [L, R, T, B] };
}

function showRoomCamera(): void {
  if (state.room === game.interviewRoom && talkingTo) {
    const head = headPosition(game, talkingTo);
    const eye: [number, number, number] = [head[0] + 0.15, head[1] + 0.12, head[2] + 1.6];
    const t = lookAtTransform(eye, [head[0], head[1] - 0.02, head[2]], [0, 1, 0]);
    scene.setLocalTransform(CLOSEUP_CAMERA, t.position, t.rotation, [1, 1, 1]);
    scene.activeCamera = CLOSEUP_CAMERA;
  } else {
    if (state.room === game.interviewRoom) fitInterviewCamera();
    scene.activeCamera = cameraId(state.room);
  }
}

/** What the pointer is over in the 3D view (an object id or a suspect id), if anything. */
let hovered: { kind: "object"; id: string } | { kind: "suspect"; id: SuspectId } | null = null;

function highlightExamined(): void {
  for (const o of game.objects) {
    const e = scene.get(objectEntityId(o.id));
    const mesh = e?.components[Tag.Mesh];
    if (!mesh || mesh.material.kind !== "pbr") continue;
    // Found clues glow faintly so the room reads as searched; the hovered one glows brighter.
    const found = o.clue && state.notebook.includes(o.clue);
    const over = hovered?.kind === "object" && hovered.id === o.id;
    mesh.material.emissive = over ? [0.22, 0.17, 0.07] : found ? [0.07, 0.05, 0.015] : [0, 0, 0];
  }
  for (const sid of game.suspectOrder) {
    const over = hovered?.kind === "suspect" && hovered.id === sid;
    for (const eid of suspectEntityIds(sid)) {
      const mesh = scene.get(eid)?.components[Tag.Mesh];
      if (mesh && mesh.material.kind === "pbr") mesh.material.emissive = over ? [0.09, 0.07, 0.03] : [0, 0, 0];
    }
  }
}

// ── click-to-interact in the 3D view ───────────────────────────────────
/** Pickable boxes for the current room: its clue objects, and in the interview room the suspects' busts, plinths and plates. */
function pickBoxes(): PickBox[] {
  const boxes: PickBox[] = [];
  const add = (eid: string, id: string, halfOf: (scale: V3, shape: string) => V3) => {
    const e = scene.get(eid as never);
    const tr = e?.components[Tag.Transform];
    const mesh = e?.components[Tag.Mesh];
    if (!tr || !mesh) return;
    const shape = mesh.geometry.kind === "primitive" ? mesh.geometry.shape : mesh.geometry.kind;
    boxes.push({ id, center: tr.position as V3, rotation: tr.rotation as Q4, half: halfOf(tr.scale as V3, shape) });
  };
  // Unit box primitives span ±0.5; sphere and cylinder primitives have radius 1 (cylinder height 2).
  const primitiveHalf = (s: V3, shape: string): V3 => (shape === "box" ? [s[0] / 2, s[1] / 2, s[2] / 2] : [s[0], s[1], s[2]]);
  for (const o of game.objects.filter((x) => x.room === state.room)) add(objectEntityId(o.id), `object:${o.id}`, primitiveHalf);
  if (state.room === game.interviewRoom) {
    for (const sid of game.suspectOrder) {
      add(headId(sid), `suspect:${sid}`, (s) => [BUST_HALF[0] * s[0], BUST_HALF[1] * s[1], BUST_HALF[2] * s[2]]);
      for (const eid of suspectEntityIds(sid).slice(1)) add(eid, `suspect:${sid}`, primitiveHalf);
    }
  }
  return boxes;
}

function pickAt(clientX: number, clientY: number): typeof hovered {
  const camEnt = scene.get(scene.activeCamera as never);
  const tr = camEnt?.components[Tag.Transform];
  const cam = camEnt?.components[Tag.Camera];
  if (!tr || !cam) return null;
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
  const hit = pick({ position: tr.position as V3, rotation: tr.rotation as Q4, fovY: cam.fovY, aspect: rect.width / rect.height }, ndcX, ndcY, pickBoxes());
  if (!hit) return null;
  const [kind, id] = hit.id.split(":") as ["object" | "suspect", string];
  return kind === "object" ? { kind, id } : { kind, id: id as SuspectId };
}

function setHovered(next: typeof hovered, clientX = 0, clientY = 0): void {
  const label = $("hover-label");
  const same = hovered?.kind === next?.kind && hovered?.id === next?.id;
  hovered = next;
  canvas.style.cursor = next ? "pointer" : "default";
  if (next) {
    const name = next.kind === "object" ? game.objects.find((o) => o.id === next.id)?.name ?? "" : suspect(next.id).name;
    label.textContent = next.kind === "object" ? `Examine: ${name}` : `Question ${name}`;
    label.style.left = `${Math.min(clientX + 14, window.innerWidth - 200)}px`;
    label.style.top = `${clientY + 12}px`;
    label.style.display = "block";
  } else {
    label.style.display = "none";
  }
  if (!same) highlightExamined();
}

function wirePicking(): void {
  canvas.addEventListener("pointermove", (ev) => setHovered(busy ? null : pickAt(ev.clientX, ev.clientY), ev.clientX, ev.clientY));
  canvas.addEventListener("pointerleave", () => setHovered(null));
  canvas.addEventListener("click", (ev) => {
    if (busy) return;
    const hit = pickAt(ev.clientX, ev.clientY);
    if (!hit) return;
    if (hit.kind === "object") doAction({ type: "examine", objectId: hit.id });
    else selectSuspect(hit.id);
  });
}

function requestBusts(): void {
  if (!renderer || params.get("busts") === "0") return;
  for (const sid of game.suspectOrder) {
    const url = bustUrl(game, sid);
    if (bustsRequested.has(url)) continue;
    bustsRequested.add(url);
    loadBust(renderer, url).catch((err) => console.warn(`[detective] ${err.message}`));
  }
}

async function bootRenderer(): Promise<void> {
  sizeCanvas();
  if (!("gpu" in navigator)) {
    toast("This browser has no WebGPU — the building won't render, but the case is fully playable.");
    return;
  }
  try {
    const r = createLiveRenderer(canvas);
    await r.init();
    r.setSky(false);
    r.setIbl(true, { irradiance: [0.2, 0.19, 0.18], intensity: 1.0 });
    r.setShadowRange(10);
    r.setImageQuality?.({ taa: true, renderScale: 0.85 });
    renderer = r;
    // Busts stream in; until one arrives its entity simply doesn't draw. `?busts=0` skips them (perf A/B).
    requestBusts();
    window.addEventListener("resize", () => {
      sizeCanvas();
      r.resize(canvas.width, canvas.height);
    });
    requestAnimationFrame(frame);
  } catch (err) {
    toast(`3D view unavailable (${(err as Error).message}). The case is still playable.`);
  }
}

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  frameMs.push(now - last);
  if (frameMs.length > 600) frameMs.shift();
  last = now;
  if (renderer) {
    try {
      renderer.render(scene, { clearColor: [0.03, 0.035, 0.05], ambient: [0.16, 0.15, 0.15], lightDir: [-0.3, -0.9, -0.4], dt });
    } catch {
      /* first frames while pipelines compile */
    }
  }
  if (frameMs.length % 30 === 0) $("perf").textContent = perfLine();
  requestAnimationFrame(frame);
}

function perfStats() {
  const s = [...frameMs].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? 0;
  return { frames: s.length, p50: q(0.5), p95: q(0.95), max: s[s.length - 1] ?? 0, fps: s.length ? 1000 / q(0.5) : 0 };
}

function perfLine(): string {
  const p = perfStats();
  return `${mode === "llm" ? "LLM dialogue" : "offline dialogue"} · ${p.fps.toFixed(0)} fps · p95 ${p.p95.toFixed(1)} ms`;
}

// ── UI ─────────────────────────────────────────────────────────────────
function toast(text: string, ms = 3800): void {
  const el = $("toast");
  el.textContent = text;
  el.style.display = "block";
  window.clearTimeout((toast as unknown as { t?: number }).t);
  (toast as unknown as { t?: number }).t = window.setTimeout(() => (el.style.display = "none"), ms);
}

function renderRooms(): void {
  $("case-title").textContent = game.title;
  const list = $("room-list");
  list.innerHTML = "";
  for (const r of game.rooms) {
    const b = document.createElement("button");
    b.textContent = r.name;
    b.dataset.testid = `room-${r.id}`;
    b.className = r.id === state.room ? "active" : "";
    b.onclick = () => doAction({ type: "move", room: r.id });
    list.appendChild(b);
  }
  const room = game.rooms.find((r) => r.id === state.room)!;
  $("room-name").textContent = room.name;
  $("room-now").textContent = room.name;
  $("room-blurb").textContent = room.blurb;
  const objs = $("objects");
  objs.innerHTML = "";
  for (const o of game.objects.filter((x) => x.room === state.room)) {
    const b = document.createElement("button");
    const seen = state.examined.includes(o.id);
    b.textContent = `${seen ? "✓ " : ""}Examine: ${o.name}`;
    b.dataset.testid = `obj-${o.id}`;
    b.onclick = () => doAction({ type: "examine", objectId: o.id });
    objs.appendChild(b);
  }
}

function renderNotebook(): void {
  const ul = $("notebook-list");
  ul.innerHTML = "";
  $("notebook-count").textContent = String(state.notebook.length);
  if (state.notebook.length === 0) {
    ul.innerHTML = '<li class="empty">Nothing yet. Search the rooms.</li>';
  }
  for (const e of state.notebook) {
    const li = document.createElement("li");
    li.className = e.startsWith("T_") ? "testimony" : "";
    li.dataset.testid = `nb-${e}`;
    li.innerHTML = `<span class="name"></span><br><span class="text"></span>`;
    (li.querySelector(".name") as HTMLElement).textContent = evidence(e).name;
    (li.querySelector(".text") as HTMLElement).textContent = evidence(e).text;
    ul.appendChild(li);
  }
  const sel = $<HTMLSelectElement>("present-select");
  const keep = sel.value;
  sel.innerHTML = '<option value="">Show evidence…</option>';
  for (const e of state.notebook) {
    const o = document.createElement("option");
    o.value = e;
    o.textContent = evidence(e).name;
    sel.appendChild(o);
  }
  if (state.notebook.includes(keep)) sel.value = keep;
}

function renderSuspects(): void {
  const box = $("suspects");
  box.innerHTML = "";
  const inRoom = state.room === game.interviewRoom;
  for (const sid of game.suspectOrder) {
    const s = suspect(sid);
    const b = document.createElement("button");
    b.className = "suspect-btn";
    b.innerHTML = `<img alt="" src="${assetUrl(portraitPath(sid, "calm", game.id))}"><span></span>`;
    (b.querySelector("span") as HTMLElement).textContent = firstName(sid);
    b.title = `${s.name} — ${s.role}`;
    b.setAttribute("aria-label", `${s.name}, ${s.role}`);
    b.dataset.testid = `suspect-${sid}`;
    if (talkingTo === sid) b.classList.add("active");
    b.disabled = !inRoom || busy;
    b.onclick = () => selectSuspect(sid);
    box.appendChild(b);
  }
  const can = inRoom && talkingTo !== null && !busy;
  $<HTMLInputElement>("ask-input").disabled = !can;
  $<HTMLButtonElement>("ask-send").disabled = !can;
  $<HTMLSelectElement>("present-select").disabled = !can;
  $<HTMLButtonElement>("present-send").disabled = !can;
  if (!inRoom) $("memory-note").textContent = "";
  showPortrait(inRoom ? talkingTo : null, talkingTo ? moods[talkingTo] : "calm");
}

function renderAll(): void {
  renderRooms();
  renderNotebook();
  renderSuspects();
  highlightExamined();
  showRoomCamera();
  if (started && state.turn > 0) saveProgress(pid, state);
}

function setSpeech(who: string, text: string, source: string): void {
  const el = $("speech");
  el.innerHTML = `<span class="who"></span>: <span class="line" data-testid="speech-line"></span><span class="src"></span>`;
  (el.querySelector(".who") as HTMLElement).textContent = who;
  (el.querySelector(".line") as HTMLElement).textContent = text;
  (el.querySelector(".src") as HTMLElement).textContent = source;
}

function selectSuspect(sid: SuspectId): void {
  talkingTo = sid;
  const line = greeted.has(sid) ? null : rememberLine(memory, sid);
  $("memory-note").textContent = line ? `${firstName(sid)} remembers you: “${line}”` : "";
  if (window.innerWidth <= 900) collapse("rooms", true);
  renderAll();
  setSpeech(suspect(sid).name, "…", "");
  void converse({ type: "ask", suspect: sid, text: "Good evening. Who are you?" }, "(you introduce yourself)");
}

async function converse(action: Extract<Action, { type: "ask" | "present" }>, playerText: string): Promise<void> {
  if (busy) return;
  const r = act(state, action);
  if (!r.ok) {
    toast(r.error);
    return;
  }
  state = r.state;
  const u = r.utterance as Utterance;
  const sid = u.suspect;
  const memLine = greeted.has(sid) ? null : rememberLine(memory, sid);
  greeted.add(sid);
  memory = action.type === "ask" ? recordAsk(memory, sid, u.topic, u.cracked) : recordPresent(memory, sid, action.evidence, u.cracked);
  void store.save(memory);
  const ctx: DialogueContext = { utterance: u, playerText, notebook: [...state.notebook], memoryLine: memLine };
  busy = true;
  // The rules, not the model, decide the face the suspect shows.
  setMood(sid, moodFor({ utterance: u, presented: action.type === "present" ? action.evidence : undefined, playerText: action.type === "ask" ? action.text : undefined }));
  renderAll();
  const spoken = await speak(ctx, mode, sessionId, (partial) => {
    // Never display text the guardrail would reject.
    if (validateLine(partial, ctx).ok) setSpeech(suspect(sid).name, partial, "");
  });
  setSpeech(suspect(sid).name, spoken.text, spoken.source === "llm" ? "· voiced by the model, checked by the rules" : spoken.rejected.length ? "· model line rejected by guardrail" : "");
  transcript.push({ who: suspect(sid).name, text: spoken.text, source: spoken.source });
  if (u.cracked.length) toast(`${firstName(sid)}'s story just changed.`);
  for (const t of u.revealed) toast(`New testimony in your notebook: ${evidence(t).name}`);
  busy = false;
  renderAll();
}

function doAction(a: Action): void {
  if (busy) return;
  const r = act(state, a);
  if (!r.ok) {
    toast(r.error);
    return;
  }
  state = r.state;
  if (a.type === "move" && a.room !== game.interviewRoom) talkingTo = null;
  if (a.type === "examine") {
    setSpeech("Notebook", r.message, "");
    if (r.clue) toast(`Clue found: ${evidence(r.clue).name}`);
  }
  if (a.type === "move") setSpeech(game.rooms.find((x) => x.id === a.room)!.name, r.message, "");
  renderAll();
}

// ── collapsible panels (mobile, and handy on desktop) ──────────────────
function collapse(panel: "rooms" | "notebook", on: boolean): void {
  const el = $(panel);
  el.classList.toggle("collapsed", on);
  el.querySelector(".panel-toggle")?.setAttribute("aria-expanded", String(!on));
  requestAnimationFrame(() => showRoomCamera());
}

function wirePanels(): void {
  for (const panel of ["rooms", "notebook"] as const) {
    $(`toggle-${panel}`).onclick = () => collapse(panel, !$(panel).classList.contains("collapsed"));
  }
  if (window.innerWidth <= 640) collapse("notebook", true);
  window.addEventListener("resize", () => showRoomCamera());
}

// ── accusation ─────────────────────────────────────────────────────────
function openAccuse(): void {
  if (state.notebook.length === 0) {
    toast("You have no evidence yet.");
    return;
  }
  const fill = (sel: HTMLSelectElement, items: [string, string][]) => {
    sel.innerHTML = "";
    for (const [v, label] of items) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      sel.appendChild(o);
    }
  };
  fill($<HTMLSelectElement>("acc-suspect"), game.suspectOrder.map((s) => [s, suspect(s).name]));
  fill($<HTMLSelectElement>("acc-method"), Object.entries(game.methods));
  fill($<HTMLSelectElement>("acc-motive"), Object.entries(game.motives));
  const list = $("acc-evidence");
  list.innerHTML = "";
  for (const e of state.notebook) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${e}" data-testid="acc-ev-${e}"> <span></span>`;
    (label.querySelector("span") as HTMLElement).textContent = evidence(e).name;
    list.appendChild(label);
  }
  $("acc-error").textContent = "";
  $("accuse").classList.add("show");
}

let shareBlobUrl: string | null = null;

function confirmAccuse(): void {
  const chosen = [...$("acc-evidence").querySelectorAll<HTMLInputElement>("input:checked")].map((i) => i.value);
  const r = act(state, {
    type: "accuse",
    suspect: $<HTMLSelectElement>("acc-suspect").value,
    method: $<HTMLSelectElement>("acc-method").value as MethodId,
    motive: $<HTMLSelectElement>("acc-motive").value as MotiveId,
    evidence: chosen,
  });
  if (!r.ok) {
    $("acc-error").textContent = r.error;
    return;
  }
  state = r.state;
  saveProgress(pid, state);
  const res = r.accusation!;
  memory = recordAccusation(memory, res);
  void store.save(memory);
  $("accuse").classList.remove("show");
  const h = $("result-headline");
  h.textContent = res.headline;
  h.className = `verdict-${res.verdict}`;
  const body = $("result-body");
  body.innerHTML = "";
  const links = document.createElement("p");
  links.className = "muted";
  links.textContent = `Your case — tied to the suspect: ${res.links.suspect ? "yes" : "no"} · method shown: ${res.links.method ? "yes" : "no"} · motive shown: ${res.links.motive ? "yes" : "no"}`;
  body.appendChild(links);
  const h3 = document.createElement("h3");
  h3.textContent = "What really happened";
  body.appendChild(h3);
  for (const para of res.explanation) {
    const p = document.createElement("p");
    p.textContent = para;
    body.appendChild(p);
  }
  const mem = document.createElement("p");
  mem.className = "muted";
  mem.textContent = `The ${game.id === DEFAULT_CASE_ID ? "household" : "staff"} will remember this. Cases solved: ${memory.cases.solved} · failed: ${memory.cases.failed}.`;
  body.appendChild(mem);
  void renderShare(res.verdict);
  $("result").classList.add("show");
}

/** The share block: spoiler-free text and a picture card. Nothing leaves the device unless the player sends it. */
async function renderShare(verdict: "solved" | "weak" | "wrong"): Promise<void> {
  const stats = caseStats(state);
  const url = `${location.origin}${import.meta.env.BASE_URL}?case=${game.id}`;
  const text = shareText({ title: game.title, verdict, stats, url });
  $<HTMLTextAreaElement>("share-text").value = text;
  $("share-status").textContent = "";
  const img = $<HTMLImageElement>("share-img");
  try {
    const blob = await drawShareCard({
      title: game.title,
      tagline: game.tagline,
      verdict,
      stats,
      portraitUrl: assetUrl(portraitPath(game.victim.portraitId, "calm", game.id)),
      footer: url.replace(/^https?:\/\//, ""),
    });
    if (shareBlobUrl) URL.revokeObjectURL(shareBlobUrl);
    shareBlobUrl = URL.createObjectURL(blob);
    img.src = shareBlobUrl;
    img.alt = `${game.title}: ${verdictLabel(verdict)}`;
    const dl = $<HTMLAnchorElement>("share-download");
    dl.href = shareBlobUrl;
    dl.download = `${game.id}-verdict.png`;
    const nativeBtn = $<HTMLButtonElement>("share-native");
    const file = new File([blob], `${game.id}-verdict.png`, { type: "image/png" });
    const canShareFile = typeof navigator.canShare === "function" && navigator.canShare({ files: [file] });
    nativeBtn.style.display = typeof navigator.share === "function" ? "" : "none";
    nativeBtn.onclick = () => {
      const data: ShareData = canShareFile ? { text, files: [file] } : { text };
      navigator.share(data).catch(() => undefined);
    };
  } catch (err) {
    $("share-status").textContent = `Card unavailable (${(err as Error).message}); the text still works.`;
  }
}

function wireShare(): void {
  $("share-copy").onclick = async () => {
    const ta = $<HTMLTextAreaElement>("share-text");
    try {
      await navigator.clipboard.writeText(ta.value);
      $("share-status").textContent = "Copied.";
    } catch {
      ta.select();
      $("share-status").textContent = "Select the text and copy it.";
    }
  };
}

async function newCase(): Promise<void> {
  state = newGame(game.id);
  clearProgress(pid, game.id);
  sessionId = crypto.randomUUID();
  sessionByCase.set(game.id, sessionId);
  memory = startSession(memory, sessionId);
  void store.save(memory);
  greeted.clear();
  talkingTo = null;
  resetMoods();
  transcript.length = 0;
  $("result").classList.remove("show");
  setSpeech(capital(game.authority), "You have the night. Start wherever you like.", "");
  renderAll();
}

// ── cases ──────────────────────────────────────────────────────────────
/** Switch the whole game to a case: memory, progress, scene and busts. */
async function loadCase(caseId: string, resume: boolean): Promise<void> {
  game = caseById(caseId);
  try {
    localStorage.setItem(LAST_CASE_KEY, game.id);
  } catch {
    /* ignore */
  }
  document.title = `${game.title} — an AugEngine detective game`;
  store = new BrowserMemoryStore(game.id);
  memory = (await store.load(pid)) ?? blankMemory(pid, game.id);
  const existing = sessionByCase.get(game.id);
  sessionId = existing ?? crypto.randomUUID();
  if (!existing) {
    sessionByCase.set(game.id, sessionId);
    memory = startSession(memory, sessionId);
    void store.save(memory);
  }
  const saved = resume ? loadProgress(pid, game.id) : null;
  if (!resume) clearProgress(pid, game.id);
  state = saved ?? newGame(game.id);
  greeted.clear();
  talkingTo = null;
  resetMoods();
  transcript.length = 0;
  scene = new Scene(buildCaseSpec(game));
  requestBusts();
  setSpeech(capital(game.authority), saved ? "Welcome back. Your notebook is as you left it." : "You have the night. Start wherever you like.", "");
  renderAll();
}

interface CaseProgress {
  label: string;
  inProgress: boolean;
}

async function progressFor(c: CaseDef): Promise<CaseProgress> {
  const saved = loadProgress(pid, c.id);
  const mem = await new BrowserMemoryStore(c.id).load(pid);
  const best = mem?.best ?? null;
  const parts: string[] = [];
  if (saved) {
    const s = caseStats(saved);
    parts.push(`In progress · ${s.cluesFound}/${s.cluesTotal} clues`);
  }
  if (best) parts.push(best === "solved" ? "Solved" : best === "weak" ? "Best: right suspect, weak case" : "Not solved yet");
  if (!saved && !best) parts.push(mem && mem.sessions > 0 ? "Visited" : "New case");
  return { label: parts.join(" · "), inProgress: !!saved };
}

let selectedCase = DEFAULT_CASE_ID;

async function renderCaseSelect(): Promise<void> {
  const list = $("case-list");
  list.innerHTML = "";
  for (const c of CASES) {
    const prog = await progressFor(c);
    const b = document.createElement("button");
    b.className = `case-card${c.id === selectedCase ? " active" : ""}`;
    b.dataset.testid = `case-card-${c.id}`;
    b.setAttribute("aria-pressed", String(c.id === selectedCase));
    b.innerHTML = `<img alt="" src="${assetUrl(portraitPath(c.victim.portraitId, "calm", c.id))}"><span class="case-meta"><span class="case-name"></span><span class="case-tag"></span><span class="case-progress" data-testid="case-progress-${c.id}"></span></span>`;
    (b.querySelector(".case-name") as HTMLElement).textContent = c.title;
    (b.querySelector(".case-tag") as HTMLElement).textContent = c.tagline;
    (b.querySelector(".case-progress") as HTMLElement).textContent = prog.label;
    b.onclick = () => {
      selectedCase = c.id;
      void renderCaseSelect();
    };
    list.appendChild(b);
  }
  const c = caseById(selectedCase);
  const prog = await progressFor(c);
  $("intro-title").textContent = c.title;
  $("intro-text").textContent = c.intro;
  const mem = await new BrowserMemoryStore(c.id).load(pid);
  const prior = mem?.sessions ?? 0;
  $("intro-memory").textContent =
    prior > 0 ? `You have been here before (${prior} earlier visit${prior === 1 ? "" : "s"}; solved ${mem!.cases.solved}, failed ${mem!.cases.failed}). They remember you.` : "";
  $("start").textContent = prog.inProgress ? "Resume the investigation" : "Begin the investigation";
  $("restart").style.display = prog.inProgress ? "" : "none";
}

function openCaseSelect(): void {
  if (started && state.turn > 0) saveProgress(pid, state);
  selectedCase = game.id;
  void renderCaseSelect();
  $("intro").classList.add("show");
}

// ── boot ───────────────────────────────────────────────────────────────
async function boot(): Promise<void> {
  const asked = params.get("case");
  let last: string | null = null;
  try {
    last = localStorage.getItem(LAST_CASE_KEY);
  } catch {
    /* ignore */
  }
  selectedCase = isCaseId(asked) ? asked : isCaseId(last) ? last : DEFAULT_CASE_ID;
  // Prepare the selected case behind the intro so the 3D view shows it straight away.
  await loadCase(selectedCase, true);
  mode = await detectMode(FORCE_OFFLINE);
  await renderCaseSelect();
  $("mode-note").textContent =
    mode === "llm"
      ? "Suspects speak through a language model, but every fact they say is decided by the case rules and checked before you see it."
      : "Offline dialogue: suspects answer with their scripted lines. Every fact is the same as in model mode.";
  const begin = async (resume: boolean) => {
    if (selectedCase !== game.id || !resume) await loadCase(selectedCase, resume);
    started = true;
    $("intro").classList.remove("show");
    renderAll();
  };
  $("start").onclick = () => void begin(true);
  $("restart").onclick = () => void begin(false);
  $("cases-open").onclick = openCaseSelect;
  $("accuse-open").onclick = openAccuse;
  $("acc-cancel").onclick = () => $("accuse").classList.remove("show");
  $("acc-confirm").onclick = confirmAccuse;
  $("new-case").onclick = () => void newCase();
  $("result-cases").onclick = () => {
    $("result").classList.remove("show");
    void newCase().then(openCaseSelect);
  };
  const sendAsk = () => {
    const input = $<HTMLInputElement>("ask-input");
    const text = input.value.trim();
    if (!text || !talkingTo) return;
    input.value = "";
    void converse({ type: "ask", suspect: talkingTo, text }, text);
  };
  $("ask-send").onclick = sendAsk;
  $<HTMLInputElement>("ask-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendAsk();
  });
  $("present-send").onclick = () => {
    const ev = $<HTMLSelectElement>("present-select").value;
    if (!ev || !talkingTo) return;
    void converse({ type: "present", suspect: talkingTo, evidence: ev }, `The detective shows you: ${evidence(ev).name}. ${evidence(ev).text}`);
  };
  window.addEventListener("keydown", (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || $("intro").classList.contains("show")) return;
    const n = Number(e.key);
    if (n >= 1 && n <= game.rooms.length) doAction({ type: "move", room: game.rooms[n - 1]!.id });
  });
  wirePicking();
  wirePanels();
  wireShare();
  renderAll();
  await bootRenderer();
}

// Agent / test hook: read-only view of the session.
(window as unknown as Record<string, unknown>).__detective = {
  view: () => publicView(state),
  caseId: () => game.id,
  mode: () => mode,
  perf: () => ({ ...perfStats(), renderer: renderer?.getFrameStats() ?? null }),
  memory: () => memory,
  moods: () => ({ ...moods }),
  fit: () => lastFit,
  /** Screen point (client px) of an entity's centre in the active camera, for click tests. */
  screenPointOf: (entityId: string) => {
    const camEnt = scene.get(scene.activeCamera as never);
    const tr = camEnt?.components[Tag.Transform];
    const cam = camEnt?.components[Tag.Camera];
    const target = scene.get(entityId as never)?.components[Tag.Transform];
    if (!tr || !cam || !target) return null;
    const [x, y, z, w] = tr.rotation as Q4;
    const d: V3 = [target.position[0] - tr.position[0], target.position[1] - tr.position[1], target.position[2] - tr.position[2]];
    const c = rotate([-x, -y, -z, w], d); // into camera space
    if (c[2] >= 0) return null;
    const rect = canvas.getBoundingClientRect();
    const t = Math.tan(cam.fovY / 2);
    const ndcX = c[0] / (-c[2] * t * (rect.width / rect.height));
    const ndcY = c[1] / (-c[2] * t);
    return { x: rect.left + ((ndcX + 1) / 2) * rect.width, y: rect.top + ((1 - ndcY) / 2) * rect.height };
  },
  transcript: () => [...transcript],
};

void boot();

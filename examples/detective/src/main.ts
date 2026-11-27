/**
 * The Wrenfield Observatory — browser shell.
 *
 * State flows one way: UI action → `act()` (rule engine) → new GameState →
 * dialogue (LLM restates the rule gist, guardrailed; offline shows the gist)
 * → memory recorded → UI re-rendered. The renderer only reads the scene.
 */
import { Scene, Tag } from "@mnemopay/augengine";
import { assetUrl } from "./asset-url.js";
import { createLiveRenderer, lookAtTransform, type WebGpuLiveRenderer } from "@mnemopay/augengine/runtime/live";
import {
  CASE_INTRO,
  EVIDENCE,
  METHODS,
  MOTIVES,
  OBJECTS,
  ROOMS,
  SUSPECTS,
  act,
  blankMemory,
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
  type RoomId,
  type SuspectId,
  type Utterance,
} from "@mnemopay/augengine-detective";
import { BrowserMemoryStore, playerId } from "./memory-store.js";
import { detectMode, speak, type DialogueMode } from "./llm-client.js";
import { CLOSEUP_CAMERA, SUSPECT_ORDER, BUST_HALF, buildHouseSpec, bustUrl, cameraId, headId, headPosition, objectEntityId, suspectEntityIds } from "./scene.js";
import { loadBust } from "./busts.js";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
const FORCE_OFFLINE = params.has("offline");

// ── state ──────────────────────────────────────────────────────────────
let state: GameState = newGame();
let memory: PlayerMemory = blankMemory("anon");
let mode: DialogueMode = "offline";
let sessionId = crypto.randomUUID();
let talkingTo: SuspectId | null = null;
let busy = false;
const store = new BrowserMemoryStore();
const pid = playerId();
/** Suspects already greeted with their memory line this session. */
const greeted = new Set<SuspectId>();
const transcript: { who: string; text: string; source: string }[] = [];

// ── renderer ───────────────────────────────────────────────────────────
const canvas = $<HTMLCanvasElement>("game");
let renderer: WebGpuLiveRenderer | null = null;
const scene = new Scene(buildHouseSpec());
/** Current portrait mood per suspect, set only from the rule engine's utterances. */
const moods: Record<SuspectId, Mood> = { mara: "calm", desmond: "calm", ines: "calm", teodor: "calm" };
const frameMs: number[] = [];

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
  const src = assetUrl(portraitPath(sid, mood));
  img.alt = `${SUSPECTS[sid].name}, ${mood}`;
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

function showRoomCamera(): void {
  if (state.room === "parlour" && talkingTo) {
    const head = headPosition(talkingTo);
    const eye: [number, number, number] = [head[0] + 0.15, head[1] + 0.12, head[2] + 1.6];
    const t = lookAtTransform(eye, [head[0], head[1] - 0.02, head[2]], [0, 1, 0]);
    scene.setLocalTransform(CLOSEUP_CAMERA, t.position, t.rotation, [1, 1, 1]);
    scene.activeCamera = CLOSEUP_CAMERA;
  } else {
    scene.activeCamera = cameraId(state.room);
  }
}

/** What the pointer is over in the 3D view (an object id or a suspect id), if anything. */
let hovered: { kind: "object"; id: string } | { kind: "suspect"; id: SuspectId } | null = null;

function highlightExamined(): void {
  for (const o of OBJECTS) {
    const e = scene.get(objectEntityId(o.id));
    const mesh = e?.components[Tag.Mesh];
    if (!mesh || mesh.material.kind !== "pbr") continue;
    // Found clues glow faintly so the room reads as searched; the hovered one glows brighter.
    const found = o.clue && state.notebook.includes(o.clue);
    const over = hovered?.kind === "object" && hovered.id === o.id;
    mesh.material.emissive = over ? [0.22, 0.17, 0.07] : found ? [0.07, 0.05, 0.015] : [0, 0, 0];
  }
  for (const sid of SUSPECT_ORDER) {
    const over = hovered?.kind === "suspect" && hovered.id === sid;
    for (const eid of suspectEntityIds(sid)) {
      const mesh = scene.get(eid)?.components[Tag.Mesh];
      if (mesh && mesh.material.kind === "pbr") mesh.material.emissive = over ? [0.09, 0.07, 0.03] : [0, 0, 0];
    }
  }
}

// ── click-to-interact in the 3D view ───────────────────────────────────
/** Pickable boxes for the current room: its clue objects, and in the parlour the suspects' busts, plinths and plates. */
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
  for (const o of OBJECTS.filter((x) => x.room === state.room)) add(objectEntityId(o.id), `object:${o.id}`, primitiveHalf);
  if (state.room === "parlour") {
    for (const sid of SUSPECT_ORDER) {
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
  const same = (hovered?.kind === next?.kind && hovered?.id === next?.id);
  hovered = next;
  canvas.style.cursor = next ? "pointer" : "default";
  if (next) {
    const name = next.kind === "object" ? OBJECTS.find((o) => o.id === next.id)?.name ?? "" : SUSPECTS[next.id].name;
    label.textContent = next.kind === "object" ? `Examine: ${name}` : `Question ${name}`;
    label.style.left = `${clientX + 14}px`;
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

async function bootRenderer(): Promise<void> {
  sizeCanvas();
  if (!("gpu" in navigator)) {
    toast("This browser has no WebGPU — the house won't render, but the case is fully playable.");
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
    if (params.get("busts") !== "0") {
      for (const sid of SUSPECT_ORDER) loadBust(r, bustUrl(sid)).catch((err) => console.warn(`[detective] ${err.message}`));
    }
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
  const list = $("room-list");
  list.innerHTML = "";
  for (const r of ROOMS) {
    const b = document.createElement("button");
    b.textContent = r.name;
    b.dataset.testid = `room-${r.id}`;
    b.className = r.id === state.room ? "active" : "";
    b.onclick = () => doAction({ type: "move", room: r.id });
    list.appendChild(b);
  }
  const room = ROOMS.find((r) => r.id === state.room)!;
  $("room-name").textContent = room.name;
  $("room-blurb").textContent = room.blurb;
  const objs = $("objects");
  objs.innerHTML = "";
  for (const o of OBJECTS.filter((x) => x.room === state.room)) {
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
  if (state.notebook.length === 0) {
    ul.innerHTML = '<li class="empty">Nothing yet. Search the rooms.</li>';
  }
  for (const e of state.notebook) {
    const li = document.createElement("li");
    li.className = e.startsWith("T_") ? "testimony" : "";
    li.dataset.testid = `nb-${e}`;
    li.innerHTML = `<span class="name"></span><br><span class="text"></span>`;
    (li.querySelector(".name") as HTMLElement).textContent = EVIDENCE[e].name;
    (li.querySelector(".text") as HTMLElement).textContent = EVIDENCE[e].text;
    ul.appendChild(li);
  }
  const sel = $<HTMLSelectElement>("present-select");
  const keep = sel.value;
  sel.innerHTML = '<option value="">Show evidence…</option>';
  for (const e of state.notebook) {
    const o = document.createElement("option");
    o.value = e;
    o.textContent = EVIDENCE[e].name;
    sel.appendChild(o);
  }
  if (state.notebook.includes(keep as EvidenceId)) sel.value = keep;
}

function renderSuspects(): void {
  const box = $("suspects");
  box.innerHTML = "";
  const inParlour = state.room === "parlour";
  for (const sid of SUSPECT_ORDER) {
    const s = SUSPECTS[sid];
    const b = document.createElement("button");
    b.className = "suspect-btn";
    b.innerHTML = `<img alt="" src="${assetUrl(portraitPath(sid, "calm"))}"><span></span>`;
    (b.querySelector("span") as HTMLElement).textContent = s.name.split(" ")[0]!;
    b.title = `${s.name} — ${s.role}`;
    b.setAttribute("aria-label", `${s.name}, ${s.role}`);
    b.dataset.testid = `suspect-${sid}`;
    if (talkingTo === sid) b.classList.add("active");
    b.disabled = !inParlour || busy;
    b.onclick = () => selectSuspect(sid);
    box.appendChild(b);
  }
  const can = inParlour && talkingTo !== null && !busy;
  $<HTMLInputElement>("ask-input").disabled = !can;
  $<HTMLButtonElement>("ask-send").disabled = !can;
  $<HTMLSelectElement>("present-select").disabled = !can;
  $<HTMLButtonElement>("present-send").disabled = !can;
  if (!inParlour) $("memory-note").textContent = "";
  showPortrait(inParlour ? talkingTo : null, talkingTo ? moods[talkingTo] : "calm");
}

function renderAll(): void {
  renderRooms();
  renderNotebook();
  renderSuspects();
  highlightExamined();
  showRoomCamera();
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
  $("memory-note").textContent = line ? `${SUSPECTS[sid].name.split(" ")[0]} remembers you: “${line}”` : "";
  renderAll();
  setSpeech(SUSPECTS[sid].name, "…", "");
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
    if (validateLine(partial, ctx).ok) setSpeech(SUSPECTS[sid].name, partial, "");
  });
  setSpeech(SUSPECTS[sid].name, spoken.text, spoken.source === "llm" ? "· voiced by the model, checked by the rules" : spoken.rejected.length ? "· model line rejected by guardrail" : "");
  transcript.push({ who: SUSPECTS[sid].name, text: spoken.text, source: spoken.source });
  if (u.cracked.length) toast(`${SUSPECTS[sid].name.split(" ")[0]}'s story just changed.`);
  for (const t of u.revealed) toast(`New testimony in your notebook: ${EVIDENCE[t].name}`);
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
  if (a.type === "move" && a.room !== "parlour") talkingTo = null;
  if (a.type === "examine") {
    setSpeech("Notebook", r.message, "");
    if (r.clue) toast(`Clue found: ${EVIDENCE[r.clue].name}`);
  }
  if (a.type === "move") setSpeech(ROOMS.find((x) => x.id === a.room)!.name, r.message, "");
  renderAll();
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
  fill($<HTMLSelectElement>("acc-suspect"), SUSPECT_ORDER.map((s) => [s, SUSPECTS[s].name]));
  fill($<HTMLSelectElement>("acc-method"), Object.entries(METHODS));
  fill($<HTMLSelectElement>("acc-motive"), Object.entries(MOTIVES));
  const list = $("acc-evidence");
  list.innerHTML = "";
  for (const e of state.notebook) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${e}" data-testid="acc-ev-${e}"> <span></span>`;
    (label.querySelector("span") as HTMLElement).textContent = EVIDENCE[e].name;
    list.appendChild(label);
  }
  $("acc-error").textContent = "";
  $("accuse").classList.add("show");
}

function confirmAccuse(): void {
  const evidence = [...$("acc-evidence").querySelectorAll<HTMLInputElement>("input:checked")].map((i) => i.value as EvidenceId);
  const r = act(state, {
    type: "accuse",
    suspect: $<HTMLSelectElement>("acc-suspect").value as SuspectId,
    method: $<HTMLSelectElement>("acc-method").value as MethodId,
    motive: $<HTMLSelectElement>("acc-motive").value as MotiveId,
    evidence,
  });
  if (!r.ok) {
    $("acc-error").textContent = r.error;
    return;
  }
  state = r.state;
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
  mem.textContent = `The household will remember this. Cases solved: ${memory.cases.solved} · failed: ${memory.cases.failed}.`;
  body.appendChild(mem);
  $("result").classList.add("show");
}

async function newCase(): Promise<void> {
  state = newGame();
  sessionId = crypto.randomUUID();
  memory = startSession(memory, sessionId);
  void store.save(memory);
  greeted.clear();
  talkingTo = null;
  for (const sid of SUSPECT_ORDER) setMood(sid, "calm");
  transcript.length = 0;
  $("result").classList.remove("show");
  setSpeech("Magistrate", "You have the night. Start wherever you like.", "");
  renderAll();
}

// ── boot ───────────────────────────────────────────────────────────────
async function boot(): Promise<void> {
  memory = (await store.load(pid)) ?? blankMemory(pid);
  memory = startSession(memory, sessionId);
  void store.save(memory);
  mode = await detectMode(FORCE_OFFLINE);
  $("intro-text").textContent = CASE_INTRO;
  const prior = memory.sessions - 1;
  $("intro-memory").textContent =
    prior > 0 ? `You have been here before (${prior} earlier visit${prior === 1 ? "" : "s"}; solved ${memory.cases.solved}, failed ${memory.cases.failed}). The household remembers.` : "";
  $("mode-note").textContent =
    mode === "llm"
      ? "Suspects speak through a language model, but every fact they say is decided by the case rules and checked before you see it."
      : "Offline dialogue: suspects answer with their scripted lines. Every fact is the same as in model mode.";
  $("start").onclick = () => {
    $("intro").classList.remove("show");
  };
  $("accuse-open").onclick = openAccuse;
  $("acc-cancel").onclick = () => $("accuse").classList.remove("show");
  $("acc-confirm").onclick = confirmAccuse;
  $("new-case").onclick = () => void newCase();
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
    const ev = $<HTMLSelectElement>("present-select").value as EvidenceId;
    if (!ev || !talkingTo) return;
    void converse({ type: "present", suspect: talkingTo, evidence: ev }, `The detective shows you: ${EVIDENCE[ev].name}. ${EVIDENCE[ev].text}`);
  };
  window.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
    const n = Number(e.key);
    if (n >= 1 && n <= ROOMS.length) doAction({ type: "move", room: ROOMS[n - 1]!.id as RoomId });
  });
  wirePicking();
  renderAll();
  await bootRenderer();
}

// Agent / test hook: read-only view of the session.
(window as unknown as Record<string, unknown>).__detective = {
  view: () => publicView(state),
  mode: () => mode,
  perf: () => ({ ...perfStats(), renderer: renderer?.getFrameStats() ?? null }),
  memory: () => memory,
  moods: () => ({ ...moods }),
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

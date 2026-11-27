/**
 * Rule engine. Pure functions over plain data: the browser game, the MCP
 * server and the tests all drive the same `act()`.
 *
 * The rules decide every fact a suspect conveys (the "gist"). Dialogue layers
 * may only rephrase that gist; they never change state.
 */

import {
  EVIDENCE,
  OBJECTS,
  ROOMS,
  SUSPECTS,
  TRUTH,
  type ClueId,
  type EvidenceId,
  type LiePolicy,
  type MethodId,
  type MotiveId,
  type RoomId,
  type SuspectId,
  type TestimonyId,
  type Topic,
} from "./case.js";

export const SUSPECT_IDS: SuspectId[] = ["mara", "desmond", "ines", "teodor"];

export interface LogEntry {
  turn: number;
  kind: "move" | "examine" | "ask" | "present" | "accuse" | "system";
  text: string;
}

export type Verdict = "solved" | "weak" | "wrong";

export interface AccusationResult {
  verdict: Verdict;
  suspect: SuspectId;
  method: MethodId;
  motive: MotiveId;
  evidence: EvidenceId[];
  /** Which links the evidence covered for the accused. */
  links: { suspect: boolean; method: boolean; motive: boolean };
  headline: string;
  explanation: string[];
}

export interface GameState {
  turn: number;
  room: RoomId;
  examined: string[];
  notebook: EvidenceId[];
  presented: Record<SuspectId, EvidenceId[]>;
  /** Lie topics cracked per suspect. */
  cracked: Record<SuspectId, Topic[]>;
  asked: Record<SuspectId, Topic[]>;
  accusation: AccusationResult | null;
  log: LogEntry[];
}

export type Action =
  | { type: "move"; room: RoomId }
  | { type: "examine"; objectId: string }
  | { type: "ask"; suspect: SuspectId; text: string }
  | { type: "present"; suspect: SuspectId; evidence: EvidenceId }
  | { type: "accuse"; suspect: SuspectId; method: MethodId; motive: MotiveId; evidence: EvidenceId[] };

/** What a suspect conveys, decided by the rules. */
export interface Utterance {
  suspect: SuspectId;
  topic: Topic;
  /** The facts to convey. Offline mode shows this verbatim. */
  gist: string;
  /** Testimony that entered the notebook because of this answer. */
  revealed: TestimonyId[];
  /** Lie topics that cracked on this turn. */
  cracked: Topic[];
  /** True when the gist is a scripted lie. */
  lying: boolean;
}

export type ActResult =
  | { ok: true; state: GameState; message: string; clue?: ClueId; utterance?: Utterance; accusation?: AccusationResult }
  | { ok: false; state: GameState; error: string };

function emptyPerSuspect<T>(): Record<SuspectId, T[]> {
  return { mara: [], desmond: [], ines: [], teodor: [] };
}

export function newGame(): GameState {
  return {
    turn: 0,
    room: "parlour",
    examined: [],
    notebook: [],
    presented: emptyPerSuspect<EvidenceId>(),
    cracked: emptyPerSuspect<Topic>(),
    asked: emptyPerSuspect<Topic>(),
    accusation: null,
    log: [],
  };
}

function clone(s: GameState): GameState {
  return structuredClone(s);
}

export function isSuspectId(v: unknown): v is SuspectId {
  return typeof v === "string" && (SUSPECT_IDS as string[]).includes(v);
}

export function isEvidenceId(v: unknown): v is EvidenceId {
  return typeof v === "string" && Object.prototype.hasOwnProperty.call(EVIDENCE, v);
}

export function isRoomId(v: unknown): v is RoomId {
  return typeof v === "string" && ROOMS.some((r) => r.id === v);
}

export function objectsIn(room: RoomId) {
  return OBJECTS.filter((o) => o.room === room);
}

/** Map free text to a topic. Deterministic keyword rules, no model. */
export function classifyQuestion(text: string, asking: SuspectId, notebook?: readonly EvidenceId[]): Topic {
  const t = ` ${text.toLowerCase().replace(/[^a-z0-9:.' ]+/g, " ")} `;
  const has = (...words: string[]) => words.some((w) => t.includes(w));
  // Evidence first: the most specific signal. With a notebook, only evidence
  // already found counts (or something this suspect lies about), so a casual
  // "tea" in "where were you when the tea was made" doesn't become a question
  // about an unfound clue.
  for (const ev of Object.values(EVIDENCE)) {
    if (notebook && !notebook.includes(ev.id) && !lieFor(asking, `clue:${ev.id}`)) continue;
    if (ev.keywords.some((k) => t.includes(` ${k}`) || t.includes(`${k} `))) {
      if (ev.id.startsWith("C_")) return `clue:${ev.id}`;
    }
  }
  for (const id of SUSPECT_IDS) {
    if (id === asking) continue;
    const s = SUSPECTS[id];
    const [first, last] = s.name.toLowerCase().split(" ");
    if (has(` ${first}`, ` ${last}`, ` ${id} `)) return `suspect:${id}`;
  }
  if (has("kitchen")) return "room:kitchen";
  if (has("dome", "telescope", "upstairs")) return "room:dome";
  if (has("library")) return "room:library";
  if (has("conservatory", "garden", "orchid")) return "room:conservatory";
  if (has("where were", "alibi", "what were you doing", "where did you", "your evening", "last night", "where have you")) return "alibi";
  if (has("when", "what time", "o'clock", " time ")) return "time";
  if (has("why", "motive", "who would", "who killed", "who did", "hate", "enemies")) return "motive";
  if (has("vance", "professor", "victim", "uncle", "aurel", "employer")) return "victim";
  if (has(" hello", " hi ", "good evening", "introduce", "who are you")) return "greeting";
  return "unknown";
}

function lieFor(suspect: SuspectId, topic: Topic): LiePolicy | undefined {
  return SUSPECTS[suspect].lies.find((l) => l.topic === topic);
}

/** Resolve the gist for a topic given current cracked state. Pure. */
export function gistFor(
  state: GameState,
  suspect: SuspectId,
  topic: Topic,
): { gist: string; reveals: TestimonyId | undefined; lying: boolean } {
  const s = SUSPECTS[suspect];
  const lie = lieFor(suspect, topic);
  // Evidence the detective hasn't found: suspects won't volunteer it. A liar
  // keeps to the lie (which reveals nothing new); everyone else deflects.
  if (topic.startsWith("clue:") && !state.notebook.includes(topic.slice(5) as EvidenceId)) {
    if (lie) return { gist: lie.lie, reveals: undefined, lying: true };
    return { gist: s.shrug, reveals: undefined, lying: false };
  }
  if (lie) {
    const cracked = state.cracked[suspect].includes(topic);
    return { gist: cracked ? lie.truth : lie.lie, reveals: undefined, lying: !cracked };
  }
  const ans = s.answers[topic];
  if (ans) return { gist: ans.line, reveals: ans.reveals, lying: false };
  if (topic === "unknown") {
    return { gist: "I'm not sure what you're asking, detective. Ask me about tonight.", reveals: undefined, lying: false };
  }
  return { gist: s.shrug, reveals: undefined, lying: false };
}

function crackable(state: GameState, suspect: SuspectId): LiePolicy[] {
  const shown = new Set(state.presented[suspect]);
  return SUSPECTS[suspect].lies.filter(
    (l) => !state.cracked[suspect].includes(l.topic) && l.crackWhen.some((set) => set.every((e) => shown.has(e))),
  );
}

function addEvidence(state: GameState, id: EvidenceId): boolean {
  if (state.notebook.includes(id)) return false;
  state.notebook.push(id);
  return true;
}

export function scoreAccusation(
  suspect: SuspectId,
  method: MethodId,
  motive: MotiveId,
  evidence: EvidenceId[],
): AccusationResult {
  const evs = evidence.map((e) => EVIDENCE[e]);
  const links = {
    suspect: evs.some((e) => e.supports.includes("suspect") && e.pointsAt === suspect),
    method: evs.some((e) => e.supports.includes("method")),
    motive: evs.some((e) => e.supports.includes("motive") && (e.pointsAt === undefined || e.pointsAt === suspect)),
  };
  const right = suspect === TRUTH.killer && method === TRUTH.method && motive === TRUTH.motive;
  const covered = links.suspect && links.method && links.motive;
  const name = SUSPECTS[suspect].name;
  let verdict: Verdict;
  let headline: string;
  if (right && covered) {
    verdict = "solved";
    headline = `Case closed. ${name} poisoned Professor Vance's tea to keep the inheritance.`;
  } else if (suspect === TRUTH.killer) {
    verdict = "weak";
    headline = `You named the right person — but the magistrate releases ${name} for want of a complete case.`;
  } else {
    verdict = "wrong";
    headline = `${name} is innocent. The real killer walks free tonight.`;
  }
  return { verdict, suspect, method, motive, evidence: [...evidence], links, headline, explanation: [...TRUTH.explanation] };
}

export function act(prev: GameState, action: Action): ActResult {
  if (prev.accusation) {
    return { ok: false, state: prev, error: "The case is over — you have made your accusation. Start a new case to play again." };
  }
  const state = clone(prev);
  state.turn += 1;
  const fail = (error: string): ActResult => ({ ok: false, state: prev, error });

  switch (action.type) {
    case "move": {
      if (!isRoomId(action.room)) return fail(`No room called "${String(action.room)}".`);
      state.room = action.room;
      const room = ROOMS.find((r) => r.id === action.room)!;
      state.log.push({ turn: state.turn, kind: "move", text: `You go to the ${room.name}.` });
      return { ok: true, state, message: `${room.name}. ${room.blurb}` };
    }
    case "examine": {
      const obj = OBJECTS.find((o) => o.id === action.objectId);
      if (!obj) return fail(`Nothing called "${action.objectId}" here.`);
      if (obj.room !== state.room) return fail(`The ${obj.name.toLowerCase()} is not in this room.`);
      if (!state.examined.includes(obj.id)) state.examined.push(obj.id);
      let clue: ClueId | undefined;
      if (obj.clue && addEvidence(state, obj.clue)) clue = obj.clue;
      state.log.push({ turn: state.turn, kind: "examine", text: `${obj.name}: ${obj.text}` });
      return clue
        ? { ok: true, state, message: obj.text, clue }
        : { ok: true, state, message: obj.text };
    }
    case "ask": {
      if (!isSuspectId(action.suspect)) return fail(`No suspect called "${String(action.suspect)}".`);
      if (state.room !== "parlour") return fail("The household is in the parlour. Go there to question them.");
      const text = String(action.text ?? "").slice(0, 400);
      if (text.trim().length === 0) return fail("Ask a question.");
      const topic = classifyQuestion(text, action.suspect, state.notebook);
      const { gist, reveals, lying } = gistFor(state, action.suspect, topic);
      const revealed: TestimonyId[] = [];
      if (reveals && addEvidence(state, reveals)) revealed.push(reveals);
      if (!state.asked[action.suspect].includes(topic)) state.asked[action.suspect].push(topic);
      state.log.push({ turn: state.turn, kind: "ask", text: `You ask ${SUSPECTS[action.suspect].name}: "${text}"` });
      const utterance: Utterance = { suspect: action.suspect, topic, gist, revealed, cracked: [], lying };
      return { ok: true, state, message: gist, utterance };
    }
    case "present": {
      if (!isSuspectId(action.suspect)) return fail(`No suspect called "${String(action.suspect)}".`);
      if (state.room !== "parlour") return fail("The household is in the parlour. Go there to question them.");
      if (!isEvidenceId(action.evidence)) return fail(`No evidence called "${String(action.evidence)}".`);
      if (!state.notebook.includes(action.evidence)) return fail("You haven't found that yet.");
      const sid = action.suspect;
      if (!state.presented[sid].includes(action.evidence)) state.presented[sid].push(action.evidence);
      const cracking = crackable(state, sid);
      const topic: Topic = `clue:${action.evidence}`;
      const revealed: TestimonyId[] = [];
      let gist: string;
      let lying = false;
      if (cracking.length > 0) {
        for (const l of cracking) state.cracked[sid].push(l.topic);
        gist = cracking.map((l) => l.crackLine).join(" ");
      } else {
        const g = gistFor(state, sid, topic);
        gist = g.gist;
        lying = g.lying;
        if (g.reveals && addEvidence(state, g.reveals)) revealed.push(g.reveals);
      }
      state.log.push({ turn: state.turn, kind: "present", text: `You show ${SUSPECTS[sid].name}: ${EVIDENCE[action.evidence].name}.` });
      const utterance: Utterance = { suspect: sid, topic, gist, revealed, cracked: cracking.map((l) => l.topic), lying };
      return { ok: true, state, message: gist, utterance };
    }
    case "accuse": {
      if (!isSuspectId(action.suspect)) return fail(`No suspect called "${String(action.suspect)}".`);
      const evidence = [...new Set(action.evidence ?? [])];
      if (evidence.length === 0) return fail("An accusation needs evidence. Pick at least one item from your notebook.");
      if (evidence.length > 4) return fail("Pick at most four pieces of evidence.");
      for (const e of evidence) {
        if (!isEvidenceId(e)) return fail(`No evidence called "${String(e)}".`);
        if (!state.notebook.includes(e)) return fail(`You haven't found "${EVIDENCE[e].name}".`);
      }
      const result = scoreAccusation(action.suspect, action.method, action.motive, evidence);
      state.accusation = result;
      state.log.push({ turn: state.turn, kind: "accuse", text: `You accuse ${SUSPECTS[action.suspect].name}. ${result.headline}` });
      return { ok: true, state, message: result.headline, accusation: result };
    }
  }
}

/** Public view for agents/MCP: never includes the truth until the case is over. */
export function publicView(state: GameState, opts: { revealTruth?: boolean } = {}) {
  const room = ROOMS.find((r) => r.id === state.room)!;
  return {
    turn: state.turn,
    room: { id: room.id, name: room.name, blurb: room.blurb },
    rooms: ROOMS.map((r) => ({ id: r.id, name: r.name })),
    objectsHere: objectsIn(state.room).map((o) => ({ id: o.id, name: o.name, examined: state.examined.includes(o.id) })),
    suspects: SUSPECT_IDS.map((id) => ({ id, name: SUSPECTS[id].name, role: SUSPECTS[id].role })),
    notebook: state.notebook.map((e) => ({ id: e, name: EVIDENCE[e].name, text: EVIDENCE[e].text })),
    presented: state.presented,
    cracked: state.cracked,
    accusation: state.accusation,
    ...(opts.revealTruth || state.accusation ? { truth: TRUTH } : {}),
  };
}

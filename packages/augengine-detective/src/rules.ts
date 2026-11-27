/**
 * Rule engine. Pure functions over plain data: the browser game, the MCP
 * server and the tests all drive the same `act()`.
 *
 * The rules decide every fact a suspect conveys (the "gist"). Dialogue layers
 * may only rephrase that gist; they never change state.
 *
 * A game state carries its case id; every function reads the case from it, so
 * one engine plays every registered case.
 */

import type {
  CaseDef,
  ClueId,
  EvidenceId,
  LiePolicy,
  MethodId,
  MotiveId,
  RoomId,
  Suspect,
  SuspectId,
  TestimonyId,
  Topic,
} from "./case.js";
import { DEFAULT_CASE_ID, caseById } from "./cases.js";

/** Wrenfield's suspects (kept for older callers; prefer `caseOf(state).suspectOrder`). */
export const SUSPECT_IDS: SuspectId[] = caseById(DEFAULT_CASE_ID).suspectOrder;

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
  /** Which case this game plays (older saved states without it are Wrenfield). */
  caseId: string;
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
  /** Case the utterance belongs to (absent = the first case). */
  caseId?: string;
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

const firstCase = () => caseById(DEFAULT_CASE_ID);

function emptyPerSuspect<T>(c: CaseDef): Record<SuspectId, T[]> {
  const out: Record<SuspectId, T[]> = {};
  for (const id of c.suspectOrder) out[id] = [];
  return out;
}

/** The case a game state belongs to. */
export function caseOf(state: { caseId?: string | undefined }): CaseDef {
  return caseById(state.caseId);
}

function suspectOf(c: CaseDef, id: SuspectId): Suspect {
  const s = c.suspects[id];
  if (!s) throw new Error(`no suspect ${id} in case ${c.id}`);
  return s;
}

export function newGame(caseId: string = DEFAULT_CASE_ID): GameState {
  const c = caseById(caseId);
  return {
    caseId: c.id,
    turn: 0,
    room: c.interviewRoom,
    examined: [],
    notebook: [],
    presented: emptyPerSuspect<EvidenceId>(c),
    cracked: emptyPerSuspect<Topic>(c),
    asked: emptyPerSuspect<Topic>(c),
    accusation: null,
    log: [],
  };
}

function clone(s: GameState): GameState {
  const c = structuredClone(s);
  if (!c.caseId) c.caseId = DEFAULT_CASE_ID;
  return c;
}

export function isSuspectId(v: unknown, c: CaseDef = firstCase()): v is SuspectId {
  return typeof v === "string" && c.suspectOrder.includes(v);
}

export function isEvidenceId(v: unknown, c: CaseDef = firstCase()): v is EvidenceId {
  return typeof v === "string" && Object.prototype.hasOwnProperty.call(c.evidence, v);
}

export function isRoomId(v: unknown, c: CaseDef = firstCase()): v is RoomId {
  return typeof v === "string" && c.rooms.some((r) => r.id === v);
}

export function objectsIn(room: RoomId, c: CaseDef = firstCase()) {
  return c.objects.filter((o) => o.room === room);
}

/** Map free text to a topic. Deterministic keyword rules, no model. */
export function classifyQuestion(text: string, asking: SuspectId, notebook?: readonly EvidenceId[], c: CaseDef = firstCase()): Topic {
  const t = ` ${text.toLowerCase().replace(/[^a-z0-9:.' ]+/g, " ")} `;
  const has = (...words: string[]) => words.some((w) => t.includes(w));
  // Evidence first: the most specific signal. With a notebook, only evidence
  // already found counts (or something this suspect lies about), so a casual
  // "tea" in "where were you when the tea was made" doesn't become a question
  // about an unfound clue.
  for (const ev of Object.values(c.evidence)) {
    if (notebook && !notebook.includes(ev.id) && !lieFor(c, asking, `clue:${ev.id}`)) continue;
    if (ev.keywords.some((k) => t.includes(` ${k}`) || t.includes(`${k} `))) {
      if (ev.id.startsWith("C_")) return `clue:${ev.id}`;
    }
  }
  for (const id of c.suspectOrder) {
    if (id === asking) continue;
    const s = suspectOf(c, id);
    const [first, last] = s.name.toLowerCase().split(" ");
    if (has(` ${first}`, ` ${last}`, ` ${id} `)) return `suspect:${id}`;
  }
  for (const [room, words] of c.roomKeywords) if (has(...words)) return `room:${room}`;
  if (has("where were", "alibi", "what were you doing", "where did you", "your evening", "last night", "where have you")) return "alibi";
  if (has("when", "what time", "o'clock", " time ")) return "time";
  if (has("why", "motive", "who would", "who killed", "who did", "hate", "enemies")) return "motive";
  if (has(...c.victim.keywords)) return "victim";
  if (has(" hello", " hi ", "good evening", "introduce", "who are you")) return "greeting";
  return "unknown";
}

function lieFor(c: CaseDef, suspect: SuspectId, topic: Topic): LiePolicy | undefined {
  return c.suspects[suspect]?.lies.find((l) => l.topic === topic);
}

/** Resolve the gist for a topic given current cracked state. Pure. */
export function gistFor(
  state: GameState,
  suspect: SuspectId,
  topic: Topic,
): { gist: string; reveals: TestimonyId | undefined; lying: boolean } {
  const c = caseOf(state);
  const s = suspectOf(c, suspect);
  const lie = lieFor(c, suspect, topic);
  // Evidence the detective hasn't found: suspects won't volunteer it. A liar
  // keeps to the lie (which reveals nothing new); everyone else deflects.
  if (topic.startsWith("clue:") && !state.notebook.includes(topic.slice(5) as EvidenceId)) {
    if (lie) return { gist: lie.lie, reveals: undefined, lying: true };
    return { gist: s.shrug, reveals: undefined, lying: false };
  }
  if (lie) {
    const cracked = (state.cracked[suspect] ?? []).includes(topic);
    return { gist: cracked ? lie.truth : lie.lie, reveals: undefined, lying: !cracked };
  }
  const ans = s.answers[topic];
  if (ans) return { gist: ans.line, reveals: ans.reveals, lying: false };
  if (topic === "unknown") return { gist: c.unknownLine, reveals: undefined, lying: false };
  return { gist: s.shrug, reveals: undefined, lying: false };
}

function crackable(state: GameState, suspect: SuspectId): LiePolicy[] {
  const shown = new Set(state.presented[suspect]);
  const cracked = state.cracked[suspect] ?? [];
  return suspectOf(caseOf(state), suspect).lies.filter(
    (l) => !cracked.includes(l.topic) && l.crackWhen.some((set) => set.every((e) => shown.has(e))),
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
  c: CaseDef = firstCase(),
): AccusationResult {
  const truth = c.truth;
  const evs = evidence.map((e) => c.evidence[e]).filter((e): e is NonNullable<typeof e> => !!e);
  const links = {
    suspect: evs.some((e) => e.supports.includes("suspect") && e.pointsAt === suspect),
    method: evs.some((e) => e.supports.includes("method")),
    motive: evs.some((e) => e.supports.includes("motive") && (e.pointsAt === undefined || e.pointsAt === suspect)),
  };
  const right = suspect === truth.killer && method === truth.method && motive === truth.motive;
  const covered = links.suspect && links.method && links.motive;
  const name = suspectOf(c, suspect).name;
  let verdict: Verdict;
  let headline: string;
  if (right && covered) {
    verdict = "solved";
    headline = truth.solvedHeadline;
  } else if (suspect === truth.killer) {
    verdict = "weak";
    headline = `You named the right person — but ${c.authority} releases ${name} for want of a complete case.`;
  } else {
    verdict = "wrong";
    headline = `${name} is innocent. The real killer walks free tonight.`;
  }
  return { verdict, suspect, method, motive, evidence: [...evidence], links, headline, explanation: [...truth.explanation] };
}

export function act(prev: GameState, action: Action): ActResult {
  if (prev.accusation) {
    return { ok: false, state: prev, error: "The case is over — you have made your accusation. Start a new case to play again." };
  }
  const state = clone(prev);
  const c = caseOf(state);
  state.turn += 1;
  const fail = (error: string): ActResult => ({ ok: false, state: prev, error });

  switch (action.type) {
    case "move": {
      if (!isRoomId(action.room, c)) return fail(`No room called "${String(action.room)}".`);
      state.room = action.room;
      const room = c.rooms.find((r) => r.id === action.room)!;
      state.log.push({ turn: state.turn, kind: "move", text: `You go to the ${room.name}.` });
      return { ok: true, state, message: `${room.name}. ${room.blurb}` };
    }
    case "examine": {
      const obj = c.objects.find((o) => o.id === action.objectId);
      if (!obj) return fail(`Nothing called "${action.objectId}" here.`);
      if (obj.room !== state.room) return fail(`The ${obj.name.toLowerCase()} is not in this room.`);
      if (!state.examined.includes(obj.id)) state.examined.push(obj.id);
      let clue: ClueId | undefined;
      if (obj.clue && addEvidence(state, obj.clue)) clue = obj.clue;
      state.log.push({ turn: state.turn, kind: "examine", text: `${obj.name}: ${obj.text}` });
      return clue ? { ok: true, state, message: obj.text, clue } : { ok: true, state, message: obj.text };
    }
    case "ask": {
      if (!isSuspectId(action.suspect, c)) return fail(`No suspect called "${String(action.suspect)}".`);
      if (state.room !== c.interviewRoom) return fail(c.interviewHint);
      const text = String(action.text ?? "").slice(0, 400);
      if (text.trim().length === 0) return fail("Ask a question.");
      const topic = classifyQuestion(text, action.suspect, state.notebook, c);
      const { gist, reveals, lying } = gistFor(state, action.suspect, topic);
      const revealed: TestimonyId[] = [];
      if (reveals && addEvidence(state, reveals)) revealed.push(reveals);
      const asked = (state.asked[action.suspect] ??= []);
      if (!asked.includes(topic)) asked.push(topic);
      state.log.push({ turn: state.turn, kind: "ask", text: `You ask ${suspectOf(c, action.suspect).name}: "${text}"` });
      const utterance: Utterance = { caseId: c.id, suspect: action.suspect, topic, gist, revealed, cracked: [], lying };
      return { ok: true, state, message: gist, utterance };
    }
    case "present": {
      if (!isSuspectId(action.suspect, c)) return fail(`No suspect called "${String(action.suspect)}".`);
      if (state.room !== c.interviewRoom) return fail(c.interviewHint);
      if (!isEvidenceId(action.evidence, c)) return fail(`No evidence called "${String(action.evidence)}".`);
      if (!state.notebook.includes(action.evidence)) return fail("You haven't found that yet.");
      const sid = action.suspect;
      const presented = (state.presented[sid] ??= []);
      if (!presented.includes(action.evidence)) presented.push(action.evidence);
      const cracking = crackable(state, sid);
      const topic: Topic = `clue:${action.evidence}`;
      const revealed: TestimonyId[] = [];
      let gist: string;
      let lying = false;
      if (cracking.length > 0) {
        const cr = (state.cracked[sid] ??= []);
        for (const l of cracking) cr.push(l.topic);
        gist = cracking.map((l) => l.crackLine).join(" ");
      } else {
        const g = gistFor(state, sid, topic);
        gist = g.gist;
        lying = g.lying;
        if (g.reveals && addEvidence(state, g.reveals)) revealed.push(g.reveals);
      }
      state.log.push({ turn: state.turn, kind: "present", text: `You show ${suspectOf(c, sid).name}: ${c.evidence[action.evidence]!.name}.` });
      const utterance: Utterance = { caseId: c.id, suspect: sid, topic, gist, revealed, cracked: cracking.map((l) => l.topic), lying };
      return { ok: true, state, message: gist, utterance };
    }
    case "accuse": {
      if (!isSuspectId(action.suspect, c)) return fail(`No suspect called "${String(action.suspect)}".`);
      if (!Object.prototype.hasOwnProperty.call(c.methods, action.method)) return fail(`No method called "${String(action.method)}".`);
      if (!Object.prototype.hasOwnProperty.call(c.motives, action.motive)) return fail(`No motive called "${String(action.motive)}".`);
      const evidence = [...new Set(action.evidence ?? [])];
      if (evidence.length === 0) return fail("An accusation needs evidence. Pick at least one item from your notebook.");
      if (evidence.length > 4) return fail("Pick at most four pieces of evidence.");
      for (const e of evidence) {
        if (!isEvidenceId(e, c)) return fail(`No evidence called "${String(e)}".`);
        if (!state.notebook.includes(e)) return fail(`You haven't found "${c.evidence[e]!.name}".`);
      }
      const result = scoreAccusation(action.suspect, action.method, action.motive, evidence, c);
      state.accusation = result;
      state.log.push({ turn: state.turn, kind: "accuse", text: `You accuse ${suspectOf(c, action.suspect).name}. ${result.headline}` });
      return { ok: true, state, message: result.headline, accusation: result };
    }
  }
}

/** Public view for agents/MCP: never includes the truth until the case is over. */
export function publicView(state: GameState, opts: { revealTruth?: boolean } = {}) {
  const c = caseOf(state);
  const room = c.rooms.find((r) => r.id === state.room)!;
  return {
    case: { id: c.id, title: c.title },
    turn: state.turn,
    room: { id: room.id, name: room.name, blurb: room.blurb },
    rooms: c.rooms.map((r) => ({ id: r.id, name: r.name })),
    objectsHere: objectsIn(state.room, c).map((o) => ({ id: o.id, name: o.name, examined: state.examined.includes(o.id) })),
    suspects: c.suspectOrder.map((id) => ({ id, name: suspectOf(c, id).name, role: suspectOf(c, id).role })),
    notebook: state.notebook.map((e) => ({ id: e, name: c.evidence[e]?.name ?? e, text: c.evidence[e]?.text ?? "" })),
    presented: state.presented,
    cracked: state.cracked,
    accusation: state.accusation,
    ...(opts.revealTruth || state.accusation ? { truth: c.truth } : {}),
  };
}

/** Spoiler-free numbers for the verdict screen and the share card. */
export function caseStats(state: GameState) {
  const c = caseOf(state);
  const liesTotal = c.suspectOrder.reduce((n, id) => n + (c.suspects[id]?.lies.length ?? 0), 0);
  const liesCracked = c.suspectOrder.reduce((n, id) => n + (state.cracked[id]?.length ?? 0), 0);
  return {
    turns: state.turn,
    cluesFound: state.notebook.filter((e) => e.startsWith("C_")).length,
    cluesTotal: Object.keys(c.evidence).filter((k) => k.startsWith("C_")).length,
    testimonies: state.notebook.filter((e) => e.startsWith("T_")).length,
    liesCracked,
    liesTotal,
  };
}

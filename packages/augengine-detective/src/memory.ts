/**
 * Per-player suspect memory. Each suspect remembers what this player asked,
 * showed and accused across sessions. Storage is pluggable: IndexedDB /
 * localStorage in the browser, a JSON file for the MCP server, in-memory for
 * tests.
 *
 * Memory is per case: a PlayerMemory belongs to one case (`caseId`). Records
 * saved before cases existed have no caseId and are the first case's.
 */

import type { EvidenceId, SuspectId, Topic } from "./case.js";
import { DEFAULT_CASE_ID, caseById, isCaseId } from "./cases.js";
import type { AccusationResult } from "./rules.js";

export interface SuspectMemory {
  /** Sessions in which this player questioned this suspect. */
  sessions: number;
  lastSessionId: string | null;
  topics: Topic[];
  shown: EvidenceId[];
  /** Times the player accused this suspect. */
  accusedThem: number;
  /** Most recent accusation outcome where this suspect was named. */
  lastAccusedVerdict: AccusationResult["verdict"] | null;
  crackedTopics: Topic[];
}

export interface PlayerMemory {
  playerId: string;
  /** The case this memory belongs to. */
  caseId: string;
  sessions: number;
  currentSessionId: string | null;
  cases: { solved: number; failed: number };
  /** Best verdict this player has reached in this case. */
  best: AccusationResult["verdict"] | null;
  suspects: Record<SuspectId, SuspectMemory>;
}

export interface MemoryStore {
  load(playerId: string): Promise<PlayerMemory | null>;
  save(memory: PlayerMemory): Promise<void>;
}

function blankSuspect(): SuspectMemory {
  return { sessions: 0, lastSessionId: null, topics: [], shown: [], accusedThem: 0, lastAccusedVerdict: null, crackedTopics: [] };
}

export function blankMemory(playerId: string, caseId: string = DEFAULT_CASE_ID): PlayerMemory {
  const c = caseById(caseId);
  const suspects: Record<SuspectId, SuspectMemory> = {};
  for (const id of c.suspectOrder) suspects[id] = blankSuspect();
  return { playerId, caseId: c.id, sessions: 0, currentSessionId: null, cases: { solved: 0, failed: 0 }, best: null, suspects };
}

const VERDICTS = ["solved", "weak", "wrong"] as const;
const isVerdict = (v: unknown): v is AccusationResult["verdict"] => (VERDICTS as readonly unknown[]).includes(v);
const RANK: Record<AccusationResult["verdict"], number> = { wrong: 0, weak: 1, solved: 2 };

/** Normalise anything loaded from storage (older shapes, tampering). */
export function coerceMemory(playerId: string, raw: unknown, caseId?: string): PlayerMemory {
  const rawCase = raw && typeof raw === "object" ? (raw as { caseId?: unknown }).caseId : undefined;
  const id = caseId ?? (isCaseId(rawCase) ? rawCase : DEFAULT_CASE_ID);
  const base = blankMemory(playerId, id);
  if (!raw || typeof raw !== "object") return base;
  const c = caseById(id);
  const r = raw as Partial<PlayerMemory>;
  base.sessions = Number.isFinite(r.sessions) ? Number(r.sessions) : 0;
  base.currentSessionId = typeof r.currentSessionId === "string" ? r.currentSessionId : null;
  if (r.cases && typeof r.cases === "object") {
    base.cases.solved = Number(r.cases.solved) || 0;
    base.cases.failed = Number(r.cases.failed) || 0;
  }
  base.best = isVerdict(r.best) ? r.best : base.cases.solved > 0 ? "solved" : null;
  for (const sid of Object.keys(base.suspects)) {
    const m = r.suspects?.[sid];
    const s = base.suspects[sid];
    if (!m || !s) continue;
    s.sessions = Number(m.sessions) || 0;
    s.lastSessionId = typeof m.lastSessionId === "string" ? m.lastSessionId : null;
    s.topics = Array.isArray(m.topics) ? m.topics.filter((t) => typeof t === "string").slice(-50) : [];
    s.shown = Array.isArray(m.shown) ? m.shown.filter((e): e is EvidenceId => typeof e === "string" && e in c.evidence) : [];
    s.accusedThem = Number(m.accusedThem) || 0;
    s.lastAccusedVerdict = isVerdict(m.lastAccusedVerdict) ? m.lastAccusedVerdict : null;
    s.crackedTopics = Array.isArray(m.crackedTopics) ? m.crackedTopics.filter((t) => typeof t === "string") : [];
  }
  return base;
}

export function startSession(mem: PlayerMemory, sessionId: string): PlayerMemory {
  const m = structuredClone(mem);
  m.sessions += 1;
  m.currentSessionId = sessionId;
  return m;
}

function suspectMem(m: PlayerMemory, suspect: SuspectId): SuspectMemory {
  return (m.suspects[suspect] ??= blankSuspect());
}

/** Called the first time a suspect is spoken to in a session. */
function touch(m: PlayerMemory, suspect: SuspectId): void {
  const s = suspectMem(m, suspect);
  if (s.lastSessionId !== m.currentSessionId) {
    s.sessions += 1;
    s.lastSessionId = m.currentSessionId;
  }
}

export function recordAsk(mem: PlayerMemory, suspect: SuspectId, topic: Topic, cracked: Topic[] = []): PlayerMemory {
  const m = structuredClone(mem);
  touch(m, suspect);
  const s = suspectMem(m, suspect);
  if (!s.topics.includes(topic)) s.topics.push(topic);
  for (const c of cracked) if (!s.crackedTopics.includes(c)) s.crackedTopics.push(c);
  return m;
}

export function recordPresent(mem: PlayerMemory, suspect: SuspectId, evidence: EvidenceId, cracked: Topic[] = []): PlayerMemory {
  const m = structuredClone(mem);
  touch(m, suspect);
  const s = suspectMem(m, suspect);
  if (!s.shown.includes(evidence)) s.shown.push(evidence);
  for (const c of cracked) if (!s.crackedTopics.includes(c)) s.crackedTopics.push(c);
  return m;
}

export function recordAccusation(mem: PlayerMemory, result: AccusationResult): PlayerMemory {
  const m = structuredClone(mem);
  const s = suspectMem(m, result.suspect);
  s.accusedThem += 1;
  s.lastAccusedVerdict = result.verdict;
  if (result.verdict === "solved") m.cases.solved += 1;
  else m.cases.failed += 1;
  if (m.best === null || RANK[result.verdict] > RANK[m.best]) m.best = result.verdict;
  return m;
}

/**
 * A line the suspect says on first contact this session, based on what they
 * remember from earlier sessions. Null when there is nothing to remember yet.
 */
export function rememberLine(mem: PlayerMemory, suspect: SuspectId): string | null {
  const s = mem.suspects[suspect];
  if (!s) return null;
  // Only lines about *previous* sessions.
  const metBefore = s.sessions > 0 && s.lastSessionId !== null && s.lastSessionId !== mem.currentSessionId;
  if (!metBefore && s.accusedThem === 0) return null;
  if (s.accusedThem > 0 && s.lastAccusedVerdict === "wrong") {
    return `You accused me last time, detective. In front of everyone. I haven't forgotten.`;
  }
  if (s.accusedThem > 0) {
    return `You named me last time. And yet here I am, still talking to you.`;
  }
  if (s.shown.length > 0) {
    const last = s.shown[s.shown.length - 1]!;
    const name = caseById(mem.caseId).evidence[last]?.name ?? "evidence";
    return `Back again. Last time you showed me the ${name.toLowerCase()}. I told you what I know.`;
  }
  if (s.sessions >= 2) {
    return `We've done this ${s.sessions} times now, detective. My story hasn't changed.`;
  }
  return "You again. I remember your questions from last time.";
}

/**
 * Per-player suspect memory. Each suspect remembers what this player asked,
 * showed and accused across sessions. Storage is pluggable: IndexedDB /
 * localStorage in the browser, a JSON file for the MCP server, in-memory for
 * tests.
 */

import { EVIDENCE, type EvidenceId, type SuspectId, type Topic } from "./case.js";
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
  sessions: number;
  currentSessionId: string | null;
  cases: { solved: number; failed: number };
  suspects: Record<SuspectId, SuspectMemory>;
}

export interface MemoryStore {
  load(playerId: string): Promise<PlayerMemory | null>;
  save(memory: PlayerMemory): Promise<void>;
}

function blankSuspect(): SuspectMemory {
  return { sessions: 0, lastSessionId: null, topics: [], shown: [], accusedThem: 0, lastAccusedVerdict: null, crackedTopics: [] };
}

export function blankMemory(playerId: string): PlayerMemory {
  return {
    playerId,
    sessions: 0,
    currentSessionId: null,
    cases: { solved: 0, failed: 0 },
    suspects: { mara: blankSuspect(), desmond: blankSuspect(), ines: blankSuspect(), teodor: blankSuspect() },
  };
}

/** Normalise anything loaded from storage (older shapes, tampering). */
export function coerceMemory(playerId: string, raw: unknown): PlayerMemory {
  const base = blankMemory(playerId);
  if (!raw || typeof raw !== "object") return base;
  const r = raw as Partial<PlayerMemory>;
  base.sessions = Number.isFinite(r.sessions) ? Number(r.sessions) : 0;
  base.currentSessionId = typeof r.currentSessionId === "string" ? r.currentSessionId : null;
  if (r.cases && typeof r.cases === "object") {
    base.cases.solved = Number(r.cases.solved) || 0;
    base.cases.failed = Number(r.cases.failed) || 0;
  }
  for (const id of Object.keys(base.suspects) as SuspectId[]) {
    const m = r.suspects?.[id];
    if (!m) continue;
    const s = base.suspects[id];
    s.sessions = Number(m.sessions) || 0;
    s.lastSessionId = typeof m.lastSessionId === "string" ? m.lastSessionId : null;
    s.topics = Array.isArray(m.topics) ? m.topics.filter((t) => typeof t === "string").slice(-50) : [];
    s.shown = Array.isArray(m.shown) ? m.shown.filter((e): e is EvidenceId => typeof e === "string" && e in EVIDENCE) : [];
    s.accusedThem = Number(m.accusedThem) || 0;
    s.lastAccusedVerdict = m.lastAccusedVerdict === "solved" || m.lastAccusedVerdict === "weak" || m.lastAccusedVerdict === "wrong" ? m.lastAccusedVerdict : null;
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

/** Called the first time a suspect is spoken to in a session. */
function touch(m: PlayerMemory, suspect: SuspectId): void {
  const s = m.suspects[suspect];
  if (s.lastSessionId !== m.currentSessionId) {
    s.sessions += 1;
    s.lastSessionId = m.currentSessionId;
  }
}

export function recordAsk(mem: PlayerMemory, suspect: SuspectId, topic: Topic, cracked: Topic[] = []): PlayerMemory {
  const m = structuredClone(mem);
  touch(m, suspect);
  const s = m.suspects[suspect];
  if (!s.topics.includes(topic)) s.topics.push(topic);
  for (const c of cracked) if (!s.crackedTopics.includes(c)) s.crackedTopics.push(c);
  return m;
}

export function recordPresent(mem: PlayerMemory, suspect: SuspectId, evidence: EvidenceId, cracked: Topic[] = []): PlayerMemory {
  const m = structuredClone(mem);
  touch(m, suspect);
  const s = m.suspects[suspect];
  if (!s.shown.includes(evidence)) s.shown.push(evidence);
  for (const c of cracked) if (!s.crackedTopics.includes(c)) s.crackedTopics.push(c);
  return m;
}

export function recordAccusation(mem: PlayerMemory, result: AccusationResult): PlayerMemory {
  const m = structuredClone(mem);
  const s = m.suspects[result.suspect];
  s.accusedThem += 1;
  s.lastAccusedVerdict = result.verdict;
  if (result.verdict === "solved") m.cases.solved += 1;
  else m.cases.failed += 1;
  return m;
}

/**
 * A line the suspect says on first contact this session, based on what they
 * remember from earlier sessions. Null when there is nothing to remember yet.
 */
export function rememberLine(mem: PlayerMemory, suspect: SuspectId): string | null {
  const s = mem.suspects[suspect];
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
    return `Back again. Last time you showed me the ${EVIDENCE[last].name.toLowerCase()}. I told you what I know.`;
  }
  if (s.sessions >= 2) {
    return `We've done this ${s.sessions} times now, detective. My story hasn't changed.`;
  }
  return "You again. I remember your questions from last time.";
}

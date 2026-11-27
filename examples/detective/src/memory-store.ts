/**
 * Browser persistence, all on this device (nothing is sent anywhere):
 *
 * - Suspect memory per player per case: IndexedDB, falling back to
 *   localStorage, then to RAM. The first case keeps its original key (the bare
 *   player id) so returning players keep what the household remembers; other
 *   cases use `<playerId>::<caseId>`.
 * - The in-progress game per case (localStorage), so a player can switch cases
 *   and come back to where they were.
 */
import { DEFAULT_CASE_ID, caseById, coerceMemory, type GameState, type MemoryStore, type PlayerMemory } from "@mnemopay/augengine-detective";

const DB = "augengine-detective";
const STORE = "players";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Storage key for one player's memory of one case. */
export function memoryKey(playerId: string, caseId: string): string {
  return caseId === DEFAULT_CASE_ID ? playerId : `${playerId}::${caseId}`;
}

export class BrowserMemoryStore implements MemoryStore {
  private ram = new Map<string, PlayerMemory>();

  constructor(readonly caseId: string = DEFAULT_CASE_ID) {}

  async load(playerId: string): Promise<PlayerMemory | null> {
    const key = memoryKey(playerId, this.caseId);
    try {
      const db = await openDb();
      const raw = await new Promise<unknown>((resolve, reject) => {
        const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
        tx.onsuccess = () => resolve(tx.result);
        tx.onerror = () => reject(tx.error);
      });
      db.close();
      if (raw) return coerceMemory(playerId, raw, this.caseId);
    } catch {
      /* fall through */
    }
    try {
      const raw = localStorage.getItem(`${DB}:${key}`);
      if (raw) return coerceMemory(playerId, JSON.parse(raw), this.caseId);
    } catch {
      /* fall through */
    }
    return this.ram.get(key) ?? null;
  }

  async save(memory: PlayerMemory): Promise<void> {
    const key = memoryKey(memory.playerId, memory.caseId);
    this.ram.set(key, memory);
    try {
      const db = await openDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(memory, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
      db.close();
      return;
    } catch {
      /* fall through */
    }
    try {
      localStorage.setItem(`${DB}:${key}`, JSON.stringify(memory));
    } catch {
      /* RAM only */
    }
  }
}

export function playerId(): string {
  const key = "augengine-detective:player";
  try {
    const existing = localStorage.getItem(key);
    if (existing) return existing;
    const id = crypto.randomUUID();
    localStorage.setItem(key, id);
    return id;
  } catch {
    return crypto.randomUUID();
  }
}

const progressKey = (pid: string, caseId: string) => `${DB}:progress:${pid}:${caseId}`;

/** The saved in-progress game for a case, if it is intact and unfinished. */
export function loadProgress(pid: string, caseId: string): GameState | null {
  try {
    const raw = localStorage.getItem(progressKey(pid, caseId));
    if (!raw) return null;
    const s = JSON.parse(raw) as GameState;
    const c = caseById(caseId);
    const ok =
      s &&
      s.caseId === c.id &&
      typeof s.turn === "number" &&
      c.rooms.some((r) => r.id === s.room) &&
      Array.isArray(s.notebook) &&
      s.notebook.every((e) => typeof e === "string" && e in c.evidence) &&
      Array.isArray(s.examined) &&
      Array.isArray(s.log) &&
      c.suspectOrder.every((id) => Array.isArray(s.presented?.[id]) && Array.isArray(s.cracked?.[id]) && Array.isArray(s.asked?.[id]));
    if (!ok || s.accusation) return null;
    return s;
  } catch {
    return null;
  }
}

export function saveProgress(pid: string, state: GameState): void {
  try {
    if (state.accusation) localStorage.removeItem(progressKey(pid, state.caseId));
    else localStorage.setItem(progressKey(pid, state.caseId), JSON.stringify({ ...state, log: state.log.slice(-60) }));
  } catch {
    /* progress is a convenience */
  }
}

export function clearProgress(pid: string, caseId: string): void {
  try {
    localStorage.removeItem(progressKey(pid, caseId));
  } catch {
    /* ignore */
  }
}

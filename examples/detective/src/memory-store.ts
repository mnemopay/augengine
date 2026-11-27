/** Browser memory store: IndexedDB, falling back to localStorage, then to RAM. */
import { coerceMemory, type MemoryStore, type PlayerMemory } from "@mnemopay/augengine-detective";

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

export class BrowserMemoryStore implements MemoryStore {
  private ram = new Map<string, PlayerMemory>();

  async load(playerId: string): Promise<PlayerMemory | null> {
    try {
      const db = await openDb();
      const raw = await new Promise<unknown>((resolve, reject) => {
        const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(playerId);
        tx.onsuccess = () => resolve(tx.result);
        tx.onerror = () => reject(tx.error);
      });
      db.close();
      if (raw) return coerceMemory(playerId, raw);
    } catch {
      /* fall through */
    }
    try {
      const raw = localStorage.getItem(`${DB}:${playerId}`);
      if (raw) return coerceMemory(playerId, JSON.parse(raw));
    } catch {
      /* fall through */
    }
    return this.ram.get(playerId) ?? null;
  }

  async save(memory: PlayerMemory): Promise<void> {
    this.ram.set(memory.playerId, memory);
    try {
      const db = await openDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(memory, memory.playerId);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
      db.close();
      return;
    } catch {
      /* fall through */
    }
    try {
      localStorage.setItem(`${DB}:${memory.playerId}`, JSON.stringify(memory));
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

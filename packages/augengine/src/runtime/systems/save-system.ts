/**
 * Track 5 — Save / load wrapper.
 *
 * The engine already ships `save()` / `load()` in `persistence/save.ts`.
 * This system is a small adapter that:
 *
 *   1. captures the current scene to a `SaveFile`,
 *   2. serializes it (gzipped where supported), and
 *   3. either stores it in `localStorage` or hands it to a caller-
 *      supplied `IO` (so headless tests can pipe through Map<string>).
 *
 * The smoke-game's previous saves wrote three separate localStorage
 * keys (`smoke-game.brain.v1`, `smoke-game.story.v1`, `augengine.squad-trust`).
 * With the trust ledger + story checkpoints now on entity components,
 * one `save()` of the scene round-trips all of them — the brain stays
 * its own concern because it's keyed by `npc/maya` namespace and lives
 * inside `createInProcessBrain`'s adapter.
 */

import type { Scene } from "../../scene/scene.js";
import {
  save as captureSaveFile,
  load as loadSaveFile,
  serializeGzip,
  deserializeGzip,
  type SaveFile,
  type SaveOptions,
} from "../../persistence/save.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface SaveIO {
  read(key: string): Promise<Uint8Array | null> | Uint8Array | null;
  write(key: string, bytes: Uint8Array): Promise<void> | void;
  remove?(key: string): Promise<void> | void;
}

/** Browser localStorage adapter — base64-encodes the gzipped bytes. */
export function browserSaveIO(): SaveIO {
  return {
    read(key: string): Uint8Array | null {
      try {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        const bin = atob(raw);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
      } catch {
        return null;
      }
    },
    write(key: string, bytes: Uint8Array): void {
      try {
        let bin = "";
        for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!);
        localStorage.setItem(key, btoa(bin));
      } catch {
        /* localStorage quota or disabled */
      }
    },
    remove(key: string): void {
      try { localStorage.removeItem(key); } catch { /* swallow */ }
    },
  };
}

/** In-memory adapter — used by tests + replay deterministic runs. */
export function memorySaveIO(map: Map<string, Uint8Array> = new Map()): SaveIO {
  return {
    read(key: string): Uint8Array | null { return map.get(key) ?? null; },
    write(key: string, bytes: Uint8Array): void { map.set(key, bytes); },
    remove(key: string): void { map.delete(key); },
  };
}

export class SaveSystem implements GameSystem {
  readonly id = "save";

  constructor(
    private readonly storageKey: string,
    private readonly io: SaveIO,
  ) {}

  /** Capture + serialize + commit to the IO. Returns the SaveFile. */
  async persist(scene: Scene, opts: SaveOptions = {}): Promise<SaveFile> {
    const file = await captureSaveFile(scene, opts);
    const bytes = await serializeGzip(file);
    await this.io.write(this.storageKey, bytes);
    return file;
  }

  /** Read + decompress + verify hash. Returns null when no save exists. */
  async restore(): Promise<{ file: SaveFile; scene: Scene } | null> {
    const bytes = await this.io.read(this.storageKey);
    if (!bytes) return null;
    const file = await deserializeGzip(bytes);
    const { scene } = await loadSaveFile(file);
    return { file, scene };
  }

  /** Drop the saved file. Idempotent. */
  async clear(): Promise<void> {
    if (this.io.remove) await this.io.remove(this.storageKey);
  }

  /** Per-frame tick — no-op; SaveSystem is request-driven. */
  update(_scene: Scene, _dt: number, _ctx: SystemContext): void { /* request-driven */ }
}

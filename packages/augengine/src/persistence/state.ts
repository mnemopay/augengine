/**
 * Deterministic state snapshot — capture/restore the parts of a Scene that
 * MUST round-trip exactly (entities, transforms, components). AI-generated
 * artifacts (text, audio, images) live in the artifact manifest; this file
 * does NOT touch them.
 *
 * Decision #4: deterministic core + AI-artifact manifest, save = both.
 *
 * Hashing:
 *   - `worldStateHash()` returns SHA-256 hex of a canonicalized scene JSON.
 *     Async because crypto.subtle.digest is async in browsers.
 *   - For sync content-addressing of small payloads, use `manifest.fnv1a64`.
 */

import type { Entity, EntityId, SceneSpec } from "../scene/types.js";
import { Scene } from "../scene/scene.js";

export interface SceneSnapshot {
  schemaVersion: 1;
  capturedAt: string;
  rngSeed: number;
  spec: SceneSpec;
}

export interface SnapshotOptions {
  /** RNG seed at the moment of capture (threaded into deterministic systems). */
  rngSeed?: number;
}

export function captureSnapshot(scene: Scene, opts: SnapshotOptions = {}): SceneSnapshot {
  // Deep clone the spec by walking entities + components — JSON.parse(JSON.stringify())
  // would lose Float32Array typing on inverseBindMatrices, so we serialize
  // those explicitly into number arrays.
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: scene.spec.source,
    entities: [...scene.entities.values()].map(cloneEntity),
  };
  if (scene.spec.defaultCamera) spec.defaultCamera = scene.spec.defaultCamera;
  if (scene.spec.environment) spec.environment = scene.spec.environment;

  return {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    rngSeed: opts.rngSeed ?? 0,
    spec,
  };
}

export function restoreSnapshot(snap: SceneSnapshot): Scene {
  if (snap.schemaVersion !== 1) {
    throw new Error(`unsupported snapshot schemaVersion: ${snap.schemaVersion}`);
  }
  return new Scene(snap.spec);
}

function cloneEntity(e: Entity): Entity {
  const components: Entity["components"] = {};
  for (const [tag, comp] of Object.entries(e.components)) {
    if (comp === undefined) continue;
    components[tag as keyof Entity["components"]] = cloneComponent(comp) as never;
  }
  return { id: e.id, name: e.name, components };
}

function cloneComponent(c: unknown): unknown {
  if (c === null || typeof c !== "object") return c;
  if (c instanceof Float32Array) return Array.from(c);
  if (c instanceof Uint16Array) return Array.from(c);
  if (Array.isArray(c)) return c.map(cloneComponent);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(c as Record<string, unknown>)) {
    out[k] = cloneComponent(v);
  }
  return out;
}

/**
 * Canonical-form JSON of a snapshot — sorted keys at every level so the same
 * scene state hashes to the same string regardless of property insertion order.
 */
export function canonicalize(snap: SceneSnapshot): string {
  return canonicalStringify({
    schemaVersion: snap.schemaVersion,
    rngSeed: snap.rngSeed,
    spec: snap.spec,
    // capturedAt is intentionally excluded — we want hash to depend on game
    // state, not on the timestamp.
  });
}

function canonicalStringify(value: unknown): string {
  if (value === null) return "null";
  const t = typeof value;
  if (t === "string") return JSON.stringify(value);
  if (t === "number" || t === "boolean") return String(value);
  if (t === "undefined") return "null";
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalStringify).join(",") + "]";
  }
  if (t === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      if (obj[k] === undefined) continue;
      parts.push(JSON.stringify(k) + ":" + canonicalStringify(obj[k]));
    }
    return "{" + parts.join(",") + "}";
  }
  return "null";
}

/**
 * SHA-256 hex of canonicalized snapshot. Async — uses `crypto.subtle`,
 * which is globally available in modern browsers and Node 19+. Used as the
 * world-state cache key for AI artifact lookups.
 */
export async function worldStateHash(snap: SceneSnapshot): Promise<string> {
  const canon = canonicalize(snap);
  return await sha256Hex(canon);
}

export async function sha256Hex(s: string): Promise<string> {
  const buf = new TextEncoder().encode(s);
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Simple lookup: which entities exist in `b` but not `a`? */
export function diffEntities(a: SceneSnapshot, b: SceneSnapshot): {
  added: EntityId[];
  removed: EntityId[];
} {
  const aIds = new Set(a.spec.entities.map((e) => e.id));
  const bIds = new Set(b.spec.entities.map((e) => e.id));
  const added: EntityId[] = [];
  const removed: EntityId[] = [];
  for (const id of bIds) if (!aIds.has(id)) added.push(id);
  for (const id of aIds) if (!bIds.has(id)) removed.push(id);
  return { added, removed };
}

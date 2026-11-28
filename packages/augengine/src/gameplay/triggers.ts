/**
 * Trigger volumes, spawners and checkpoints — the runtime half of the
 * prefab `gameplay` entries. Pure math and deterministic (seeded), so they
 * replay and test without a renderer or physics.
 *
 * Triggers test subject positions (usually capsule centres) against
 * axis-aligned boxes or spheres and report enter / exit transitions.
 */

import type { Vec3 } from "../scene/types.js";
import type { PlacedCheckpoint, PlacedSpawner, PlacedTrigger } from "./prefab.js";

export interface TriggerEvent {
  kind: "enter" | "exit";
  trigger: string;
  event: string;
  subject: string;
}

export interface Subject {
  id: string;
  position: Vec3;
}

interface TriggerState {
  spec: PlacedTrigger;
  inside: Set<string>;
  done: boolean;
}

export function insideTrigger(t: Pick<PlacedTrigger, "shape" | "center" | "halfExtents" | "radius">, p: Vec3): boolean {
  if (t.shape === "sphere") return Math.hypot(p[0] - t.center[0], p[1] - t.center[1], p[2] - t.center[2]) <= (t.radius ?? 0);
  const h = t.halfExtents ?? [0, 0, 0];
  return Math.abs(p[0] - t.center[0]) <= h[0] && Math.abs(p[1] - t.center[1]) <= h[1] && Math.abs(p[2] - t.center[2]) <= h[2];
}

export class TriggerSystem {
  private readonly triggers: TriggerState[] = [];

  add(t: PlacedTrigger): void {
    this.triggers.push({ spec: t, inside: new Set(), done: false });
  }

  /** Enter/exit events since the last update, in trigger then subject order. `once` triggers fire one enter, then go quiet. */
  update(subjects: readonly Subject[]): TriggerEvent[] {
    const out: TriggerEvent[] = [];
    for (const t of this.triggers) {
      if (t.done) continue;
      const now = new Set<string>();
      for (const s of subjects) if (insideTrigger(t.spec, s.position)) now.add(s.id);
      for (const id of now) if (!t.inside.has(id)) {
        out.push({ kind: "enter", trigger: t.spec.id, event: t.spec.event, subject: id });
        if (t.spec.once) { t.done = true; break; }
      }
      if (t.done) continue;
      for (const id of t.inside) if (!now.has(id)) out.push({ kind: "exit", trigger: t.spec.id, event: t.spec.event, subject: id });
      t.inside = now;
    }
    return out;
  }

  /** Subjects currently inside a trigger. */
  occupants(triggerId: string): string[] {
    return [...(this.triggers.find((t) => t.spec.id === triggerId)?.inside ?? [])];
  }
}

export interface SpawnRequest {
  spawner: string;
  kind: string;
  position: Vec3;
  /** 0-based index of this spawn for the spawner. */
  index: number;
}

/**
 * Emits spawn requests on an interval, never above `maxAlive` alive and never
 * more than `count` in total. Points are picked by a seeded shuffle bag, so a
 * run is reproducible. Inactive until `activate()` when `activateOn` is set.
 */
export class Spawner {
  private timer = 0;
  private spawned = 0;
  private active: boolean;
  private bag: number[] = [];
  private rng: () => number;

  constructor(readonly spec: PlacedSpawner) {
    this.active = !spec.activateOn;
    this.rng = mulberry32(spec.seed ?? 1);
    this.timer = spec.interval ?? 1; // first spawn immediately on activation
  }

  get isActive(): boolean { return this.active; }
  get spawnedCount(): number { return this.spawned; }
  get exhausted(): boolean { return this.spawned >= (this.spec.count ?? Infinity); }

  activate(): void { this.active = true; }

  /** `alive` = how many of this spawner's spawns are still alive. */
  update(dt: number, alive: number): SpawnRequest[] {
    if (!this.active || this.exhausted) return [];
    this.timer += dt;
    const interval = this.spec.interval ?? 1;
    const out: SpawnRequest[] = [];
    while (this.timer >= interval && alive + out.length < (this.spec.maxAlive ?? 4) && !this.exhausted) {
      this.timer -= interval;
      out.push({ spawner: this.spec.id, kind: this.spec.kind, position: this.nextPoint(), index: this.spawned++ });
    }
    // Hold at most one interval of credit while capped, so a freed slot spawns promptly but not in a burst.
    if (this.timer > interval) this.timer = interval;
    return out;
  }

  private nextPoint(): Vec3 {
    const pts = this.spec.worldPoints;
    if (this.bag.length === 0) {
      this.bag = pts.map((_, i) => i);
      for (let i = this.bag.length - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        [this.bag[i], this.bag[j]] = [this.bag[j]!, this.bag[i]!];
      }
    }
    return pts[this.bag.pop()!]!;
  }
}

/** Highest-order checkpoint reached becomes the respawn point. */
export class CheckpointTracker {
  private readonly points: PlacedCheckpoint[];
  private reachedOrder = -Infinity;
  private current: PlacedCheckpoint | null = null;

  constructor(points: readonly PlacedCheckpoint[], private readonly fallback: Vec3) {
    this.points = [...points].sort((a, b) => a.order - b.order);
  }

  /** Newly reached checkpoint ids (only ones that raise the order). */
  update(position: Vec3): string[] {
    const out: string[] = [];
    for (const c of this.points) {
      if (c.order <= this.reachedOrder) continue;
      const r = c.radius ?? 2;
      if (Math.hypot(position[0] - c.position[0], position[2] - c.position[2]) <= r && Math.abs(position[1] - c.position[1]) <= Math.max(2, r)) {
        this.reachedOrder = c.order;
        this.current = c;
        out.push(c.id);
      }
    }
    return out;
  }

  respawnPoint(): Vec3 {
    return this.current?.position ?? this.fallback;
  }

  get reached(): string | null {
    return this.current?.id ?? null;
  }
}

/** Small seeded PRNG (mulberry32), [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

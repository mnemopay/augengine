/**
 * Track 5 — System abstraction.
 *
 * Every game-state subsystem implements `GameSystem`. The smoke-game's
 * shell drives them by calling `tick(scene, dt, ctx)` per frame inside
 * its `requestAnimationFrame` callback or by registering them with
 * `EngineRunner.registerAsyncSystem` (existing API).
 *
 * Events are a simple list the systems push to during `tick`. The shell
 * drains them after each frame and forwards them to the UI / mixer /
 * brain / portal. Keeping events as plain data means systems stay pure
 * and tests can assert the event log directly.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId, Vec3 } from "../../scene/types.js";

/** Possible sources for a damage / death event. */
export type DamageSource = "recruit" | "environment" | "friendly-fire" | "player";

/** A single side-effect emitted by a system this frame. */
export type GameEvent =
  | {
      type: "damage";
      target: EntityId;
      amount: number;
      remainingHp: number;
      source: DamageSource;
    }
  | { type: "death"; target: EntityId; source: DamageSource }
  | { type: "shot-fired"; weapon: EntityId; position: Vec3 }
  | { type: "shot-hit"; weapon: EntityId; target: EntityId; damage: number; killed: boolean }
  | { type: "reload-start"; weapon: EntityId; lockoutSec: number }
  | { type: "reload-complete"; weapon: EntityId; loaded: number }
  | { type: "objective-completed"; goalId: string; kind: "secure" | "extract" | "capture"; score: number }
  | { type: "mission-checkpoint"; checkpoint: string; mapId: string }
  | { type: "mission-ended"; won: boolean; score: number; reason: "won" | "timeout" | "death" }
  | { type: "recruit-state-changed"; recruit: EntityId; from: string; to: string }
  | { type: "recruit-killed"; recruit: EntityId; by: EntityId }
  | { type: "trust-changed"; npc: EntityId; delta: number; score: number; tier: string; reason: string }
  | { type: "maya-consequence"; line: string; betrayed: boolean }
  | { type: "race-started"; stageId: string }
  | { type: "race-checkpoint"; racerId: string; checkpointId: string; elapsedSeconds: number }
  | { type: "race-lap-completed"; racerId: string; lap: number }
  | { type: "race-finished"; racerId: string; rank: number; elapsedSeconds: number };

/**
 * Per-frame system context. The shell builds this once each frame and
 * passes it to every system in `update`. `playerPos` and `playerHpFrac`
 * are the most-used context values, so they're pre-extracted.
 */
export interface SystemContext {
  /** Current player world position. Null if the player entity isn't in scene. */
  playerPos: Vec3 | null;
  /** Player HP fraction (0..1), or 1 if no health component is attached. */
  playerHpFrac: number;
  /** Append-only event log this frame. Systems push, shell drains. */
  events: GameEvent[];
  /** Monotonic ms timestamp for damage flash / debounce logic. */
  nowMs: number;
}

/** Minimum surface every game system implements. */
export interface GameSystem {
  /** Stable identifier — used for debug logging + replay tagging. */
  readonly id: string;
  /** Per-frame tick. `dt` is seconds since the previous tick. */
  update(scene: Scene, dt: number, ctx: SystemContext): void;
}

/** Convenience helper — build a fresh, empty per-frame context. */
export function buildSystemContext(
  nowMs: number,
  playerPos: Vec3 | null,
  playerHpFrac: number,
): SystemContext {
  return {
    playerPos,
    playerHpFrac,
    events: [],
    nowMs,
  };
}

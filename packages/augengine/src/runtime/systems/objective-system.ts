/**
 * Track 5 — Objective system.
 *
 * Walks entities with `ObjectiveComponent` each tick. Three kinds:
 *
 *   - `secure`   — caller flips `completed = true` via `secure()` from
 *                  a click handler (raycast crosshair). The system just
 *                  emits the completion event.
 *   - `extract`  — completes when the player is inside the radius AND
 *                  every recruit in the scene is `down` (caller wires
 *                  the recruits-down predicate into options).
 *   - `capture`  — accumulates seconds while the player is inside the
 *                  radius; decays at half-speed when outside. Completes
 *                  once `progressSeconds >= requiredSeconds`.
 *
 * Mission HUD reads `progressSeconds` for the capture bar.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId, Vec3 } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  type MissionComponent,
  type ObjectiveComponent,
} from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface ObjectiveSystemOptions {
  /** Returns true if all hostiles in the scene are down. */
  allHostilesDown: (scene: Scene) => boolean;
  /** Entity id of the mission singleton — used to credit score on win. */
  missionEntityId: EntityId;
}

export class ObjectiveSystem implements GameSystem {
  readonly id = "objective";

  constructor(private readonly opts: ObjectiveSystemOptions) {}

  /**
   * Mark a "secure" objective as completed. Called from the shell's
   * crosshair-click handler. Emits the completion event + bumps the
   * mission score.
   */
  secure(scene: Scene, goalEntityId: EntityId, ctx: SystemContext): boolean {
    const entity = scene.get(goalEntityId);
    if (!entity) return false;
    const obj = getGameComponent<ObjectiveComponent>(entity, GameTag.Objective);
    if (!obj || obj.completed || obj.kind !== "secure") return false;
    obj.completed = true;
    this.creditScore(scene, obj.scoreReward ?? 50, ctx);
    ctx.events.push({
      type: "objective-completed",
      goalId: obj.goalId,
      kind: "secure",
      score: obj.scoreReward ?? 50,
    });
    return true;
  }

  update(scene: Scene, dt: number, ctx: SystemContext): void {
    const playerPos = ctx.playerPos;
    for (const entity of scene) {
      const obj = getGameComponent<ObjectiveComponent>(entity, GameTag.Objective);
      if (!obj || obj.completed) continue;
      if (obj.kind === "secure") continue; // event-driven
      if (!obj.position || obj.radius == null || !playerPos) continue;
      const dx = playerPos[0] - obj.position[0];
      const dz = playerPos[2] - obj.position[2];
      const inZone = Math.hypot(dx, dz) <= obj.radius;
      if (obj.kind === "capture") {
        const required = obj.requiredSeconds ?? 25;
        obj.progressSeconds ??= 0;
        if (inZone) {
          obj.progressSeconds = Math.min(required, obj.progressSeconds + dt);
        } else if (obj.progressSeconds > 0) {
          obj.progressSeconds = Math.max(0, obj.progressSeconds - dt * 0.5);
        }
        if (obj.progressSeconds >= required) {
          obj.completed = true;
          this.creditScore(scene, obj.scoreReward ?? 400, ctx);
          ctx.events.push({
            type: "objective-completed",
            goalId: obj.goalId,
            kind: "capture",
            score: obj.scoreReward ?? 400,
          });
        }
      } else if (obj.kind === "extract") {
        if (!inZone) continue;
        if (!this.opts.allHostilesDown(scene)) continue;
        obj.completed = true;
        this.creditScore(scene, obj.scoreReward ?? 200, ctx);
        ctx.events.push({
          type: "objective-completed",
          goalId: obj.goalId,
          kind: "extract",
          score: obj.scoreReward ?? 200,
        });
      }
    }
  }

  /** Best-distance read for HUD copy ("approach plaza centre"). */
  static distanceToObjective(playerPos: Vec3, obj: ObjectiveComponent): number {
    if (!obj.position) return Number.POSITIVE_INFINITY;
    const dx = playerPos[0] - obj.position[0];
    const dz = playerPos[2] - obj.position[2];
    return Math.hypot(dx, dz);
  }

  private creditScore(scene: Scene, reward: number, _ctx: SystemContext): void {
    const mission = scene.get(this.opts.missionEntityId);
    if (!mission) return;
    const m = getGameComponent<MissionComponent>(mission, GameTag.Mission);
    if (!m || m.ended) return;
    m.score += reward;
  }
}

/**
 * Track 5 — Mission / round system.
 *
 * Owns the round timer, score reconciliation, and the
 * `mission-checkpoint` event stream. A scene gets exactly one
 * `MissionComponent` attached to the singleton entity created by
 * `createMissionEntity()`.
 *
 * The system reads the per-frame events drained by the shell so it
 * can:
 *   - End the round when an `objective-completed` event completes the
 *     last open win-condition (caller specifies the goalId set in
 *     `MissionSystemOptions.winOn`).
 *   - End the round when the player dies (HealthSystem death event for
 *     the player id).
 *   - Time out when `timeLeftSec` hits zero.
 *
 * Checkpoints are author-defined: pass an `id` + `predicate` array.
 * When a predicate first returns true, the system pushes a
 * `mission-checkpoint` event and records the id in `MissionComponent`.
 * Used by the smoke-game to fire Maya consequence lines on
 * narrative beats ("first kill", "all crates secured", "extraction
 * reached").
 */

import type { Scene } from "../../scene/scene.js";
import { Tag, type Entity, type EntityId } from "../../scene/types.js";
import {
  GameTag,
  MISSION_ENTITY_ID,
  getGameComponent,
  setGameComponent,
  type MissionComponent,
} from "./components.js";
import type { GameEvent, GameSystem, SystemContext } from "./system.js";

export interface MissionCheckpoint {
  /** Stable id, used for de-dup + event payload. */
  id: string;
  /**
   * Pure predicate evaluated each frame against the scene + drained
   * events. Returns true the first time the checkpoint should fire.
   */
  predicate: (scene: Scene, events: ReadonlyArray<GameEvent>, ctx: SystemContext) => boolean;
}

export interface MissionSystemOptions {
  mapId: string;
  durationSec: number;
  playerEntityId: EntityId;
  /**
   * Goal id whose `objective-completed` event ends the round as a win.
   * When omitted, the round only ends on timeout / death.
   */
  winOnGoalId?: string;
  checkpoints?: MissionCheckpoint[];
}

/**
 * Build (or hydrate) the singleton mission entity. Returns the entity
 * id so callers can pin it on the ObjectiveSystem.
 */
export function createMissionEntity(
  scene: Scene,
  opts: { mapId: string; durationSec: number },
): EntityId {
  const existing = scene.get(MISSION_ENTITY_ID);
  if (existing) {
    const mission = getGameComponent<MissionComponent>(existing, GameTag.Mission);
    if (mission) return MISSION_ENTITY_ID;
  }
  const entity: Entity = {
    id: MISSION_ENTITY_ID,
    name: "Mission",
    components: {
      [Tag.Transform]: {
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
    },
  };
  setGameComponent<MissionComponent>(entity, GameTag.Mission, {
    mapId: opts.mapId,
    durationSec: opts.durationSec,
    timeLeftSec: opts.durationSec,
    score: 0,
    ended: false,
    won: false,
    endReason: "open",
    checkpoints: [],
  });
  scene.add(entity);
  return MISSION_ENTITY_ID;
}

export class MissionSystem implements GameSystem {
  readonly id = "mission";
  private readonly checkpoints: MissionCheckpoint[];

  constructor(private readonly opts: MissionSystemOptions) {
    this.checkpoints = (opts.checkpoints ?? []).slice();
  }

  update(scene: Scene, dt: number, ctx: SystemContext): void {
    const entity = scene.get(MISSION_ENTITY_ID);
    if (!entity) return;
    const mission = getGameComponent<MissionComponent>(entity, GameTag.Mission);
    if (!mission || mission.ended) return;

    // Tick the round timer.
    mission.timeLeftSec = Math.max(0, mission.timeLeftSec - dt);

    // React to events drained this frame. We iterate the *current*
    // event log — systems that ran before this one (recruits, weapons,
    // objectives) have already populated it.
    for (const event of ctx.events) {
      if (event.type === "objective-completed" && this.opts.winOnGoalId && event.goalId === this.opts.winOnGoalId) {
        this.end(mission, true, "won", ctx);
        return;
      }
      if (event.type === "death" && event.target === this.opts.playerEntityId) {
        this.end(mission, false, "death", ctx);
        return;
      }
    }

    if (mission.timeLeftSec <= 0) {
      this.end(mission, false, "timeout", ctx);
      return;
    }

    // Fire one-shot checkpoints.
    for (const cp of this.checkpoints) {
      if (mission.checkpoints.includes(cp.id)) continue;
      if (cp.predicate(scene, ctx.events, ctx)) {
        mission.checkpoints.push(cp.id);
        ctx.events.push({
          type: "mission-checkpoint",
          checkpoint: cp.id,
          mapId: mission.mapId,
        });
      }
    }
  }

  private end(
    mission: MissionComponent,
    won: boolean,
    reason: "won" | "timeout" | "death",
    ctx: SystemContext,
  ): void {
    mission.ended = true;
    mission.won = won;
    mission.endReason = reason;
    ctx.events.push({ type: "mission-ended", won, score: mission.score, reason });
  }
}

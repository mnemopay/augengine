/**
 * Track 5 — Recruit AI system.
 *
 * Owns the FSM (patrol / investigate / engage / retreat / down) for
 * every entity with a `RecruitComponent`. Movement is delegated to the
 * already-engine-native `KinematicCharacterController` — the system
 * keeps one controller per recruit and feeds it the per-state move
 * input each tick.
 *
 * The previous implementation (a shell-side `RecruitAI` class) encoded
 * both the data and the behaviour in one place. We split the data into
 * `RecruitComponent` and keep the behaviour here so the smoke-game shell
 * no longer owns AI state at all — the shell is purely visual mirror
 * (limb-sync + down-state tint).
 *
 * Pathfinding is optional — when a `pathfinder` is wired into options
 * the engage / patrol / retreat moves replan around obstacles. Without
 * one the recruit takes the direct line (same as the smoke-game).
 */

import type { Scene } from "../../scene/scene.js";
import { KinematicCharacterController } from "../../physics/character-controller.js";
import { Tag, type EntityId, type Vec3 } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  type HealthComponent,
  type RecruitComponent,
  type RecruitFsmState,
} from "./components.js";
import { HealthSystem } from "./health-system.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface RecruitSystemOptions {
  /** Player entity id — recruits target this for vision + melee. */
  playerEntityId: EntityId;
  /** Optional pathfinder (e.g. the smoke-game's nav A*). */
  pathfinder?: (from: Vec3, to: Vec3) => Vec3[] | null;
  /** Seconds between replans when a pathfinder is wired. */
  replanIntervalSec?: number;
}

interface RuntimeState {
  controller: KinematicCharacterController;
  activePath: Vec3[];
  activePathCursor: number;
  activePathGoal: Vec3 | null;
  replanTimer: number;
}

export class RecruitSystem implements GameSystem {
  readonly id = "recruit";
  private readonly runtime = new Map<EntityId, RuntimeState>();
  private readonly replanIntervalSec: number;

  constructor(private readonly opts: RecruitSystemOptions) {
    this.replanIntervalSec = opts.replanIntervalSec ?? 1.2;
  }

  /**
   * Tell every recruit within `radius` of `pos` that a sound was
   * heard. Recruits in engage/retreat/down ignore.
   */
  hearSound(scene: Scene, pos: Vec3, radius: number): number {
    let alerted = 0;
    for (const entity of scene) {
      const recruit = getGameComponent<RecruitComponent>(entity, GameTag.Recruit);
      if (!recruit) continue;
      if (recruit.state === "down" || recruit.state === "engage" || recruit.state === "retreat") continue;
      const tx = entity.components[Tag.Transform];
      if (!tx) continue;
      const dx = tx.position[0] - pos[0];
      const dz = tx.position[2] - pos[2];
      if (Math.hypot(dx, dz) > radius) continue;
      recruit.investigateTarget = [pos[0], pos[1], pos[2]] as Vec3;
      recruit.investigateTimer = 6.0;
      recruit.state = "investigate";
      alerted++;
    }
    return alerted;
  }

  /** True if every recruit in the scene is downed. */
  allDown(scene: Scene): boolean {
    let any = false;
    for (const entity of scene) {
      const recruit = getGameComponent<RecruitComponent>(entity, GameTag.Recruit);
      if (!recruit) continue;
      any = true;
      if (recruit.state !== "down") return false;
    }
    return any;
  }

  update(scene: Scene, dt: number, ctx: SystemContext): void {
    if (!ctx.playerPos) return;
    for (const entity of scene) {
      const recruit = getGameComponent<RecruitComponent>(entity, GameTag.Recruit);
      if (!recruit) continue;
      const tx = entity.components[Tag.Transform];
      if (!tx) continue;
      // Mirror the HealthComponent into the recruit shadow so downstream
      // checks (e.g. `state === "down"`) stay accurate.
      const health = getGameComponent<HealthComponent>(entity, GameTag.Health);
      if (health) {
        recruit.hp = health.hp;
        if (health.dead && recruit.state !== "down") {
          this.transition(entity.id, recruit, "down", ctx);
        }
      }
      this.tickOne(scene, entity.id, recruit, tx.position as Vec3, ctx, dt);
    }
  }

  // ─── internals ──────────────────────────────────────────────────────

  private getRuntime(scene: Scene, recruitId: EntityId): RuntimeState {
    let r = this.runtime.get(recruitId);
    if (r) return r;
    r = {
      controller: new KinematicCharacterController(scene, {
        entityId: recruitId,
        moveSpeed: 2.2,
        jumpSpeed: 0,
      }),
      activePath: [],
      activePathCursor: 0,
      activePathGoal: null,
      replanTimer: 0,
    };
    this.runtime.set(recruitId, r);
    return r;
  }

  private tickOne(
    scene: Scene,
    id: EntityId,
    recruit: RecruitComponent,
    myPos: Vec3,
    ctx: SystemContext,
    dt: number,
  ): void {
    const rt = this.getRuntime(scene, id);
    if (recruit.state === "down") {
      rt.controller.update(dt, [0, 0, 0]);
      return;
    }

    const playerPos = ctx.playerPos!;
    const dx = playerPos[0] - myPos[0];
    const dz = playerPos[2] - myPos[2];
    const dist = Math.hypot(dx, dz);
    const facingX = -Math.sin(recruit.facingYaw);
    const facingZ = -Math.cos(recruit.facingYaw);
    const dirX = dist > 1e-6 ? dx / dist : 0;
    const dirZ = dist > 1e-6 ? dz / dist : 0;
    const dot = facingX * dirX + facingZ * dirZ;
    const playerInCone = dist < recruit.visionRange && dot >= recruit.visionFovCos;

    const prev = recruit.state;
    if (recruit.state === "patrol") {
      if (playerInCone) {
        recruit.detectionTimer += dt;
        if (recruit.detectionTimer >= recruit.reactionDelay) {
          this.transition(id, recruit, "engage", ctx);
          recruit.detectionTimer = 0;
        }
      } else {
        recruit.detectionTimer = Math.max(0, recruit.detectionTimer - dt * 0.5);
      }
    } else if (recruit.state === "investigate") {
      if (playerInCone && dist < recruit.visionRange) {
        this.transition(id, recruit, "engage", ctx);
        recruit.detectionTimer = 0;
        recruit.investigateTarget = null;
      } else {
        recruit.investigateTimer -= dt;
        if (recruit.investigateTimer <= 0) {
          this.transition(id, recruit, "patrol", ctx);
          recruit.investigateTarget = null;
        }
      }
    } else if (recruit.state === "engage") {
      if (recruit.hp / recruit.maxHp < recruit.retreatThreshold) {
        this.transition(id, recruit, "retreat", ctx);
      } else if (dist > recruit.visionRange * 1.4) {
        this.transition(id, recruit, "patrol", ctx);
        recruit.detectionTimer = 0;
      }
    } else if (recruit.state === "retreat") {
      if (dist > recruit.visionRange * 1.6) {
        this.transition(id, recruit, "patrol", ctx);
        recruit.detectionTimer = 0;
      }
    }

    if (prev !== recruit.state) {
      rt.activePath = [];
      rt.activePathCursor = 0;
      rt.activePathGoal = null;
      rt.replanTimer = 0;
    }

    let moveX = 0;
    let moveZ = 0;
    let speed = recruit.patrolSpeed;
    if (recruit.state === "investigate" && recruit.investigateTarget) {
      const ix = recruit.investigateTarget[0] - myPos[0];
      const iz = recruit.investigateTarget[2] - myPos[2];
      const id2 = Math.hypot(ix, iz);
      if (id2 < 1.2) {
        moveX = 0;
        moveZ = 0;
      } else {
        moveX = ix / id2;
        moveZ = iz / id2;
      }
      speed = (recruit.patrolSpeed + recruit.engageSpeed) * 0.5;
    } else if (recruit.state === "patrol") {
      const tgt = recruit.patrolPath[recruit.patrolIndex]!;
      const routed = this.routeMove(rt, myPos, tgt, dt);
      if (routed.reachedGoal) {
        recruit.patrolIndex = (recruit.patrolIndex + 1) % recruit.patrolPath.length;
      } else {
        moveX = routed.moveX;
        moveZ = routed.moveZ;
      }
      speed = recruit.patrolSpeed;
    } else if (recruit.state === "engage") {
      if (this.opts.pathfinder) {
        const routed = this.routeMove(rt, myPos, playerPos, dt);
        moveX = routed.moveX;
        moveZ = routed.moveZ;
      } else {
        moveX = dirX;
        moveZ = dirZ;
      }
      speed = recruit.engageSpeed;
      if (dist < recruit.meleeRange * 0.95) {
        moveX = 0;
        moveZ = 0;
      }
      recruit.meleeTimer -= dt;
      if (dist <= recruit.meleeRange && recruit.meleeTimer <= 0) {
        HealthSystem.applyDamage(scene, this.opts.playerEntityId, recruit.meleeDamage, ctx, {
          source: "recruit",
        });
        recruit.meleeTimer = recruit.meleeCooldown;
      }
    } else if (recruit.state === "retreat") {
      // Move away from the player toward the farthest patrol waypoint.
      let bestIdx = 0;
      let bestDist = -1;
      for (let i = 0; i < recruit.patrolPath.length; i++) {
        const wp = recruit.patrolPath[i]!;
        const wd = Math.hypot(wp[0] - playerPos[0], wp[2] - playerPos[2]);
        if (wd > bestDist) {
          bestDist = wd;
          bestIdx = i;
        }
      }
      const tgt = recruit.patrolPath[bestIdx]!;
      const routed = this.routeMove(rt, myPos, tgt, dt);
      moveX = routed.moveX;
      moveZ = routed.moveZ;
      speed = recruit.retreatSpeed;
    }

    const moveMag = Math.hypot(moveX, moveZ);
    if (moveMag > 0.01) {
      const targetYaw = Math.atan2(-moveX, -moveZ);
      const turnRate = recruit.state === "engage" ? 4.5 : 2.2;
      recruit.facingYaw = shortestAngleLerp(recruit.facingYaw, targetYaw, turnRate * dt);
    }

    rt.controller.setMoveSpeed(speed);
    rt.controller.update(dt, [moveX, 0, moveZ]);
  }

  private routeMove(
    rt: RuntimeState,
    myPos: Vec3,
    goal: Vec3,
    dt: number,
  ): { moveX: number; moveZ: number; reachedGoal: boolean } {
    const dx = goal[0] - myPos[0];
    const dz = goal[2] - myPos[2];
    const dist = Math.hypot(dx, dz);
    const reachedGoal = dist < 0.6;
    if (!this.opts.pathfinder || dist < 0.9) {
      if (reachedGoal || dist < 1e-6) return { moveX: 0, moveZ: 0, reachedGoal };
      return { moveX: dx / dist, moveZ: dz / dist, reachedGoal };
    }
    rt.replanTimer -= dt;
    const goalShifted =
      !rt.activePathGoal ||
      Math.hypot(rt.activePathGoal[0] - goal[0], rt.activePathGoal[2] - goal[2]) > 1.5;
    const pathExhausted = rt.activePathCursor >= rt.activePath.length;
    if (rt.replanTimer <= 0 || goalShifted || pathExhausted) {
      const path = this.opts.pathfinder(myPos, goal);
      rt.replanTimer = this.replanIntervalSec;
      if (path && path.length > 0) {
        rt.activePath = path;
        rt.activePathCursor = 0;
        rt.activePathGoal = [goal[0], goal[1], goal[2]] as Vec3;
      } else if (goalShifted) {
        rt.activePath = [];
        rt.activePathCursor = 0;
        rt.activePathGoal = null;
      }
    }
    while (rt.activePathCursor < rt.activePath.length) {
      const wp = rt.activePath[rt.activePathCursor]!;
      const wx = wp[0] - myPos[0];
      const wz = wp[2] - myPos[2];
      const wd = Math.hypot(wx, wz);
      if (wd < 0.5) {
        rt.activePathCursor++;
        continue;
      }
      return { moveX: wx / wd, moveZ: wz / wd, reachedGoal: false };
    }
    if (dist < 1e-6) return { moveX: 0, moveZ: 0, reachedGoal };
    return { moveX: dx / dist, moveZ: dz / dist, reachedGoal };
  }

  private transition(
    id: EntityId,
    recruit: RecruitComponent,
    next: RecruitFsmState,
    ctx: SystemContext,
  ): void {
    if (recruit.state === next) return;
    const from = recruit.state;
    recruit.state = next;
    ctx.events.push({ type: "recruit-state-changed", recruit: id, from, to: next });
  }
}

function shortestAngleLerp(current: number, target: number, maxStep: number): number {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  if (Math.abs(diff) <= maxStep) return target;
  return current + Math.sign(diff) * maxStep;
}

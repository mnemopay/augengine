/**
 * NpcThinkScheduler — per-NPC LOD priority pass.
 *
 * Decides WHICH NPCs need work this frame and at what fidelity. Does
 * NOT dispatch inference — that's `AIScheduler`'s job. The two compose:
 *
 *     const ranked = thinkScheduler.schedule(scene, { playerEntityId });
 *     for (const work of ranked.slice(0, BUDGET)) {
 *       if (work.tier === "DeepReasoning") {
 *         aiScheduler.dispatch({ task: "dialogue", payload: ..., timeoutMs: 800, ... });
 *       }
 *     }
 *
 * Design lifted from the 2026-05-13 starter-pack architecture pass
 * (`AugEngineAIScheduler` in the upstream sketch). Adapted for our existing
 * `Scene` + `PersonaComponent` shape instead of the starter's
 * `AICharacter` component.
 *
 * Five tiers — each a different cost/fidelity budget:
 *
 *   Dormant       — NPC simulation stops entirely. Background facts only.
 *   Simulated     — coarse world-schedule tick (~1 Hz). No perception.
 *   Active        — per-frame perception + behavior-tree decisions.
 *   Tactical      — Active + cover / pathfinding / coordination.
 *   DeepReasoning — Tactical + LLM dispatch + memory recall + dialogue.
 *
 * Inputs to the priority score (weighted; weights documented inline):
 *   - threat-to-player from current NPC state
 *   - distance to player (decays linearly out to 80 m)
 *   - story importance (story-critical id set + role-based default)
 *   - combat urgency (boost for attack/flank/retreat states)
 *   - persona disposition (high-aggression NPCs bias up; high-fear bias
 *     down out of combat but up while retreating)
 *
 * The think-tier mapping is intentionally not a strict function of the
 * priority — it's a layered rule that biases toward upgrading rather
 * than downgrading: story-critical NPCs always Deep; in-melee-range
 * NPCs always at least Tactical; etc.
 */

import type { Entity, EntityId, PersonaComponent, Scene, Vec3 } from "../scene/index.js";
import { Tag } from "../scene/index.js";

export type ThinkTier =
  | "Dormant"
  | "Simulated"
  | "Active"
  | "Tactical"
  | "DeepReasoning";

export interface NpcThinkContext {
  /** Entity id of the player (or other "camera"-bearing focus). */
  playerEntityId: EntityId;
  /** Optional set of NPCs that always get bumped to DeepReasoning. */
  storyCriticalIds?: ReadonlySet<EntityId>;
  /**
   * Optional resolver mapping an NPC entity to a "state" string. The
   * starter uses `ai.state` ∈ {attack, flank, retreat, search, ...};
   * AugEngine entities don't carry a uniform state field, so the caller
   * supplies one (e.g. reads `RecruitComponent.state` for recruits, the
   * encounter state-machine phase for bosses, agent observers, etc.).
   * When absent, every NPC is treated as `idle` for state-driven scoring.
   */
  resolveState?: (entityId: EntityId) => NpcCombatState;
}

export type NpcCombatState =
  | "idle"
  | "patrol"
  | "search"
  | "investigate"
  | "engage"
  | "flank"
  | "retreat"
  | "attack"
  | "dialogue"
  | "down";

export interface ScheduledNpc {
  entity: Entity;
  persona: PersonaComponent;
  /** Combined priority ∈ [0, ~1.5]. */
  priority: number;
  tier: ThinkTier;
  /** Distance from player at scheduling time, in metres. */
  distance: number;
  state: NpcCombatState;
  /** Short human-readable reason for the assigned tier. */
  reason: string;
}

export interface NpcThinkSchedulerOptions {
  /**
   * Distance (m) past which an NPC drops to Dormant by default.
   * Story-critical + currently-combat NPCs override this. Default 80 m.
   */
  dormantDistance?: number;
  /** Distance (m) inside which Tactical is always at least the floor. */
  tacticalDistance?: number;
  /** Distance (m) inside which Active is always at least the floor. */
  activeDistance?: number;
  /** Tweak the weight of each priority input — defaults sum to 1.05. */
  weights?: Partial<typeof DEFAULT_WEIGHTS>;
}

const DEFAULT_WEIGHTS = {
  threatToPlayer: 0.35,
  distance: 0.15,
  storyImportance: 0.2,
  combatUrgency: 0.25,
  memoryBonus: 0.05,
  /** Per-axis adjustment from `PersonaComponent.axes`. */
  dispositionBoost: 0.05,
} as const;

const STATE_THREAT: Record<NpcCombatState, number> = {
  idle: 0.1,
  patrol: 0.15,
  search: 0.45,
  investigate: 0.4,
  engage: 0.85,
  flank: 0.8,
  retreat: 0.3,
  attack: 1.0,
  dialogue: 0.5,
  down: 0,
};

const COMBAT_STATES: ReadonlySet<NpcCombatState> = new Set([
  "engage",
  "flank",
  "retreat",
  "attack",
]);

export class NpcThinkScheduler {
  private readonly dormantDistance: number;
  private readonly tacticalDistance: number;
  private readonly activeDistance: number;
  private readonly weights: typeof DEFAULT_WEIGHTS;

  constructor(opts: NpcThinkSchedulerOptions = {}) {
    this.dormantDistance = opts.dormantDistance ?? 80;
    this.tacticalDistance = opts.tacticalDistance ?? 18;
    this.activeDistance = opts.activeDistance ?? 45;
    this.weights = { ...DEFAULT_WEIGHTS, ...opts.weights };
  }

  /**
   * Walk every persona-bearing entity in the scene, score it, bucket it
   * into a think-tier. Returns the work-items sorted by priority
   * descending — callers slice off the top-N each frame to fit their
   * inference budget.
   */
  schedule(scene: Scene, ctx: NpcThinkContext): ScheduledNpc[] {
    const playerPos = readPosition(scene, ctx.playerEntityId);
    const items: ScheduledNpc[] = [];
    for (const entity of scene) {
      if (entity.id === ctx.playerEntityId) continue;
      const persona = entity.components[Tag.Persona] as PersonaComponent | undefined;
      if (!persona) continue;
      const pos = readPosition(scene, entity.id);
      const distance = distanceOf(playerPos, pos);
      const state: NpcCombatState = ctx.resolveState
        ? ctx.resolveState(entity.id) ?? "idle"
        : "idle";
      const storyCritical = ctx.storyCriticalIds?.has(entity.id) === true;
      const priority = this.score(persona, distance, state, storyCritical);
      const { tier, reason } = this.tierFor({
        priority,
        distance,
        state,
        storyCritical,
        role: persona.role,
      });
      items.push({ entity, persona, priority, tier, distance, state, reason });
    }
    items.sort((a, b) => b.priority - a.priority);
    return items;
  }

  private score(
    persona: PersonaComponent,
    distance: number,
    state: NpcCombatState,
    storyCritical: boolean,
  ): number {
    const w = this.weights;
    const threatToPlayer = STATE_THREAT[state] ?? 0.2;
    const distanceScore = Math.max(0, 1 - distance / this.dormantDistance);
    const storyImportance = storyCritical ? 1 : roleStoryImportance(persona.role);
    const combatUrgency = COMBAT_STATES.has(state) ? 1 : 0.25;
    const memoryBonus = persona.memoryId ? 1 : 0;
    const disposition = dispositionAdjustment(persona, state);
    return (
      threatToPlayer * w.threatToPlayer +
      distanceScore * w.distance +
      storyImportance * w.storyImportance +
      combatUrgency * w.combatUrgency +
      memoryBonus * w.memoryBonus +
      disposition * w.dispositionBoost
    );
  }

  private tierFor(input: {
    priority: number;
    distance: number;
    state: NpcCombatState;
    storyCritical: boolean;
    role: string | undefined;
  }): { tier: ThinkTier; reason: string } {
    const { priority, distance, state, storyCritical, role } = input;
    if (state === "down") {
      return { tier: "Dormant", reason: "NPC is down" };
    }
    if (storyCritical || role === "boss") {
      return { tier: "DeepReasoning", reason: "story-critical / boss role" };
    }
    if (state === "dialogue") {
      return { tier: "DeepReasoning", reason: "active dialogue requires LLM dispatch" };
    }
    if (priority > 0.82) {
      return { tier: "DeepReasoning", reason: `priority ${priority.toFixed(2)} ≥ 0.82` };
    }
    if (priority > 0.65 || distance < this.tacticalDistance) {
      return {
        tier: "Tactical",
        reason: distance < this.tacticalDistance
          ? `melee range (${distance.toFixed(1)} m)`
          : `priority ${priority.toFixed(2)} ≥ 0.65`,
      };
    }
    if (priority > 0.38 || distance < this.activeDistance) {
      return {
        tier: "Active",
        reason: distance < this.activeDistance
          ? `active range (${distance.toFixed(1)} m)`
          : `priority ${priority.toFixed(2)} ≥ 0.38`,
      };
    }
    if (priority > 0.18 || distance < this.dormantDistance) {
      return {
        tier: "Simulated",
        reason: `simulated range (${distance.toFixed(1)} m, p=${priority.toFixed(2)})`,
      };
    }
    return { tier: "Dormant", reason: `far + low priority (${distance.toFixed(1)} m)` };
  }
}

function readPosition(scene: Scene, id: EntityId): Vec3 {
  const e = scene.get(id);
  if (!e) return [0, 0, 0];
  const tx = e.components[Tag.Transform];
  return tx ? tx.position : [0, 0, 0];
}

function distanceOf(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function roleStoryImportance(role: string | undefined): number {
  switch (role) {
    case "boss":
      return 0.9;
    case "ally":
      return 0.55;
    case "civilian":
      return 0.15;
    default:
      return 0.2;
  }
}

/**
 * Persona-axis adjustment ∈ [-1, 1]. Aggressive NPCs bias up in
 * combat; fearful NPCs bias up while retreating (they're working
 * harder to find safety); curious NPCs bias up on investigate;
 * loyal NPCs bias up when story-critical NPCs are in play.
 */
function dispositionAdjustment(persona: PersonaComponent, state: NpcCombatState): number {
  if (!persona.axes) return 0;
  const a = persona.axes;
  switch (state) {
    case "engage":
    case "attack":
    case "flank":
      return a.aggression - 0.5;
    case "retreat":
      return a.fear - 0.5;
    case "search":
    case "investigate":
      return a.curiosity - 0.5;
    case "dialogue":
      return a.loyalty - 0.5;
    default:
      return 0;
  }
}

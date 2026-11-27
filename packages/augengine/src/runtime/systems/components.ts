/**
 * Track 5 — Game-native ECS components.
 *
 * Smoke-game state (player health, weapons, recruits, objectives, mission
 * checkpoints, Maya memory consequences, squad trust) used to live as a
 * tangle of local closures inside `examples/smoke-game/src/main.ts`. This
 * file defines the canonical Entity/Component shapes those systems read
 * and write so the loop becomes pure ECS:
 *
 *   - `HealthComponent`      → who can take damage + die
 *   - `ArmorComponent`       → optional damage-absorber
 *   - `WeaponComponent`      → magazine / reserve / cooldown / reload state
 *   - `ObjectiveComponent`   → securable / extractable / capturable goal
 *   - `MissionComponent`     → singleton per-scene round state (timer/score)
 *   - `RecruitComponent`     → patrol/engage/retreat AI state
 *   - `SquadTrustComponent`  → per-NPC trust ledger + audit trail
 *   - `MayaMemoryComponent`  → narrative consequences mirrored to the brain
 *
 * The components are stored on `Entity.components` under string keys
 * (`GameTag.*`) to keep the renderer / physics `Tag` union surface
 * unchanged. Systems narrow with `getGameComponent()` so the cast is in
 * one place.
 *
 * Implementation rule: components hold *data only*. Behaviour lives in
 * `systems/`. Tests rely on that split — call `update()` and assert on
 * the component delta with zero rendering.
 */

import type { Entity, EntityId, Vec3 } from "../../scene/types.js";

/**
 * Game-component tag namespace. Distinct from the engine's `Tag` so
 * renderer code paths stay untouched. Keys are short to keep bundle
 * cost down (string literals deduplicate at minify time).
 */
export const GameTag = {
  Health: "game.health",
  Armor: "game.armor",
  Weapon: "game.weapon",
  Objective: "game.objective",
  Mission: "game.mission",
  Recruit: "game.recruit",
  SquadTrust: "game.squad_trust",
  MayaMemory: "game.maya_memory",
  Vehicle: "game.vehicle",
  Racer: "game.racer",
  RaceState: "game.race_state",
  UI: "game.ui",
  VfxParticle: "game.vfx_particle",
} as const;

export type GameTagKey = typeof GameTag[keyof typeof GameTag];

// ─── Health ──────────────────────────────────────────────────────────

export interface HealthComponent {
  hp: number;
  maxHp: number;
  /** Set true once `hp <= 0`. Systems should stop ticking the entity. */
  dead: boolean;
  /** ms timestamp of the last damage event; -1 if never hit. */
  lastDamageAt: number;
}

// ─── Armor ───────────────────────────────────────────────────────────

export interface ArmorComponent {
  /** Current armor pool. */
  value: number;
  /** Maximum armor pool. */
  max: number;
  /**
   * Fraction (0..1) of incoming damage armor absorbs before it drains.
   * Smoke-game default = 0.5 (armor takes half).
   */
  absorptionFraction: number;
}

// ─── Weapon ──────────────────────────────────────────────────────────

export interface WeaponComponent {
  /** Display name on the HUD. */
  name: string;
  /** Magazine ammo currently loaded. */
  ammo: number;
  /** Magazine capacity. */
  magSize: number;
  /** Reserve ammo (drawn into mag on reload). */
  reserve: number;
  /** Cap on reserve pool. */
  reserveMax: number;
  /** Damage per shot landed. */
  shotDamage: number;
  /** Seconds between shots when firing. */
  fireCooldown: number;
  /** Seconds the weapon is locked while reloading. */
  reloadLockout: number;
  /** Decremented each frame; > 0 means can't fire. */
  cooldownTimer: number;
  /** Decremented each frame; > 0 means reload in progress. */
  reloadTimer: number;
  /** Pitch kick (radians) applied this frame on fire — read once then decays. */
  recoilKick: number;
}

// ─── Objective ───────────────────────────────────────────────────────

/**
 * One securable, extractable or capturable goal in the scene. The
 * objective system walks entities with this component each tick.
 */
export interface ObjectiveComponent {
  /** What kind of objective this is. Drives the system's update branch. */
  kind: "secure" | "extract" | "capture";
  /** Identifying tag (e.g. "block13.extraction", "crate_3"). */
  goalId: string;
  /** True once the win condition for this objective fires. */
  completed: boolean;
  /**
   * Capture: world position of the ring centre. Extract: zone centre.
   * Secure: ignored.
   */
  position?: Vec3;
  /** Capture/extract radius in metres. */
  radius?: number;
  /** Seconds the player must stand inside to capture. */
  requiredSeconds?: number;
  /** Seconds accumulated so far (capture only). */
  progressSeconds?: number;
  /** Points awarded on completion. */
  scoreReward?: number;
}

// ─── Mission (round-level singleton) ─────────────────────────────────

export interface MissionComponent {
  /** Map id this mission is anchored to. Used for telemetry only. */
  mapId: string;
  /** Round duration in seconds. */
  durationSec: number;
  /** Seconds remaining; clamped to 0 at end of round. */
  timeLeftSec: number;
  /** Live score. Authoritative; HUD reads from here. */
  score: number;
  /** Set true on first win or loss; downstream commits become no-ops. */
  ended: boolean;
  /** Set true on win; false (or unset) means loss / open. */
  won: boolean;
  /** Why the round ended — surfaced in overlays. */
  endReason: "open" | "won" | "timeout" | "death";
  /** Per-objective checkpoint names that have fired this run. */
  checkpoints: string[];
}

// ─── Recruit AI ──────────────────────────────────────────────────────

export type RecruitFsmState =
  | "patrol"
  | "investigate"
  | "engage"
  | "retreat"
  | "down";

export interface RecruitComponent {
  state: RecruitFsmState;
  /** HP shadow — mirrors `HealthComponent.hp` so the FSM can branch on it cheaply. */
  hp: number;
  maxHp: number;
  /** Index into `patrolPath`. */
  patrolIndex: number;
  patrolPath: ReadonlyArray<Vec3>;
  /** World-direction facing yaw in radians. */
  facingYaw: number;
  /** Seconds the player has been continuously in cone. */
  detectionTimer: number;
  /** Seconds remaining in the current investigate trip. */
  investigateTimer: number;
  /** Position of the sound that triggered investigate, if any. */
  investigateTarget: Vec3 | null;
  /** Seconds until next melee swing is allowed. */
  meleeTimer: number;
  /** Tunables that match the smoke-game's prior defaults. */
  visionRange: number;
  visionFovCos: number;
  reactionDelay: number;
  retreatThreshold: number;
  patrolSpeed: number;
  engageSpeed: number;
  retreatSpeed: number;
  meleeRange: number;
  meleeDamage: number;
  meleeCooldown: number;
}

// ─── Squad trust ─────────────────────────────────────────────────────

export type TrustTier =
  | "hostile"
  | "distrustful"
  | "neutral"
  | "cooperative"
  | "loyal";

export interface SquadTrustComponent {
  /** Display name for HUD copy. */
  name: string;
  /** Trust value in 0..100. */
  score: number;
  /** Discrete tier derived from `score`. */
  tier: TrustTier;
  /** Bounded audit log — newest entries at the end, capped at 32. */
  auditTrail: string[];
}

// ─── Maya memory consequences ────────────────────────────────────────

/**
 * Tracks the narrative side-effects of player actions that Maya
 * remembers across sessions. Mirrored to the brain via the
 * `mission-system` so portable memory keeps working.
 */
export interface MayaMemoryComponent {
  /** Number of friendly-fire incidents seen this run. */
  friendlyFireCount: number;
  /** Number of objectives secured while Maya was nearby. */
  objectivesWitnessed: number;
  /** Number of kills the player landed with Maya in line-of-sight. */
  killsWitnessed: number;
  /** True once Maya considers the player betrayed her (≥ 2 friendly fire). */
  betrayed: boolean;
  /** Last narrative line Maya internalised about the player. */
  lastConsequence: string | null;
  /** ms timestamp of the last consequence write. */
  lastConsequenceAt: number;
}

// ─── Vehicle ─────────────────────────────────────────────────────────

export interface VehicleComponent {
  /** Throttle input: 1.0 (accelerate) to -1.0 (reverse/brake). */
  throttle: number;
  /** Steering input: -1.0 (steer left) to 1.0 (steer right). */
  steering: number;
  /** Handbrake toggle. Triggers slide/drift calculations. */
  handbrake: boolean;

  /** Maximum engine thrust force. */
  enginePower: number;
  /** Maximum speed target in meters/second. */
  maxSpeed: number;
  /** Deceleration rate applied when braking. */
  brakeForce: number;
  /** Maximum steering rotation speed in radians/second. */
  turnSpeed: number;
  /** Air resistance coefficient. Scales quadratically with speed. */
  aerodynamicDrag: number;
  /** Rolling resistance coefficient (friction coefficient). */
  rollingResistance: number;
  /** Lateral grip traction. 0.0 (ice/sliding) to 1.0 (sticky rails). */
  tireGrip: number;
  /** Suspension spring stiffness constant (Hooke's Law). */
  suspensionStiffness: number;
  /** Suspension damper constant (avoids infinite bouncy oscillation). */
  suspensionDamping: number;
  /** Suspension length at rest (distance to ground). */
  suspensionRestLength: number;
  /** Center of mass offset (usually slightly below pivot to prevent flipping). */
  centerOfMassOffset: number;
  /** Vehicle mass in kilograms. */
  mass: number;

  /** Current velocity vector. */
  velocity: Vec3;
  /** Current vehicle forward yaw heading (radians). */
  yaw: number;
  /** Grounded state flag. True if suspension springs are bearing weight. */
  isGrounded: boolean;
  /** Normal of the surface directly below the vehicle. */
  groundNormal: Vec3;
  /** Current surface type the vehicle is on. */
  surfaceType?: "road" | "water" | "timber";
  /** Current water spray/wake intensity (0.0 to 1.0). */
  waterSpray?: number;
}

// ─── Racer ───────────────────────────────────────────────────────────

export interface RacerComponent {
  /** Unique identifier of the racer (matching racers.mjs IDs or "player"). */
  racerId: string;
  /** Current lap index the racer is on. */
  currentLap: number;
  /** Index of the last passed checkpoint. */
  currentCheckpointIndex: number;
  /** Current rank position in the race. */
  currentRank: number;
  /** The text of the last active spoken dialogue/comment. */
  lastActiveDialogue?: string | null;
  /** ms timestamp or frame timer preventing dialog flooding. */
  dialogueCooldown?: number;
}

// ─── Race State ──────────────────────────────────────────────────────

export interface RaceStateComponent {
  /** Countdown timer in seconds before the race starts. Starts at 3.0, ticks down. */
  countdown: number;
  /** True once countdown reaches 0. */
  active: boolean;
  /** Array of racerIds that have finished all laps. */
  finishedRacerIds: string[];
  /** The stageId being raced. */
  stageId: string;
  /** Roster of racer IDs in current ranking order. */
  rankings: string[];
}

// ─── UI Overlay ──────────────────────────────────────────────────────

export interface UIComponent {
  /** HTML template string with placeholder bindings, e.g. "<div>HP: {{hp}}</div>" */
  template: string;
  /** Key-map linking variable names in the template to component fields. */
  bindings: Record<string, { componentTag: string; fieldKey: string }>;
  /** Cached rendered template output to prevent redundant DOM updates. */
  lastRendered?: string;
}

// ─── VFX Particle ───────────────────────────────────────────────────

export interface VfxParticleComponent {
  velocity: Vec3;
  color: [number, number, number];
  startOpacity: number;
  startScale: Vec3;
  age: number;
  lifetime: number;
}

// ─── Component accessors ─────────────────────────────────────────────

/**
 * Read a game component off an entity. Returns undefined if missing.
 * Encapsulates the one cast required to escape the renderer's `Tag`
 * union, so the rest of the codebase stays type-safe.
 */
export function getGameComponent<T>(entity: Entity, tag: GameTagKey): T | undefined {
  return (entity.components as Record<string, unknown>)[tag] as T | undefined;
}

/** Attach or replace a game component on an entity. */
export function setGameComponent<T>(entity: Entity, tag: GameTagKey, value: T): void {
  (entity.components as Record<string, unknown>)[tag] = value;
}

/** Remove a game component. Returns true if it existed. */
export function removeGameComponent(entity: Entity, tag: GameTagKey): boolean {
  const components = entity.components as Record<string, unknown>;
  if (!(tag in components)) return false;
  delete components[tag];
  return true;
}

/** ID for the singleton mission entity systems create on startup. */
export const MISSION_ENTITY_ID = "game.mission_singleton" as EntityId;

/**
 * Compute the trust tier for a numeric trust score. Pure function so
 * tests can pin the exact thresholds.
 */
export function trustTierForScore(score: number): TrustTier {
  if (score < 20) return "hostile";
  if (score < 40) return "distrustful";
  if (score < 60) return "neutral";
  if (score < 80) return "cooperative";
  return "loyal";
}

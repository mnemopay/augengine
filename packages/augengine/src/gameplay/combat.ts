/**
 * Combat primitives: bone-attached hurtboxes and hitboxes resolved by Rapier,
 * health with poise and invulnerability frames, knockback, hit-stop, camera
 * shake and a lock-on target helper.
 *
 * Hurtboxes are kinematic sensor colliders (root entities, because physics
 * requires roots) whose poses are copied from bone world matrices each step.
 * Hitboxes are not colliders: while active they query
 * `PhysicsSystem.overlapShape`, swept from the previous step's pose so a
 * 10 m/s sword tip cannot pass through a forearm between two steps. Each
 * target is hit at most once per activation.
 *
 * Order per fixed step:
 *   1. animate characters and apply poses to the scene
 *   2. `combat.syncHurtboxes()`        (bone → sensor transforms)
 *   3. `physics.stepFixed(scene, dt)`   (Rapier picks the new sensor poses up)
 *   4. `hitbox.query()` for each active hitbox → HitEvents
 */

import type { Scene } from "../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../scene/types.js";
import type { OverlapShape, PhysicsSystem } from "../physics/system.js";
import { humanoidBoneOf } from "../characters/humanoid.js";

// ──────────────────────────────────────────────────────────────────────
// Shapes attached to bones
// ──────────────────────────────────────────────────────────────────────

/**
 * A sphere on one bone, or a capsule spanning two bones (`toBone`). Bones are
 * glTF node names, or canonical humanoid ids ("leftUpperArm"). `offset` is in
 * metres along the bone's own axes (scale stripped), sphere only.
 */
export interface BoneShapeSpec {
  bone: string;
  toBone?: string;
  radius: number;
  offset?: Vec3;
  /** Extend a two-bone capsule past `toBone` by this many metres (blade tips). */
  extend?: number;
}

export interface HurtboxSpec extends BoneShapeSpec {
  /** Damage zone label ("head", "body", "limb"). */
  zone?: string;
  /** Damage multiplier for this zone. Default 1. */
  damageScale?: number;
}

export interface HitEvent {
  attacker: string;
  target: string;
  zone: string;
  damageScale: number;
  /** World point between the hitbox and the struck hurtbox. */
  point: Vec3;
  /** Horizontal unit vector attacker → target (knockback direction). */
  direction: Vec3;
}

/** Map bone names (exact, case-insensitive, or humanoid id) to entity ids of a glTF instance. */
export function resolveBones(nodeNames: readonly string[], nodeToEntity: ReadonlyMap<number, EntityId>, bones: readonly string[]): EntityId[] {
  return bones.map((b) => {
    const lower = b.toLowerCase();
    let i = nodeNames.findIndex((n) => n === b);
    if (i < 0) i = nodeNames.findIndex((n) => n.toLowerCase() === lower);
    if (i < 0) i = nodeNames.findIndex((n) => humanoidBoneOf(n)?.toLowerCase() === lower);
    const id = i >= 0 ? nodeToEntity.get(i) : undefined;
    if (!id) throw new Error(`combat: bone "${b}" not found`);
    return id;
  });
}

interface ResolvedShape {
  a: EntityId;
  b: EntityId | null;
  radius: number;
  offset: Vec3 | null;
  extend: number;
}

interface ShapePose {
  center: [number, number, number];
  rotation: [number, number, number, number];
  shape: OverlapShape;
}

function worldOf(scene: Scene, id: EntityId): Float32Array | null {
  const wt = scene.worldTransforms();
  const i = wt.indexById.get(id);
  return i === undefined ? null : wt.worldMatrices.subarray(i * 16, i * 16 + 16);
}

/** Shortest rotation taking +Y onto unit vector d. */
function quatFromY(dx: number, dy: number, dz: number): [number, number, number, number] {
  // cross(Y, d) = (dz, 0, -dx), dot = dy
  if (dy < -0.999999) return [1, 0, 0, 0];
  const x = dz, y = 0, z = -dx, w = 1 + dy;
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
}

/** Unit quaternion of a (possibly scaled) column-major matrix's rotation. */
function quatOfMatrix(m: Float32Array): [number, number, number, number] {
  const sx = Math.hypot(m[0]!, m[1]!, m[2]!) || 1, sy = Math.hypot(m[4]!, m[5]!, m[6]!) || 1, sz = Math.hypot(m[8]!, m[9]!, m[10]!) || 1;
  const r00 = m[0]! / sx, r10 = m[1]! / sx, r20 = m[2]! / sx, r01 = m[4]! / sy, r11 = m[5]! / sy, r21 = m[6]! / sy, r02 = m[8]! / sz, r12 = m[9]! / sz, r22 = m[10]! / sz;
  const tr = r00 + r11 + r22;
  let q: [number, number, number, number];
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s]; }
  else if (r00 > r11 && r00 > r22) { const s = Math.sqrt(1 + r00 - r11 - r22) * 2; q = [0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s]; }
  else if (r11 > r22) { const s = Math.sqrt(1 + r11 - r00 - r22) * 2; q = [(r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s]; }
  else { const s = Math.sqrt(1 + r22 - r00 - r11) * 2; q = [(r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s]; }
  const l = Math.hypot(...q) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

function poseOf(scene: Scene, s: ResolvedShape): ShapePose | null {
  const ma = worldOf(scene, s.a);
  if (!ma) return null;
  if (s.b) {
    const mb = worldOf(scene, s.b);
    if (!mb) return null;
    const ax = ma[12]!, ay = ma[13]!, az = ma[14]!;
    let bx = mb[12]!, by = mb[13]!, bz = mb[14]!;
    let dx = bx - ax, dy = by - ay, dz = bz - az;
    let len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) return { center: [ax, ay, az], rotation: [0, 0, 0, 1], shape: { kind: "sphere", radius: s.radius } };
    if (s.extend) {
      const k = (len + s.extend) / len;
      bx = ax + dx * k; by = ay + dy * k; bz = az + dz * k;
      dx = bx - ax; dy = by - ay; dz = bz - az;
      len = Math.hypot(dx, dy, dz);
    }
    return {
      center: [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2],
      rotation: quatFromY(dx / len, dy / len, dz / len),
      shape: { kind: "capsule", radius: s.radius, halfHeight: len / 2 },
    };
  }
  let cx = ma[12]!, cy = ma[13]!, cz = ma[14]!;
  if (s.offset) {
    // Offset along the bone's normalised axes (metres, scale stripped).
    for (let col = 0; col < 3; col++) {
      const sc = Math.hypot(ma[col * 4]!, ma[col * 4 + 1]!, ma[col * 4 + 2]!) || 1;
      const o = s.offset[col]!;
      cx += (ma[col * 4]! / sc) * o;
      cy += (ma[col * 4 + 1]! / sc) * o;
      cz += (ma[col * 4 + 2]! / sc) * o;
    }
  }
  return { center: [cx, cy, cz], rotation: quatOfMatrix(ma), shape: { kind: "sphere", radius: s.radius } };
}

// ──────────────────────────────────────────────────────────────────────
// Combat world: hurtbox registry
// ──────────────────────────────────────────────────────────────────────

interface HurtboxRecord {
  owner: string;
  zone: string;
  damageScale: number;
  shape: ResolvedShape;
  enabled: boolean;
}

export interface CombatantRig {
  /** glTF node names in node order (`AnimRig.names` or the JSON nodes' names). */
  nodeNames: readonly string[];
  nodeToEntity: ReadonlyMap<number, EntityId>;
}

export class CombatWorld {
  private readonly hurtboxes = new Map<EntityId, HurtboxRecord>();
  private readonly owners = new Map<string, { team: string; ids: EntityId[] }>();

  constructor(readonly scene: Scene, readonly physics: PhysicsSystem) {}

  /** Add a combatant's hurtboxes as kinematic sensor entities (`<owner>_hurt_<n>`). */
  register(owner: string, team: string, rig: CombatantRig, specs: readonly HurtboxSpec[]): EntityId[] {
    if (this.owners.has(owner)) throw new Error(`combat: "${owner}" already registered`);
    const ids: EntityId[] = [];
    specs.forEach((spec, n) => {
      const [a, b] = resolveBones(rig.nodeNames, rig.nodeToEntity, spec.toBone ? [spec.bone, spec.toBone] : [spec.bone]);
      const shape: ResolvedShape = { a: a!, b: b ?? null, radius: spec.radius, offset: spec.offset ?? null, extend: spec.extend ?? 0 };
      const id = `${owner}_hurt_${n}` as EntityId;
      const pose = poseOf(this.scene, shape) ?? { center: [0, -1000, 0] as [number, number, number], rotation: [0, 0, 0, 1] as [number, number, number, number], shape: { kind: "sphere", radius: spec.radius } as OverlapShape };
      this.scene.add({
        id,
        name: `${owner} hurtbox ${spec.zone ?? n}`,
        components: {
          [Tag.Transform]: { position: pose.center, rotation: pose.rotation, scale: [1, 1, 1], parent: null },
          [Tag.Collider]: pose.shape.kind === "capsule"
            ? { shape: "capsule", radius: spec.radius, halfHeight: pose.shape.halfHeight, sensor: true }
            : { shape: "sphere", radius: spec.radius, sensor: true },
          [Tag.RigidBody]: { velocity: [0, 0, 0], angularVelocity: [0, 0, 0], mass: 1, isKinematic: true },
        },
      });
      this.hurtboxes.set(id, { owner, zone: spec.zone ?? "body", damageScale: spec.damageScale ?? 1, shape, enabled: true });
      ids.push(id);
    });
    this.owners.set(owner, { team, ids });
    return ids;
  }

  unregister(owner: string): void {
    const o = this.owners.get(owner);
    if (!o) return;
    for (const id of o.ids) {
      this.hurtboxes.delete(id);
      this.scene.remove(id);
    }
    this.owners.delete(owner);
  }

  /** Disable (dead, dodging) or re-enable every hurtbox of an owner. */
  setEnabled(owner: string, enabled: boolean): void {
    for (const id of this.owners.get(owner)?.ids ?? []) {
      const h = this.hurtboxes.get(id);
      if (h) h.enabled = enabled;
    }
  }

  teamOf(owner: string): string | undefined {
    return this.owners.get(owner)?.team;
  }

  /** Hurtbox entity → owner record (undefined for anything else). */
  hurtbox(id: EntityId): Readonly<{ owner: string; zone: string; damageScale: number; enabled: boolean }> | undefined {
    return this.hurtboxes.get(id);
  }

  /**
   * Copy bone poses into the sensor transforms. Call after animation is
   * applied and before `physics.stepFixed`. All poses are read from one
   * world-transform pass before any sensor is written (writing dirties the
   * scene, so interleaving would recompute every world matrix per hurtbox).
   * Capsule length stays at its registration value: rebuilding Rapier
   * colliders every time an elbow bends costs more than the < 10 % length
   * error is worth.
   */
  syncHurtboxes(): void {
    const poses: Array<[EntityId, ShapePose]> = [];
    for (const [id, h] of this.hurtboxes) {
      const pose = poseOf(this.scene, h.shape);
      if (pose) poses.push([id, pose]);
    }
    for (const [id, pose] of poses) this.scene.setLocalTransform(id, pose.center, pose.rotation, [1, 1, 1]);
  }
}

// ──────────────────────────────────────────────────────────────────────
// Hitboxes
// ──────────────────────────────────────────────────────────────────────

export class Hitbox {
  private readonly shape: ResolvedShape;
  private readonly hit = new Set<string>();
  private prev: ShapePose | null = null;
  private on = false;
  /** Owners never hit by this hitbox (besides the attacker itself). */
  readonly ignoreOwners = new Set<string>();
  /** Hit allies too. Default false: same-team hurtboxes are skipped. */
  friendlyFire = false;

  constructor(private readonly world: CombatWorld, readonly owner: string, rig: CombatantRig, spec: BoneShapeSpec) {
    const [a, b] = resolveBones(rig.nodeNames, rig.nodeToEntity, spec.toBone ? [spec.bone, spec.toBone] : [spec.bone]);
    this.shape = { a: a!, b: b ?? null, radius: spec.radius, offset: spec.offset ?? null, extend: spec.extend ?? 0 };
  }

  get active(): boolean {
    return this.on;
  }

  /** Start an activation (swing). Targets hit in a previous activation can be hit again. */
  begin(): void {
    this.on = true;
    this.hit.clear();
    this.prev = poseOf(this.world.scene, this.shape);
  }

  end(): void {
    this.on = false;
    this.prev = null;
  }

  /** Current world pose of the hitbox shape (for debug drawing). */
  pose(): ShapePose | null {
    return poseOf(this.world.scene, this.shape);
  }

  /**
   * New hits since the last query. Sweeps from the previous pose in steps no
   * longer than the hitbox radius. Call after `physics.stepFixed`.
   */
  query(): HitEvent[] {
    if (!this.on) return [];
    const cur = poseOf(this.world.scene, this.shape);
    if (!cur) return [];
    const prev = this.prev ?? cur;
    const travel = Math.hypot(cur.center[0] - prev.center[0], cur.center[1] - prev.center[1], cur.center[2] - prev.center[2]);
    const steps = Math.max(1, Math.min(8, Math.ceil(travel / Math.max(0.05, this.shape.radius))));
    const out: HitEvent[] = [];
    const myTeam = this.world.teamOf(this.owner);
    const filter = (id: EntityId) => {
      const h = this.world.hurtbox(id);
      if (!h || !h.enabled || h.owner === this.owner || this.hit.has(h.owner) || this.ignoreOwners.has(h.owner)) return false;
      if (!this.friendlyFire && myTeam !== undefined && this.world.teamOf(h.owner) === myTeam) return false;
      return true;
    };
    for (let k = 1; k <= steps; k++) {
      const t = k / steps;
      const c: Vec3 = [prev.center[0] + (cur.center[0] - prev.center[0]) * t, prev.center[1] + (cur.center[1] - prev.center[1]) * t, prev.center[2] + (cur.center[2] - prev.center[2]) * t];
      const rot = t < 1 ? nlerpQ(prev.rotation, cur.rotation, t) : cur.rotation;
      const shape: OverlapShape = cur.shape.kind === "capsule" && prev.shape.kind === "capsule"
        ? { kind: "capsule", radius: cur.shape.radius, halfHeight: prev.shape.halfHeight + (cur.shape.halfHeight - prev.shape.halfHeight) * t }
        : cur.shape;
      for (const id of this.world.physics.overlapShape(shape, c, rot, { filter })) {
        const h = this.world.hurtbox(id)!;
        if (this.hit.has(h.owner)) continue;
        this.hit.add(h.owner);
        const tp = this.world.scene.get(id)?.components[Tag.Transform]?.position ?? c;
        const ap = attackerRootPos(this.world.scene, this.shape.a);
        out.push({
          attacker: this.owner,
          target: h.owner,
          zone: h.zone,
          damageScale: h.damageScale,
          point: [(c[0] + tp[0]) / 2, (c[1] + tp[1]) / 2, (c[2] + tp[2]) / 2],
          direction: horizontalDir(ap, tp),
        });
      }
    }
    this.prev = cur;
    return out;
  }
}

function attackerRootPos(scene: Scene, bone: EntityId): Vec3 {
  // Walk to the top of the bone's hierarchy: the character root.
  let id: EntityId | null = bone;
  let last: EntityId = bone;
  for (let guard = 0; id && guard < 64; guard++) {
    last = id;
    id = scene.get(id)?.components[Tag.Transform]?.parent ?? null;
  }
  const m = worldOf(scene, last);
  return m ? [m[12]!, m[13]!, m[14]!] : [0, 0, 0];
}

function nlerpQ(a: readonly number[], b: readonly number[], t: number): [number, number, number, number] {
  const s = a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]! + a[3]! * b[3]! < 0 ? -1 : 1;
  const q: [number, number, number, number] = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) q[i] = a[i]! + (b[i]! * s - a[i]!) * t;
  const l = Math.hypot(...q) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/** Horizontal unit vector from a to b (falls back to +Z when they coincide). */
export function horizontalDir(a: Vec3, b: Vec3): Vec3 {
  const dx = b[0] - a[0], dz = b[2] - a[2];
  const l = Math.hypot(dx, dz);
  return l > 1e-6 ? [dx / l, 0, dz / l] : [0, 0, 1];
}

// ──────────────────────────────────────────────────────────────────────
// Health, damage, knockback
// ──────────────────────────────────────────────────────────────────────

export interface Health {
  hp: number;
  max: number;
  /** Stagger resistance: damage drains it; at 0 the target staggers and poise refills. */
  poise: number;
  maxPoise: number;
  /** Poise regained per second when not hit for `poiseDelay` seconds. */
  poiseRegen: number;
  poiseDelay: number;
  /** Seconds of invulnerability left. */
  invulnerable: number;
  /** Seconds since the last damage. */
  sinceHit: number;
  dead: boolean;
}

export function createHealth(max: number, opts: { poise?: number; poiseRegen?: number; poiseDelay?: number } = {}): Health {
  const poise = opts.poise ?? max * 0.4;
  return { hp: max, max, poise, maxPoise: poise, poiseRegen: opts.poiseRegen ?? poise * 0.5, poiseDelay: opts.poiseDelay ?? 1.5, invulnerable: 0, sinceHit: Infinity, dead: false };
}

export interface DamageOptions {
  /** Poise damage (default = the HP damage). */
  poiseDamage?: number;
  /** Invulnerability seconds granted after the hit (default 0). */
  iFrames?: number;
}

export interface DamageResult {
  /** HP actually removed (0 when invulnerable or already dead). */
  applied: number;
  killed: boolean;
  staggered: boolean;
  /** True when the hit was ignored (i-frames or dead). */
  ignored: boolean;
}

export function applyDamage(h: Health, amount: number, opts: DamageOptions = {}): DamageResult {
  if (h.dead || h.invulnerable > 0 || !(amount > 0)) return { applied: 0, killed: false, staggered: false, ignored: true };
  const applied = Math.min(h.hp, amount);
  h.hp -= applied;
  h.sinceHit = 0;
  h.invulnerable = opts.iFrames ?? 0;
  let staggered = false;
  h.poise -= opts.poiseDamage ?? amount;
  if (h.poise <= 0) {
    staggered = true;
    h.poise = h.maxPoise;
  }
  const killed = h.hp <= 0;
  if (killed) h.dead = true;
  return { applied, killed, staggered: staggered && !killed, ignored: false };
}

export function tickHealth(h: Health, dt: number): void {
  h.invulnerable = Math.max(0, h.invulnerable - dt);
  h.sinceHit += dt;
  // Regenerate only for the part of dt past the delay.
  const regenTime = Math.min(dt, h.sinceHit - h.poiseDelay);
  if (!h.dead && regenTime > 0) h.poise = Math.min(h.maxPoise, h.poise + h.poiseRegen * regenTime);
}

/** Knockback velocity: `strength` m/s along the hit direction, plus an optional upward pop. */
export function knockbackVelocity(direction: Vec3, strength: number, up = 0): Vec3 {
  return [direction[0] * strength, up, direction[2] * strength];
}

// ──────────────────────────────────────────────────────────────────────
// Hit-stop
// ──────────────────────────────────────────────────────────────────────

/**
 * Per-entity freeze timers. Gameplay multiplies each participant's dt by
 * `scale(id)` so attacker and target hang for a few frames on impact while
 * the rest of the world keeps running.
 */
export class HitStop {
  private readonly timers = new Map<string, number>();
  /** Time scale while frozen (0 = full stop). */
  frozenScale = 0;

  add(ids: readonly string[], seconds: number): void {
    for (const id of ids) this.timers.set(id, Math.max(this.timers.get(id) ?? 0, seconds));
  }

  /** dt multiplier for an entity right now. */
  scale(id: string): number {
    return (this.timers.get(id) ?? 0) > 0 ? this.frozenScale : 1;
  }

  /** Advance wall-clock timers (not scaled). */
  step(dt: number): void {
    for (const [id, t] of this.timers) {
      const n = t - dt;
      if (n <= 0) this.timers.delete(id);
      else this.timers.set(id, n);
    }
  }

  frozen(id: string): boolean {
    return (this.timers.get(id) ?? 0) > 0;
  }
}

// ──────────────────────────────────────────────────────────────────────
// Camera shake
// ──────────────────────────────────────────────────────────────────────

/**
 * Trauma-based shake: hits add trauma (0..1), it decays linearly, and the
 * offset scales with trauma² so small hits barely move the camera while a
 * kill thumps. Smooth deterministic noise per axis (seeded).
 */
export class CameraShake {
  trauma = 0;
  /** Trauma lost per second. */
  decay = 1.8;
  /** Offset at trauma 1, metres. */
  maxOffset = 0.22;
  /** Noise frequency, Hz. */
  frequency = 22;
  private t = 0;

  constructor(private readonly seed = 1) {}

  add(amount: number): void {
    this.trauma = Math.min(1, this.trauma + Math.max(0, amount));
  }

  /** Advance and return this frame's offset. */
  update(dt: number): [number, number, number] {
    this.t += dt;
    this.trauma = Math.max(0, this.trauma - this.decay * dt);
    const k = this.trauma * this.trauma * this.maxOffset;
    if (k === 0) return [0, 0, 0];
    const f = this.t * this.frequency;
    return [k * noise1(f, this.seed), k * noise1(f, this.seed + 17), k * 0.5 * noise1(f, this.seed + 31)];
  }
}

/** Smooth value noise in [-1, 1]. */
function noise1(x: number, seed: number): number {
  const i = Math.floor(x), f = x - i;
  const h = (n: number) => {
    let v = Math.imul(n ^ Math.imul(seed, 0x9e3779b1), 0x85ebca6b);
    v ^= v >>> 13;
    v = Math.imul(v, 0xc2b2ae35);
    v ^= v >>> 16;
    return ((v >>> 0) / 4294967295) * 2 - 1;
  };
  const u = f * f * (3 - 2 * f);
  return h(i) + (h(i + 1) - h(i)) * u;
}

// ──────────────────────────────────────────────────────────────────────
// Lock-on
// ──────────────────────────────────────────────────────────────────────

export interface LockCandidate {
  id: string;
  position: Vec3;
}

export interface LockOptions {
  /** Default 15 m. */
  maxDistance?: number;
  /** Half-angle of the acquisition cone around `forward`, radians. Default 70 degrees. */
  maxAngle?: number;
  /** Optional line-of-sight test. */
  visible?: (c: LockCandidate) => boolean;
}

/** Signed horizontal angle from forward to (to − origin), radians, positive = to the right (clockwise seen from above). */
export function signedAngle(origin: Vec3, forward: Vec3, to: Vec3): number {
  const dx = to[0] - origin[0], dz = to[2] - origin[2];
  const a = Math.atan2(forward[0], forward[2]);
  const b = Math.atan2(dx, dz);
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

/**
 * Best target in the cone: lowest 0.5·angle/maxAngle + 0.5·distance/maxDistance
 * (a near enemy a little off-centre beats a far one dead ahead, which suits melee).
 * Returns null when nothing qualifies.
 */
export function pickLockTarget(origin: Vec3, forward: Vec3, candidates: readonly LockCandidate[], opts: LockOptions = {}): string | null {
  const maxD = opts.maxDistance ?? 15, maxA = opts.maxAngle ?? (70 * Math.PI) / 180;
  let best: string | null = null, bestScore = Infinity;
  for (const c of candidates) {
    const d = Math.hypot(c.position[0] - origin[0], c.position[2] - origin[2]);
    if (d > maxD) continue;
    const a = Math.abs(signedAngle(origin, forward, c.position));
    if (a > maxA) continue;
    if (opts.visible && !opts.visible(c)) continue;
    const score = 0.5 * (a / maxA) + 0.5 * (d / maxD);
    if (score < bestScore) { bestScore = score; best = c.id; }
  }
  return best;
}

/** Next target to the right (dir 1) or left (dir -1) of the current one, wrapping. */
export function cycleLockTarget(origin: Vec3, forward: Vec3, candidates: readonly LockCandidate[], current: string | null, dir: 1 | -1, opts: LockOptions = {}): string | null {
  const maxD = opts.maxDistance ?? 15;
  const inRange = candidates
    .filter((c) => Math.hypot(c.position[0] - origin[0], c.position[2] - origin[2]) <= maxD && (!opts.visible || opts.visible(c)))
    .map((c) => ({ id: c.id, a: signedAngle(origin, forward, c.position) }))
    .sort((x, y) => x.a - y.a);
  if (inRange.length === 0) return null;
  const i = inRange.findIndex((c) => c.id === current);
  if (i < 0) return pickLockTarget(origin, forward, candidates, opts) ?? inRange[0]!.id;
  return inRange[(i + dir + inRange.length) % inRange.length]!.id;
}

/** Break the lock when the target is too far (or gone). */
export function shouldBreakLock(origin: Vec3, target: Vec3 | null, breakDistance = 20): boolean {
  return !target || Math.hypot(target[0] - origin[0], target[2] - origin[2]) > breakDistance;
}

/**
 * Kinematic character controller — capsule against static colliders.
 *
 * Tier 2 game-readiness (see PHASE-LEDGER.md). Walks the controller's capsule
 * through scene colliders one axis at a time so the resolution slides along
 * surfaces instead of stopping dead. Routes everything through the existing
 * `Tag.Collider` shapes so the smoke-game's PBR floor/container/crates are
 * physical without re-authoring them as meshes.
 *
 * Not a full physics body — the controller is deterministic, frame-paced, and
 * deliberately uncoupled from the dynamics solver. Other rigid bodies still
 * run through `PhysicsSystem`.
 */
import type { Scene } from "../scene/scene.js";
import { Tag, type EntityId, type HeightfieldShape, type Vec3, type Quat } from "../scene/types.js";
import { sampleHeightfieldAt } from "./heightfield.js";
import {
  capsuleAabb,
  capsuleCapsule,
  sphereCapsule,
  type AABB,
  type Capsule,
  type ContactManifold,
} from "./colliders.js";

export interface CharacterControllerOptions {
  /** Controller's entity in the scene — must own Transform + Collider(capsule). */
  entityId: EntityId;
  /** Horizontal speed in m/s. Default 4. */
  moveSpeed?: number;
  /** Upward velocity applied when jump is requested while grounded. Default 5. */
  jumpSpeed?: number;
  /** Gravity (negative = down). Default -20 (snappier than 9.81 for FPS feel). */
  gravity?: number;
  /** Auto-step over obstacles up to this height (meters). Default 0.3. */
  stepHeight?: number;
  /** Keep the capsule this far off colliders to avoid jitter. Default 0.005. */
  skinWidth?: number;
  /** Treat ground hits with normal.y >= this as walkable. Default 0.7 (~45°). */
  maxGroundSlope?: number;
  /**
   * When the controller was grounded last frame and walkable ground lies at
   * most this far below, glue it down instead of letting it fall in little
   * hops (walking down a hill). Default 0 (off).
   */
  snapDistance?: number;
}

export class KinematicCharacterController {
  private readonly entityId: EntityId;
  private moveSpeed: number;
  private readonly jumpSpeed: number;
  private readonly gravity: number;
  private readonly stepHeight: number;
  private readonly skinWidth: number;
  private readonly minGroundNormalY: number;
  private readonly snapDistance: number;
  private verticalVel = 0;
  private _grounded = false;
  // Slide budget per axis sweep — caps the inner loop in pathological geometry.
  private static readonly MAX_SLIDE_ITERATIONS = 4;

  constructor(private readonly scene: Scene, opts: CharacterControllerOptions) {
    this.entityId = opts.entityId;
    this.moveSpeed = opts.moveSpeed ?? 4;
    this.jumpSpeed = opts.jumpSpeed ?? 5;
    this.gravity = opts.gravity ?? -20;
    this.stepHeight = opts.stepHeight ?? 0.3;
    this.skinWidth = opts.skinWidth ?? 0.005;
    this.minGroundNormalY = opts.maxGroundSlope ?? 0.7;
    this.snapDistance = Math.max(0, opts.snapDistance ?? 0);
  }

  /** Reset vertical velocity — useful after teleport/respawn. */
  reset(): void {
    this.verticalVel = 0;
    this._grounded = false;
  }

  /** Runtime walk speed setter — used for sprint/crouch/slowdown effects. */
  setMoveSpeed(metersPerSecond: number): void {
    this.moveSpeed = Math.max(0, metersPerSecond);
  }

  get grounded(): boolean { return this._grounded; }
  get velocity(): Vec3 { return [0, this.verticalVel, 0]; }

  /**
   * Advance one frame.
   *
   * @param dt        Seconds since last update. Clamped to 0.1 internally.
   * @param moveInput World-space XZ movement direction (caller resolves yaw).
   *                  Magnitude becomes the input intensity (0..1 typical).
   * @param jump      If true and grounded, fires a single jump impulse.
   */
  update(dt: number, moveInput: Vec3, jump = false): void {
    const clampedDt = Math.min(dt, 0.1);
    const entity = this.scene.get(this.entityId);
    if (!entity) return;
    const tx = entity.components[Tag.Transform];
    const col = entity.components[Tag.Collider];
    if (!tx || !col || col.shape !== "capsule" || col.radius == null || col.halfHeight == null) {
      return;
    }
    const radius = col.radius;
    const halfHeight = col.halfHeight;
    let pos: Vec3 = [tx.position[0], tx.position[1], tx.position[2]];
    const wasGrounded = this._grounded;
    const jumped = this._grounded && jump;

    if (this._grounded && jump) {
      this.verticalVel = this.jumpSpeed;
      this._grounded = false;
    } else if (!this._grounded) {
      this.verticalVel += this.gravity * clampedDt;
    } else if (this.verticalVel < 0) {
      this.verticalVel = 0;
    }

    const staticColliders = this.collectStatics();

    // Horizontal pass — slide along contact normals.
    const dx = moveInput[0] * this.moveSpeed * clampedDt;
    const dz = moveInput[2] * this.moveSpeed * clampedDt;
    pos = this.slideMove(pos, [dx, 0, dz], radius, halfHeight, staticColliders);

    // Vertical pass — gravity + jump. Kill vertical velocity on hit.
    const dy = this.verticalVel * clampedDt;
    const verticalResult = this.sweep(pos, [0, dy, 0], radius, halfHeight, staticColliders);
    pos = verticalResult.position;
    if (verticalResult.hit) {
      if (dy <= 0 && verticalResult.normal[1] >= this.minGroundNormalY) {
        this._grounded = true;
      } else if (dy > 0 && verticalResult.normal[1] <= -0.5) {
        // Bumped head — kill upward velocity.
        this.verticalVel = 0;
      } else {
        this._grounded = false;
      }
      if ((dy <= 0 && verticalResult.normal[1] > 0) || (dy > 0 && verticalResult.normal[1] < 0)) {
        this.verticalVel = 0;
      }
    } else {
      this._grounded = false;
    }

    // Final ground probe — small downward cast so the controller stays glued
    // to slopes when moveInput pushed it off a step.
    if (!this._grounded && this.verticalVel <= 0) {
      const probe = this.sweep(pos, [0, -this.skinWidth * 4, 0], radius, halfHeight, staticColliders);
      if (probe.hit && probe.normal[1] >= this.minGroundNormalY) {
        pos = probe.position;
        this._grounded = true;
        this.verticalVel = 0;
      }
    }

    // Ground snap — stay on a downhill slope instead of hopping off it.
    if (!this._grounded && wasGrounded && !jumped && this.snapDistance > 0 && this.verticalVel <= 0) {
      const snap = this.sweep(pos, [0, -this.snapDistance, 0], radius, halfHeight, staticColliders);
      if (snap.hit && snap.normal[1] >= this.minGroundNormalY) {
        pos = snap.position;
        this._grounded = true;
        this.verticalVel = 0;
      }
    }

    const rot: Quat = tx.rotation;
    this.scene.setLocalTransform(this.entityId, pos, rot, tx.scale);
  }

  // ────────────────────────────────────────────────────────────────────
  // Internals
  // ────────────────────────────────────────────────────────────────────

  /** Move along `delta`, sliding along contact normals up to N iterations. */
  private slideMove(
    start: Vec3,
    delta: Vec3,
    radius: number,
    halfHeight: number,
    statics: ResolvedCollider[],
  ): Vec3 {
    let pos: Vec3 = start;
    let remaining: Vec3 = [delta[0], delta[1], delta[2]];
    for (let iter = 0; iter < KinematicCharacterController.MAX_SLIDE_ITERATIONS; iter++) {
      const lenSq = remaining[0] * remaining[0] + remaining[1] * remaining[1] + remaining[2] * remaining[2];
      if (lenSq < 1e-10) break;
      const stepResult = this.sweep(pos, remaining, radius, halfHeight, statics);
      pos = stepResult.position;
      if (!stepResult.hit) break;
      // Project remaining motion onto the contact plane.
      const dot = remaining[0] * stepResult.normal[0]
                + remaining[1] * stepResult.normal[1]
                + remaining[2] * stepResult.normal[2];
      remaining = [
        remaining[0] - stepResult.normal[0] * dot,
        remaining[1] - stepResult.normal[1] * dot,
        remaining[2] - stepResult.normal[2] * dot,
      ];
    }
    return pos;
  }

  /**
   * Move the capsule from `start` by `delta`, resolving the deepest contact.
   * Returns the final position and the hit normal if anything was struck.
   */
  private sweep(
    start: Vec3,
    delta: Vec3,
    radius: number,
    halfHeight: number,
    statics: ResolvedCollider[],
  ): SweepResult {
    let pos: Vec3 = [start[0] + delta[0], start[1] + delta[1], start[2] + delta[2]];
    let hit = false;
    let normal: Vec3 = [0, 1, 0];
    for (let i = 0; i < KinematicCharacterController.MAX_SLIDE_ITERATIONS; i++) {
      const cap = capsuleAt(pos, radius, halfHeight);
      const deepest = this.deepestContact(cap, statics);
      if (!deepest) break;
      const push = deepest.depth + this.skinWidth;
      pos = [
        pos[0] + deepest.normal[0] * push,
        pos[1] + deepest.normal[1] * push,
        pos[2] + deepest.normal[2] * push,
      ];
      normal = deepest.normal;
      hit = true;
    }
    return { position: pos, hit, normal };
  }

  /** Deepest manifold across all colliders (most-penetrating wins). */
  private deepestContact(cap: Capsule, statics: ResolvedCollider[]): ContactManifold | null {
    let deepest: ContactManifold | null = null;
    for (const c of statics) {
      const m = contactCapsuleAgainst(cap, c);
      if (!m) continue;
      if (!deepest || m.depth > deepest.depth) deepest = m;
    }
    return deepest;
  }

  /** Pull every other colliding entity out of the scene as a resolved shape. */
  private collectStatics(): ResolvedCollider[] {
    const out: ResolvedCollider[] = [];
    for (const entity of this.scene.withComponent(Tag.Collider)) {
      if (entity.id === this.entityId) continue;
      const col = entity.components[Tag.Collider]!;
      const tx = entity.components[Tag.Transform];
      if (!tx) continue;
      const resolved = resolveCollider(col, tx.position);
      if (resolved) out.push(resolved);
    }
    return out;
  }
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

interface SweepResult {
  position: Vec3;
  hit: boolean;
  normal: Vec3;
}

type ResolvedCollider =
  | { kind: "box"; aabb: AABB }
  | { kind: "sphere"; center: Vec3; radius: number }
  | { kind: "capsule"; capsule: Capsule }
  | { kind: "heightfield"; hf: HeightfieldShape; position: Vec3 };

function capsuleAt(position: Vec3, radius: number, halfHeight: number): Capsule {
  return {
    kind: "capsule",
    p0: [position[0], position[1] - halfHeight, position[2]],
    p1: [position[0], position[1] + halfHeight, position[2]],
    radius,
  };
}

/** Sizes are world-space (ColliderComponent); `centerOffset` shifts the shape as PhysicsSystem does. */
function resolveCollider(
  col: { shape: string; radius?: number; halfHeight?: number; extents?: Vec3; centerOffset?: Vec3; heightfield?: HeightfieldShape },
  at: Vec3,
): ResolvedCollider | null {
  const off = col.centerOffset;
  const pos: Vec3 = off ? [at[0] + off[0], at[1] + off[1], at[2] + off[2]] : at;
  if (col.shape === "heightfield" && col.heightfield) {
    return { kind: "heightfield", hf: col.heightfield, position: pos };
  }
  if (col.shape === "box" && col.extents) {
    return {
      kind: "box",
      aabb: {
        min: [pos[0] - col.extents[0], pos[1] - col.extents[1], pos[2] - col.extents[2]],
        max: [pos[0] + col.extents[0], pos[1] + col.extents[1], pos[2] + col.extents[2]],
      },
    };
  }
  if (col.shape === "sphere" && col.radius != null) {
    return { kind: "sphere", center: pos, radius: col.radius };
  }
  if (col.shape === "capsule" && col.radius != null && col.halfHeight != null) {
    return {
      kind: "capsule",
      capsule: {
        kind: "capsule",
        p0: [pos[0], pos[1] - col.halfHeight, pos[2]],
        p1: [pos[0], pos[1] + col.halfHeight, pos[2]],
        radius: col.radius,
      },
    };
  }
  return null;
}

function contactCapsuleAgainst(cap: Capsule, target: ResolvedCollider): ContactManifold | null {
  if (target.kind === "box") {
    const m = capsuleAabb(cap, target.aabb);
    if (!m) return null;
    // capsuleAabb returns normal pointing from box toward capsule (the way the
    // capsule should be pushed). That's already what we want.
    return m;
  }
  if (target.kind === "sphere") {
    const m = sphereCapsule({ kind: "sphere", center: target.center, radius: target.radius }, cap);
    if (!m) return null;
    // sphereCapsule normal is from sphere toward capsule? sphereSphere returns
    // normal from a toward b, so from sphere toward closest-point-on-capsule.
    // We want push direction = away from sphere, so flip.
    return { normal: [-m.normal[0], -m.normal[1], -m.normal[2]], depth: m.depth, point: m.point };
  }
  if (target.kind === "capsule") {
    const m = capsuleCapsule(cap, target.capsule);
    if (!m) return null;
    // capsuleCapsule via sphereSphere: normal points from cap (a) toward
    // target (b). Flip so it pushes cap away from target.
    return { normal: [-m.normal[0], -m.normal[1], -m.normal[2]], depth: m.depth, point: m.point };
  }
  if (target.kind === "heightfield") {
    return capsuleHeightfield(cap, target.hf, target.position);
  }
  return null;
}

/**
 * Capsule against terrain: the lower end-sphere against the plane of the
 * triangle beneath it. A sphere that has sunk below the surface (deep
 * penetration, a long frame) still gets pushed back up along the normal, so
 * the capsule cannot tunnel through the ground.
 */
function capsuleHeightfield(cap: Capsule, hf: HeightfieldShape, position: Vec3): ContactManifold | null {
  const low = cap.p0[1] <= cap.p1[1] ? cap.p0 : cap.p1;
  const hit = sampleHeightfieldAt(hf, position, low[0], low[2]);
  if (!hit) return null;
  const n = hit.normal;
  // Signed distance from the sphere centre to the triangle's plane.
  const d = (low[1] - hit.height) * n[1];
  if (d >= cap.radius) return null;
  const depth = cap.radius - d;
  return {
    normal: n,
    depth,
    point: [low[0] - n[0] * d, low[1] - n[1] * d, low[2] - n[2] * d],
  };
}

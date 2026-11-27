/**
 * Two-bone ragdoll — Phase 11.
 *
 * A minimal rigid-body chain solver that takes over a character's
 * skeleton on death-impact-frame and lets gravity + impulse drive the
 * body until it settles. Each ragdoll bone is a capsule with mass +
 * angular inertia; constraints between bones are pin joints with
 * optional swing/twist limits.
 *
 * Phase 11 ships the data structures + Verlet-integration step. Joint
 * limits beyond simple pin are deferred to Phase 11.1.
 */

import type { Vec3 } from "../scene/types.js";
import type { Capsule } from "./colliders.js";
import {
  v3Add,
  v3Sub,
  v3Length,
  v3Scale,
  type MutVec3,
} from "../scene/math.js";

export interface RagdollBone {
  /** Source skeleton joint index this bone is anchored to. */
  jointIndex: number;
  /** Capsule collider in world space; updated each step. */
  capsule: Capsule;
  /** Total mass; used for impulse application. */
  mass: number;
  /** Linear velocity. */
  velocity: MutVec3;
  /** Previous world position of capsule midpoint (for Verlet). */
  prevMid: MutVec3;
}

export interface RagdollPinJoint {
  /** Bone A index in `Ragdoll.bones`. */
  a: number;
  /** Bone B index. */
  b: number;
  /** Anchor point on A in world space (typically midpoint or capsule.p1). */
  anchorA: Vec3;
  /** Anchor point on B (typically capsule.p0). */
  anchorB: Vec3;
}

export interface RagdollOptions {
  bones: RagdollBone[];
  joints: RagdollPinJoint[];
  /** Gravity acceleration. Default [0, -9.81, 0]. */
  gravity?: Vec3;
  /** Linear damping (0..1 per second). Default 0.04. */
  damping?: number;
  /** Constraint solver iterations per step. */
  solverIterations?: number;
}

/** Verlet-integrated ragdoll. Step deterministically under fixed dt. */
export class Ragdoll {
  bones: RagdollBone[];
  joints: RagdollPinJoint[];
  gravity: Vec3;
  damping: number;
  solverIterations: number;

  constructor(opts: RagdollOptions) {
    this.bones = opts.bones;
    this.joints = opts.joints;
    this.gravity = opts.gravity ?? [0, -9.81, 0];
    this.damping = opts.damping ?? 0.04;
    this.solverIterations = opts.solverIterations ?? 6;
  }

  /** Apply an impulse at a world point on a bone (e.g., bullet hit). */
  applyImpulse(boneIdx: number, impulse: Vec3): void {
    const b = this.bones[boneIdx];
    if (!b) return;
    b.velocity[0] += impulse[0] / b.mass;
    b.velocity[1] += impulse[1] / b.mass;
    b.velocity[2] += impulse[2] / b.mass;
  }

  /**
   * Integrate one step of dt seconds. Calls in order:
   *   1. apply gravity + damping → update velocities
   *   2. integrate positions
   *   3. iterate constraint solver `solverIterations` times
   *   4. recompute velocities from new positions (Verlet)
   */
  step(dt: number): void {
    if (dt <= 0) return;

    // Save previous mid before stepping for Verlet velocity update.
    const prev: MutVec3[] = this.bones.map((b) => [
      midX(b.capsule), midY(b.capsule), midZ(b.capsule),
    ]);

    // 1. Apply gravity + damping → 2. integrate.
    for (let i = 0; i < this.bones.length; i++) {
      const b = this.bones[i]!;
      // Damping (1 - damping)^dt approximated linearly for small dt.
      const k = 1 - this.damping * dt;
      b.velocity[0] = b.velocity[0] * k + this.gravity[0] * dt;
      b.velocity[1] = b.velocity[1] * k + this.gravity[1] * dt;
      b.velocity[2] = b.velocity[2] * k + this.gravity[2] * dt;
      // Translate the capsule midpoint by velocity * dt.
      const mx = midX(b.capsule) + b.velocity[0] * dt;
      const my = midY(b.capsule) + b.velocity[1] * dt;
      const mz = midZ(b.capsule) + b.velocity[2] * dt;
      setMid(b.capsule, mx, my, mz);
    }

    // 3. Constraint solver — pin joints: enforce |posA - posB| ≈ |restLength|.
    // Pin joints here are treated as zero-length constraints between anchors.
    for (let iter = 0; iter < this.solverIterations; iter++) {
      for (const j of this.joints) {
        const A = this.bones[j.a]!;
        const B = this.bones[j.b]!;
        const aWorld = anchorWorld(A.capsule, j.anchorA);
        const bWorld = anchorWorld(B.capsule, j.anchorB);
        const diff: MutVec3 = [
          bWorld[0] - aWorld[0],
          bWorld[1] - aWorld[1],
          bWorld[2] - aWorld[2],
        ];
        const dist = v3Length(diff);
        if (dist < 1e-9) continue;
        // Move both anchors halfway toward each other (mass-weighted).
        const totalMass = A.mass + B.mass;
        const wA = B.mass / totalMass;
        const wB = A.mass / totalMass;
        translateCapsuleMid(A.capsule, diff[0] * wA, diff[1] * wA, diff[2] * wA);
        translateCapsuleMid(B.capsule, -diff[0] * wB, -diff[1] * wB, -diff[2] * wB);
      }
    }

    // 4. Recompute velocities (Verlet implicit form).
    for (let i = 0; i < this.bones.length; i++) {
      const b = this.bones[i]!;
      const mx = midX(b.capsule);
      const my = midY(b.capsule);
      const mz = midZ(b.capsule);
      b.velocity[0] = (mx - prev[i]![0]) / dt;
      b.velocity[1] = (my - prev[i]![1]) / dt;
      b.velocity[2] = (mz - prev[i]![2]) / dt;
      b.prevMid[0] = mx;
      b.prevMid[1] = my;
      b.prevMid[2] = mz;
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// Capsule helpers (mid-point math)
// ──────────────────────────────────────────────────────────────────────

function midX(c: Capsule): number { return (c.p0[0] + c.p1[0]) * 0.5; }
function midY(c: Capsule): number { return (c.p0[1] + c.p1[1]) * 0.5; }
function midZ(c: Capsule): number { return (c.p0[2] + c.p1[2]) * 0.5; }

function setMid(c: Capsule, x: number, y: number, z: number): void {
  const cx = midX(c), cy = midY(c), cz = midZ(c);
  const dx = x - cx, dy = y - cy, dz = z - cz;
  (c.p0 as MutVec3)[0] += dx; (c.p0 as MutVec3)[1] += dy; (c.p0 as MutVec3)[2] += dz;
  (c.p1 as MutVec3)[0] += dx; (c.p1 as MutVec3)[1] += dy; (c.p1 as MutVec3)[2] += dz;
}

function translateCapsuleMid(c: Capsule, dx: number, dy: number, dz: number): void {
  (c.p0 as MutVec3)[0] += dx; (c.p0 as MutVec3)[1] += dy; (c.p0 as MutVec3)[2] += dz;
  (c.p1 as MutVec3)[0] += dx; (c.p1 as MutVec3)[1] += dy; (c.p1 as MutVec3)[2] += dz;
}

function anchorWorld(c: Capsule, local: Vec3): Vec3 {
  // Phase 11 simplification: treat anchor as already in world space (caller
  // updates it post-step). Phase 11.1 will properly transform local-to-world
  // through the capsule's orientation when we add per-bone rotation state.
  void c;
  return local;
}

void v3Add; void v3Sub; void v3Scale;

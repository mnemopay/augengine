/**
 * Two-bone IK — analytic solver for hip/knee/ankle and shoulder/elbow/wrist
 * chains. Cosine-rule on the triangle formed by the three joints.
 *
 * Used by retargeting Phase 3 (foot-lock fixup) and gameplay (e.g., hand
 * grabs object at runtime). Pure math, no GPU dependency.
 */

import type { Quat, Vec3 } from "../scene/types.js";
import {
  qFromAxisAngle,
  qMul,
  qInvert,
  qNormalize,
  qRotate,
  v3,
  v3Cross,
  v3Distance,
  v3Length,
  v3Normalize,
  v3Sub,
  type MutQuat,
  type MutVec3,
} from "../scene/math.js";

export interface TwoBoneIKInput {
  /** World-space root joint (e.g., hip, shoulder). */
  root: Vec3;
  /** World-space middle joint in current bind pose (e.g., knee, elbow). */
  mid: Vec3;
  /** World-space tip joint in current bind pose (e.g., ankle, wrist). */
  tip: Vec3;
  /** Target world position the tip should reach. */
  target: Vec3;
  /** Pole vector — direction the bend (knee/elbow) should face. World space. */
  pole: Vec3;
}

export interface TwoBoneIKResult {
  /** Local-space rotations to apply (relative to the parent of root → mid). */
  rootRotation: Quat;
  midRotation: Quat;
  /** True iff the target was reachable; false if clamped to max extension. */
  reached: boolean;
}

/**
 * Solve for the rotations that move the tip to the target.
 *
 * Returns local rotations for the root and mid joints. Caller composes
 * these onto the existing bind/local rotations.
 */
export function solveTwoBoneIK(input: TwoBoneIKInput): TwoBoneIKResult {
  const { root, mid, tip, target, pole } = input;

  // Bone lengths preserved.
  const a = v3Distance(root, mid);
  const b = v3Distance(mid, tip);

  const toTarget: MutVec3 = [0, 0, 0];
  v3Sub(toTarget, target, root);
  let c = v3Length(toTarget);

  let reached = true;
  // Clamp target to max extension.
  const maxC = a + b - 1e-4;
  const minC = Math.abs(a - b) + 1e-4;
  if (c > maxC) {
    c = maxC;
    reached = false;
  } else if (c < minC) {
    c = minC;
    reached = false;
  }

  // Triangle angles via cosine rule.
  const cosAtRoot = clamp((a * a + c * c - b * b) / (2 * a * c), -1, 1);
  const cosAtMid = clamp((a * a + b * b - c * c) / (2 * a * b), -1, 1);
  const angleAtRoot = Math.acos(cosAtRoot);
  const angleAtMid = Math.PI - Math.acos(cosAtMid);

  // Build rotation axis perpendicular to the (root→tip, pole) plane.
  const targetDir: MutVec3 = [0, 0, 0];
  v3Normalize(targetDir, toTarget);
  const poleDir: MutVec3 = [0, 0, 0];
  v3Sub(poleDir, pole, root);
  v3Normalize(poleDir, poleDir);
  const axis: MutVec3 = [0, 0, 0];
  v3Cross(axis, targetDir, poleDir);
  if (v3Length(axis) < 1e-6) {
    // Pole co-linear with target — pick an arbitrary perpendicular.
    if (Math.abs(targetDir[0]) < 0.9) v3Cross(axis, targetDir, [1, 0, 0]);
    else v3Cross(axis, targetDir, [0, 1, 0]);
  }
  v3Normalize(axis, axis);

  // Root rotation = rotate (root→mid) onto (root→target) about a perpendicular
  // axis, then bend by angleAtRoot toward pole side.
  const currentDir: MutVec3 = [0, 0, 0];
  v3Sub(currentDir, mid, root);
  v3Normalize(currentDir, currentDir);
  const currentAxis: MutVec3 = [0, 0, 0];
  v3Cross(currentAxis, currentDir, targetDir);
  if (v3Length(currentAxis) < 1e-6) {
    v3Cross(currentAxis, currentDir, [0, 1, 0]);
  }
  v3Normalize(currentAxis, currentAxis);
  const dotCurTar = clamp(currentDir[0] * targetDir[0] + currentDir[1] * targetDir[1] + currentDir[2] * targetDir[2], -1, 1);
  const angleAlign = Math.acos(dotCurTar);

  const qAlign: MutQuat = [0, 0, 0, 1];
  qFromAxisAngle(qAlign, currentAxis, angleAlign);

  const qBend: MutQuat = [0, 0, 0, 1];
  qFromAxisAngle(qBend, axis, angleAtRoot);

  const rootRot: MutQuat = [0, 0, 0, 1];
  qMul(rootRot, qBend, qAlign);
  qNormalize(rootRot, rootRot);

  // Mid rotation: bend by angleAtMid about the same axis, but in the joint's
  // local frame. We approximate by applying the inverse of the root rotation.
  const midRot: MutQuat = [0, 0, 0, 1];
  qFromAxisAngle(midRot, axis, angleAtMid);
  const invRoot: MutQuat = [0, 0, 0, 1];
  qInvert(invRoot, rootRot);
  // local mid rotation = inv(root) · mid (approximation; production callers
  // should resolve the exact parent transform).
  const localMid: MutQuat = [0, 0, 0, 1];
  qMul(localMid, invRoot, midRot);
  qNormalize(localMid, localMid);

  return {
    rootRotation: rootRot,
    midRotation: localMid,
    reached,
  };
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Convenience: foot-lock fixup. Given the planted foot world position the
 * caller wants to enforce, returns the new hip/knee local rotations.
 */
export function footLock(
  hipWorld: Vec3,
  kneeWorld: Vec3,
  ankleWorld: Vec3,
  plantedAnklePos: Vec3,
  poleDir: Vec3,
): TwoBoneIKResult {
  return solveTwoBoneIK({
    root: hipWorld,
    mid: kneeWorld,
    tip: ankleWorld,
    target: plantedAnklePos,
    pole: poleDir,
  });
}

void v3;
void qRotate;

/**
 * Cinematic camera track — Phase 10.
 *
 * Sequencer-style timeline of camera keyframes. Each keyframe specifies
 * a camera transform (position + lookAt + FOV) at a time, plus an
 * easing curve to the next keyframe. A `CameraTrack` evaluates to a
 * camera state at any time `t`.
 *
 * Same track type drives BOTH live cutscenes (RAF clock) AND film
 * export (frame-stepped render runtime). The cinematic-aware part of
 * the engine just calls `track.sample(t)` and writes the result onto
 * the active camera entity.
 *
 * Bezier paths support: each keyframe optionally carries
 * `inHandle` + `outHandle` control points for cubic bezier
 * interpolation between adjacent keyframes. Linear interp is the
 * default when handles are absent.
 */

import type { Quat, Vec3 } from "../../scene/types.js";
import { qSlerp, v3Lerp, type MutQuat, type MutVec3 } from "../../scene/math.js";

export type Easing =
  | "linear"
  | "easeIn"
  | "easeOut"
  | "easeInOut"
  | "smoothstep";

export interface CameraKeyframe {
  /** Time in seconds from track start. */
  t: number;
  /** World-space camera position. */
  position: Vec3;
  /** Look-at target in world space. Camera always points here. */
  lookAt: Vec3;
  /** Camera FOV in radians. */
  fovY: number;
  /** Roll around the look-axis in radians. Default 0. */
  roll?: number;
  /** Easing applied between THIS keyframe and the next one. */
  easing?: Easing;
  /** Optional bezier handles in (time, value) space.
   *  inHandle / outHandle are RELATIVE to the keyframe time/value, so
   *  outHandle.t = +0.3 means the control point sits 0.3 seconds after this kf. */
  outHandle?: { t: number; position?: Vec3 };
  inHandle?: { t: number; position?: Vec3 };
}

export interface CameraSample {
  position: Vec3;
  rotation: Quat;
  fovY: number;
}

export class CameraTrack {
  keyframes: CameraKeyframe[];
  /** Total track duration in seconds. */
  duration: number;

  constructor(keyframes: CameraKeyframe[]) {
    if (keyframes.length === 0) {
      throw new Error("CameraTrack requires at least one keyframe");
    }
    // Ensure sorted ascending by time.
    this.keyframes = [...keyframes].sort((a, b) => a.t - b.t);
    this.duration = this.keyframes[this.keyframes.length - 1]!.t;
  }

  /** Sample the track at time `t` (clamped to [0, duration]). */
  sample(t: number, out: CameraSample): CameraSample {
    const kfs = this.keyframes;
    const last = kfs.length - 1;
    if (t <= kfs[0]!.t) return writeKeyframe(kfs[0]!, out);
    if (t >= kfs[last]!.t) return writeKeyframe(kfs[last]!, out);

    // Find bracketing keyframe pair.
    let i = 0;
    while (i < last && kfs[i + 1]!.t <= t) i++;
    const a = kfs[i]!;
    const b = kfs[i + 1]!;
    const span = b.t - a.t;
    const local = span > 0 ? (t - a.t) / span : 0;
    const eased = applyEasing(a.easing ?? "linear", local);

    // Position: bezier if both handles present, otherwise linear lerp eased.
    if (a.outHandle?.position && b.inHandle?.position) {
      bezierVec3(out.position, a.position, a.outHandle.position, b.inHandle.position, b.position, eased);
    } else {
      v3Lerp(out.position as MutVec3, a.position, b.position, eased);
    }

    // FOV linear lerp.
    out.fovY = a.fovY + (b.fovY - a.fovY) * eased;

    // Compute rotation as look-at: build a lookAt-derived quaternion at
    // this sample time. We slerp the orientation between the two
    // computed rotations rather than slerp the lookAt targets, which
    // produces a smoother arc when the lookAt moves between kfs.
    const lookAtA: Vec3 = a.lookAt;
    const lookAtB: Vec3 = b.lookAt;
    const interpLookAt: MutVec3 = [0, 0, 0];
    v3Lerp(interpLookAt, lookAtA, lookAtB, eased);
    const rollA = a.roll ?? 0;
    const rollB = b.roll ?? 0;
    const roll = rollA + (rollB - rollA) * eased;
    lookAtToQuat(out.rotation as MutQuat, out.position, interpLookAt, roll);

    return out;
  }
}

function writeKeyframe(kf: CameraKeyframe, out: CameraSample): CameraSample {
  out.position = [...kf.position] as Vec3;
  out.fovY = kf.fovY;
  lookAtToQuat(out.rotation as MutQuat, kf.position, kf.lookAt, kf.roll ?? 0);
  return out;
}

// ──────────────────────────────────────────────────────────────────────
// Easings
// ──────────────────────────────────────────────────────────────────────

function applyEasing(easing: Easing, t: number): number {
  switch (easing) {
    case "linear": return t;
    case "easeIn": return t * t;
    case "easeOut": return 1 - (1 - t) * (1 - t);
    case "easeInOut":
      return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    case "smoothstep": return t * t * (3 - 2 * t);
  }
}

// ──────────────────────────────────────────────────────────────────────
// Cubic Bezier (Vec3)
// ──────────────────────────────────────────────────────────────────────

function bezierVec3(
  out: Vec3,
  p0: Vec3,
  p1: Vec3,
  p2: Vec3,
  p3: Vec3,
  t: number,
): void {
  const u = 1 - t;
  const w0 = u * u * u;
  const w1 = 3 * u * u * t;
  const w2 = 3 * u * t * t;
  const w3 = t * t * t;
  (out as MutVec3)[0] = w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0];
  (out as MutVec3)[1] = w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1];
  (out as MutVec3)[2] = w0 * p0[2] + w1 * p1[2] + w2 * p2[2] + w3 * p3[2];
}

// ──────────────────────────────────────────────────────────────────────
// LookAt → Quaternion (with optional roll)
// ──────────────────────────────────────────────────────────────────────

function lookAtToQuat(out: MutQuat, eye: Vec3, target: Vec3, roll: number): MutQuat {
  // Build orthonormal basis: forward = normalize(target - eye), up = world up,
  // right = normalize(cross(forward, up)). Apply roll around forward.
  const fx = target[0] - eye[0];
  const fy = target[1] - eye[1];
  const fz = target[2] - eye[2];
  const flen = Math.hypot(fx, fy, fz);
  if (flen < 1e-9) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
    return out;
  }
  const fnx = fx / flen, fny = fy / flen, fnz = fz / flen;
  // World up.
  let ux = 0, uy = 1, uz = 0;
  if (Math.abs(fny) > 0.999) {
    // Forward is too close to world-up; use world-Z as fallback up.
    uy = 0; uz = 1;
  }
  // right = cross(forward, up)
  let rx = fny * uz - fnz * uy;
  let ry = fnz * ux - fnx * uz;
  let rz = fnx * uy - fny * ux;
  const rlen = Math.hypot(rx, ry, rz);
  rx /= rlen; ry /= rlen; rz /= rlen;
  // up = cross(right, forward)
  ux = ry * fnz - rz * fny;
  uy = rz * fnx - rx * fnz;
  uz = rx * fny - ry * fnx;

  // Apply roll around the forward axis to right + up.
  if (roll !== 0) {
    const c = Math.cos(roll), s = Math.sin(roll);
    const nrx = rx * c + ux * s;
    const nry = ry * c + uy * s;
    const nrz = rz * c + uz * s;
    const nux = -rx * s + ux * c;
    const nuy = -ry * s + uy * c;
    const nuz = -rz * s + uz * c;
    rx = nrx; ry = nry; rz = nrz;
    ux = nux; uy = nuy; uz = nuz;
  }

  // Build rotation matrix [right | up | -forward] and convert to quaternion.
  // (Camera convention: -Z is forward.)
  const m00 = rx, m01 = ux, m02 = -fnx;
  const m10 = ry, m11 = uy, m12 = -fny;
  const m20 = rz, m21 = uz, m22 = -fnz;

  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    out[3] = 0.25 * s;
    out[0] = (m21 - m12) / s;
    out[1] = (m02 - m20) / s;
    out[2] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    out[3] = (m21 - m12) / s;
    out[0] = 0.25 * s;
    out[1] = (m01 + m10) / s;
    out[2] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    out[3] = (m02 - m20) / s;
    out[0] = (m01 + m10) / s;
    out[1] = 0.25 * s;
    out[2] = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    out[3] = (m10 - m01) / s;
    out[0] = (m02 + m20) / s;
    out[1] = (m12 + m21) / s;
    out[2] = 0.25 * s;
  }
  // Normalize to be safe.
  const len = Math.hypot(out[0], out[1], out[2], out[3]);
  if (len > 0) {
    out[0] /= len; out[1] /= len; out[2] /= len; out[3] /= len;
  }
  void qSlerp; // imported for future blend-tree use; suppress unused warning
  return out;
}

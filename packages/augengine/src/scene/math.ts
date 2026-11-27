/**
 * Math primitives — Vec3, Quat, Mat4. Pure functions, no allocation in hot
 * paths where possible (out-param style). Used by every subsystem:
 * transform hierarchy, skinning, retargeting, IK, camera.
 *
 * Conventions:
 *   Quat = [x, y, z, w]      (Hamilton, right-handed)
 *   Mat4 = column-major Float32Array(16)
 *   Vec3 = readonly [x, y, z] (immutable triples for component types)
 *   MutVec3 = [x, y, z]      (used internally for working storage)
 */

import type { Vec3, Quat, Mat4 } from "./types.js";

export type MutVec3 = [number, number, number];
export type MutQuat = [number, number, number, number];

// ──────────────────────────────────────────────────────────────────────
// Vec3
// ──────────────────────────────────────────────────────────────────────

export function v3(x = 0, y = 0, z = 0): MutVec3 {
  return [x, y, z];
}

export function v3Copy(out: MutVec3, a: Vec3): MutVec3 {
  out[0] = a[0]!;
  out[1] = a[1]!;
  out[2] = a[2]!;
  return out;
}

export function v3Add(out: MutVec3, a: Vec3, b: Vec3): MutVec3 {
  out[0] = a[0]! + b[0]!;
  out[1] = a[1]! + b[1]!;
  out[2] = a[2]! + b[2]!;
  return out;
}

export function v3Sub(out: MutVec3, a: Vec3, b: Vec3): MutVec3 {
  out[0] = a[0]! - b[0]!;
  out[1] = a[1]! - b[1]!;
  out[2] = a[2]! - b[2]!;
  return out;
}

export function v3Scale(out: MutVec3, a: Vec3, s: number): MutVec3 {
  out[0] = a[0]! * s;
  out[1] = a[1]! * s;
  out[2] = a[2]! * s;
  return out;
}

export function v3Dot(a: Vec3, b: Vec3): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

export function v3Cross(out: MutVec3, a: Vec3, b: Vec3): MutVec3 {
  const ax = a[0]!,
    ay = a[1]!,
    az = a[2]!;
  const bx = b[0]!,
    by = b[1]!,
    bz = b[2]!;
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

export function v3Length(a: Vec3): number {
  return Math.hypot(a[0]!, a[1]!, a[2]!);
}

export function v3LengthSq(a: Vec3): number {
  return a[0]! * a[0]! + a[1]! * a[1]! + a[2]! * a[2]!;
}

export function v3Normalize(out: MutVec3, a: Vec3): MutVec3 {
  const len = v3Length(a);
  if (len < 1e-12) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
  }
  const inv = 1 / len;
  out[0] = a[0]! * inv;
  out[1] = a[1]! * inv;
  out[2] = a[2]! * inv;
  return out;
}

export function v3Distance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
}

export function v3Lerp(out: MutVec3, a: Vec3, b: Vec3, t: number): MutVec3 {
  out[0] = a[0]! + (b[0]! - a[0]!) * t;
  out[1] = a[1]! + (b[1]! - a[1]!) * t;
  out[2] = a[2]! + (b[2]! - a[2]!) * t;
  return out;
}

// ──────────────────────────────────────────────────────────────────────
// Quat (x, y, z, w)
// ──────────────────────────────────────────────────────────────────────

export function qIdentity(out: MutQuat = [0, 0, 0, 1]): MutQuat {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = 1;
  return out;
}

export function qCopy(out: MutQuat, a: Quat): MutQuat {
  out[0] = a[0]!;
  out[1] = a[1]!;
  out[2] = a[2]!;
  out[3] = a[3]!;
  return out;
}

/** Hamilton product q = a * b (apply b, then a). */
export function qMul(out: MutQuat, a: Quat, b: Quat): MutQuat {
  const ax = a[0]!,
    ay = a[1]!,
    az = a[2]!,
    aw = a[3]!;
  const bx = b[0]!,
    by = b[1]!,
    bz = b[2]!,
    bw = b[3]!;
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

/** Inverse of a unit quaternion is its conjugate. */
export function qInvert(out: MutQuat, a: Quat): MutQuat {
  out[0] = -a[0]!;
  out[1] = -a[1]!;
  out[2] = -a[2]!;
  out[3] = a[3]!;
  return out;
}

export function qNormalize(out: MutQuat, a: Quat): MutQuat {
  const len = Math.hypot(a[0]!, a[1]!, a[2]!, a[3]!);
  if (len < 1e-12) return qIdentity(out);
  const inv = 1 / len;
  out[0] = a[0]! * inv;
  out[1] = a[1]! * inv;
  out[2] = a[2]! * inv;
  out[3] = a[3]! * inv;
  return out;
}

export function qDot(a: Quat, b: Quat): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]! + a[3]! * b[3]!;
}

/** Rotate vec by quat — out = q * v * q^-1. */
export function qRotate(out: MutVec3, q: Quat, v: Vec3): MutVec3 {
  const qx = q[0]!,
    qy = q[1]!,
    qz = q[2]!,
    qw = q[3]!;
  const vx = v[0]!,
    vy = v[1]!,
    vz = v[2]!;
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  // out = v + qw * t + cross(q.xyz, t)
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

/** Spherical linear interpolation. */
export function qSlerp(out: MutQuat, a: Quat, b: Quat, t: number): MutQuat {
  let cosTheta = qDot(a, b);
  let bx = b[0]!,
    by = b[1]!,
    bz = b[2]!,
    bw = b[3]!;
  if (cosTheta < 0) {
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
    cosTheta = -cosTheta;
  }
  if (cosTheta > 0.9995) {
    // Use linear lerp for very close quaternions to avoid divide-by-zero.
    out[0] = a[0]! + t * (bx - a[0]!);
    out[1] = a[1]! + t * (by - a[1]!);
    out[2] = a[2]! + t * (bz - a[2]!);
    out[3] = a[3]! + t * (bw - a[3]!);
    return qNormalize(out, out);
  }
  const theta = Math.acos(cosTheta);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  out[0] = wa * a[0]! + wb * bx;
  out[1] = wa * a[1]! + wb * by;
  out[2] = wa * a[2]! + wb * bz;
  out[3] = wa * a[3]! + wb * bw;
  return out;
}

/** Quaternion that rotates from→to (both unit vectors). */
export function qFromTo(out: MutQuat, from: Vec3, to: Vec3): MutQuat {
  const d = v3Dot(from, to);
  if (d > 0.999999) return qIdentity(out);
  if (d < -0.999999) {
    // 180° rotation; pick any orthogonal axis.
    let axis: MutVec3 = [1, 0, 0];
    if (Math.abs(from[0]!) > 0.9) axis = [0, 1, 0];
    const ortho = v3Cross([0, 0, 0], from, axis);
    v3Normalize(ortho, ortho);
    out[0] = ortho[0]!;
    out[1] = ortho[1]!;
    out[2] = ortho[2]!;
    out[3] = 0;
    return out;
  }
  const cross = v3Cross([0, 0, 0], from, to);
  out[0] = cross[0]!;
  out[1] = cross[1]!;
  out[2] = cross[2]!;
  out[3] = 1 + d;
  return qNormalize(out, out);
}

export function qFromAxisAngle(out: MutQuat, axis: Vec3, angleRad: number): MutQuat {
  const half = angleRad * 0.5;
  const s = Math.sin(half);
  out[0] = axis[0]! * s;
  out[1] = axis[1]! * s;
  out[2] = axis[2]! * s;
  out[3] = Math.cos(half);
  return out;
}

// ──────────────────────────────────────────────────────────────────────
// Mat4 — column-major, Float32Array(16)
// ──────────────────────────────────────────────────────────────────────

export function m4Create(): Mat4 {
  return new Float32Array(16);
}

export function m4Identity(out: Mat4 = m4Create()): Mat4 {
  out[0] = 1;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = 1;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = 1;
  out[11] = 0;
  out[12] = 0;
  out[13] = 0;
  out[14] = 0;
  out[15] = 1;
  return out;
}

/** Compose translation + rotation + scale into a column-major matrix. */
export function m4FromTRS(
  out: Mat4,
  t: Vec3,
  r: Quat,
  s: Vec3,
): Mat4 {
  const x = r[0]!,
    y = r[1]!,
    z = r[2]!,
    w = r[3]!;
  const x2 = x + x,
    y2 = y + y,
    z2 = z + z;
  const xx = x * x2,
    xy = x * y2,
    xz = x * z2;
  const yy = y * y2,
    yz = y * z2,
    zz = z * z2;
  const wx = w * x2,
    wy = w * y2,
    wz = w * z2;
  const sx = s[0]!,
    sy = s[1]!,
    sz = s[2]!;
  out[0] = (1 - (yy + zz)) * sx;
  out[1] = (xy + wz) * sx;
  out[2] = (xz - wy) * sx;
  out[3] = 0;
  out[4] = (xy - wz) * sy;
  out[5] = (1 - (xx + zz)) * sy;
  out[6] = (yz + wx) * sy;
  out[7] = 0;
  out[8] = (xz + wy) * sz;
  out[9] = (yz - wx) * sz;
  out[10] = (1 - (xx + yy)) * sz;
  out[11] = 0;
  out[12] = t[0]!;
  out[13] = t[1]!;
  out[14] = t[2]!;
  out[15] = 1;
  return out;
}

/** Multiply two column-major mat4: out = a * b. */
export function m4Mul(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!;
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!;
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!;
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!;
  const b00 = b[0]!,
    b01 = b[1]!,
    b02 = b[2]!,
    b03 = b[3]!;
  const b10 = b[4]!,
    b11 = b[5]!,
    b12 = b[6]!,
    b13 = b[7]!;
  const b20 = b[8]!,
    b21 = b[9]!,
    b22 = b[10]!,
    b23 = b[11]!;
  const b30 = b[12]!,
    b31 = b[13]!,
    b32 = b[14]!,
    b33 = b[15]!;
  out[0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
  out[1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
  out[2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
  out[3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
  out[4] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
  out[5] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
  out[6] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
  out[7] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
  out[8] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
  out[9] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
  out[10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
  out[11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
  out[12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
  out[13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
  out[14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
  out[15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
  return out;
}

/** General Mat4 inverse via cofactors. Returns null on singular matrices. */
export function m4Invert(out: Mat4, a: Mat4): Mat4 | null {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!;
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!;
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!;
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!;
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
}

/** Decompose a column-major mat4 back into TRS. Assumes uniform/non-skew scale. */
export function m4DecomposeTRS(
  m: Mat4,
  outT: MutVec3,
  outR: MutQuat,
  outS: MutVec3,
): void {
  outT[0] = m[12]!;
  outT[1] = m[13]!;
  outT[2] = m[14]!;
  const sx = Math.hypot(m[0]!, m[1]!, m[2]!);
  const sy = Math.hypot(m[4]!, m[5]!, m[6]!);
  const sz = Math.hypot(m[8]!, m[9]!, m[10]!);
  outS[0] = sx;
  outS[1] = sy;
  outS[2] = sz;
  // Rotation matrix (3x3)
  const isx = sx > 1e-12 ? 1 / sx : 0;
  const isy = sy > 1e-12 ? 1 / sy : 0;
  const isz = sz > 1e-12 ? 1 / sz : 0;
  const r00 = m[0]! * isx,
    r10 = m[1]! * isx,
    r20 = m[2]! * isx;
  const r01 = m[4]! * isy,
    r11 = m[5]! * isy,
    r21 = m[6]! * isy;
  const r02 = m[8]! * isz,
    r12 = m[9]! * isz,
    r22 = m[10]! * isz;
  // Convert rotation matrix to quaternion (Shepperd's method).
  const trace = r00 + r11 + r22;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    outR[3] = 0.25 * s;
    outR[0] = (r21 - r12) / s;
    outR[1] = (r02 - r20) / s;
    outR[2] = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
    outR[3] = (r21 - r12) / s;
    outR[0] = 0.25 * s;
    outR[1] = (r01 + r10) / s;
    outR[2] = (r02 + r20) / s;
  } else if (r11 > r22) {
    const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
    outR[3] = (r02 - r20) / s;
    outR[0] = (r01 + r10) / s;
    outR[1] = 0.25 * s;
    outR[2] = (r12 + r21) / s;
  } else {
    const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
    outR[3] = (r10 - r01) / s;
    outR[0] = (r02 + r20) / s;
    outR[1] = (r12 + r21) / s;
    outR[2] = 0.25 * s;
  }
}

/** Standard right-handed perspective projection. */
export function m4Perspective(
  out: Mat4,
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): Mat4 {
  const f = 1 / Math.tan(fovY * 0.5);
  const nf = 1 / (near - far);
  out[0] = f / aspect;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = f;
  out[6] = 0;
  out[7] = 0;
  out[8] = 0;
  out[9] = 0;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[12] = 0;
  out[13] = 0;
  out[14] = 2 * far * near * nf;
  out[15] = 0;
  return out;
}

export function m4LookAt(out: Mat4, eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z: MutVec3 = [0, 0, 0];
  v3Sub(z, eye, target);
  v3Normalize(z, z);
  const x: MutVec3 = [0, 0, 0];
  v3Cross(x, up, z);
  v3Normalize(x, x);
  const y: MutVec3 = [0, 0, 0];
  v3Cross(y, z, x);
  out[0] = x[0]!;
  out[1] = y[0]!;
  out[2] = z[0]!;
  out[3] = 0;
  out[4] = x[1]!;
  out[5] = y[1]!;
  out[6] = z[1]!;
  out[7] = 0;
  out[8] = x[2]!;
  out[9] = y[2]!;
  out[10] = z[2]!;
  out[11] = 0;
  out[12] = -v3Dot(x, eye);
  out[13] = -v3Dot(y, eye);
  out[14] = -v3Dot(z, eye);
  out[15] = 1;
  return out;
}

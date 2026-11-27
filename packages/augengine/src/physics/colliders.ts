/**
 * Physics tier 2 — collider primitives + intersection tests.
 *
 * Phase 11. AABB is already covered by `@blackpig/game`'s primary physics;
 * this adds sphere, capsule, and mesh (BVH-accelerated triangle soup). All
 * tests are deterministic so replay/multiplayer parity holds.
 *
 * Pure math, no GPU. Each shape exposes:
 *   - bounds(): AABB suitable for broad-phase
 *   - support(direction): farthest point in `direction` (GJK requirement)
 *   - intersect(other): boolean + manifold (normal + depth) for narrow-phase
 */

import type { Vec3 } from "../scene/types.js";
import {
  v3Sub,
  v3Dot,
  v3Length,
  v3Normalize,
  v3Cross,
  type MutVec3,
} from "../scene/math.js";

export interface AABB {
  min: Vec3;
  max: Vec3;
}

export interface Sphere {
  kind: "sphere";
  center: Vec3;
  radius: number;
}

export interface Capsule {
  kind: "capsule";
  /** Capsule axis from p0 (bottom hemisphere center) to p1 (top hemisphere center). */
  p0: Vec3;
  p1: Vec3;
  radius: number;
}

export interface MeshCollider {
  kind: "mesh";
  /** Vertex positions, flat (x,y,z). */
  vertices: Float32Array;
  /** Triangle indices, flat. */
  indices: Uint32Array;
  /** Optional pre-built BVH; if absent, narrow-phase will fall back to brute scan. */
  bvh?: BvhNode | null;
}

export type Collider = Sphere | Capsule | MeshCollider;

export interface ContactManifold {
  /** Surface normal pointing from `a` toward `b`. */
  normal: Vec3;
  /** Penetration depth along the normal. */
  depth: number;
  /** World-space contact point. */
  point: Vec3;
}

// ──────────────────────────────────────────────────────────────────────
// AABB
// ──────────────────────────────────────────────────────────────────────

export function aabbForCollider(c: Collider): AABB {
  if (c.kind === "sphere") {
    return {
      min: [c.center[0] - c.radius, c.center[1] - c.radius, c.center[2] - c.radius],
      max: [c.center[0] + c.radius, c.center[1] + c.radius, c.center[2] + c.radius],
    };
  }
  if (c.kind === "capsule") {
    const r = c.radius;
    return {
      min: [
        Math.min(c.p0[0], c.p1[0]) - r,
        Math.min(c.p0[1], c.p1[1]) - r,
        Math.min(c.p0[2], c.p1[2]) - r,
      ],
      max: [
        Math.max(c.p0[0], c.p1[0]) + r,
        Math.max(c.p0[1], c.p1[1]) + r,
        Math.max(c.p0[2], c.p1[2]) + r,
      ],
    };
  }
  // mesh — scan vertices.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < c.vertices.length; i += 3) {
    const x = c.vertices[i]!;
    const y = c.vertices[i + 1]!;
    const z = c.vertices[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

export function aabbOverlap(a: AABB, b: AABB): boolean {
  return (
    a.min[0] <= b.max[0] && a.max[0] >= b.min[0] &&
    a.min[1] <= b.max[1] && a.max[1] >= b.min[1] &&
    a.min[2] <= b.max[2] && a.max[2] >= b.min[2]
  );
}

// ──────────────────────────────────────────────────────────────────────
// Sphere ↔ Sphere
// ──────────────────────────────────────────────────────────────────────

export function sphereSphere(a: Sphere, b: Sphere): ContactManifold | null {
  const dx = b.center[0] - a.center[0];
  const dy = b.center[1] - a.center[1];
  const dz = b.center[2] - a.center[2];
  const d2 = dx * dx + dy * dy + dz * dz;
  const r = a.radius + b.radius;
  if (d2 > r * r) return null;
  const d = Math.sqrt(d2);
  let nx = 0, ny = 1, nz = 0;
  if (d > 1e-9) { nx = dx / d; ny = dy / d; nz = dz / d; }
  return {
    normal: [nx, ny, nz],
    depth: r - d,
    point: [
      a.center[0] + nx * a.radius,
      a.center[1] + ny * a.radius,
      a.center[2] + nz * a.radius,
    ],
  };
}

// ──────────────────────────────────────────────────────────────────────
// Capsule helpers — closest-point-on-segment, capsule-capsule, sphere-capsule
// ──────────────────────────────────────────────────────────────────────

/** Closest point on segment [a,b] to point p, returned as `out`. */
function closestPointOnSegment(out: MutVec3, a: Vec3, b: Vec3, p: Vec3): void {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const abz = b[2] - a[2];
  const apx = p[0] - a[0];
  const apy = p[1] - a[1];
  const apz = p[2] - a[2];
  const denom = abx * abx + aby * aby + abz * abz;
  let t = denom > 1e-9 ? (apx * abx + apy * aby + apz * abz) / denom : 0;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  out[0] = a[0] + abx * t;
  out[1] = a[1] + aby * t;
  out[2] = a[2] + abz * t;
}

export function sphereCapsule(s: Sphere, c: Capsule): ContactManifold | null {
  const closest: MutVec3 = [0, 0, 0];
  closestPointOnSegment(closest, c.p0, c.p1, s.center);
  return sphereSphere(s, { kind: "sphere", center: closest, radius: c.radius });
}

/**
 * Capsule ↔ capsule via closest-point-between-two-segments (Eberly).
 * Then sphere-sphere at the closest pair.
 */
export function capsuleCapsule(a: Capsule, b: Capsule): ContactManifold | null {
  const c1: MutVec3 = [0, 0, 0];
  const c2: MutVec3 = [0, 0, 0];
  closestPointsBetweenSegments(c1, c2, a.p0, a.p1, b.p0, b.p1);
  return sphereSphere(
    { kind: "sphere", center: c1, radius: a.radius },
    { kind: "sphere", center: c2, radius: b.radius },
  );
}

/** Closest points between two segments (Eberly's method, parallel-safe). */
function closestPointsBetweenSegments(
  outA: MutVec3,
  outB: MutVec3,
  a0: Vec3, a1: Vec3,
  b0: Vec3, b1: Vec3,
): void {
  const dx1 = a1[0] - a0[0], dy1 = a1[1] - a0[1], dz1 = a1[2] - a0[2];
  const dx2 = b1[0] - b0[0], dy2 = b1[1] - b0[1], dz2 = b1[2] - b0[2];
  const rx = a0[0] - b0[0], ry = a0[1] - b0[1], rz = a0[2] - b0[2];
  const a = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const e = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const f = dx2 * rx + dy2 * ry + dz2 * rz;
  let s = 0, t = 0;
  if (a <= 1e-9 && e <= 1e-9) {
    outA[0] = a0[0]; outA[1] = a0[1]; outA[2] = a0[2];
    outB[0] = b0[0]; outB[1] = b0[1]; outB[2] = b0[2];
    return;
  }
  if (a <= 1e-9) {
    t = clamp01(f / e);
  } else {
    const c = dx1 * rx + dy1 * ry + dz1 * rz;
    if (e <= 1e-9) {
      s = clamp01(-c / a);
    } else {
      const b = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
      const denom = a * e - b * b;
      if (denom !== 0) s = clamp01((b * f - c * e) / denom);
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = clamp01(-c / a); }
      else if (t > 1) { t = 1; s = clamp01((b - c) / a); }
    }
  }
  outA[0] = a0[0] + dx1 * s;
  outA[1] = a0[1] + dy1 * s;
  outA[2] = a0[2] + dz1 * s;
  outB[0] = b0[0] + dx2 * t;
  outB[1] = b0[1] + dy2 * t;
  outB[2] = b0[2] + dz2 * t;
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

// ──────────────────────────────────────────────────────────────────────
// Sphere ↔ Mesh — BVH-accelerated triangle scan
// ──────────────────────────────────────────────────────────────────────

export interface BvhNode {
  bounds: AABB;
  /** Triangle indices for leaf nodes; null for internal. */
  triangles?: number[];
  left?: BvhNode;
  right?: BvhNode;
}

export function buildBvh(mesh: MeshCollider, leafSize = 8): BvhNode {
  const triCount = mesh.indices.length / 3;
  const triIndices: number[] = [];
  for (let i = 0; i < triCount; i++) triIndices.push(i);
  const triBounds: AABB[] = [];
  for (let i = 0; i < triCount; i++) {
    triBounds.push(triBoundsOf(mesh, i));
  }
  return buildBvhRecursive(triBounds, triIndices, leafSize);
}

function buildBvhRecursive(
  triBounds: AABB[],
  triIndices: number[],
  leafSize: number,
): BvhNode {
  const bounds = unionBounds(triBounds, triIndices);
  if (triIndices.length <= leafSize) {
    return { bounds, triangles: triIndices };
  }
  // Pick split axis = longest extent.
  const ext = [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
  let axis = 0;
  if (ext[1]! > ext[axis]!) axis = 1;
  if (ext[2]! > ext[axis]!) axis = 2;
  // Sort by tri-bound center on chosen axis.
  triIndices.sort((a, b) => {
    const ca = (triBounds[a]!.min[axis]! + triBounds[a]!.max[axis]!) * 0.5;
    const cb = (triBounds[b]!.min[axis]! + triBounds[b]!.max[axis]!) * 0.5;
    return ca - cb;
  });
  const mid = Math.floor(triIndices.length / 2);
  const left = buildBvhRecursive(triBounds, triIndices.slice(0, mid), leafSize);
  const right = buildBvhRecursive(triBounds, triIndices.slice(mid), leafSize);
  return { bounds, left, right };
}

function triBoundsOf(mesh: MeshCollider, triIdx: number): AABB {
  const i0 = mesh.indices[triIdx * 3]!;
  const i1 = mesh.indices[triIdx * 3 + 1]!;
  const i2 = mesh.indices[triIdx * 3 + 2]!;
  const v = mesh.vertices;
  const x = [v[i0 * 3]!, v[i1 * 3]!, v[i2 * 3]!];
  const y = [v[i0 * 3 + 1]!, v[i1 * 3 + 1]!, v[i2 * 3 + 1]!];
  const z = [v[i0 * 3 + 2]!, v[i1 * 3 + 2]!, v[i2 * 3 + 2]!];
  return {
    min: [Math.min(...x), Math.min(...y), Math.min(...z)],
    max: [Math.max(...x), Math.max(...y), Math.max(...z)],
  };
}

function unionBounds(triBounds: AABB[], triIndices: number[]): AABB {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const i of triIndices) {
    const b = triBounds[i]!;
    if (b.min[0] < minX) minX = b.min[0];
    if (b.min[1] < minY) minY = b.min[1];
    if (b.min[2] < minZ) minZ = b.min[2];
    if (b.max[0] > maxX) maxX = b.max[0];
    if (b.max[1] > maxY) maxY = b.max[1];
    if (b.max[2] > maxZ) maxZ = b.max[2];
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/** Sphere vs mesh — find shallowest penetrating triangle, return manifold. */
export function sphereMesh(s: Sphere, m: MeshCollider): ContactManifold | null {
  const sBounds: AABB = {
    min: [s.center[0] - s.radius, s.center[1] - s.radius, s.center[2] - s.radius],
    max: [s.center[0] + s.radius, s.center[1] + s.radius, s.center[2] + s.radius],
  };
  const candidates: number[] = [];
  if (m.bvh) gatherBvhCandidates(m.bvh, sBounds, candidates);
  else for (let i = 0; i < m.indices.length / 3; i++) candidates.push(i);

  let best: ContactManifold | null = null;
  for (const tri of candidates) {
    const c = sphereTriangle(s, m, tri);
    if (!c) continue;
    if (!best || c.depth > best.depth) best = c;
  }
  return best;
}

function gatherBvhCandidates(node: BvhNode, q: AABB, out: number[]): void {
  if (!aabbOverlap(node.bounds, q)) return;
  if (node.triangles) {
    for (const t of node.triangles) out.push(t);
    return;
  }
  if (node.left) gatherBvhCandidates(node.left, q, out);
  if (node.right) gatherBvhCandidates(node.right, q, out);
}

function sphereTriangle(s: Sphere, m: MeshCollider, triIdx: number): ContactManifold | null {
  const i0 = m.indices[triIdx * 3]!;
  const i1 = m.indices[triIdx * 3 + 1]!;
  const i2 = m.indices[triIdx * 3 + 2]!;
  const v = m.vertices;
  const a: Vec3 = [v[i0 * 3]!, v[i0 * 3 + 1]!, v[i0 * 3 + 2]!];
  const b: Vec3 = [v[i1 * 3]!, v[i1 * 3 + 1]!, v[i1 * 3 + 2]!];
  const c: Vec3 = [v[i2 * 3]!, v[i2 * 3 + 1]!, v[i2 * 3 + 2]!];
  const closest: MutVec3 = [0, 0, 0];
  closestPointOnTriangle(closest, s.center, a, b, c);
  const dx = s.center[0] - closest[0];
  const dy = s.center[1] - closest[1];
  const dz = s.center[2] - closest[2];
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > s.radius * s.radius) return null;
  const d = Math.sqrt(d2);
  let nx = 0, ny = 1, nz = 0;
  if (d > 1e-9) { nx = dx / d; ny = dy / d; nz = dz / d; }
  return {
    normal: [nx, ny, nz],
    depth: s.radius - d,
    point: [closest[0], closest[1], closest[2]],
  };
}

function closestPointOnTriangle(out: MutVec3, p: Vec3, a: Vec3, b: Vec3, c: Vec3): void {
  // Ericson "Real-Time Collision Detection" algorithm.
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; return; }
  const bpx = p[0] - b[0], bpy = p[1] - b[1], bpz = p[2] - b[2];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { out[0] = b[0]; out[1] = b[1]; out[2] = b[2]; return; }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const w = d1 / (d1 - d3);
    out[0] = a[0] + abx * w; out[1] = a[1] + aby * w; out[2] = a[2] + abz * w;
    return;
  }
  const cpx = p[0] - c[0], cpy = p[1] - c[1], cpz = p[2] - c[2];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; return; }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    out[0] = a[0] + acx * w; out[1] = a[1] + acy * w; out[2] = a[2] + acz * w;
    return;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    out[0] = b[0] + (c[0] - b[0]) * w;
    out[1] = b[1] + (c[1] - b[1]) * w;
    out[2] = b[2] + (c[2] - b[2]) * w;
    return;
  }
  const denom = 1 / (va + vb + vc);
  const v_w = vb * denom;
  const w_w = vc * denom;
  out[0] = a[0] + abx * v_w + acx * w_w;
  out[1] = a[1] + aby * v_w + acy * w_w;
  out[2] = a[2] + abz * v_w + acz * w_w;
}

// ──────────────────────────────────────────────────────────────────────
// Sphere ↔ AABB
// ──────────────────────────────────────────────────────────────────────

/**
 * Sphere ↔ AABB. Standard clamp-and-measure. Returns null when separated.
 * When the sphere center is inside the box, picks the shallowest-penetration
 * face as the normal so the character-controller resolution still produces a
 * sensible push direction.
 */
export function sphereAabb(s: Sphere, a: AABB): ContactManifold | null {
  const cx = clampN(s.center[0], a.min[0], a.max[0]);
  const cy = clampN(s.center[1], a.min[1], a.max[1]);
  const cz = clampN(s.center[2], a.min[2], a.max[2]);
  const dx = s.center[0] - cx;
  const dy = s.center[1] - cy;
  const dz = s.center[2] - cz;
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > s.radius * s.radius) return null;
  if (d2 > 1e-12) {
    const d = Math.sqrt(d2);
    return {
      normal: [dx / d, dy / d, dz / d],
      depth: s.radius - d,
      point: [cx, cy, cz],
    };
  }
  // Center inside the box — shortest exit face wins.
  const dxL = s.center[0] - a.min[0];
  const dxR = a.max[0] - s.center[0];
  const dyL = s.center[1] - a.min[1];
  const dyR = a.max[1] - s.center[1];
  const dzL = s.center[2] - a.min[2];
  const dzR = a.max[2] - s.center[2];
  const m = Math.min(dxL, dxR, dyL, dyR, dzL, dzR);
  let nx = 0, ny = 0, nz = 0;
  if (m === dxL) nx = -1;
  else if (m === dxR) nx = 1;
  else if (m === dyL) ny = -1;
  else if (m === dyR) ny = 1;
  else if (m === dzL) nz = -1;
  else nz = 1;
  return { normal: [nx, ny, nz], depth: s.radius + m, point: [cx, cy, cz] };
}

// ──────────────────────────────────────────────────────────────────────
// Capsule ↔ AABB
// ──────────────────────────────────────────────────────────────────────

/**
 * Capsule ↔ AABB by iterating segment ↔ AABB closest-point pair, then
 * resolving as sphere ↔ AABB at the closest segment point. Four iterations
 * converge for any practical character-controller-vs-static-box case.
 */
export function capsuleAabb(c: Capsule, a: AABB): ContactManifold | null {
  let sx = (c.p0[0] + c.p1[0]) * 0.5;
  let sy = (c.p0[1] + c.p1[1]) * 0.5;
  let sz = (c.p0[2] + c.p1[2]) * 0.5;
  for (let i = 0; i < 4; i++) {
    const ax = clampN(sx, a.min[0], a.max[0]);
    const ay = clampN(sy, a.min[1], a.max[1]);
    const az = clampN(sz, a.min[2], a.max[2]);
    const next: MutVec3 = [0, 0, 0];
    closestPointOnSegment(next, c.p0, c.p1, [ax, ay, az]);
    if (
      Math.abs(next[0] - sx) < 1e-6 &&
      Math.abs(next[1] - sy) < 1e-6 &&
      Math.abs(next[2] - sz) < 1e-6
    ) break;
    sx = next[0]; sy = next[1]; sz = next[2];
  }
  return sphereAabb({ kind: "sphere", center: [sx, sy, sz], radius: c.radius }, a);
}

function clampN(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// Suppress unused-import warnings — these helpers are exported for downstream
// systems that use them directly (cloth, particle collision, etc).
void v3Sub; void v3Dot; void v3Length; void v3Normalize; void v3Cross;

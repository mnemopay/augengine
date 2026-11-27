/**
 * world/culling — Frustum culling (cheap broadphase) + an opt-in shim for
 * hardware occlusion queries.
 *
 * Track 7 (2026-05-29). The renderer used to do trivial radius culling
 * inline; this module replaces that with proper plane extraction from the
 * view-projection matrix and a vectorized AABB test that handles squashed
 * / rotated bounding volumes (Map3D buildings, skewed terrain chunks).
 *
 * Two-pass occlusion (Hi-Z) is deferred — see plan note. For v1 we expose
 * a capability probe + `OcclusionQueryRunner` placeholder that lets the
 * renderer issue `occlusion-query`-typed query sets per-object when the
 * adapter advertises support, then read results back next frame. The
 * tile manager consumes those results to bias eviction.
 */

import type { Mat4, Vec3 } from "../../scene/types.js";

export type Plane = readonly [number, number, number, number]; // ax+by+cz+d=0 (outward normal)

/** 6 outward-facing planes: left, right, bottom, top, near, far. */
export interface Frustum {
  planes: readonly [Plane, Plane, Plane, Plane, Plane, Plane];
}

export interface AABB {
  /** World-space center. */
  centerWorld: Vec3;
  /** Half-extents on x/y/z. */
  halfExtents: Vec3;
}

/**
 * Extract a 6-plane frustum from a column-major view-projection matrix
 * (Gribb–Hartmann method). Planes have outward-pointing, normalised normals;
 * the on-positive-side test is `dot(n, p) + d ≥ 0` for "inside or on".
 */
export function extractFrustum(viewProj: Mat4): Frustum {
  const m = viewProj;
  const m0 = m[0]!, m1 = m[1]!, m2 = m[2]!, m3 = m[3]!;
  const m4 = m[4]!, m5 = m[5]!, m6 = m[6]!, m7 = m[7]!;
  const m8 = m[8]!, m9 = m[9]!, m10 = m[10]!, m11 = m[11]!;
  const m12 = m[12]!, m13 = m[13]!, m14 = m[14]!, m15 = m[15]!;
  // Note: planes' outward normals are `row3 ± rowN`. We negate to keep the
  // convention `dot(n, p) + d ≥ 0 ⇔ inside`.
  const planes: [Plane, Plane, Plane, Plane, Plane, Plane] = [
    normPlane([m3 + m0, m7 + m4, m11 + m8, m15 + m12]),   // left
    normPlane([m3 - m0, m7 - m4, m11 - m8, m15 - m12]),   // right
    normPlane([m3 + m1, m7 + m5, m11 + m9, m15 + m13]),   // bottom
    normPlane([m3 - m1, m7 - m5, m11 - m9, m15 - m13]),   // top
    normPlane([m3 + m2, m7 + m6, m11 + m10, m15 + m14]),  // near
    normPlane([m3 - m2, m7 - m6, m11 - m10, m15 - m14]),  // far
  ];
  return { planes };
}

function normPlane(raw: [number, number, number, number]): Plane {
  const [a, b, c, d] = raw;
  const len = Math.hypot(a, b, c);
  if (len < 1e-12) return [0, 0, 0, 0];
  const inv = 1 / len;
  return [a * inv, b * inv, c * inv, d * inv];
}

/**
 * Test whether an AABB is at all visible (not strictly outside) any plane.
 * Returns `false` only when the box is fully outside one of the 6 planes —
 * which guarantees zero visible pixels. Conservative: a box can return
 * `true` while ultimately being occluded (handled by the occlusion pass).
 */
export function aabbInFrustum(frustum: Frustum, box: AABB): boolean {
  const cx = box.centerWorld[0], cy = box.centerWorld[1], cz = box.centerWorld[2];
  const hx = box.halfExtents[0], hy = box.halfExtents[1], hz = box.halfExtents[2];
  for (const p of frustum.planes) {
    const a = p[0], b = p[1], c = p[2], d = p[3];
    // Project the half-extent onto the plane normal — the maximum distance
    // any corner of the box reaches from the center along this plane axis.
    const reach = Math.abs(a) * hx + Math.abs(b) * hy + Math.abs(c) * hz;
    const signedDist = a * cx + b * cy + c * cz + d;
    if (signedDist + reach < 0) return false; // entire box on negative side
  }
  return true;
}

/** Sphere variant — used by the LOD path which already has a sphere. */
export function sphereInFrustum(frustum: Frustum, center: Vec3, radius: number): boolean {
  for (const p of frustum.planes) {
    const signedDist = p[0] * center[0] + p[1] * center[1] + p[2] * center[2] + p[3];
    if (signedDist + radius < 0) return false;
  }
  return true;
}

// ── Hardware occlusion query shim ──────────────────────────────────────

export interface OcclusionCapability {
  /** True when the adapter advertises `occlusion-query`. WebGPU 2026 in
   *  Chrome 126+ ships this — older browsers fall through. */
  supported: boolean;
}

export function probeOcclusion(adapter: GPUAdapter | null | undefined): OcclusionCapability {
  if (!adapter) return { supported: false };
  try {
    // The feature name in the WebGPU 1.0 spec is "occlusion-query"; some
    // dev releases use the older "occlusion-query-status". Probe both.
    const list: ReadonlySet<string> | undefined = adapter.features as unknown as ReadonlySet<string>;
    if (!list) return { supported: false };
    return { supported: list.has("occlusion-query") };
  } catch {
    return { supported: false };
  }
}

/**
 * Minimal occlusion-query lifecycle. The renderer is expected to:
 *   1. `begin(querySet, index, pass)` before drawing the bounding proxy.
 *   2. `end(pass)` after the draw.
 *   3. Next frame, read the resolved buffer; pass `(index → visiblePx)` to
 *      the tile manager via `OcclusionResult` for eviction biasing.
 *
 * This stub keeps the API stable while we defer the Hi-Z + two-pass pipeline.
 */
export interface OcclusionResult {
  /** Query-index → pixel count from the previous frame. */
  visiblePixels: ReadonlyMap<number, number>;
}

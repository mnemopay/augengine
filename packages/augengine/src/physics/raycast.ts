/**
 * Raycast — fire a ray through the scene, return the nearest collider hit.
 *
 * Tier 2 game-readiness. Wired to mouse click in the smoke-game so the player
 * can "look at Maya and click" to interact. Ray-vs-AABB / sphere / capsule /
 * heightfield. Returns the entity id + world-space contact point + surface
 * normal + parametric distance.
 */
import type { Scene } from "../scene/scene.js";
import { Tag, type EntityId, type HeightfieldShape, type Vec3, type Quat } from "../scene/types.js";
import { aabbOverlap, type AABB, type Capsule } from "./colliders.js";
import { sampleHeightfieldAt } from "./heightfield.js";
import { qInvert, qRotate, type MutQuat, type MutVec3 } from "../scene/math.js";

export interface RayHit {
  /** Entity whose collider the ray struck. */
  entityId: EntityId;
  /** Parametric distance from origin along direction. */
  distance: number;
  /** World-space contact point. */
  point: Vec3;
  /** Approximate surface normal at the contact point. */
  normal: Vec3;
}

export interface RaycastOptions {
  /** Max distance along direction to consider. Default 100m. */
  maxDistance?: number;
  /** Entity ids to skip (e.g. the player firing the ray). */
  ignore?: ReadonlyArray<EntityId>;
}

/**
 * Raycast against every collider-bearing entity in the scene. The ray
 * direction does not need to be normalized — internally we normalize and use
 * `maxDistance` as the cap.
 */
export function raycast(
  scene: Scene,
  origin: Vec3,
  direction: Vec3,
  opts: RaycastOptions = {},
): RayHit | null {
  const maxDist = opts.maxDistance ?? 100;
  const ignore = new Set(opts.ignore ?? []);
  const len = Math.hypot(direction[0], direction[1], direction[2]);
  if (len < 1e-9) return null;
  const dir: Vec3 = [direction[0] / len, direction[1] / len, direction[2] / len];

  let nearest: RayHit | null = null;
  for (const entity of scene.withComponent(Tag.Collider)) {
    if (ignore.has(entity.id)) continue;
    const col = entity.components[Tag.Collider]!;
    const tx = entity.components[Tag.Transform];
    if (!tx) continue;
    const hit = hitForCollider(origin, dir, maxDist, col, tx.position, tx.rotation);
    if (!hit) continue;
    if (!nearest || hit.distance < nearest.distance) {
      nearest = {
        entityId: entity.id,
        distance: hit.distance,
        point: [
          origin[0] + dir[0] * hit.distance,
          origin[1] + dir[1] * hit.distance,
          origin[2] + dir[2] * hit.distance,
        ],
        normal: hit.normal,
      };
    }
  }
  return nearest;
}

interface PrimitiveHit { distance: number; normal: Vec3 }

function hitForCollider(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  col: { shape: string; radius?: number; halfHeight?: number; extents?: Vec3; heightfield?: HeightfieldShape },
  pos: Vec3,
  rot?: Quat,
): PrimitiveHit | null {
  if (col.shape === "box" && col.extents) {
    if (rot && (rot[0] !== 0 || rot[1] !== 0 || rot[2] !== 0 || rot[3] !== 1)) {
      const invRot: MutQuat = [0, 0, 0, 1];
      qInvert(invRot, rot);

      const diff: MutVec3 = [
        origin[0] - pos[0],
        origin[1] - pos[1],
        origin[2] - pos[2],
      ];
      const localOrigin: MutVec3 = [0, 0, 0];
      qRotate(localOrigin, invRot, diff);

      const localDir: MutVec3 = [0, 0, 0];
      qRotate(localDir, invRot, dir);

      const box: AABB = {
        min: [-col.extents[0], -col.extents[1], -col.extents[2]],
        max: [col.extents[0], col.extents[1], col.extents[2]],
      };

      const hit = rayAabb(localOrigin, localDir, maxDist, box);
      if (hit) {
        const worldNormal: MutVec3 = [0, 0, 0];
        qRotate(worldNormal, rot, hit.normal);
        return {
          distance: hit.distance,
          normal: worldNormal,
        };
      }
      return null;
    } else {
      const box: AABB = {
        min: [pos[0] - col.extents[0], pos[1] - col.extents[1], pos[2] - col.extents[2]],
        max: [pos[0] + col.extents[0], pos[1] + col.extents[1], pos[2] + col.extents[2]],
      };
      return rayAabb(origin, dir, maxDist, box);
    }
  }
  if (col.shape === "sphere" && col.radius != null) {
    return raySphere(origin, dir, maxDist, pos, col.radius);
  }
  if (col.shape === "capsule" && col.radius != null && col.halfHeight != null) {
    const cap: Capsule = {
      kind: "capsule",
      p0: [pos[0], pos[1] - col.halfHeight, pos[2]],
      p1: [pos[0], pos[1] + col.halfHeight, pos[2]],
      radius: col.radius,
    };
    return rayCapsule(origin, dir, maxDist, cap);
  }
  if (col.shape === "heightfield" && col.heightfield) {
    return rayHeightfield(origin, dir, maxDist, col.heightfield, pos);
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────
// Ray ↔ Heightfield — clip to the field's bounds, march half a cell at a
// time until the ray drops below the surface, then bisect the crossing.
// ──────────────────────────────────────────────────────────────────────

export function rayHeightfield(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  hf: HeightfieldShape,
  pos: Vec3,
): PrimitiveHit | null {
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < hf.heights.length; k++) {
    const h = hf.heights[k]!;
    if (h < lo) lo = h;
    if (h > hi) hi = h;
  }
  const box: AABB = {
    min: [pos[0] - hf.sizeX / 2, pos[1] + lo - 1e-3, pos[2] - hf.sizeZ / 2],
    max: [pos[0] + hf.sizeX / 2, pos[1] + hi + 1e-3, pos[2] + hf.sizeZ / 2],
  };
  // Entry/exit along the ray (origin may already be inside the box).
  let t0 = 0, t1 = maxDist;
  for (let i = 0; i < 3; i++) {
    const o = origin[i]!, d = dir[i]!;
    if (Math.abs(d) < 1e-12) {
      if (o < box.min[i]! || o > box.max[i]!) return null;
      continue;
    }
    let a = (box.min[i]! - o) / d, b = (box.max[i]! - o) / d;
    if (a > b) { const t = a; a = b; b = t; }
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    if (t0 > t1) return null;
  }
  const above = (t: number): number | null => {
    const x = origin[0] + dir[0] * t, y = origin[1] + dir[1] * t, z = origin[2] + dir[2] * t;
    const s = sampleHeightfieldAt(hf, pos, x, z);
    return s ? y - s.height : null;
  };
  const step = Math.min(hf.sizeX / (hf.samplesX - 1), hf.sizeZ / (hf.samplesZ - 1)) * 0.5;
  let prevT = t0;
  let prev = above(t0);
  if (prev !== null && prev <= 0) return null; // started under the surface
  for (let t = t0 + step; ; t += step) {
    const tt = Math.min(t, t1);
    const cur = above(tt);
    if (prev !== null && cur !== null && cur <= 0) {
      let a = prevT, b = tt;
      for (let k = 0; k < 24; k++) {
        const m = (a + b) / 2;
        const v = above(m);
        if (v !== null && v <= 0) b = m; else a = m;
      }
      const x = origin[0] + dir[0] * b, z = origin[2] + dir[2] * b;
      const s = sampleHeightfieldAt(hf, pos, x, z)!;
      return { distance: b, normal: s.normal };
    }
    prev = cur;
    prevT = tt;
    if (tt >= t1) return null;
  }
}

// ──────────────────────────────────────────────────────────────────────
// Ray ↔ AABB — slab method (Kay & Kajiya). Returns nearest non-negative t.
// ──────────────────────────────────────────────────────────────────────

export function rayAabb(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  box: AABB,
): PrimitiveHit | null {
  let tmin = 0;
  let tmax = maxDist;
  let hitAxis = 0;
  let hitSign = 0;
  for (let i = 0; i < 3; i++) {
    const o = origin[i]!;
    const d = dir[i]!;
    const mn = box.min[i]!;
    const mx = box.max[i]!;
    if (Math.abs(d) < 1e-9) {
      if (o < mn || o > mx) return null;
    } else {
      const inv = 1 / d;
      let t1 = (mn - o) * inv;
      let t2 = (mx - o) * inv;
      let sign = -1;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; sign = 1; }
      if (t1 > tmin) { tmin = t1; hitAxis = i; hitSign = sign; }
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return null;
    }
  }
  if (tmin <= 0 || tmin > maxDist) return null;
  const normal: Vec3 = [
    hitAxis === 0 ? hitSign : 0,
    hitAxis === 1 ? hitSign : 0,
    hitAxis === 2 ? hitSign : 0,
  ];
  return { distance: tmin, normal };
}

// ──────────────────────────────────────────────────────────────────────
// Ray ↔ Sphere — quadratic solve.
// ──────────────────────────────────────────────────────────────────────

export function raySphere(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  center: Vec3,
  radius: number,
): PrimitiveHit | null {
  const ox = origin[0] - center[0];
  const oy = origin[1] - center[1];
  const oz = origin[2] - center[2];
  const b = ox * dir[0] + oy * dir[1] + oz * dir[2];
  const c = ox * ox + oy * oy + oz * oz - radius * radius;
  if (c > 0 && b > 0) return null;
  const disc = b * b - c;
  if (disc < 0) return null;
  const sqrtDisc = Math.sqrt(disc);
  let t = -b - sqrtDisc;
  if (t < 0) t = -b + sqrtDisc;
  if (t < 0 || t > maxDist) return null;
  const hx = origin[0] + dir[0] * t;
  const hy = origin[1] + dir[1] * t;
  const hz = origin[2] + dir[2] * t;
  const nx = (hx - center[0]) / radius;
  const ny = (hy - center[1]) / radius;
  const nz = (hz - center[2]) / radius;
  return { distance: t, normal: [nx, ny, nz] };
}

// ──────────────────────────────────────────────────────────────────────
// Ray ↔ Capsule — clamp ray-line / capsule-segment closest pair, then sphere.
// ──────────────────────────────────────────────────────────────────────

export function rayCapsule(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  cap: Capsule,
): PrimitiveHit | null {
  // Approximate: ray vs cylinder body, then vs two end-sphere caps. Take the
  // nearest valid hit. Good enough for character-radius capsules.
  let nearest: PrimitiveHit | null = null;
  const cyl = rayCylinder(origin, dir, maxDist, cap.p0, cap.p1, cap.radius);
  if (cyl) nearest = cyl;
  const sphereA = raySphere(origin, dir, maxDist, cap.p0, cap.radius);
  if (sphereA && (!nearest || sphereA.distance < nearest.distance)) nearest = sphereA;
  const sphereB = raySphere(origin, dir, maxDist, cap.p1, cap.radius);
  if (sphereB && (!nearest || sphereB.distance < nearest.distance)) nearest = sphereB;
  return nearest;
}

function rayCylinder(
  origin: Vec3,
  dir: Vec3,
  maxDist: number,
  p0: Vec3,
  p1: Vec3,
  radius: number,
): PrimitiveHit | null {
  // Solve |(O + tD) - (p0 + s(p1-p0))|² = r² with s ∈ [0,1].
  const ax = p1[0] - p0[0];
  const ay = p1[1] - p0[1];
  const az = p1[2] - p0[2];
  const aa = ax * ax + ay * ay + az * az;
  if (aa < 1e-9) return null;
  const dx = origin[0] - p0[0];
  const dy = origin[1] - p0[1];
  const dz = origin[2] - p0[2];
  const dotDA = dir[0] * ax + dir[1] * ay + dir[2] * az;
  const dotDeltaA = dx * ax + dy * ay + dz * az;
  const Ax = dir[0] - (dotDA / aa) * ax;
  const Ay = dir[1] - (dotDA / aa) * ay;
  const Az = dir[2] - (dotDA / aa) * az;
  const Bx = dx - (dotDeltaA / aa) * ax;
  const By = dy - (dotDeltaA / aa) * ay;
  const Bz = dz - (dotDeltaA / aa) * az;
  const A = Ax * Ax + Ay * Ay + Az * Az;
  const B = 2 * (Ax * Bx + Ay * By + Az * Bz);
  const C = Bx * Bx + By * By + Bz * Bz - radius * radius;
  if (Math.abs(A) < 1e-9) return null;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return null;
  const sqrtDisc = Math.sqrt(disc);
  let t = (-B - sqrtDisc) / (2 * A);
  if (t < 0) t = (-B + sqrtDisc) / (2 * A);
  if (t < 0 || t > maxDist) return null;
  // Confirm the hit is on the finite segment.
  const hx = origin[0] + dir[0] * t;
  const hy = origin[1] + dir[1] * t;
  const hz = origin[2] + dir[2] * t;
  const sNumer = (hx - p0[0]) * ax + (hy - p0[1]) * ay + (hz - p0[2]) * az;
  const s = sNumer / aa;
  if (s < 0 || s > 1) return null;
  // Surface normal: from segment-projection to hit point.
  const projX = p0[0] + ax * s;
  const projY = p0[1] + ay * s;
  const projZ = p0[2] + az * s;
  const nx = (hx - projX) / radius;
  const ny = (hy - projY) / radius;
  const nz = (hz - projZ) / radius;
  return { distance: t, normal: [nx, ny, nz] };
}

// Re-export aabbOverlap so callers writing custom culling don't have to dual-import.
export { aabbOverlap };

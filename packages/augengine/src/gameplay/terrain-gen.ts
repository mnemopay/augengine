/**
 * Procedural terrain + scatter for small levels (an arena, a valley), as
 * opposed to `runtime/world/terrain.ts`, which is tied to the 1 km streamed
 * level's tiles.
 *
 *   generateTerrain(opts) → heights grid, Rapier `heightfield` collider,
 *                           a render mesh (vertex-coloured by slope/height),
 *                           heightAt / normalAt queries on the exact triangles
 *                           Rapier uses (cells split along (x1,z0)–(x0,z1)).
 *   scatter(terrain, opts) → seeded, spacing-respecting placements filtered by
 *                           slope and height, for entities that share one mesh
 *                           key (the renderer instances groups of ≥ 4).
 *   grassTuftMesh / rockMesh → small procedural meshes to scatter.
 *
 * Everything is deterministic for a given seed.
 */

import type { ColliderComponent, HeightfieldShape, Quat, Vec3 } from "../scene/types.js";
import type { MeshData } from "../runtime/live/webgpu/primitives.js";
import { sampleHeightfield } from "../physics/heightfield.js";
import { mulberry32 } from "./triggers.js";

export interface TerrainOptions {
  seed?: number;
  /** World size (metres) along x and z. Default 80. */
  size?: number;
  /** Quads per edge. Default 64. */
  cells?: number;
  /** Peak height above the base (metres). Default 4. */
  amplitude?: number;
  /** Largest feature size (metres). Default 28. */
  wavelength?: number;
  octaves?: number;
  /** Keep a flat disc (height 0) of this radius around `flatCenter`, blending over `flatBlend`. */
  flatRadius?: number;
  flatBlend?: number;
  flatCenter?: [number, number];
  /** Raise the rim so the playable area reads as a bowl. Default 0. */
  rim?: number;
}

export interface Terrain {
  readonly size: number;
  readonly cells: number;
  /** (cells+1)² heights, row-major by z: index = j * (cells + 1) + i. */
  readonly heights: Float32Array;
  readonly shape: HeightfieldShape;
  /** Collider for an entity whose transform position is [0, 0, 0] (the grid is centred on it). */
  readonly collider: ColliderComponent;
  readonly minHeight: number;
  readonly maxHeight: number;
  heightAt(x: number, z: number): number;
  normalAt(x: number, z: number): Vec3;
  /** Render mesh in world space around the origin (8-float vertices + RGBA colours). */
  mesh(): MeshData;
}

function hash2(ix: number, iz: number, seed: number): number {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}

function valueNoise(x: number, z: number, seed: number): number {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const u = fx * fx * fx * (fx * (fx * 6 - 15) + 10), w = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
  const a = hash2(ix, iz, seed), b = hash2(ix + 1, iz, seed), c = hash2(ix, iz + 1, seed), d = hash2(ix + 1, iz + 1, seed);
  return (a + (b - a) * u + (c + (d - c) * u - (a + (b - a) * u)) * w) * 2 - 1;
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

export function generateTerrain(opts: TerrainOptions = {}): Terrain {
  const seed = opts.seed ?? 1, size = opts.size ?? 80, cells = opts.cells ?? 64;
  const amp = opts.amplitude ?? 4, wl = opts.wavelength ?? 28, oct = opts.octaves ?? 4;
  const [fcx, fcz] = opts.flatCenter ?? [0, 0];
  const flatR = opts.flatRadius ?? 0, flatB = opts.flatBlend ?? 10, rim = opts.rim ?? 0;
  const n = cells + 1, step = size / cells, half = size / 2;
  const heights = new Float32Array(n * n);
  let minH = Infinity, maxH = -Infinity;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = -half + i * step, z = -half + j * step;
      let sum = 0, a = 1, f = 1 / wl, norm = 0;
      for (let o = 0; o < oct; o++) {
        sum += a * valueNoise(x * f + o * 31.7, z * f - o * 17.3, seed + o);
        norm += a;
        a *= 0.5;
        f *= 2.02;
      }
      let h = (sum / norm) * amp;
      const d = Math.hypot(x - fcx, z - fcz);
      if (flatR > 0) h *= smoothstep(flatR, flatR + flatB, d);
      if (rim > 0) h += rim * smoothstep(half * 0.55, half * 0.95, Math.max(Math.abs(x), Math.abs(z)));
      h = Math.fround(h);
      heights[j * n + i] = h;
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;
    }
  }
  const shape: HeightfieldShape = { samplesX: n, samplesZ: n, heights, sizeX: size, sizeZ: size };
  const heightAt = (x: number, z: number) => sampleHeightfield(shape, -half, -half, 0, x, z)?.height ?? 0;
  const normalAt = (x: number, z: number): Vec3 => sampleHeightfield(shape, -half, -half, 0, x, z)?.normal ?? [0, 1, 0];
  return {
    size, cells, heights, shape,
    collider: { shape: "heightfield", heightfield: shape },
    minHeight: minH, maxHeight: maxH,
    heightAt, normalAt,
    mesh: () => terrainMesh(heights, n, size, minH, maxH),
  };
}

/** Grass on flat ground, dirt on slopes, lighter rock high up and on steep faces. */
function terrainMesh(h: Float32Array, n: number, size: number, minH: number, maxH: number): MeshData {
  const step = size / (n - 1), half = size / 2;
  const verts = new Float32Array(n * n * 8);
  const colors = new Float32Array(n * n * 4);
  const at = (i: number, j: number) => h[Math.min(n - 1, Math.max(0, j)) * n + Math.min(n - 1, Math.max(0, i))]!;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const o = (j * n + i) * 8;
      const dx = (at(i + 1, j) - at(i - 1, j)) / (2 * step), dz = (at(i, j + 1) - at(i, j - 1)) / (2 * step);
      const l = Math.hypot(dx, 1, dz);
      const ny = 1 / l;
      verts[o] = -half + i * step; verts[o + 1] = at(i, j); verts[o + 2] = -half + j * step;
      verts[o + 3] = -dx / l; verts[o + 4] = ny; verts[o + 5] = -dz / l;
      verts[o + 6] = i / 8; verts[o + 7] = j / 8;
      // Colour: slope 0 → grass, steep → dirt/rock; height adds a little lightness. Small hash jitter breaks banding.
      const slope = smoothstep(0.93, 0.78, ny);
      const hi = maxH > minH ? (at(i, j) - minH) / (maxH - minH) : 0;
      const jit = (hash2(i, j, 99) - 0.5) * 0.06;
      const grass: Vec3 = [0.3 + jit, 0.46 + jit * 1.5, 0.2 + jit];
      const dirt: Vec3 = [0.45 + jit, 0.37 + jit, 0.26 + jit];
      const rock: Vec3 = [0.55 + jit, 0.53 + jit, 0.5 + jit];
      const mid: Vec3 = [grass[0] + (dirt[0] - grass[0]) * slope, grass[1] + (dirt[1] - grass[1]) * slope, grass[2] + (dirt[2] - grass[2]) * slope];
      const rk = smoothstep(0.75, 0.6, ny) * 0.8 + smoothstep(0.7, 1, hi) * 0.25;
      const c = o / 2;
      colors[c] = mid[0] + (rock[0] - mid[0]) * rk;
      colors[c + 1] = mid[1] + (rock[1] - mid[1]) * rk;
      colors[c + 2] = mid[2] + (rock[2] - mid[2]) * rk;
      colors[c + 3] = 1;
    }
  }
  const idx = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let j = 0; j < n - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      // Diagonal b–c (Rapier's split), counter-clockwise seen from +Y.
      idx[k++] = a; idx[k++] = c; idx[k++] = b;
      idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
  }
  return { vertices: verts, indices: idx, vertexCount: n * n, indexCount: idx.length, colors };
}

// ──────────────────────────────────────────────────────────────────────
// Scatter
// ──────────────────────────────────────────────────────────────────────

export interface ScatterOptions {
  seed?: number;
  /** Target number of placements (fewer when spacing/filters reject candidates). */
  count: number;
  /** Minimum horizontal distance between placements of this call. */
  minSpacing?: number;
  /** Placement area (centre-based square half-size). Default: the whole terrain minus a 1 m margin. */
  halfExtent?: number;
  center?: [number, number];
  /** Skip points closer than this to any of `avoid` (e.g. the arena centre, paths). */
  avoid?: Array<{ x: number; z: number; radius: number }>;
  /** Only where the surface normal's y ≥ this (1 = flat). Default 0.8. */
  minNormalY?: number;
  /** Only within this height range. */
  heightRange?: [number, number];
  scaleRange?: [number, number];
  /** Tilt to follow the slope (0 = upright, 1 = fully aligned to the normal). Default 0. */
  alignToNormal?: number;
  /** Push the placement this far into the ground (rocks sit in, not on). Default 0. */
  sink?: number;
  /** Candidate tries per wanted placement. Default 8. */
  tries?: number;
}

export interface Placement {
  position: Vec3;
  rotation: Quat;
  scale: number;
}

/** Dart-throwing with a spatial hash: deterministic, O(count · tries). */
export function scatter(terrain: Pick<Terrain, "size" | "heightAt" | "normalAt">, o: ScatterOptions): Placement[] {
  const rng = mulberry32(o.seed ?? 7);
  const half = o.halfExtent ?? terrain.size / 2 - 1;
  const [cx, cz] = o.center ?? [0, 0];
  const spacing = o.minSpacing ?? 0;
  const minNy = o.minNormalY ?? 0.8;
  const [s0, s1] = o.scaleRange ?? [1, 1];
  const align = o.alignToNormal ?? 0;
  const cell = Math.max(spacing, 0.5);
  const grid = new Map<string, Array<[number, number]>>();
  const out: Placement[] = [];
  const tries = o.count * (o.tries ?? 8);
  for (let t = 0; t < tries && out.length < o.count; t++) {
    const x = cx + (rng() * 2 - 1) * half, z = cz + (rng() * 2 - 1) * half;
    const yawR = rng() * Math.PI * 2, sR = rng();
    if (o.avoid?.some((a) => Math.hypot(x - a.x, z - a.z) < a.radius)) continue;
    const nrm = terrain.normalAt(x, z);
    if (nrm[1] < minNy) continue;
    const y = terrain.heightAt(x, z);
    if (o.heightRange && (y < o.heightRange[0] || y > o.heightRange[1])) continue;
    if (spacing > 0) {
      const gx = Math.floor(x / cell), gz = Math.floor(z / cell);
      let clash = false;
      for (let dz = -1; dz <= 1 && !clash; dz++) for (let dx = -1; dx <= 1 && !clash; dx++) {
        for (const [px, pz] of grid.get(`${gx + dx},${gz + dz}`) ?? []) if (Math.hypot(px - x, pz - z) < spacing) { clash = true; break; }
      }
      if (clash) continue;
      const key = `${gx},${gz}`;
      let list = grid.get(key);
      if (!list) grid.set(key, (list = []));
      list.push([x, z]);
    }
    const scale = s0 + (s1 - s0) * sR;
    // Yaw about +Y, then tilt from +Y toward the normal by `align`.
    const yawQ: Quat = [0, Math.sin(yawR / 2), 0, Math.cos(yawR / 2)];
    let rot = yawQ;
    if (align > 0) {
      const tx = nrm[0] * align, ty = 1 + (nrm[1] - 1) * align, tz = nrm[2] * align;
      const l = Math.hypot(tx, ty, tz);
      const nx = tx / l, ny = ty / l, nz = tz / l;
      // from +Y to n: axis (nz, 0, -nx), w = 1 + ny
      let q: [number, number, number, number] = [nz, 0, -nx, 1 + ny];
      const ql = Math.hypot(...q);
      q = [q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql];
      rot = qmul(q, yawQ);
    }
    out.push({ position: [x, y - (o.sink ?? 0) * scale, z], rotation: rot, scale });
  }
  return out;
}

function qmul(a: Quat, b: Quat): Quat {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

// ──────────────────────────────────────────────────────────────────────
// Scatter meshes
// ──────────────────────────────────────────────────────────────────────

/**
 * A grass tuft: `blades` tapered blades fanned around the origin, each emitted
 * with both windings (the renderer culls back faces). Base at y = 0, ~0.5 m tall.
 * Vertex colours darken toward the root.
 */
export function grassTuftMesh(seed = 3, blades = 7): MeshData {
  const rng = mulberry32(seed);
  const v: number[] = [], c: number[] = [], idx: number[] = [];
  for (let b = 0; b < blades; b++) {
    const a = (b / blades) * Math.PI * 2 + rng() * 0.6;
    const r = 0.05 + rng() * 0.12;
    const bx = Math.cos(a) * r, bz = Math.sin(a) * r;
    const h = 0.32 + rng() * 0.28, w = 0.035 + rng() * 0.02;
    const lean = 0.08 + rng() * 0.14;
    // Blade plane faces the radial direction rotated 90 degrees.
    const px = -Math.sin(a), pz = Math.cos(a);
    const tipX = bx + Math.cos(a) * lean, tipZ = bz + Math.sin(a) * lean;
    const nx = Math.cos(a) * 0.4, ny = 0.9, nz = Math.sin(a) * 0.4;
    const base = v.length / 8;
    const push = (x: number, y: number, z: number, shade: number) => {
      v.push(x, y, z, nx, ny, nz, 0, 0);
      c.push(0.24 * shade, 0.42 * shade, 0.16 * shade, 1);
    };
    push(bx - px * w, 0, bz - pz * w, 0.55);
    push(bx + px * w, 0, bz + pz * w, 0.55);
    push(tipX, h, tipZ, 1.15);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 1);
  }
  return { vertices: Float32Array.from(v), indices: Uint32Array.from(idx), vertexCount: v.length / 8, indexCount: idx.length, colors: Float32Array.from(c) };
}

/** A low-poly rock: an octahedron subdivided once, vertices jittered by a seeded radius, flattened a little. ~0.5 m across. */
export function rockMesh(seed = 5): MeshData {
  const rng = mulberry32(seed);
  const base: Vec3[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const faces: Array<[number, number, number]> = [[0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0], [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5]];
  const pts: Vec3[] = [...base];
  const mid = new Map<string, number>();
  const midpoint = (a: number, b: number) => {
    const k = a < b ? `${a}-${b}` : `${b}-${a}`;
    let m = mid.get(k);
    if (m === undefined) {
      const p = pts[a]!, q = pts[b]!;
      const x = (p[0] + q[0]) / 2, y = (p[1] + q[1]) / 2, z = (p[2] + q[2]) / 2, l = Math.hypot(x, y, z);
      m = pts.push([x / l, y / l, z / l]) - 1;
      mid.set(k, m);
    }
    return m;
  };
  const tris: Array<[number, number, number]> = [];
  for (const [a, b, c] of faces) {
    const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
    tris.push([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]);
  }
  const radius = pts.map(() => 0.22 + rng() * 0.1);
  const P = pts.map((p, i) => [p[0] * radius[i]!, p[1] * radius[i]! * 0.7, p[2] * radius[i]!] as Vec3);
  // Flat-shaded: three vertices per triangle.
  const v = new Float32Array(tris.length * 3 * 8);
  const col = new Float32Array(tris.length * 3 * 4);
  const idx = new Uint32Array(tris.length * 3);
  tris.forEach(([a, b, c], t) => {
    const A = P[a]!, B = P[b]!, C = P[c]!;
    const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2], wx = C[0] - A[0], wy = C[1] - A[1], wz = C[2] - A[2];
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    // Outward winding check (centroid direction).
    const cx = A[0] + B[0] + C[0], cy = A[1] + B[1] + C[1], cz = A[2] + B[2] + C[2];
    const flip = nx * cx + ny * cy + nz * cz < 0;
    const order = flip ? [A, C, B] : [A, B, C];
    if (flip) { nx = -nx; ny = -ny; nz = -nz; }
    const shade = 0.42 + rng() * 0.12;
    order.forEach((p, k) => {
      const o = (t * 3 + k) * 8;
      v[o] = p[0]; v[o + 1] = p[1] + 0.12; v[o + 2] = p[2];
      v[o + 3] = nx; v[o + 4] = ny; v[o + 5] = nz;
      const ci = (t * 3 + k) * 4;
      col[ci] = shade; col[ci + 1] = shade * 0.97; col[ci + 2] = shade * 0.92; col[ci + 3] = 1;
      idx[t * 3 + k] = t * 3 + k;
    });
  });
  return { vertices: v, indices: idx, vertexCount: tris.length * 3, indexCount: idx.length, colors: col };
}

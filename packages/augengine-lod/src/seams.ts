/**
 * seams — UV-seam handling for textured simplification.
 *
 * A textured mesh is split wherever its UV map is cut: the same position
 * appears once per UV island ("wedge") that meets there. Image-to-3D output
 * (TRELLIS, generated characters) is cut into hundreds of islands, so 20-45%
 * of positions sit on a seam. 0.2.0-alpha.0 pinned every one of those
 * vertices, which stopped simplification near LOD1 on such meshes.
 *
 * What the bake does instead:
 *
 *   1. `weldUvSeams` (pre-pass): wedges at one position whose UVs differ by
 *      less than a tolerance (default half a texel of the material's
 *      baseColor image) are snapped together. Such "seams" are exporter
 *      rounding, not cuts, and splitting there only blocks collapses.
 *   2. Real seams are left to meshoptimizer: a seam vertex with two wedges may
 *      collapse only along its seam, onto another seam vertex, with both
 *      wedges moving together, so the seam stays closed and every surviving
 *      corner keeps a UV from its own island (no texture swimming). Junctions
 *      (three or more islands, or a material boundary) never collapse.
 *   3. `seamClasses` tells the two apart for `uvSeams: "lock"`, which pins
 *      only junctions — not every vertex with a UV split.
 *
 * `uvIslandBreaks` counts output triangles whose corners come from different
 * UV islands: each one stretches the texture across the atlas. The bake
 * reports it per level; with "preserve" and "lock" it is 0.
 */
import { triangleMaterials, type KitMesh } from "./mesh.js";

const Q = 1e4;

/**
 * Vertices grouped by quantized position (1e-4), optionally also by `key`:
 * returns each group as an array of vertex indices. Sort-based (no string
 * keys), so it stays fast on 100k-vertex scans.
 */
function groupByPosition(p: Float32Array, key?: ArrayLike<number>): number[][] {
  const n = p.length / 3;
  const q = new Int32Array(n * 3);
  for (let i = 0; i < n * 3; i++) q[i] = Math.round(p[i]! * Q);
  const order = Uint32Array.from({ length: n }, (_, i) => i);
  const cmp = (a: number, b: number) =>
    (key ? key[a]! - key[b]! : 0) || q[a * 3]! - q[b * 3]! || q[a * 3 + 1]! - q[b * 3 + 1]! || q[a * 3 + 2]! - q[b * 3 + 2]!;
  order.sort(cmp);
  const groups: number[][] = [];
  let cur: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = order[i]!;
    if (cur.length && cmp(cur[0]!, v) !== 0) { groups.push(cur); cur = []; }
    cur.push(v);
  }
  if (cur.length) groups.push(cur);
  return groups;
}

export interface SeamWeldResult {
  mesh: KitMesh;
  /** Vertex corners whose UV was snapped onto a neighbouring wedge. */
  snapped: number;
}

/**
 * Snap UV wedges at the same position (and material) that lie within
 * `tolerance[material]` (UV units, Chebyshev distance) onto one UV. Returns a
 * copy; the input is untouched. Positions, normals and colours are unchanged.
 */
export function weldUvSeams(m: KitMesh, tolerance: (material: number) => number): SeamWeldResult {
  const triMat = triangleMaterials(m);
  const vMat = new Int32Array(m.positions.length / 3).fill(-1);
  const uvs = m.uvs.slice();
  // Material of each vertex: the first triangle that uses it (UVs are per vertex).
  for (let t = 0; t < m.indices.length; t++) {
    const v = m.indices[t]!;
    if (vMat[v] === -1) vMat[v] = triMat[Math.floor(t / 3)]!;
  }
  let snapped = 0;
  for (const g of groupByPosition(m.positions, vMat)) {
    if (g.length < 2 || vMat[g[0]!] === -1) continue;
    const tol = tolerance(vMat[g[0]!]!);
    if (!(tol > 0)) continue;
    // Representatives: first vertex of each cluster.
    const reps: number[] = [];
    for (const v of g) {
      const u = uvs[v * 2]!, w = uvs[v * 2 + 1]!;
      const r = reps.find((r) => Math.abs(uvs[r * 2]! - u) <= tol && Math.abs(uvs[r * 2 + 1]! - w) <= tol);
      if (r === undefined) { reps.push(v); continue; }
      if (uvs[r * 2] !== u || uvs[r * 2 + 1] !== w) {
        uvs[v * 2] = uvs[r * 2]!;
        uvs[v * 2 + 1] = uvs[r * 2 + 1]!;
        snapped++;
      }
    }
  }
  return { mesh: { ...m, uvs }, snapped };
}

export const SEAM_INTERIOR = 0;
/** Two UV wedges meet here: may collapse along the seam. */
export const SEAM_EDGE = 1;
/** Three or more wedges, or more than one material: a junction. */
export const SEAM_JUNCTION = 2;

export interface SeamStats {
  /** Distinct positions. */
  positions: number;
  /** Positions on a UV/material seam (two or more wedges). */
  seamPositions: number;
  /** Seam positions where three or more wedges or two materials meet. */
  junctions: number;
}

/**
 * Classify each vertex of a welded mesh (one vertex per position + material
 * + UV wedge, as `bakeLods` builds it). Wedges are counted by distinct UV and
 * material only: a split that exists only because of a hard normal or a
 * colour change is not a texture seam and stays SEAM_INTERIOR here.
 */
export function seamClasses(m: KitMesh, vertexMaterial: ArrayLike<number>): { classes: Uint8Array; stats: SeamStats } {
  const n = m.positions.length / 3;
  const byPos = groupByPosition(m.positions);
  const classes = new Uint8Array(n);
  let seamPositions = 0, junctions = 0;
  for (const g of byPos) {
    if (g.length < 2) continue;
    const wedges = new Set<string>(), mats = new Set<number>();
    for (const v of g) {
      mats.add(vertexMaterial[v]!);
      wedges.add(`${vertexMaterial[v]}|${Math.round(m.uvs[v * 2]! * Q)},${Math.round(m.uvs[v * 2 + 1]! * Q)}`);
    }
    if (wedges.size < 2) continue;
    seamPositions++;
    const cls = wedges.size >= 3 || mats.size >= 2 ? SEAM_JUNCTION : SEAM_EDGE;
    if (cls === SEAM_JUNCTION) junctions++;
    for (const v of g) classes[v] = cls;
  }
  return { classes, stats: { positions: byPos.length, seamPositions, junctions } };
}

/**
 * UV island of every vertex: connected through triangles, and through
 * vertices that share position, material and UV (so splits made only for
 * normals or colours do not separate an island).
 */
export function uvIslands(m: KitMesh, vertexMaterial: ArrayLike<number>): Int32Array {
  const n = m.positions.length / 3;
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]!]!; x = parent[x]!; }
    return x;
  };
  const union = (a: number, b: number) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
  for (let t = 0; t < m.indices.length; t += 3) {
    union(m.indices[t]!, m.indices[t + 1]!);
    union(m.indices[t]!, m.indices[t + 2]!);
  }
  for (const g of groupByPosition(m.positions, vertexMaterial)) {
    for (let i = 1; i < g.length; i++) for (let j = 0; j < i; j++) {
      const a = g[i]!, b = g[j]!;
      if (Math.round(m.uvs[a * 2]! * Q) === Math.round(m.uvs[b * 2]! * Q) && Math.round(m.uvs[a * 2 + 1]! * Q) === Math.round(m.uvs[b * 2 + 1]! * Q)) union(a, b);
    }
  }
  const out = new Int32Array(n);
  for (let v = 0; v < n; v++) out[v] = find(v);
  return out;
}

/** Triangles of `indices` whose three corners are not all from one UV island. */
export function uvIslandBreaks(indices: ArrayLike<number>, island: ArrayLike<number>): number {
  let breaks = 0;
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = island[indices[t]!], b = island[indices[t + 1]!], c = island[indices[t + 2]!];
    if (a !== b || b !== c) breaks++;
  }
  return breaks;
}

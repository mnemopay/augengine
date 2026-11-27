/**
 * world/terrain — heightfield ground that streams with The Loading Range tiles.
 *
 * One global height function (`terrainHeight`) defines the whole 1 km map, so
 * two neighbouring tiles sample identical heights on their shared edge and the
 * seam is continuous by construction. Each tile turns that function into:
 *
 *  - `samples`: an (N+1)² grid at `TERRAIN.spacing` metres — the single source
 *    for the visual mesh, the physics heightfield collider and nav slope.
 *  - three mesh LODs (every 1st / 2nd / 4th sample) with skirts hanging below
 *    every border edge, so a high-detail tile next to a low-detail tile never
 *    shows a crack.
 *  - a `heightfield` collider (ColliderComponent) that both the Rapier
 *    PhysicsSystem and the KinematicCharacterController read.
 *
 * Cells are split along the (x1,z0)–(x0,z1) diagonal everywhere — the split
 * Rapier's heightfield uses (checked against Rapier raycasts in the tests).
 * `sampleHeight` interpolates on exactly those triangles, so the collider
 * surface, the mesh surface and the analytic queries agree to float precision.
 *
 * The road and the landmark plazas are flattened to y = 0, so the walk route
 * and its landmarks stay where they were; hills rise away from the road.
 */
import { sampleHeightfield } from "../../physics/heightfield.js";
import type { ColliderComponent, HeightfieldShape, Vec3 } from "../../scene/types.js";
import type { MeshData } from "../live/webgpu/primitives.js";
import { LANDMARKS, LEVEL, TILE_SIZE } from "./level-content.js";
import type { NavBlocker } from "./tile-nav.js";

export const TERRAIN = {
  /** Quads per tile edge at LOD0 (2 m spacing on a 64 m tile). */
  cells: 32,
  spacing: TILE_SIZE / 32,
  /** Peak hill height in metres. */
  amplitude: 22,
  /** Lowest-octave wavelength in metres. */
  wavelength: 230,
  /** Flat band either side of the road centreline before hills start. */
  roadFlat: LEVEL.roadHalfWidth + 6,
  roadBlend: 46,
  /** Flat plaza radius around each landmark, then a blend out. */
  plazaFlat: 34,
  plazaBlend: 40,
  /** Skirt depth below each border vertex (covers LOD height error). */
  skirt: 4,
  /** LOD steps in samples and the camera distances (m) that switch them. */
  lodSteps: [1, 2, 4] as const,
  lodThresholds: [110, 220] as const,
  /** Cells steeper than this (normal.y) are not walkable for nav. */
  maxWalkNormalY: 0.7,
} as const;

export type TerrainLod = 0 | 1 | 2;

// ── height function ─────────────────────────────────────────────────────

function hash2(ix: number, iz: number): number {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1) ^ 0x5eed7e44;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295; // [0, 1]
}

const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);

/** Smooth value noise in [-1, 1]. */
function valueNoise(x: number, z: number): number {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const u = fade(fx), w = fade(fz);
  const a = hash2(ix, iz), b = hash2(ix + 1, iz), c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
  const top = a + (b - a) * u;
  const bottom = c + (d - c) * u;
  return (top + (bottom - top) * w) * 2 - 1;
}

function fbm(x: number, z: number): number {
  let sum = 0, amp = 0.55, freq = 1, norm = 0;
  for (let o = 0; o < 4; o++) {
    sum += amp * valueNoise(x * freq + o * 17.31, z * freq - o * 9.17);
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return sum / norm;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** Plaza centres: each landmark sits near the road in its tile. */
const PLAZAS: ReadonlyArray<readonly [number, number]> = Object.keys(LANDMARKS).map((k) => {
  const [tx] = k.split("|").map(Number) as [number, number];
  return [LEVEL.origin[0] + (tx + 0.5) * TILE_SIZE, LEVEL.roadZ + 6] as const;
});

/** 0 on the road and plazas, 1 where full hills are allowed. */
export function terrainMask(x: number, z: number): number {
  let m = smoothstep(TERRAIN.roadFlat, TERRAIN.roadFlat + TERRAIN.roadBlend, Math.abs(z - LEVEL.roadZ));
  for (const [px, pz] of PLAZAS) {
    const d = Math.hypot(x - px, z - pz);
    m = Math.min(m, smoothstep(TERRAIN.plazaFlat, TERRAIN.plazaFlat + TERRAIN.plazaBlend, d));
  }
  return m;
}

/** Continuous ground height at any world XZ (the analytic surface). */
export function terrainHeight(x: number, z: number): number {
  const m = terrainMask(x, z);
  if (m <= 0) return 0;
  const n = fbm(x / TERRAIN.wavelength, z / TERRAIN.wavelength);
  // Bias upward so most of the map is hills rather than pits.
  return TERRAIN.amplitude * m * Math.max(-0.25, n * 1.15 + 0.15);
}

// ── tiles ───────────────────────────────────────────────────────────────

export interface TerrainTile {
  tx: number;
  tz: number;
  /** World XZ of the tile's min corner. */
  x0: number;
  z0: number;
  /** (cells+1)² heights, row-major: index = j * (cells + 1) + i, i along x, j along z. */
  samples: Float32Array;
  minHeight: number;
  maxHeight: number;
}

/** Sample the global height function on this tile's grid (shared edges are identical). */
export function buildTerrainTile(tx: number, tz: number): TerrainTile {
  const n = TERRAIN.cells, s = TERRAIN.spacing;
  const x0 = LEVEL.origin[0] + tx * TILE_SIZE;
  const z0 = LEVEL.origin[2] + tz * TILE_SIZE;
  const samples = new Float32Array((n + 1) * (n + 1));
  let minHeight = Infinity, maxHeight = -Infinity;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const h = Math.fround(terrainHeight(x0 + i * s, z0 + j * s));
      samples[j * (n + 1) + i] = h;
      if (h < minHeight) minHeight = h;
      if (h > maxHeight) maxHeight = h;
    }
  }
  return { tx, tz, x0, z0, samples, minHeight, maxHeight };
}

/**
 * Height of the tile's triangulated surface at world (x, z), or null when the
 * point is outside the tile. Matches the mesh, the collider and Rapier.
 */
export function sampleHeight(tile: TerrainTile, x: number, z: number): number | null {
  const tri = locate(tile, x, z);
  if (!tri) return null;
  return tri.h;
}

/** Unit surface normal of the triangle under (x, z), or null outside the tile. */
export function sampleNormal(tile: TerrainTile, x: number, z: number): Vec3 | null {
  const tri = locate(tile, x, z);
  return tri ? tri.n : null;
}

function locate(tile: TerrainTile, x: number, z: number): { h: number; n: Vec3 } | null {
  const hit = sampleHeightfield(terrainCollider(tile).heightfield!, tile.x0, tile.z0, 0, x, z);
  return hit ? { h: hit.height, n: hit.normal } : null;
}

/** Lowest ground under a footprint — props sit here so no edge floats. */
export function groundUnder(minX: number, maxX: number, minZ: number, maxZ: number): number {
  let h = Infinity;
  const steps = 3;
  for (let a = 0; a <= steps; a++) {
    for (let b = 0; b <= steps; b++) {
      h = Math.min(h, terrainHeight(minX + ((maxX - minX) * a) / steps, minZ + ((maxZ - minZ) * b) / steps));
    }
  }
  return h;
}

// ── collider ────────────────────────────────────────────────────────────

/**
 * Heightfield collider for this tile. The owning entity's transform position
 * must be the tile centre at y = 0, so the heights read as world y.
 */
export function terrainCollider(tile: TerrainTile): ColliderComponent {
  return {
    shape: "heightfield",
    heightfield: {
      samplesX: TERRAIN.cells + 1,
      samplesZ: TERRAIN.cells + 1,
      heights: tile.samples,
      sizeX: TILE_SIZE,
      sizeZ: TILE_SIZE,
    },
  };
}

export function terrainTileCenter(tile: TerrainTile): Vec3 {
  return [tile.x0 + TILE_SIZE / 2, 0, tile.z0 + TILE_SIZE / 2];
}

// ── nav ─────────────────────────────────────────────────────────────────

/** Cells too steep to walk, as nav blockers (merged per row to keep the list short). */
export function terrainSlopeBlockers(tile: TerrainTile, navCell: number): NavBlocker[] {
  const out: NavBlocker[] = [];
  const per = Math.round(TILE_SIZE / navCell);
  for (let j = 0; j < per; j++) {
    let runStart = -1;
    for (let i = 0; i <= per; i++) {
      let steep = false;
      if (i < per) {
        const cx = tile.x0 + (i + 0.5) * navCell, cz = tile.z0 + (j + 0.5) * navCell;
        const nrm = sampleNormal(tile, cx, cz);
        steep = !!nrm && nrm[1] < TERRAIN.maxWalkNormalY;
      }
      if (steep && runStart < 0) runStart = i;
      if (!steep && runStart >= 0) {
        out.push({
          minX: tile.x0 + runStart * navCell + 0.01, maxX: tile.x0 + i * navCell - 0.01,
          minZ: tile.z0 + j * navCell + 0.01, maxZ: tile.z0 + (j + 1) * navCell - 0.01,
        });
        runStart = -1;
      }
    }
  }
  return out;
}

// ── mesh ────────────────────────────────────────────────────────────────

/** Mesh key the renderer uploads this tile's LOD under. */
export function terrainMeshKey(tx: number, tz: number, lod: TerrainLod): string {
  return `proc:stream-terrain-${tx}-${tz}:${lod}`;
}

export function terrainGenerator(tx: number, tz: number): string {
  return `stream-terrain-${tx}-${tz}`;
}

/**
 * Per-sample normals for a tile, (cells+1)² × 3, from central differences one
 * sample spacing apart. Interior neighbours come from the tile's own samples;
 * only the ring just outside the tile calls the height function, so a tile
 * costs (cells+1)² + 4·(cells+1) height evaluations instead of 4 per vertex
 * per LOD. Neighbours one spacing apart are exactly the grid samples, so the
 * result matches evaluating the global function (to float rounding) and
 * lighting stays continuous across tile seams.
 */
export function terrainNormals(tile: TerrainTile): Float32Array {
  const n = TERRAIN.cells, s = TERRAIN.spacing;
  const out = new Float32Array((n + 1) * (n + 1) * 3);
  const h = (i: number, j: number): number =>
    i >= 0 && j >= 0 && i <= n && j <= n ? tile.samples[j * (n + 1) + i]! : terrainHeight(tile.x0 + i * s, tile.z0 + j * s);
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const dx = (h(i + 1, j) - h(i - 1, j)) / (2 * s);
      const dz = (h(i, j + 1) - h(i, j - 1)) / (2 * s);
      const len = Math.hypot(dx, 1, dz);
      const o = (j * (n + 1) + i) * 3;
      out[o] = -dx / len; out[o + 1] = 1 / len; out[o + 2] = -dz / len;
    }
  }
  return out;
}

/**
 * Mesh for one LOD of a tile, in tile-local coordinates (origin = tile centre,
 * y absolute). Border vertices get a skirt hanging `TERRAIN.skirt` metres down
 * so neighbouring LODs never open a crack. Normals come from the global height
 * function (via `terrainNormals`, pass them in to share across LODs), so
 * lighting is continuous across tile seams.
 */
export function terrainTileMesh(tile: TerrainTile, lod: TerrainLod, normals: Float32Array = terrainNormals(tile)): MeshData {
  const n = TERRAIN.cells;
  const step = TERRAIN.lodSteps[lod];
  const q = n / step; // quads per edge
  const s = TERRAIN.spacing;
  const half = TILE_SIZE / 2;
  const top = (q + 1) * (q + 1);
  const skirtVerts = 4 * (q + 1);
  const verts = new Float32Array((top + skirtVerts) * 8);
  const indices: number[] = [];
  const heightAt = (gi: number, gj: number) => tile.samples[gj * (n + 1) + gi]!;
  const normalAt = (gi: number, gj: number): Vec3 => {
    const o = (gj * (n + 1) + gi) * 3;
    return [normals[o]!, normals[o + 1]!, normals[o + 2]!];
  };
  let v = 0;
  const put = (lx: number, y: number, lz: number, nrm: Vec3, u: number, w: number) => {
    verts[v++] = lx; verts[v++] = y; verts[v++] = lz;
    verts[v++] = nrm[0]; verts[v++] = nrm[1]; verts[v++] = nrm[2];
    verts[v++] = u; verts[v++] = w;
  };
  for (let j = 0; j <= q; j++) {
    for (let i = 0; i <= q; i++) {
      const gi = i * step, gj = j * step;
      const wx = tile.x0 + gi * s, wz = tile.z0 + gj * s;
      put(wx - tile.x0 - half, heightAt(gi, gj), wz - tile.z0 - half, normalAt(gi, gj), gi / n, gj / n);
    }
  }
  const topIndex = (i: number, j: number) => j * (q + 1) + i;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < q; i++) {
      const a = topIndex(i, j), b = topIndex(i + 1, j), c = topIndex(i, j + 1), d = topIndex(i + 1, j + 1);
      // Diagonal b–c (Rapier's split); both triangles wind counter-clockwise seen from +Y.
      indices.push(a, c, b, b, c, d);
    }
  }
  // Skirts: one strip per edge, each top vertex duplicated `skirt` metres lower.
  const edges: Array<Array<[number, number]>> = [
    Array.from({ length: q + 1 }, (_, i) => [i, 0] as [number, number]), // z-min edge
    Array.from({ length: q + 1 }, (_, i) => [q, i] as [number, number]), // x-max edge
    Array.from({ length: q + 1 }, (_, i) => [q - i, q] as [number, number]), // z-max edge
    Array.from({ length: q + 1 }, (_, i) => [0, q - i] as [number, number]), // x-min edge
  ];
  let base = top;
  for (const edge of edges) {
    for (const [i, j] of edge) {
      const ti = topIndex(i, j) * 8;
      put(verts[ti]!, verts[ti + 1]! - TERRAIN.skirt, verts[ti + 2]!, [verts[ti + 3]!, verts[ti + 4]!, verts[ti + 5]!], verts[ti + 6]!, verts[ti + 7]!);
    }
    for (let k = 0; k < q; k++) {
      const t0 = topIndex(...edge[k]!), t1 = topIndex(...edge[k + 1]!);
      const s0 = base + k, s1 = base + k + 1;
      // A crack can be seen from either tile, and the renderer culls back
      // faces, so each skirt quad is emitted with both windings.
      indices.push(t0, s0, s1, t0, s1, t1, t0, s1, s0, t0, t1, s1);
    }
    base += q + 1;
  }
  const idx = Uint32Array.from(indices);
  return { vertices: verts, indices: idx, vertexCount: top + skirtVerts, indexCount: idx.length };
}

/** The tile's surface as drawn at a given LOD, as a heightfield (every `step`-th sample). */
export function terrainLodHeightfield(tile: TerrainTile, lod: TerrainLod): HeightfieldShape {
  const n = TERRAIN.cells, step = TERRAIN.lodSteps[lod], q = n / step;
  const heights = new Float32Array((q + 1) * (q + 1));
  for (let j = 0; j <= q; j++) for (let i = 0; i <= q; i++) heights[j * (q + 1) + i] = tile.samples[j * step * (n + 1) + i * step]!;
  return { samplesX: q + 1, samplesZ: q + 1, heights, sizeX: TILE_SIZE, sizeZ: TILE_SIZE };
}

/**
 * How far the coarse (LOD2) surface rises above the lowest drawn surface of
 * any LOD, over the whole tile. Lowering LOD2 occluder triangles by this much
 * keeps them on or under whatever is actually drawn, so a hill never hides
 * something that would be visible over the real ground. All LOD triangles
 * split along the same anti-diagonals, so the difference is piecewise linear
 * on the fine grid and its maximum lies on a fine sample.
 */
export function terrainOccluderDrop(tile: TerrainTile): number {
  const n = TERRAIN.cells, s = TERRAIN.spacing;
  const l1 = terrainLodHeightfield(tile, 1), l2 = terrainLodHeightfield(tile, 2);
  let drop = 0;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const x = tile.x0 + i * s, z = tile.z0 + j * s;
      const fine = tile.samples[j * (n + 1) + i]!;
      const mid = sampleHeightfield(l1, tile.x0, tile.z0, 0, x, z)!.height;
      const coarse = sampleHeightfield(l2, tile.x0, tile.z0, 0, x, z)!.height;
      drop = Math.max(drop, coarse - Math.min(fine, mid));
    }
  }
  return drop + 1e-3;
}

/** Triangle count of one LOD, skirts included (for stats). */
export function terrainLodTriangles(lod: TerrainLod): number {
  const q = TERRAIN.cells / TERRAIN.lodSteps[lod];
  return q * q * 2 + 4 * q * 4;
}

/** Bytes a tile's terrain holds resident (samples + three mesh LODs). */
export function terrainTileBytes(): number {
  const n = TERRAIN.cells;
  let bytes = (n + 1) * (n + 1) * 4;
  for (const step of TERRAIN.lodSteps) {
    const q = n / step;
    bytes += ((q + 1) * (q + 1) + 4 * (q + 1)) * 32 + (q * q * 6 + 4 * q * 12) * 4;
  }
  return bytes;
}

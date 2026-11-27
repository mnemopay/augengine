/**
 * world/lod — Level-of-detail selection.
 *
 * Track 7 (2026-05-29). Four LOD levels per asset:
 *   LOD0 = full triangle count (close-up)
 *   LOD1 = ~1/4 tris
 *   LOD2 = ~1/16 tris
 *   LOD3 = imposter billboard (a single quad with a baked texture)
 *
 * Selection blends two signals:
 *   (a) Camera distance vs `distanceThresholds` (default tuned for 64m tiles
 *       at 1km extent — covers human-scale POV up to a "flying over" cam).
 *   (b) Screen-space error: if the assumed projected size in pixels falls
 *       below `pixelErrorThreshold` we kick to the next level regardless of
 *       distance. Cheap proxy: `(radius / dist) * focalPx`. No geometry
 *       sampling required.
 *
 * The renderer-side resolution of which mesh handle each LOD level points
 * to lives in the tile manager; this module is pure math + chain config so
 * it stays unit-testable in plain Node.
 *
 * Pre-bakability — `tools/augengine-asset-pack/pack.mjs --emit-lods` writes
 * `<asset>.lod1.glb`, `<asset>.lod2.glb`, `<asset>.lod3.glb` siblings via
 * gltfpack `-si` simplify ratios. At load time, the tile manager fetches
 * whichever LOD the camera distance currently demands.
 */

import type { Vec3 } from "../../scene/types.js";

/** 0-3 inclusive — 0 is highest detail. */
export type LodLevel = 0 | 1 | 2 | 3;

export interface LodChainConfig {
  /**
   * Camera distance (world units) at which each LOD step kicks in.
   * `distanceThresholds[i]` = distance at which LOD `i+1` becomes active.
   * Length 3 — for a 4-LOD chain. Defaults tuned for tiles ~64 m wide.
   */
  distanceThresholds: readonly [number, number, number];
  /**
   * Below this projected pixel size we promote to the next LOD level even
   * if the distance threshold is not crossed (small objects far away).
   * Default 40 px (eye can't resolve below that on a typical 1080p HUD).
   */
  pixelErrorThreshold: number;
  /**
   * Focal length in pixels. Computed from camera FOV + screen height:
   * `focalPx = 0.5 * screenHeight / tan(fovY / 2)`. Caller updates this
   * once per resize; the selector reads it per call.
   */
  focalPx: number;
}

export const DEFAULT_LOD_CHAIN: LodChainConfig = {
  // 32 m → LOD1, 96 m → LOD2, 256 m → LOD3 (imposter).
  distanceThresholds: [32, 96, 256],
  pixelErrorThreshold: 40,
  // 540 px focal ≈ 60° FOV at 1080p.
  focalPx: 540,
};

/**
 * Pick a LOD for an entity given the camera position and a bounding sphere
 * around the entity. Pure function — no allocation, safe in a hot loop.
 *
 * `bounds` is `{ centerWorld, radius }`. The renderer caches these on the
 * tile + entity registration so we avoid recomputing per frame.
 */
export function selectLod(
  cameraPos: Vec3,
  centerWorld: Vec3,
  radius: number,
  cfg: LodChainConfig = DEFAULT_LOD_CHAIN,
): LodLevel {
  const dx = cameraPos[0] - centerWorld[0];
  const dy = cameraPos[1] - centerWorld[1];
  const dz = cameraPos[2] - centerWorld[2];
  const dist = Math.hypot(dx, dy, dz);
  // Distance gate — coarsest categorical bin.
  let level: LodLevel = 0;
  if (dist > cfg.distanceThresholds[0]) level = 1;
  if (dist > cfg.distanceThresholds[1]) level = 2;
  if (dist > cfg.distanceThresholds[2]) level = 3;
  // Screen-space override — small things far away get bumped sooner.
  // `radius / dist * focalPx` ≈ projected radius in pixels (small-angle).
  if (dist > 0.0001) {
    const projectedPx = (radius / dist) * cfg.focalPx;
    if (projectedPx < cfg.pixelErrorThreshold && level < 3) {
      level = Math.min(3, level + 1) as LodLevel;
    }
    if (projectedPx < cfg.pixelErrorThreshold * 0.25 && level < 3) {
      level = Math.min(3, level + 1) as LodLevel;
    }
  }
  return level;
}

/**
 * Distance-only LOD with hysteresis, for props that switch every frame as
 * the player walks. A level steps coarser only once `dist` passes the
 * threshold by `band` (fraction), and steps finer only once it is `band`
 * inside it, so an object parked on a boundary does not flicker between
 * meshes. `prev` is the level this object drew last frame (or null).
 */
export function selectLodHysteresis(
  prev: LodLevel | null,
  dist: number,
  thresholds: readonly [number, number, number],
  band = 0.12,
): LodLevel {
  let raw: LodLevel = 0;
  if (dist > thresholds[0]) raw = 1;
  if (dist > thresholds[1]) raw = 2;
  if (dist > thresholds[2]) raw = 3;
  if (prev === null || raw === prev) return raw;
  let level: number = prev;
  // Step one level at a time so a big jump still stops at the right level.
  while (level < raw && dist > thresholds[level]! * (1 + band)) level += 1;
  while (level > raw && dist < thresholds[level - 1]! * (1 - band)) level -= 1;
  return level as LodLevel;
}

/**
 * LOD index suffix used in mesh-key resolution. `mesh:foo` → `mesh:foo`,
 * `mesh:foo` + LOD2 → `mesh:foo#lod2`. Wired through to the renderer's
 * mesh map so each LOD level draws against its own uploaded buffer.
 */
export function lodMeshKey(baseKey: string, lod: LodLevel): string {
  return lod === 0 ? baseKey : `${baseKey}#lod${lod}`;
}

/**
 * Estimate a per-LOD vertex count given the LOD0 baseline. Used by the
 * memory accountant in the tile manager (we don't fetch LOD bytes until
 * load time — this estimate lets us pre-budget residency).
 *
 * Conventions match gltfpack `-si 0.25` / `-si 0.0625` / billboard
 * (8 vertices) outputs.
 */
export function estimateLodVertexCount(lod0Vertices: number, lod: LodLevel): number {
  switch (lod) {
    case 0: return lod0Vertices;
    case 1: return Math.max(64, Math.round(lod0Vertices * 0.25));
    case 2: return Math.max(32, Math.round(lod0Vertices * 0.0625));
    case 3: return 8; // billboard quad — 2 triangles, 4 verts duplicated for normals
  }
}

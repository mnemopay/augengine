/**
 * Heightfield surface queries shared by the terrain, the character controller
 * and the engine raycast, so every consumer reads the same triangles Rapier
 * builds: each cell split along its (x1,z0)–(x0,z1) diagonal.
 */
import type { HeightfieldShape, Vec3 } from "../scene/types.js";

export interface HeightfieldHit {
  /** Surface height (world y) at the query point. */
  height: number;
  /** Unit normal of the triangle under the point. */
  normal: Vec3;
}

/**
 * Surface under world (x, z) for a heightfield whose grid's min corner sits
 * at (minX, minZ) and whose heights are offsets from `baseY`. Null outside.
 */
export function sampleHeightfield(
  hf: HeightfieldShape,
  minX: number,
  minZ: number,
  baseY: number,
  x: number,
  z: number,
): HeightfieldHit | null {
  const cx = hf.samplesX - 1, cz = hf.samplesZ - 1;
  const sx = hf.sizeX / cx, sz = hf.sizeZ / cz;
  const lx = (x - minX) / sx, lz = (z - minZ) / sz;
  if (!(lx >= 0 && lz >= 0 && lx <= cx && lz <= cz)) return null;
  const i = Math.min(cx - 1, Math.floor(lx));
  const j = Math.min(cz - 1, Math.floor(lz));
  const u = lx - i, w = lz - j;
  const at = (ii: number, jj: number) => hf.heights[jj * hf.samplesX + ii]!;
  const h00 = at(i, j), h10 = at(i + 1, j), h01 = at(i, j + 1), h11 = at(i + 1, j + 1);
  let h: number, dhdx: number, dhdz: number;
  if (u + w <= 1) {
    // Triangle (p00, p01, p10).
    dhdx = (h10 - h00) / sx;
    dhdz = (h01 - h00) / sz;
    h = h00 + (h10 - h00) * u + (h01 - h00) * w;
  } else {
    // Triangle (p10, p01, p11).
    dhdx = (h11 - h01) / sx;
    dhdz = (h11 - h10) / sz;
    h = h11 + (h01 - h11) * (1 - u) + (h10 - h11) * (1 - w);
  }
  const len = Math.hypot(dhdx, 1, dhdz);
  return { height: baseY + h, normal: [-dhdx / len, 1 / len, -dhdz / len] };
}

/** Same query, for a heightfield collider whose entity sits at `position` (grid centre). */
export function sampleHeightfieldAt(hf: HeightfieldShape, position: Vec3, x: number, z: number): HeightfieldHit | null {
  return sampleHeightfield(hf, position[0] - hf.sizeX / 2, position[2] - hf.sizeZ / 2, position[1], x, z);
}

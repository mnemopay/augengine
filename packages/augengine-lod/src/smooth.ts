/**
 * smooth — vertex compaction and area-weighted smooth normals, shared by the
 * reader (sources without normals) and the baker (smooth chains).
 */
import type { KitMesh } from "./mesh.js";

/** Drop vertices no triangle uses (simplification leaves them behind) and renumber. */
export function compactVertices(m: KitMesh): KitMesh {
  const count = m.positions.length / 3;
  const remap = new Int32Array(count).fill(-1);
  let next = 0;
  for (const i of m.indices) if (remap[i] === -1) remap[i] = next++;
  if (next === count) return m;
  const P = new Float32Array(next * 3), N = new Float32Array(next * 3), UV = new Float32Array(next * 2), C = new Float32Array(next * 4);
  for (let v = 0; v < count; v++) {
    const r = remap[v]!;
    if (r < 0) continue;
    P.set(m.positions.subarray(v * 3, v * 3 + 3), r * 3);
    N.set(m.normals.subarray(v * 3, v * 3 + 3), r * 3);
    UV.set(m.uvs.subarray(v * 2, v * 2 + 2), r * 2);
    C.set(m.colors.subarray(v * 4, v * 4 + 4), r * 4);
  }
  // Index order is unchanged, so material groups carry over.
  return { ...m, positions: P, normals: N, uvs: UV, colors: C, indices: m.indices.map((i) => remap[i]!) };
}

/**
 * Area-weighted vertex normals. Vertices that share a position (UV or colour
 * seams keep them as separate vertices) share one normal, so seams don't
 * show as a shading crease.
 */
export function withSmoothNormals(m: KitMesh): KitMesh {
  const p = m.positions;
  const count = p.length / 3;
  const keyOf = (v: number) => `${Math.round(p[v * 3]! * 1e4)},${Math.round(p[v * 3 + 1]! * 1e4)},${Math.round(p[v * 3 + 2]! * 1e4)}`;
  const slot = new Uint32Array(count);
  const slots = new Map<string, number>();
  for (let v = 0; v < count; v++) {
    const k = keyOf(v);
    let s2 = slots.get(k);
    if (s2 === undefined) { s2 = slots.size; slots.set(k, s2); }
    slot[v] = s2;
  }
  const acc = new Float64Array(slots.size * 3);
  for (let t = 0; t + 2 < m.indices.length; t += 3) {
    const ia = m.indices[t]!, ib = m.indices[t + 1]!, ic = m.indices[t + 2]!;
    const a = ia * 3, b = ib * 3, c = ic * 3;
    const ux = p[b]! - p[a]!, uy = p[b + 1]! - p[a + 1]!, uz = p[b + 2]! - p[a + 2]!;
    const vx = p[c]! - p[a]!, vy = p[c + 1]! - p[a + 1]!, vz = p[c + 2]! - p[a + 2]!;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const i of [ia, ib, ic]) {
      const o = slot[i]! * 3;
      acc[o] = acc[o]! + nx; acc[o + 1] = acc[o + 1]! + ny; acc[o + 2] = acc[o + 2]! + nz;
    }
  }
  const n = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) {
    const o = slot[v]! * 3;
    const x = acc[o]!, y = acc[o + 1]!, z = acc[o + 2]!;
    const l = Math.hypot(x, y, z);
    if (l > 1e-20) { n[v * 3] = x / l; n[v * 3 + 1] = y / l; n[v * 3 + 2] = z / l; } else n[v * 3 + 1] = 1;
  }
  return { ...m, normals: n };
}

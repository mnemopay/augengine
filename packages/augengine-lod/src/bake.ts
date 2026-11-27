/**
 * bake — LOD0..LOD3 for one mesh.
 *
 *   LOD0  the source, flat-shaded.
 *   LOD1  meshoptimizer `simplify` to ~1/4 of the triangles, error-bounded.
 *   LOD2  ~1/16, with `Prune` so small disconnected parts (window panes,
 *         sills, rooftop plant) can disappear instead of blocking the target.
 *   LOD3  impostor tinted with the mesh's area-weighted average colour:
 *         "cross" = two crossed, two-sided quads (foliage, poles); "box" =
 *         the bounds box (buildings and other solid shapes, which read as
 *         thin slabs from the side as crossed cards). No baked texture yet
 *         (the renderer has no offscreen capture path for it).
 *
 * Simplification works on a welded copy: vertices are merged when position
 * and colour (and UV, when `preserveUvSeams`) match, so meshoptimizer sees
 * connected surfaces. Where the weld keeps vertices apart (different colour
 * or UV), meshoptimizer treats the shared position as an attribute seam and
 * keeps the seam intact. Output is re-emitted flat-shaded, which recomputes
 * every normal from the simplified triangles.
 */
import { MeshoptSimplifier, type SimplifierFlags } from "meshoptimizer";
import { MeshBuilder } from "./builder.js";
import {
  averageColor,
  bounds,
  boundingRadius,
  flatShade,
  removeDegenerates,
  triangleCount,
  type Aabb,
  type KitMesh,
  type Rgba,
} from "./mesh.js";

export interface BakeOptions {
  /** Target triangle fractions for LOD1 and LOD2 (of LOD0). */
  ratios?: [number, number];
  /** Max simplification error as a fraction of the mesh's bounding radius, for LOD1 and LOD2. */
  maxError?: [number, number];
  /** Keep UV seams (weld by UV too). Turn on for textured input. */
  preserveUvSeams?: boolean;
  /** Never move open-border vertices (for tiles that must meet neighbours). */
  lockBorder?: boolean;
  /**
   * LOD3 impostor: "cross" (two crossed quads — foliage, poles), "box" (the
   * bounds box — solid, boxy things; reads right from every angle), or false.
   * Default "cross"; `true` means "cross".
   */
  impostor?: boolean | "cross" | "box";
}

export interface LodLevelOut {
  level: 0 | 1 | 2 | 3;
  mesh: KitMesh;
  triangles: number;
  /** Relative error reported by meshoptimizer (0 for LOD0 and the impostor). */
  error: number;
  kind: "source" | "simplified" | "impostor";
}

export interface BakeResult {
  levels: LodLevelOut[];
  bounds: Aabb;
  radius: number;
}

export async function bakeLods(source: KitMesh, opts: BakeOptions = {}): Promise<BakeResult> {
  await MeshoptSimplifier.ready;
  const ratios = opts.ratios ?? [0.25, 0.0625];
  const maxError = opts.maxError ?? [0.02, 0.08];
  const lod0 = flatShade(removeDegenerates(source));
  const b = bounds(lod0);
  const radius = boundingRadius(b);
  const levels: LodLevelOut[] = [{ level: 0, mesh: lod0, triangles: triangleCount(lod0), error: 0, kind: "source" }];

  const welded = weld(lod0, opts.preserveUvSeams ?? false);
  const baseTris = triangleCount(lod0);
  for (const step of [0, 1] as const) {
    const target = Math.max(3, Math.floor(baseTris * ratios[step]) * 3);
    const flags: SimplifierFlags[] = [];
    if (opts.lockBorder) flags.push("LockBorder");
    if (step === 1) flags.push("Prune");
    const [idx, err] = MeshoptSimplifier.simplify(welded.indices, welded.positions, 3, target, maxError[step], flags);
    let mesh = flatShade(removeDegenerates({ ...welded, indices: idx }));
    // Never let a coarser level end up with more triangles than the one before it.
    const prev = levels[levels.length - 1]!;
    if (triangleCount(mesh) >= prev.triangles) mesh = prev.mesh;
    levels.push({ level: (step + 1) as 1 | 2, mesh, triangles: triangleCount(mesh), error: err, kind: "simplified" });
  }
  if (opts.impostor !== false) {
    const color = averageColor(lod0);
    const imp = opts.impostor === "box" ? boxImpostor(b, color) : impostor(b, color);
    levels.push({ level: 3, mesh: imp, triangles: triangleCount(imp), error: 0, kind: "impostor" });
  }
  return { levels, bounds: b, radius };
}

/**
 * Merge vertices that share position + colour (+ UV). Normals are dropped
 * on purpose: they are recomputed after simplification.
 */
export function weld(m: KitMesh, byUv: boolean): KitMesh {
  const map = new Map<string, number>();
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [];
  const remap = new Uint32Array(m.positions.length / 3);
  const q = (x: number, s: number) => Math.round(x * s);
  for (let v = 0; v < remap.length; v++) {
    const key = [
      q(m.positions[v * 3]!, 1e4), q(m.positions[v * 3 + 1]!, 1e4), q(m.positions[v * 3 + 2]!, 1e4),
      q(m.colors[v * 4]!, 1e3), q(m.colors[v * 4 + 1]!, 1e3), q(m.colors[v * 4 + 2]!, 1e3), q(m.colors[v * 4 + 3]!, 1e3),
      ...(byUv ? [q(m.uvs[v * 2]!, 1e4), q(m.uvs[v * 2 + 1]!, 1e4)] : []),
    ].join(",");
    let id = map.get(key);
    if (id === undefined) {
      id = P.length / 3;
      map.set(key, id);
      P.push(m.positions[v * 3]!, m.positions[v * 3 + 1]!, m.positions[v * 3 + 2]!);
      N.push(0, 1, 0);
      UV.push(m.uvs[v * 2]!, m.uvs[v * 2 + 1]!);
      C.push(m.colors[v * 4]!, m.colors[v * 4 + 1]!, m.colors[v * 4 + 2]!, m.colors[v * 4 + 3]!);
    }
    remap[v] = id;
  }
  return {
    positions: Float32Array.from(P),
    normals: Float32Array.from(N),
    uvs: Float32Array.from(UV),
    colors: Float32Array.from(C),
    indices: m.indices.map((i) => remap[i]!),
  };
}

/** The bounds box (bottom face omitted), one colour, flat normals: 10 triangles. */
export function boxImpostor(b: Aabb, color: Rgba): KitMesh {
  return new MeshBuilder().box(b.min, b.max, color, { ny: false }).build();
}

/** Two crossed vertical quads through the bounds centre, both sides, one colour. */
export function impostor(b: Aabb, color: Rgba): KitMesh {
  const cx = (b.min[0] + b.max[0]) / 2, cz = (b.min[2] + b.max[2]) / 2;
  const y0 = b.min[1], y1 = b.max[1];
  const hx = (b.max[0] - b.min[0]) / 2, hz = (b.max[2] - b.min[2]) / 2;
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [];
  const quad = (a: number[], bb: number[], c: number[], d: number[], n: number[]) => {
    const base = P.length / 3;
    for (const [k, v] of [a, bb, c, d].entries()) {
      P.push(...v);
      N.push(...n);
      UV.push(k === 1 || k === 2 ? 1 : 0, k >= 2 ? 1 : 0);
      C.push(...color);
    }
    I.push(base, base + 1, base + 2, base, base + 2, base + 3);
  };
  // X-aligned plane (normal ±z) and Z-aligned plane (normal ±x), each double-sided.
  quad([cx - hx, y0, cz], [cx + hx, y0, cz], [cx + hx, y1, cz], [cx - hx, y1, cz], [0, 0, 1]);
  quad([cx + hx, y0, cz], [cx - hx, y0, cz], [cx - hx, y1, cz], [cx + hx, y1, cz], [0, 0, -1]);
  quad([cx, y0, cz + hz], [cx, y0, cz - hz], [cx, y1, cz - hz], [cx, y1, cz + hz], [1, 0, 0]);
  quad([cx, y0, cz - hz], [cx, y0, cz + hz], [cx, y1, cz + hz], [cx, y1, cz - hz], [-1, 0, 0]);
  return {
    positions: Float32Array.from(P),
    normals: Float32Array.from(N),
    uvs: Float32Array.from(UV),
    colors: Float32Array.from(C),
    indices: Uint32Array.from(I),
  };
}

export interface SmoothSimplifyResult {
  mesh: KitMesh;
  triangles: number;
  /** Relative error reported by meshoptimizer. */
  error: number;
}

/**
 * Simplify an organic mesh (characters, props) to about `targetTriangles`
 * while keeping it smooth: unlike `bakeLods` (flat-shaded, for kit
 * architecture) normals are recomputed area-weighted per welded vertex, and
 * UV seams are preserved so textures still line up. Returns the input
 * unchanged when it is already under the target.
 */
export async function simplifySmooth(source: KitMesh, targetTriangles: number, maxError = 0.01): Promise<SmoothSimplifyResult> {
  await MeshoptSimplifier.ready;
  const clean = removeDegenerates(source);
  const tris = triangleCount(clean);
  if (tris <= targetTriangles) return { mesh: withSmoothNormals(compactVertices(weld(clean, true))), triangles: tris, error: 0 };
  const welded = weld(clean, true);
  const [idx, err] = MeshoptSimplifier.simplify(welded.indices, welded.positions, 3, Math.max(3, Math.floor(targetTriangles) * 3), maxError, []);
  const mesh = withSmoothNormals(compactVertices(removeDegenerates({ ...welded, indices: idx })));
  return { mesh, triangles: triangleCount(mesh), error: err };
}

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
  return { positions: P, normals: N, uvs: UV, colors: C, indices: m.indices.map((i) => remap[i]!) };
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

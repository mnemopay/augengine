/**
 * bake — LOD0..LOD3 for one mesh.
 *
 *   LOD0  the source (degenerate triangles dropped), normals as authored.
 *   LOD1  meshoptimizer attribute-aware simplification to ~1/4 of the
 *         triangles, error-bounded.
 *   LOD2  ~1/16, with `Prune` so small disconnected parts (window panes,
 *         sills, rooftop plant) can disappear instead of blocking the target.
 *   LOD3  an impostor: "box" (the bounds box — buildings and other solid
 *         shapes) or "cross" (two crossed cards — foliage, poles), showing
 *         the asset's real colours (see impostor.ts: a baked atlas, per-face
 *         vertex colours, or one flat tint).
 *
 * Materials. Triangles keep their source material (`KitMesh.groups`); every
 * LOD references the same materials, so textures are never duplicated.
 *
 * Shading follows the source (`shading: "auto"`):
 *   flat   (faceted source — the kit, low-poly): simplification output is
 *          re-emitted flat-shaded, recomputing face normals.
 *   smooth (shared normals — scans, characters, organic props): the
 *          authored normals of the surviving vertices are kept; nothing is
 *          recomputed (the reader computes normals only when the source has none).
 *
 * Simplification runs on a welded copy: vertices merge when position,
 * material, colour, UV (textured / `preserveUvSeams`) and — for smooth
 * chains — normal all match, so meshoptimizer sees connected surfaces. Where
 * the weld keeps vertices apart (a UV, material, colour or hard-normal seam)
 * meshoptimizer keeps the seam closed; `uvSeams: "lock"` (the default for
 * textured input) also pins every seam vertex so texture islands cannot
 * slide. UVs, normals and colours are passed as weighted attributes
 * (`simplifyWithAttributes`), so collapses that would smear a texture or bend
 * shading cost error like geometric ones do.
 */
import { MeshoptSimplifier, type SimplifierFlags } from "meshoptimizer";
import { MeshBuilder } from "./builder.js";
import {
  averageColor,
  bounds,
  boundingRadius,
  flatShade,
  isFlatShaded,
  regroup,
  removeDegenerates,
  triangleCount,
  triangleMaterials,
  type Aabb,
  type KitMesh,
  type Rgba,
} from "./mesh.js";
import { compactVertices, withSmoothNormals } from "./smooth.js";
import { buildImpostor, type ImpostorKind, type ImpostorOut, type ImpostorTexture } from "./impostor.js";
import { hasAnyTexture, type LodImage, type LodMaterialSet } from "./material.js";
import type { RgbaImage } from "./png.js";

export { compactVertices, withSmoothNormals } from "./smooth.js";

export interface BakeOptions {
  /** Target triangle fractions for LOD1 and LOD2 (of LOD0). */
  ratios?: [number, number];
  /** Max simplification error as a fraction of the mesh's bounding radius, for LOD1 and LOD2. */
  maxError?: [number, number];
  /** Keep UV seams (weld by UV too). Implied when `materials` has textures. */
  preserveUvSeams?: boolean;
  /**
   * UV/material seams: "lock" pins seam vertices (no texture swimming, less
   * reduction on heavily chopped atlases); "preserve" lets meshoptimizer
   * collapse along a seam while keeping it closed. Default "lock" when
   * textured, else "preserve".
   */
  uvSeams?: "lock" | "preserve";
  /** Never move open-border vertices (for tiles that must meet neighbours). */
  lockBorder?: boolean;
  /**
   * LOD3 impostor: "cross" (two crossed quads — foliage, poles), "box" (the
   * bounds box — solid, boxy things; reads right from every angle), or false.
   * Default "cross"; `true` means "cross".
   */
  impostor?: boolean | ImpostorKind;
  /**
   * How the impostor shows colour: "atlas" (baked texture from LOD0),
   * "vertex" (per-face colour from the same render, untextured), "none"
   * (one average tint, the 0.1 behaviour). Default "vertex" here; the file
   * bake (`bakeGlbFile` / `writeBaked` / the CLI) defaults to "atlas".
   */
  impostorTexture?: ImpostorTexture;
  /** Atlas pixels per impostor view. Default 64. */
  atlasTile?: number;
  /** Shading of the chain: "auto" (from the source's normals), "flat" or "smooth". */
  shading?: "auto" | "flat" | "smooth";
  /** The source's materials: textured materials switch on UV-aware simplification and feed the impostor. */
  materials?: LodMaterialSet;
  /** Decode an image for the impostor render (the Node entry supplies a PNG decoder). */
  decodeImage?: (img: LodImage) => RgbaImage | null;
  /** Attribute weights for simplification. Defaults: uv 1 (textured), normal 0.5 (smooth), color 1 (varying colours). */
  weights?: { uv?: number; normal?: number; color?: number };
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
  /** Shading the chain was baked with. */
  shading: "flat" | "smooth";
  /** The impostor's material, atlas and layout (absent with `impostor: false`). */
  impostor?: ImpostorOut & { kind: ImpostorKind };
  warnings: string[];
}

export async function bakeLods(source: KitMesh, opts: BakeOptions = {}): Promise<BakeResult> {
  await MeshoptSimplifier.ready;
  const ratios = opts.ratios ?? [0.25, 0.0625];
  const maxError = opts.maxError ?? [0.02, 0.08];
  const warnings: string[] = [];
  const textured = !!opts.materials && hasAnyTexture(opts.materials);
  const clean = removeDegenerates(source);
  const sourceFlat = isFlatShaded(clean);
  const shading: "flat" | "smooth" = opts.shading === "flat" || opts.shading === "smooth" ? opts.shading : sourceFlat ? "flat" : "smooth";
  // LOD0 keeps the authored normals unless a shading override disagrees with them.
  const lod0 = shading === "flat"
    ? (sourceFlat ? clean : flatShade(clean))
    : (sourceFlat ? withSmoothNormals(compactVertices(weld(clean, true))) : clean);
  const b = bounds(lod0);
  const radius = boundingRadius(b);
  const levels: LodLevelOut[] = [{ level: 0, mesh: lod0, triangles: triangleCount(lod0), error: 0, kind: "source" }];

  const byUv = textured || (opts.preserveUvSeams ?? false);
  const welded = weldFull(lod0, { byUv, byNormal: shading === "smooth" });
  const colorsVary = varies(welded.mesh.colors, 4);
  const w = {
    uv: byUv ? (opts.weights?.uv ?? 1) : 0,
    normal: shading === "smooth" ? (opts.weights?.normal ?? 0.5) : 0,
    color: colorsVary ? (opts.weights?.color ?? 1) : 0,
  };
  const attr = attributeStream(welded.mesh, w);
  const lockSeams = (opts.uvSeams ?? (textured ? "lock" : "preserve")) === "lock";
  const lock = lockSeams ? seamLocks(welded.mesh) : null;
  const baseTris = triangleCount(lod0);
  for (const step of [0, 1] as const) {
    const target = Math.max(3, Math.floor(baseTris * ratios[step]) * 3);
    const flags: SimplifierFlags[] = [];
    if (opts.lockBorder) flags.push("LockBorder");
    if (step === 1) flags.push("Prune");
    const [idx, err] = attr.stride > 0 || lock
      ? MeshoptSimplifier.simplifyWithAttributes(welded.mesh.indices, welded.mesh.positions, 3, attr.data, attr.stride, attr.weights, lock, target, maxError[step], flags)
      : MeshoptSimplifier.simplify(welded.mesh.indices, welded.mesh.positions, 3, target, maxError[step], flags);
    // Triangles keep one material each (collapses only follow same-material edges): group by the first corner's.
    const triMat = new Uint32Array(idx.length / 3);
    for (let t = 0; t < triMat.length; t++) triMat[t] = welded.vertexMaterial[idx[t * 3]!]!;
    const grouped = { ...welded.mesh, ...regroup(idx, triMat) };
    let mesh = shading === "flat" ? flatShade(removeDegenerates(grouped)) : compactVertices(removeDegenerates(grouped));
    // Never let a coarser level end up with more triangles than the one before it.
    const prev = levels[levels.length - 1]!;
    if (triangleCount(mesh) >= prev.triangles) mesh = prev.mesh;
    levels.push({ level: (step + 1) as 1 | 2, mesh, triangles: triangleCount(mesh), error: err, kind: "simplified" });
  }
  let impostorOut: BakeResult["impostor"];
  if (opts.impostor !== false) {
    const kind: ImpostorKind = opts.impostor === "box" ? "box" : "cross";
    const out = buildImpostor(lod0, b, {
      kind,
      texture: opts.impostorTexture ?? "vertex",
      ...(opts.materials ? { materials: opts.materials } : {}),
      ...(opts.decodeImage ? { decodeImage: opts.decodeImage } : {}),
      ...(opts.atlasTile ? { tile: opts.atlasTile } : {}),
    });
    warnings.push(...out.warnings);
    impostorOut = { ...out, kind };
    levels.push({ level: 3, mesh: out.mesh, triangles: triangleCount(out.mesh), error: 0, kind: "impostor" });
  }
  return { levels, bounds: b, radius, shading, ...(impostorOut ? { impostor: impostorOut } : {}), warnings };
}

/**
 * Merge vertices that share position + colour (+ UV). Normals are dropped
 * on purpose: they are recomputed after simplification. Material groups are kept.
 */
export function weld(m: KitMesh, byUv: boolean): KitMesh {
  const out = weldFull(m, { byUv, byNormal: false }).mesh;
  return { ...out, normals: new Float32Array(out.normals.length).map((_, i) => (i % 3 === 1 ? 1 : 0)) };
}

/**
 * Weld for simplification. Keys: position, material, colour, and optionally
 * UV and normal. Returns the welded mesh (normals of the first vertex in each
 * class) and each welded vertex's material.
 */
function weldFull(m: KitMesh, o: { byUv: boolean; byNormal: boolean }): { mesh: KitMesh; vertexMaterial: Uint32Array } {
  const triMat = triangleMaterials(m);
  // A vertex used by two materials is split per material (the key includes it).
  const map = new Map<string, number>();
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], VM: number[] = [];
  const q = (x: number, s: number) => Math.round(x * s);
  const indices = new Uint32Array(m.indices.length);
  // Normals are clustered per position with a tolerance (not quantized), so
  // near-identical normals never split on a rounding boundary while real
  // hard edges (> ~2.5 deg) stay apart.
  const clusters = new Map<string, number[]>();
  const normalCluster = (v: number, posKey: string): number => {
    const list = clusters.get(posKey) ?? [];
    const nx = m.normals[v * 3]!, ny = m.normals[v * 3 + 1]!, nz = m.normals[v * 3 + 2]!;
    for (let c = 0; c < list.length; c += 3) if (list[c]! * nx + list[c + 1]! * ny + list[c + 2]! * nz >= 0.999) return c / 3;
    list.push(nx, ny, nz);
    clusters.set(posKey, list);
    return list.length / 3 - 1;
  };
  for (let t = 0; t < m.indices.length; t++) {
    const v = m.indices[t]!;
    const mat = triMat[Math.floor(t / 3)]!;
    const posKey = `${q(m.positions[v * 3]!, 1e4)},${q(m.positions[v * 3 + 1]!, 1e4)},${q(m.positions[v * 3 + 2]!, 1e4)}`;
    const key = [
      mat,
      q(m.positions[v * 3]!, 1e4), q(m.positions[v * 3 + 1]!, 1e4), q(m.positions[v * 3 + 2]!, 1e4),
      q(m.colors[v * 4]!, 1e3), q(m.colors[v * 4 + 1]!, 1e3), q(m.colors[v * 4 + 2]!, 1e3), q(m.colors[v * 4 + 3]!, 1e3),
      ...(o.byUv ? [q(m.uvs[v * 2]!, 1e4), q(m.uvs[v * 2 + 1]!, 1e4)] : []),
      ...(o.byNormal ? [normalCluster(v, posKey)] : []),
    ].join(",");
    let id = map.get(key);
    if (id === undefined) {
      id = P.length / 3;
      map.set(key, id);
      P.push(m.positions[v * 3]!, m.positions[v * 3 + 1]!, m.positions[v * 3 + 2]!);
      N.push(m.normals[v * 3]!, m.normals[v * 3 + 1]!, m.normals[v * 3 + 2]!);
      UV.push(m.uvs[v * 2]!, m.uvs[v * 2 + 1]!);
      C.push(m.colors[v * 4]!, m.colors[v * 4 + 1]!, m.colors[v * 4 + 2]!, m.colors[v * 4 + 3]!);
      VM.push(mat);
    }
    indices[t] = id;
  }
  return {
    mesh: {
      positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV), colors: Float32Array.from(C), indices,
      ...(m.groups ? { groups: m.groups.map((g) => ({ ...g })) } : {}),
    },
    vertexMaterial: Uint32Array.from(VM),
  };
}

function varies(a: Float32Array, stride: number): boolean {
  for (let i = stride; i < a.length; i++) if (Math.abs(a[i]! - a[i % stride]!) > 1e-4) return true;
  return false;
}

/** Interleave the weighted attributes meshoptimizer should respect (zero-weight ones are left out). */
function attributeStream(m: KitMesh, w: { uv: number; normal: number; color: number }): { data: Float32Array; stride: number; weights: number[] } {
  const parts: { src: Float32Array; n: number; weight: number }[] = [];
  if (w.uv > 0) parts.push({ src: m.uvs, n: 2, weight: w.uv });
  if (w.normal > 0) parts.push({ src: m.normals, n: 3, weight: w.normal });
  if (w.color > 0) parts.push({ src: m.colors, n: 4, weight: w.color });
  const stride = parts.reduce((s, p) => s + p.n, 0);
  const count = m.positions.length / 3;
  const data = new Float32Array(count * stride);
  const weights: number[] = [];
  let o = 0;
  for (const p of parts) {
    for (let v = 0; v < count; v++) for (let k = 0; k < p.n; k++) data[v * stride + o + k] = p.src[v * p.n + k]!;
    for (let k = 0; k < p.n; k++) weights.push(p.weight);
    o += p.n;
  }
  return { data, stride, weights };
}

/** 1 for every welded vertex that shares its position with another (a UV / material / colour / hard-normal seam). */
function seamLocks(m: KitMesh): Uint8Array {
  const count = m.positions.length / 3;
  const byPos = new Map<string, number>();
  const key = (v: number) => `${Math.round(m.positions[v * 3]! * 1e4)},${Math.round(m.positions[v * 3 + 1]! * 1e4)},${Math.round(m.positions[v * 3 + 2]! * 1e4)}`;
  for (let v = 0; v < count; v++) { const k = key(v); byPos.set(k, (byPos.get(k) ?? 0) + 1); }
  const lock = new Uint8Array(count);
  for (let v = 0; v < count; v++) if (byPos.get(key(v))! > 1) lock[v] = 1;
  return lock;
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

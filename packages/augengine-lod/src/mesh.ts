/**
 * mesh — the package's working mesh format and the conversion to the
 * engine's `MeshData` (8 floats per vertex: position, normal, uv, plus an
 * optional RGBA vertex-colour stream that the renderer multiplies into the
 * material base colour).
 */

export type Vec3 = [number, number, number];
export type Rgba = [number, number, number, number];

/** Separate attribute streams; one vertex = one entry in every stream. */
export interface KitMesh {
  positions: Float32Array; // 3 per vertex
  normals: Float32Array; // 3 per vertex
  uvs: Float32Array; // 2 per vertex
  colors: Float32Array; // 4 per vertex (RGBA, linear)
  indices: Uint32Array; // 3 per triangle
}

/** Simple PBR parameters for a whole mesh (per-part colour lives in vertex colours). */
export interface KitMaterial {
  baseColor: Vec3;
  metallic: number;
  roughness: number;
  emissive?: Vec3;
}

/** Same shape as the engine's `MeshData` (runtime/live/webgpu/primitives.ts). */
export interface EngineMeshData {
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  colors?: Float32Array;
}

export function vertexCount(m: KitMesh): number {
  return m.positions.length / 3;
}

export function triangleCount(m: KitMesh): number {
  return m.indices.length / 3;
}

/** Interleave into the engine layout. Upload with `renderer.uploadMesh(key, data, 8)`. */
export function toEngineMesh(m: KitMesh): EngineMeshData {
  const n = vertexCount(m);
  const vertices = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) {
    vertices.set(m.positions.subarray(i * 3, i * 3 + 3), i * 8);
    vertices.set(m.normals.subarray(i * 3, i * 3 + 3), i * 8 + 3);
    vertices.set(m.uvs.subarray(i * 2, i * 2 + 2), i * 8 + 6);
  }
  return {
    vertices,
    indices: m.indices.slice(),
    vertexCount: n,
    indexCount: m.indices.length,
    colors: m.colors.slice(),
  };
}

export interface Aabb {
  min: Vec3;
  max: Vec3;
}

export function bounds(m: KitMesh): Aabb {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  // Only vertices referenced by triangles count, so stale vertices left by
  // simplification never widen the box.
  for (let t = 0; t < m.indices.length; t++) {
    const v = m.indices[t]! * 3;
    for (let k = 0; k < 3; k++) {
      const p = m.positions[v + k]!;
      if (p < min[k]!) min[k] = p;
      if (p > max[k]!) max[k] = p;
    }
  }
  return { min, max };
}

export function boundingRadius(b: Aabb): number {
  return Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) / 2;
}

/** Twice the area of triangle t (0 means degenerate). */
export function triangleArea2(m: KitMesh, t: number): number {
  const ia = m.indices[t * 3]! * 3;
  const ib = m.indices[t * 3 + 1]! * 3;
  const ic = m.indices[t * 3 + 2]! * 3;
  const p = m.positions;
  const ux = p[ib]! - p[ia]!, uy = p[ib + 1]! - p[ia + 1]!, uz = p[ib + 2]! - p[ia + 2]!;
  const vx = p[ic]! - p[ia]!, vy = p[ic + 1]! - p[ia + 1]!, vz = p[ic + 2]! - p[ia + 2]!;
  return Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
}

/** Drop triangles with repeated indices or (near-)zero area. */
export function removeDegenerates(m: KitMesh, minArea2 = 1e-10): KitMesh {
  const keep: number[] = [];
  for (let t = 0; t < m.indices.length / 3; t++) {
    const a = m.indices[t * 3]!, b = m.indices[t * 3 + 1]!, c = m.indices[t * 3 + 2]!;
    if (a === b || b === c || a === c) continue;
    if (triangleArea2(m, t) <= minArea2) continue;
    keep.push(a, b, c);
  }
  return { ...m, indices: Uint32Array.from(keep) };
}

/**
 * Re-emit the mesh flat-shaded: every triangle gets its own three vertices
 * and its face normal. Used after simplification, where collapsed edges
 * leave stale smooth normals; architecture and low-poly foliage both read
 * best faceted. UVs and colours are carried from the source corners.
 */
export function flatShade(m: KitMesh): KitMesh {
  const tris = m.indices.length / 3;
  const positions = new Float32Array(tris * 9);
  const normals = new Float32Array(tris * 9);
  const uvs = new Float32Array(tris * 6);
  const colors = new Float32Array(tris * 12);
  const indices = new Uint32Array(tris * 3);
  for (let t = 0; t < tris; t++) {
    const corner = [m.indices[t * 3]!, m.indices[t * 3 + 1]!, m.indices[t * 3 + 2]!];
    const pa = corner.map((v) => [m.positions[v * 3]!, m.positions[v * 3 + 1]!, m.positions[v * 3 + 2]!]);
    const ux = pa[1]![0]! - pa[0]![0]!, uy = pa[1]![1]! - pa[0]![1]!, uz = pa[1]![2]! - pa[0]![2]!;
    const vx = pa[2]![0]! - pa[0]![0]!, vy = pa[2]![1]! - pa[0]![1]!, vz = pa[2]![2]! - pa[0]![2]!;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    for (let k = 0; k < 3; k++) {
      const v = corner[k]!;
      const o = t * 3 + k;
      positions.set(pa[k]!, o * 3);
      normals.set([nx, ny, nz], o * 3);
      uvs.set([m.uvs[v * 2]!, m.uvs[v * 2 + 1]!], o * 2);
      colors.set(m.colors.subarray(v * 4, v * 4 + 4), o * 4);
      indices[o] = o;
    }
  }
  return { positions, normals, uvs, colors, indices };
}

/** Area-weighted average colour, used to tint impostors. */
export function averageColor(m: KitMesh): Rgba {
  const acc = [0, 0, 0, 0];
  let total = 0;
  for (let t = 0; t < m.indices.length / 3; t++) {
    const w = triangleArea2(m, t);
    for (let k = 0; k < 3; k++) {
      const v = m.indices[t * 3 + k]! * 4;
      for (let c = 0; c < 4; c++) acc[c]! += (m.colors[v + c]! * w) / 3;
    }
    total += w;
  }
  return total > 0 ? (acc.map((x) => x / total) as Rgba) : [1, 1, 1, 1];
}

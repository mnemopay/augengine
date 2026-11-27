/**
 * gltf — minimal, dependency-free GLB reader/writer for the bake tool.
 *
 * Reader: every mesh primitive reachable from the default scene, node
 * transforms applied, merged into one KitMesh. Material baseColorFactor is
 * folded into vertex colours. Textures are NOT carried (reported in
 * `warnings`); Draco/meshopt-compressed and sparse accessors are rejected.
 *
 * Writer: one mesh per LOD, each a single triangle primitive with POSITION,
 * NORMAL, TEXCOORD_0, COLOR_0 (float RGBA) and uint32 indices, one material,
 * one root node per LOD carrying `extras.lod`. Mesh names follow the engine's
 * `lodMeshKey` convention (`name`, `name#lod1`, ...).
 */
import { flatShade, type KitMaterial, type KitMesh } from "./mesh.js";

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const COMPONENT_SIZE: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNT: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

export interface ReadResult {
  mesh: KitMesh;
  material: KitMaterial;
  warnings: string[];
  hasTextures: boolean;
}

export function readGlb(buf: Uint8Array): ReadResult {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error("not a GLB file (bad magic). For .gltf + .bin, convert to .glb first.");
  if (dv.getUint32(4, true) !== 2) throw new Error("only glTF 2.0 GLB is supported");
  let off = 12;
  let json: any = null;
  let bin: Uint8Array | null = null;
  while (off < buf.byteLength) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const body = buf.subarray(off + 8, off + 8 + len);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(body));
    else if (type === CHUNK_BIN) bin = body;
    off += 8 + len;
  }
  if (!json) throw new Error("GLB has no JSON chunk");
  return readGltfJson(json, bin);
}

function readGltfJson(g: any, bin: Uint8Array | null): ReadResult {
  const warnings: string[] = [];
  const used = new Set<string>([...(g.extensionsUsed ?? [])]);
  for (const ext of ["KHR_draco_mesh_compression", "EXT_meshopt_compression"]) {
    if (used.has(ext)) throw new Error(`${ext} is not supported: export uncompressed glTF`);
  }
  const accessor = (i: number): { data: Float32Array | Uint32Array; count: number; comps: number } => {
    const a = g.accessors[i];
    if (a.sparse) throw new Error("sparse accessors are not supported");
    const comps = TYPE_COUNT[a.type]!;
    const count = a.count as number;
    const view = g.bufferViews[a.bufferView];
    if (!bin || (view.buffer ?? 0) !== 0) throw new Error("only GLB-embedded buffers are supported");
    const csize = COMPONENT_SIZE[a.componentType]!;
    const stride = view.byteStride ?? comps * csize;
    const base = (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
    const isIndex = a.componentType !== 5126 && !a.normalized && comps === 1;
    const out = isIndex ? new Uint32Array(count) : new Float32Array(count * comps);
    for (let e = 0; e < count; e++) {
      for (let c = 0; c < comps; c++) {
        const p = base + e * stride + c * csize;
        let v: number;
        switch (a.componentType) {
          case 5126: v = dv.getFloat32(p, true); break;
          case 5125: v = dv.getUint32(p, true); break;
          case 5123: v = dv.getUint16(p, true); if (a.normalized) v /= 65535; break;
          case 5121: v = dv.getUint8(p); if (a.normalized) v /= 255; break;
          case 5122: v = dv.getInt16(p, true); if (a.normalized) v = Math.max(v / 32767, -1); break;
          case 5120: v = dv.getInt8(p); if (a.normalized) v = Math.max(v / 127, -1); break;
          default: throw new Error(`unsupported componentType ${a.componentType}`);
        }
        out[e * comps + c] = v;
      }
    }
    return { data: out, count, comps };
  };

  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [];
  let hasTextures = false;
  let metallic = 1, roughness = 1, mats = 0;
  let hasNormals = true;

  const visit = (nodeIdx: number, parent: number[]) => {
    const node = g.nodes[nodeIdx];
    const world = mul(parent, nodeMatrix(node));
    if (node.mesh !== undefined) {
      for (const prim of g.meshes[node.mesh].primitives) {
        if ((prim.mode ?? 4) !== 4) { warnings.push(`skipped non-triangle primitive (mode ${prim.mode})`); continue; }
        const mat = prim.material !== undefined ? g.materials?.[prim.material] : undefined;
        const pbr = mat?.pbrMetallicRoughness ?? {};
        const factor: number[] = pbr.baseColorFactor ?? [1, 1, 1, 1];
        if (pbr.baseColorTexture || mat?.normalTexture) hasTextures = true;
        metallic = mats === 0 ? (pbr.metallicFactor ?? 1) : (metallic + (pbr.metallicFactor ?? 1)) / 2;
        roughness = mats === 0 ? (pbr.roughnessFactor ?? 1) : (roughness + (pbr.roughnessFactor ?? 1)) / 2;
        mats++;
        const pos = accessor(prim.attributes.POSITION);
        const nrm = prim.attributes.NORMAL !== undefined ? accessor(prim.attributes.NORMAL) : null;
        if (!nrm) hasNormals = false;
        const uv = prim.attributes.TEXCOORD_0 !== undefined ? accessor(prim.attributes.TEXCOORD_0) : null;
        const col = prim.attributes.COLOR_0 !== undefined ? accessor(prim.attributes.COLOR_0) : null;
        const base = P.length / 3;
        const nm = normalMatrix(world);
        for (let v = 0; v < pos.count; v++) {
          const x = pos.data[v * 3]!, y = pos.data[v * 3 + 1]!, z = pos.data[v * 3 + 2]!;
          P.push(world[0]! * x + world[4]! * y + world[8]! * z + world[12]!,
                 world[1]! * x + world[5]! * y + world[9]! * z + world[13]!,
                 world[2]! * x + world[6]! * y + world[10]! * z + world[14]!);
          if (nrm) {
            const a = nrm.data[v * 3]!, b = nrm.data[v * 3 + 1]!, c = nrm.data[v * 3 + 2]!;
            const nx = nm[0]! * a + nm[3]! * b + nm[6]! * c, ny = nm[1]! * a + nm[4]! * b + nm[7]! * c, nz = nm[2]! * a + nm[5]! * b + nm[8]! * c;
            const l = Math.hypot(nx, ny, nz) || 1;
            N.push(nx / l, ny / l, nz / l);
          } else N.push(0, 1, 0);
          UV.push(uv ? uv.data[v * 2]! : 0, uv ? uv.data[v * 2 + 1]! : 0);
          const cr = col ? col.data[v * col.comps]! : 1, cg = col ? col.data[v * col.comps + 1]! : 1, cb = col ? col.data[v * col.comps + 2]! : 1;
          const ca = col && col.comps === 4 ? col.data[v * 4 + 3]! : 1;
          C.push(cr * factor[0]!, cg * factor[1]!, cb * factor[2]!, ca * factor[3]!);
        }
        if (prim.indices !== undefined) {
          const ix = accessor(prim.indices).data;
          for (let k = 0; k < ix.length; k++) I.push(ix[k]! + base);
        } else for (let k = 0; k < pos.count; k++) I.push(base + k);
        if (det3(world) < 0) for (let k = I.length - (prim.indices !== undefined ? accessor(prim.indices).count : pos.count); k < I.length; k += 3) {
          const t = I[k + 1]!; I[k + 1] = I[k + 2]!; I[k + 2] = t; // mirrored node: keep outward winding
        }
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  const scene = g.scenes?.[g.scene ?? 0];
  const roots: number[] = scene?.nodes ?? (g.nodes ?? []).map((_: unknown, i: number) => i);
  for (const r of roots) visit(r, IDENTITY);
  if (P.length === 0) throw new Error("no triangle geometry found in the GLB");
  if (hasTextures) warnings.push("textures are not carried through the bake; base colour factors and vertex colours are kept");

  let mesh: KitMesh = {
    positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV),
    colors: Float32Array.from(C), indices: Uint32Array.from(I),
  };
  if (!hasNormals) { mesh = flatShade(mesh); warnings.push("source had no normals; generated flat normals"); }
  return { mesh, material: { baseColor: [1, 1, 1], metallic, roughness }, warnings, hasTextures };
}

export interface GlbLevel {
  name: string;
  lod: number;
  mesh: KitMesh;
}

/** Write one GLB holding every LOD as its own mesh + root node. */
export function writeGlb(levels: GlbLevel[], material: KitMaterial, generator = "@mnemopay/augengine-lod"): Uint8Array {
  const chunks: Uint8Array[] = [];
  let binLen = 0;
  const views: any[] = [];
  const accessors: any[] = [];
  const push = (arr: Float32Array | Uint32Array, target: number) => {
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    const pad = (4 - (binLen % 4)) % 4;
    if (pad) { chunks.push(new Uint8Array(pad)); binLen += pad; }
    views.push({ buffer: 0, byteOffset: binLen, byteLength: bytes.byteLength, target });
    chunks.push(bytes.slice());
    binLen += bytes.byteLength;
    return views.length - 1;
  };
  const meshes: any[] = [];
  const nodes: any[] = [];
  for (const lv of levels) {
    const m = lv.mesh;
    const n = m.positions.length / 3;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < n; v++) for (let k = 0; k < 3; k++) {
      const p = m.positions[v * 3 + k]!;
      if (p < min[k]!) min[k] = p;
      if (p > max[k]!) max[k] = p;
    }
    const acc = (view: number, componentType: number, count: number, type: string, extra: object = {}) => {
      accessors.push({ bufferView: view, componentType, count, type, ...extra });
      return accessors.length - 1;
    };
    const attributes = {
      POSITION: acc(push(m.positions, 34962), 5126, n, "VEC3", { min, max }),
      NORMAL: acc(push(m.normals, 34962), 5126, n, "VEC3"),
      TEXCOORD_0: acc(push(m.uvs, 34962), 5126, n, "VEC2"),
      COLOR_0: acc(push(m.colors, 34962), 5126, n, "VEC4"),
    };
    const indices = acc(push(m.indices, 34963), 5125, m.indices.length, "SCALAR");
    meshes.push({ name: lv.name, primitives: [{ attributes, indices, material: 0, mode: 4 }] });
    nodes.push({ name: lv.name, mesh: meshes.length - 1, extras: { lod: lv.lod } });
  }
  const gltf = {
    asset: { version: "2.0", generator },
    scene: 0,
    scenes: [{ nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes,
    materials: [{
      name: "kit",
      pbrMetallicRoughness: { baseColorFactor: [...material.baseColor, 1], metallicFactor: material.metallic, roughnessFactor: material.roughness },
      ...(material.emissive ? { emissiveFactor: material.emissive } : {}),
      doubleSided: true,
    }],
    buffers: [{ byteLength: binLen }],
    bufferViews: views,
    accessors,
  };
  let jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  if (jpad) { const t = new Uint8Array(jsonBytes.length + jpad).fill(0x20); t.set(jsonBytes); jsonBytes = t; }
  const bpad = (4 - (binLen % 4)) % 4;
  const total = 12 + 8 + jsonBytes.length + 8 + binLen + bpad;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, GLB_MAGIC, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true);
  dv.setUint32(16, CHUNK_JSON, true);
  out.set(jsonBytes, 20);
  let o = 20 + jsonBytes.length;
  dv.setUint32(o, binLen + bpad, true);
  dv.setUint32(o + 4, CHUNK_BIN, true);
  o += 8;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function nodeMatrix(n: any): number[] {
  if (n.matrix) return n.matrix;
  const [tx, ty, tz] = n.translation ?? [0, 0, 0];
  const [qx, qy, qz, qw] = n.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = n.scale ?? [1, 1, 1];
  const xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
  return [
    (1 - 2 * (yy + zz)) * sx, 2 * (xy + wz) * sx, 2 * (xz - wy) * sx, 0,
    2 * (xy - wz) * sy, (1 - 2 * (xx + zz)) * sy, 2 * (yz + wx) * sy, 0,
    2 * (xz + wy) * sz, 2 * (yz - wx) * sz, (1 - 2 * (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function mul(a: number[], b: number[]): number[] {
  const o = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r]! * b[c * 4 + k]!;
    o[c * 4 + r] = s;
  }
  return o;
}

function det3(m: number[]): number {
  return m[0]! * (m[5]! * m[10]! - m[9]! * m[6]!) - m[4]! * (m[1]! * m[10]! - m[9]! * m[2]!) + m[8]! * (m[1]! * m[6]! - m[5]! * m[2]!);
}

/** Inverse-transpose of the upper 3x3 (column-major), for transforming normals. */
function normalMatrix(m: number[]): number[] {
  const a = m[0]!, b = m[1]!, c = m[2]!, d = m[4]!, e = m[5]!, f = m[6]!, g = m[8]!, h = m[9]!, i = m[10]!;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
  const G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;
  const det = a * A + b * B + c * C || 1;
  // inverse = adjugate / det; its transpose is the cofactor matrix / det.
  return [A / det, B / det, C / det, D / det, E / det, F / det, G / det, H / det, I / det];
}

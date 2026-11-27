/**
 * gltf — minimal, dependency-free GLB reader/writer for the bake tool.
 *
 * Reader: every mesh primitive reachable from the default scene, node
 * transforms applied, merged into one KitMesh whose `groups` record which
 * glTF material each triangle range uses. Materials, textures, samplers and
 * images are carried verbatim (see material.ts), so textures survive the
 * bake. `materials: "fold"` restores the 0.1 behaviour instead: one material,
 * baseColorFactor multiplied into vertex colours, textures dropped.
 * Draco/meshopt-compressed and sparse accessors are rejected.
 *
 * Writer: one mesh per LOD (named with the engine's `lodMeshKey` convention:
 * `name`, `name#lod1`, ...), one primitive per material group, one root node
 * per LOD carrying `extras.lod`. Every level references the same materials,
 * so each image is stored once.
 */
import { flatShade, regroup, isFlatShaded, type KitMaterial, type KitMesh, type MeshGroup } from "./mesh.js";
import { withSmoothNormals } from "./smooth.js";
import { materialSetFromKit, summarizeMaterials, hasAnyTexture, TEXTURE_SLOTS, textureInfo, type LodImage, type LodMaterialSet } from "./material.js";
import { encodePngStored, type RgbaImage } from "./png.js";

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const COMPONENT_SIZE: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNT: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
/** Extensions that belong to materials/textures and are carried into the output. */
const MATERIAL_EXTENSION_PREFIXES = ["KHR_materials_", "KHR_texture_", "EXT_texture_"];

export interface ReadOptions {
  /**
   * "keep" (default): one group per source material, materials + textures carried.
   * "fold": 0.1 behaviour — baseColorFactor folded into vertex colours, one material, no textures.
   */
  materials?: "keep" | "fold";
  /**
   * What to do when the source has no NORMAL attribute. "auto"/"flat": flat
   * normals (what the glTF spec says a renderer must do). "smooth": area-
   * weighted smooth normals (organic scans such as TRELLIS output, which ship
   * without normals). Authored normals are always kept when present.
   */
  missingNormals?: "auto" | "flat" | "smooth";
}

export interface ReadResult {
  mesh: KitMesh;
  /** Average of the used materials' factors (the v1 manifest's `material`). */
  material: KitMaterial;
  /** The source materials/textures/images, ready to write back out. */
  materials: LodMaterialSet;
  warnings: string[];
  hasTextures: boolean;
  /** Every primitive carried a NORMAL attribute. */
  hasNormals: boolean;
  /** Some primitive carried COLOR_0. */
  hasColors: boolean;
  /** How the mesh is shaded: faceted ("flat") or with shared smooth normals. */
  shading: "flat" | "smooth";
}

export function readGlb(buf: Uint8Array, opts: ReadOptions = {}): ReadResult {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf.byteLength < 12 || dv.getUint32(0, true) !== GLB_MAGIC) throw new Error("not a GLB file (bad magic). For .gltf + .bin, convert to .glb first.");
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
  return readGltfJson(json, bin, opts);
}

function readGltfJson(g: any, bin: Uint8Array | null, opts: ReadOptions): ReadResult {
  const fold = opts.materials === "fold";
  const warnings: string[] = [];
  const warnOnce = (w: string) => { if (!warnings.includes(w)) warnings.push(w); };
  const used = new Set<string>([...(g.extensionsUsed ?? [])]);
  for (const ext of ["KHR_draco_mesh_compression", "EXT_meshopt_compression"]) {
    if (used.has(ext)) throw new Error(`${ext} is not supported: export uncompressed glTF`);
  }
  const viewBytes = (viewIdx: number): Uint8Array => {
    const view = g.bufferViews[viewIdx];
    if (!bin || (view.buffer ?? 0) !== 0) throw new Error("only GLB-embedded buffers are supported");
    return bin.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength);
  };
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

  // Materials: verbatim, plus glTF's default material for primitives without one.
  const srcMaterials: any[] = fold ? [] : structuredClone(g.materials ?? []);
  let defaultMaterial = -1;
  const materialFor = (prim: any): number => {
    if (fold) return 0;
    if (prim.material !== undefined && srcMaterials[prim.material]) return prim.material;
    if (defaultMaterial < 0) { defaultMaterial = srcMaterials.length; srcMaterials.push({ name: "default" }); }
    return defaultMaterial;
  };

  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [], TM: number[] = [];
  let foldTextures = false;
  let metallic = 1, roughness = 1, mats = 0;
  let hasNormals = true, hasColors = false;

  const visit = (nodeIdx: number, parent: number[]) => {
    const node = g.nodes[nodeIdx];
    const world = mul(parent, nodeMatrix(node));
    if (node.mesh !== undefined) {
      for (const prim of g.meshes[node.mesh].primitives) {
        if ((prim.mode ?? 4) !== 4) { warnings.push(`skipped non-triangle primitive (mode ${prim.mode})`); continue; }
        const mat = prim.material !== undefined ? g.materials?.[prim.material] : undefined;
        const pbr = mat?.pbrMetallicRoughness ?? {};
        const factor: number[] = fold ? (pbr.baseColorFactor ?? [1, 1, 1, 1]) : [1, 1, 1, 1];
        if (TEXTURE_SLOTS.some((s) => textureInfo(mat, s) !== undefined)) foldTextures = true;
        for (const s of TEXTURE_SLOTS) {
          const ti = textureInfo(mat, s);
          if (!fold && ti && (ti.texCoord ?? 0) !== 0) warnOnce(`a ${s} texture uses TEXCOORD_${ti.texCoord}; only TEXCOORD_0 is carried, so it will sample TEXCOORD_0`);
        }
        if (!fold && prim.attributes.TANGENT !== undefined) warnOnce("TANGENT attributes are dropped (they would not survive simplification); renderers derive tangents (AugEngine from screen-space derivatives, glTF viewers with MikkTSpace)");
        metallic = mats === 0 ? (pbr.metallicFactor ?? 1) : (metallic + (pbr.metallicFactor ?? 1)) / 2;
        roughness = mats === 0 ? (pbr.roughnessFactor ?? 1) : (roughness + (pbr.roughnessFactor ?? 1)) / 2;
        mats++;
        const matIndex = materialFor(prim);
        const pos = accessor(prim.attributes.POSITION);
        const nrm = prim.attributes.NORMAL !== undefined ? accessor(prim.attributes.NORMAL) : null;
        if (!nrm) hasNormals = false;
        const uv = prim.attributes.TEXCOORD_0 !== undefined ? accessor(prim.attributes.TEXCOORD_0) : null;
        const col = prim.attributes.COLOR_0 !== undefined ? accessor(prim.attributes.COLOR_0) : null;
        if (col) hasColors = true;
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
        const first = I.length;
        if (prim.indices !== undefined) {
          const ix = accessor(prim.indices).data;
          for (let k = 0; k < ix.length; k++) I.push(ix[k]! + base);
        } else for (let k = 0; k < pos.count; k++) I.push(base + k);
        if (det3(world) < 0) for (let k = first; k < I.length; k += 3) {
          const t = I[k + 1]!; I[k + 1] = I[k + 2]!; I[k + 2] = t; // mirrored node: keep outward winding
        }
        for (let k = first; k < I.length; k += 3) TM.push(matIndex);
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  const scene = g.scenes?.[g.scene ?? 0];
  const roots: number[] = scene?.nodes ?? (g.nodes ?? []).map((_: unknown, i: number) => i);
  for (const r of roots) visit(r, IDENTITY);
  if (P.length === 0) throw new Error("no triangle geometry found in the GLB");
  if (fold && foldTextures) warnings.push("textures are not carried with materials: \"fold\"; base colour factors and vertex colours are kept");

  let materials: LodMaterialSet;
  if (fold) {
    materials = materialSetFromKit({ baseColor: [1, 1, 1], metallic, roughness });
  } else {
    const images: LodImage[] = (g.images ?? []).map((img: any): LodImage => {
      const mimeType: string = img.mimeType ?? guessMime(img.uri);
      const named = img.name ? { name: img.name as string } : {};
      if (img.bufferView !== undefined) return { bytes: viewBytes(img.bufferView).slice(), mimeType, ...named };
      if (typeof img.uri === "string" && img.uri.startsWith("data:")) return { bytes: decodeDataUri(img.uri), mimeType, ...named };
      warnOnce("an image references an external file; its URI is kept as-is (relative to the source GLB)");
      return { bytes: new Uint8Array(0), mimeType, uri: img.uri, ...named };
    });
    const isMatExt = (e: string) => MATERIAL_EXTENSION_PREFIXES.some((p) => e.startsWith(p));
    materials = {
      materials: srcMaterials,
      textures: structuredClone(g.textures ?? []),
      samplers: structuredClone(g.samplers ?? []),
      images,
      extensionsUsed: (g.extensionsUsed ?? []).filter(isMatExt),
      extensionsRequired: (g.extensionsRequired ?? []).filter(isMatExt),
    };
  }

  let mesh: KitMesh = {
    positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV),
    colors: Float32Array.from(C), indices: Uint32Array.from(I),
  };
  if (!fold) mesh = { ...mesh, ...regroup(mesh.indices, Uint32Array.from(TM)) };
  if (!hasNormals) {
    if (opts.missingNormals === "smooth") { mesh = withSmoothNormals(mesh); warnings.push("source had no normals; generated smooth normals"); }
    else { mesh = flatShade(mesh); warnings.push("source had no normals; generated flat normals (pass missingNormals: \"smooth\" / --normals smooth for organic meshes)"); }
  }
  const summary = fold ? { baseColor: [1, 1, 1] as [number, number, number], metallic, roughness } : summarizeMaterials(materials, [...new Set(TM)]);
  return {
    mesh,
    material: summary,
    materials,
    warnings,
    hasTextures: fold ? foldTextures : hasAnyTexture(materials),
    hasNormals,
    hasColors,
    shading: isFlatShaded(mesh) ? "flat" : "smooth",
  };
}

function guessMime(uri: unknown): string {
  const u = typeof uri === "string" ? uri.toLowerCase() : "";
  if (u.startsWith("data:")) return u.slice(5, u.indexOf(";")) || "image/png";
  if (u.endsWith(".jpg") || u.endsWith(".jpeg")) return "image/jpeg";
  if (u.endsWith(".ktx2")) return "image/ktx2";
  if (u.endsWith(".webp")) return "image/webp";
  return "image/png";
}

function decodeDataUri(uri: string): Uint8Array {
  const bin = atob(uri.slice(uri.indexOf(",") + 1));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export interface GlbLevel {
  name: string;
  lod: number;
  mesh: KitMesh;
}

export interface WriteOptions {
  generator?: string;
  /**
   * Write COLOR_0: "auto" (default) only for levels whose colours are not all
   * white (a white stream multiplies by 1 and only costs memory); true/false forces it.
   */
  colors?: "auto" | boolean;
}

/** Write one GLB holding every LOD as its own mesh + root node (kit / v1 style: one material). */
export function writeGlb(levels: GlbLevel[], material: KitMaterial, generator = "@mnemopay/augengine-lod"): Uint8Array {
  return writeLodGlb(levels, materialSetFromKit(material), { generator, colors: true });
}

/**
 * Write one GLB holding every LOD as its own mesh + root node, one primitive
 * per material group, all levels sharing `set`'s materials and images.
 */
export function writeLodGlb(levels: GlbLevel[], set: LodMaterialSet, opts: WriteOptions = {}): Uint8Array {
  const chunks: Uint8Array[] = [];
  let binLen = 0;
  const views: any[] = [];
  const accessors: any[] = [];
  const pushBytes = (bytes: Uint8Array, target?: number) => {
    const pad = (4 - (binLen % 4)) % 4;
    if (pad) { chunks.push(new Uint8Array(pad)); binLen += pad; }
    views.push({ buffer: 0, byteOffset: binLen, byteLength: bytes.byteLength, ...(target ? { target } : {}) });
    chunks.push(bytes.slice());
    binLen += bytes.byteLength;
    return views.length - 1;
  };
  const push = (arr: Float32Array | Uint32Array, target: number) => pushBytes(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), target);
  const meshes: any[] = [];
  const nodes: any[] = [];
  const colorMode = opts.colors ?? "auto";
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
    const writeColors = colorMode === true || (colorMode === "auto" && !allWhite(m.colors));
    const attributes = {
      POSITION: acc(push(m.positions, 34962), 5126, n, "VEC3", { min, max }),
      NORMAL: acc(push(m.normals, 34962), 5126, n, "VEC3"),
      TEXCOORD_0: acc(push(m.uvs, 34962), 5126, n, "VEC2"),
      ...(writeColors ? { COLOR_0: acc(push(m.colors, 34962), 5126, n, "VEC4") } : {}),
    };
    const groups: MeshGroup[] = m.groups && m.groups.length > 0 ? m.groups : [{ material: 0, start: 0, count: m.indices.length }];
    const primitives = groups.filter((gr) => gr.count > 0).map((gr) => ({
      attributes,
      indices: acc(push(m.indices.subarray(gr.start, gr.start + gr.count), 34963), 5125, gr.count, "SCALAR"),
      material: gr.material,
      mode: 4,
    }));
    meshes.push({ name: lv.name, primitives });
    nodes.push({ name: lv.name, mesh: meshes.length - 1, extras: { lod: lv.lod } });
  }
  const images = set.images.map((img) => {
    const named = img.name ? { name: img.name } : {};
    if (img.bytes.length === 0 && img.uri) return { uri: img.uri, mimeType: img.mimeType, ...named };
    return { bufferView: pushBytes(img.bytes), mimeType: img.mimeType, ...named };
  });
  const gltf: any = {
    asset: { version: "2.0", generator: opts.generator ?? "@mnemopay/augengine-lod" },
    ...(set.extensionsUsed.length ? { extensionsUsed: set.extensionsUsed } : {}),
    ...(set.extensionsRequired.length ? { extensionsRequired: set.extensionsRequired } : {}),
    scene: 0,
    scenes: [{ nodes: nodes.map((_, i) => i) }],
    nodes,
    meshes,
    materials: set.materials,
    ...(set.textures.length ? { textures: set.textures } : {}),
    ...(set.samplers.length ? { samplers: set.samplers } : {}),
    ...(images.length ? { images } : {}),
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

/**
 * Add the impostor's material (and its atlas image, PNG-encoded with
 * `encode`) to a set. Returns the new set and the impostor's material index.
 */
export function withImpostorMaterial(
  set: LodMaterialSet,
  impostor: { material: any; atlas?: RgbaImage | null },
  encode: (img: RgbaImage) => Uint8Array = encodePngStored,
): { set: LodMaterialSet; material: number } {
  const next: LodMaterialSet = {
    ...set,
    materials: [...set.materials],
    textures: [...set.textures],
    samplers: [...set.samplers],
    images: [...set.images],
  };
  const mat = structuredClone(impostor.material);
  if (impostor.atlas) {
    next.images.push({ bytes: encode(impostor.atlas), mimeType: "image/png", name: "lod3-impostor-atlas" });
    next.samplers.push({ magFilter: 9729, minFilter: 9729, wrapS: 33071, wrapT: 33071 });
    next.textures.push({ source: next.images.length - 1, sampler: next.samplers.length - 1 });
    mat.pbrMetallicRoughness = { ...(mat.pbrMetallicRoughness ?? {}), baseColorTexture: { index: next.textures.length - 1 } };
  }
  next.materials.push(mat);
  return { set: next, material: next.materials.length - 1 };
}

function allWhite(c: Float32Array): boolean {
  for (let i = 0; i < c.length; i++) if (Math.abs(c[i]! - 1) > 1e-4) return false;
  return true;
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

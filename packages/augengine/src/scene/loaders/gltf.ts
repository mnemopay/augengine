/**
 * GLB binary loader — parses the .glb container into a parsed root +
 * a single binary buffer, then extracts meshes, skins, and animations
 * into AugEngine-native data structures.
 *
 * Phase 3 — validation-gate work.
 *
 * GLB layout (per Khronos spec):
 *   12-byte header: magic 'glTF' (0x46546C67) | version u32 | length u32
 *   Chunk 1: length u32 | type 'JSON' (0x4E4F534A) | UTF-8 JSON bytes
 *   Chunk 2 (optional): length u32 | type 'BIN\0' (0x004E4942) | bytes
 *
 * Per spec: only one BIN chunk; only one buffer entry references it as
 * `buffer 0` with no `uri`.
 *
 * No third-party deps. Pure ES + standard browser/Node Buffer APIs.
 */

import {
  COMPONENT_BYTES,
  TYPE_COMPONENTS,
  type GltfRoot,
  type GltfAccessor,
  type GltfPrimitive,
  type GltfSkin,
  type GltfAnimation,
} from "./gltf-types.js";

// Magic numbers (little-endian u32)
const GLB_MAGIC = 0x46546c67;
const GLB_CHUNK_JSON = 0x4e4f534a;
const GLB_CHUNK_BIN = 0x004e4942;

export interface GltfParsed {
  root: GltfRoot;
  /** Raw BIN-chunk bytes (or null if the GLB had no BIN chunk). */
  bin: Uint8Array | null;
}

/** Parse a .glb byte buffer into the JSON root + BIN chunk. */
export function parseGlb(bytes: Uint8Array | ArrayBuffer): GltfParsed {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf.length < 12) throw new Error("GLB: too short for header");

  const magic = view.getUint32(0, true);
  if (magic !== GLB_MAGIC) {
    throw new Error(`GLB: bad magic 0x${magic.toString(16)} (expected 'glTF')`);
  }
  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`GLB: unsupported version ${version} (only 2.0 supported)`);
  }
  const totalLength = view.getUint32(8, true);
  if (totalLength > buf.length) {
    throw new Error(`GLB: declared length ${totalLength} exceeds buffer ${buf.length}`);
  }

  // Read chunks. Spec: first MUST be JSON; second MAY be BIN.
  let offset = 12;
  let root: GltfRoot | null = null;
  let bin: Uint8Array | null = null;
  while (offset < totalLength) {
    if (offset + 8 > totalLength) {
      throw new Error("GLB: truncated chunk header");
    }
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const dataStart = offset + 8;
    const dataEnd = dataStart + chunkLength;
    if (dataEnd > totalLength) {
      throw new Error(`GLB: chunk extends past total (chunk type 0x${chunkType.toString(16)})`);
    }
    if (chunkType === GLB_CHUNK_JSON) {
      const text = new TextDecoder("utf-8").decode(buf.subarray(dataStart, dataEnd));
      root = JSON.parse(text) as GltfRoot;
    } else if (chunkType === GLB_CHUNK_BIN) {
      bin = buf.subarray(dataStart, dataEnd);
    } else {
      // Unknown chunk type — spec says skip, don't fail.
      // Common future chunk types include compression and meta.
    }
    offset = dataEnd;
  }
  if (!root) throw new Error("GLB: missing JSON chunk");
  return { root, bin };
}

// ──────────────────────────────────────────────────────────────────────
// Accessor reader
// ──────────────────────────────────────────────────────────────────────

/**
 * Read a glTF accessor as a typed array. The returned array references
 * `bin` (no copy) when stride is contiguous; otherwise copies into a
 * compact array.
 */
export function readAccessor(
  parsed: GltfParsed,
  accessorIndex: number,
): Float32Array | Uint32Array | Uint16Array | Uint8Array | Int8Array | Int16Array {
  const root = parsed.root;
  const bin = parsed.bin;
  const accessor = root.accessors?.[accessorIndex];
  if (!accessor) throw new Error(`accessor ${accessorIndex} missing`);
  if (accessor.bufferView === undefined) {
    // Spec allows sparse-only accessors; we don't yet support those.
    throw new Error(`accessor ${accessorIndex} has no bufferView (sparse-only — not yet supported)`);
  }
  const bv = root.bufferViews?.[accessor.bufferView];
  if (!bv) throw new Error(`bufferView ${accessor.bufferView} missing`);
  if (bv.buffer !== 0) {
    throw new Error(`only buffer 0 (BIN chunk) supported; got ${bv.buffer}`);
  }
  if (!bin) throw new Error(`accessor ${accessorIndex} needs BIN chunk but none present`);

  const components = TYPE_COMPONENTS[accessor.type];
  const compBytes = COMPONENT_BYTES[accessor.componentType];
  if (!components || !compBytes) {
    throw new Error(
      `accessor ${accessorIndex}: bad type ${accessor.type}/${accessor.componentType}`,
    );
  }
  const elementBytes = components * compBytes;
  const stride = bv.byteStride ?? elementBytes;
  const start = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const count = accessor.count;

  const totalBytes = (count - 1) * stride + elementBytes;
  if (start + totalBytes > bin.byteLength) {
    throw new Error(`accessor ${accessorIndex} extends past BIN`);
  }

  // Fast path — contiguous stride === element size, return a typed view.
  if (stride === elementBytes) {
    const totalElems = count * components;
    const buf = bin.buffer;
    const off = bin.byteOffset + start;
    return makeTypedView(accessor.componentType, buf, off, totalElems);
  }

  // Strided path — copy into compact storage.
  const out = makeTypedArray(accessor.componentType, count * components);
  for (let i = 0; i < count; i++) {
    const srcOff = start + i * stride;
    const view = makeTypedView(accessor.componentType, bin.buffer, bin.byteOffset + srcOff, components);
    out.set(view as never, i * components);
  }
  return out;
}

function makeTypedView(
  componentType: number,
  buffer: ArrayBufferLike,
  byteOffset: number,
  length: number,
):
  | Float32Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Int8Array
  | Int16Array {
  switch (componentType) {
    case 5126:
      return new Float32Array(buffer as ArrayBuffer, byteOffset, length);
    case 5125:
      return new Uint32Array(buffer as ArrayBuffer, byteOffset, length);
    case 5123:
      return new Uint16Array(buffer as ArrayBuffer, byteOffset, length);
    case 5122:
      return new Int16Array(buffer as ArrayBuffer, byteOffset, length);
    case 5121:
      return new Uint8Array(buffer as ArrayBuffer, byteOffset, length);
    case 5120:
      return new Int8Array(buffer as ArrayBuffer, byteOffset, length);
    default:
      throw new Error(`unsupported componentType ${componentType}`);
  }
}

function makeTypedArray(
  componentType: number,
  length: number,
):
  | Float32Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Int8Array
  | Int16Array {
  switch (componentType) {
    case 5126: return new Float32Array(length);
    case 5125: return new Uint32Array(length);
    case 5123: return new Uint16Array(length);
    case 5122: return new Int16Array(length);
    case 5121: return new Uint8Array(length);
    case 5120: return new Int8Array(length);
    default: throw new Error(`unsupported componentType ${componentType}`);
  }
}

// ──────────────────────────────────────────────────────────────────────
// Mesh extraction — produces interleaved (pos+normal) Float32Array + indices
// ──────────────────────────────────────────────────────────────────────

export interface AugEngineMeshFromGltf {
  /** Interleaved position + normal, plus TEXCOORD_0 when present. */
  vertices: Float32Array;
  vertexStride: 6 | 8;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  /** Optional per-vertex skin attributes (4 joints + 4 weights). */
  joints?: Uint16Array;
  weights?: Float32Array;
  colors?: Float32Array;
  /** Optional morph-target deltas (Phase 6). One entry per target
   *  (shape key); `positions` required, `normals` optional. Each
   *  Float32Array has length 3 * vertexCount. The renderer's
   *  `BlendshapeUniformPool` weights select which targets to mix. */
  morphTargets?: AugEngineMorphTarget[];
  /** glTF metallic/roughness factors and first native texture source. */
  pbrMaterial?: AugEnginePbrMaterialFromGltf;
}

export interface AugEngineMorphTarget {
  positions: Float32Array;
  normals?: Float32Array;
}

export interface AugEngineTextureSource {
  bytes?: Uint8Array;
  mimeType?: string;
  uri?: string;
}

export interface AugEnginePbrMaterialFromGltf {
  baseColor: readonly [number, number, number];
  metallic: number;
  roughness: number;
  /**
   * glTF `alphaMode: "MASK"`: the cutoff the base colour's alpha (texture ×
   * factor) is compared against; fragments below it are discarded in the
   * main and shadow passes. Absent for OPAQUE (and BLEND, which the renderer
   * draws opaque).
   */
  alphaCutoff?: number;
  /** Base colour factor alpha (only kept for MASK, where it scales the texture alpha). */
  baseColorAlpha?: number;
  /** glTF `doubleSided`: drawn without back-face culling, back faces lit with the flipped normal. */
  doubleSided?: boolean;
  baseColorTexture?: AugEngineTextureSource;
  metallicRoughnessTexture?: AugEngineTextureSource;
  normalTexture?: AugEngineTextureSource;
}

/** Extract a single primitive from a mesh into AugEngine-native MeshData shape. */
export function extractPrimitive(
  parsed: GltfParsed,
  primitive: GltfPrimitive,
): AugEngineMeshFromGltf {
  const positions = readAccessor(parsed, requireAttr(primitive, "POSITION")) as Float32Array;
  const normals = primitive.attributes.NORMAL !== undefined
    ? (readAccessor(parsed, primitive.attributes.NORMAL) as Float32Array)
    : null;
  const texcoords = primitive.attributes.TEXCOORD_0 !== undefined
    ? (readAccessor(parsed, primitive.attributes.TEXCOORD_0) as Float32Array)
    : null;
  const indicesRaw = primitive.indices !== undefined
    ? readAccessor(parsed, primitive.indices)
    : null;

  const vertexCount = positions.length / 3;
  const vertexStride: 6 | 8 = texcoords ? 8 : 6;
  const vertices = new Float32Array(vertexCount * vertexStride);
  for (let i = 0; i < vertexCount; i++) {
    const base = i * vertexStride;
    vertices[base] = positions[i * 3]!;
    vertices[base + 1] = positions[i * 3 + 1]!;
    vertices[base + 2] = positions[i * 3 + 2]!;
    if (normals) {
      vertices[base + 3] = normals[i * 3]!;
      vertices[base + 4] = normals[i * 3 + 1]!;
      vertices[base + 5] = normals[i * 3 + 2]!;
    } else {
      // Fallback flat normal — not great but avoids undefined data.
      vertices[base + 3] = 0;
      vertices[base + 4] = 1;
      vertices[base + 5] = 0;
    }
    if (texcoords) {
      vertices[base + 6] = texcoords[i * 2]!;
      vertices[base + 7] = texcoords[i * 2 + 1]!;
    }
  }

  let indices: Uint32Array;
  if (indicesRaw) {
    if (indicesRaw instanceof Uint32Array) {
      indices = indicesRaw;
    } else if (
      indicesRaw instanceof Uint16Array ||
      indicesRaw instanceof Uint8Array
    ) {
      indices = new Uint32Array(indicesRaw.length);
      for (let i = 0; i < indicesRaw.length; i++) indices[i] = indicesRaw[i]!;
    } else {
      throw new Error("unsupported indices componentType");
    }
  } else {
    // Non-indexed — generate trivial indices.
    indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
  }

  const result: AugEngineMeshFromGltf = {
    vertices,
    vertexStride,
    indices,
    vertexCount,
    indexCount: indices.length,
  };

  const pbrMaterial = extractPbrMaterial(parsed, primitive);
  if (pbrMaterial) result.pbrMaterial = pbrMaterial;

  if (primitive.attributes.COLOR_0 !== undefined) {
    const accessorIdx = primitive.attributes.COLOR_0;
    const accessor = parsed.root.accessors?.[accessorIdx];
    const raw = readAccessor(parsed, accessorIdx);
    if (accessor) {
      const numComponents = accessor.type === "VEC4" ? 4 : 3;
      const count = accessor.count;
      const colors = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        let r = 0, g = 0, b = 0, a = 1;
        if (accessor.componentType === 5126) { // float32
          r = raw[i * numComponents] ?? 0;
          g = raw[i * numComponents + 1] ?? 0;
          b = raw[i * numComponents + 2] ?? 0;
          if (numComponents === 4) {
            a = raw[i * numComponents + 3] ?? 1;
          }
        } else if (accessor.componentType === 5123) { // uint16 normalized
          r = (raw[i * numComponents] ?? 0) / 65535;
          g = (raw[i * numComponents + 1] ?? 0) / 65535;
          b = (raw[i * numComponents + 2] ?? 0) / 65535;
          if (numComponents === 4) {
            a = (raw[i * numComponents + 3] ?? 65535) / 65535;
          }
        } else if (accessor.componentType === 5121) { // uint8 normalized
          r = (raw[i * numComponents] ?? 0) / 255;
          g = (raw[i * numComponents + 1] ?? 0) / 255;
          b = (raw[i * numComponents + 2] ?? 0) / 255;
          if (numComponents === 4) {
            a = (raw[i * numComponents + 3] ?? 255) / 255;
          }
        } else {
          r = raw[i * numComponents] ?? 0;
          g = raw[i * numComponents + 1] ?? 0;
          b = raw[i * numComponents + 2] ?? 0;
          if (numComponents === 4) {
            a = raw[i * numComponents + 3] ?? 1;
          }
        }
        colors[i * 4] = r;
        colors[i * 4 + 1] = g;
        colors[i * 4 + 2] = b;
        colors[i * 4 + 3] = a;
      }
      result.colors = colors;
    }
  }

  if (primitive.attributes.JOINTS_0 !== undefined && primitive.attributes.WEIGHTS_0 !== undefined) {
    const jointsRaw = readAccessor(parsed, primitive.attributes.JOINTS_0);
    const weightsRaw = readAccessor(parsed, primitive.attributes.WEIGHTS_0);
    let joints: Uint16Array;
    if (jointsRaw instanceof Uint16Array) {
      joints = jointsRaw;
    } else {
      joints = new Uint16Array(jointsRaw.length);
      for (let i = 0; i < jointsRaw.length; i++) joints[i] = jointsRaw[i]!;
    }
    const weights = weightsRaw instanceof Float32Array
      ? weightsRaw
      : new Float32Array(Array.from(weightsRaw as Iterable<number>));
    result.joints = joints;
    result.weights = weights;
  }

  // Phase 6 — morph targets (shape keys). One target per shape; each
  // carries a Float32Array of per-vertex position deltas plus optional
  // normal deltas. The renderer's `BlendshapeUniformPool` provides the
  // weights that mix these per draw.
  if (primitive.targets && primitive.targets.length > 0) {
    const morphTargets: AugEngineMorphTarget[] = [];
    for (const t of primitive.targets) {
      if (t.POSITION === undefined) {
        // Targets without a POSITION delta are unusable for a vertex
        // displacement system — skip with a console.warn so authors see
        // it but don't crash on partial exports.
        // eslint-disable-next-line no-console
        console.warn("glTF morph target missing POSITION accessor; skipping");
        continue;
      }
      const posDeltas = readAccessor(parsed, t.POSITION) as Float32Array;
      const positions = posDeltas instanceof Float32Array
        ? posDeltas
        : new Float32Array(Array.from(posDeltas as Iterable<number>));
      const entry: AugEngineMorphTarget = { positions };
      if (t.NORMAL !== undefined) {
        const normRaw = readAccessor(parsed, t.NORMAL);
        entry.normals = normRaw instanceof Float32Array
          ? normRaw
          : new Float32Array(Array.from(normRaw as Iterable<number>));
      }
      morphTargets.push(entry);
    }
    if (morphTargets.length > 0) result.morphTargets = morphTargets;
  }

  return result;
}

function extractPbrMaterial(
  parsed: GltfParsed,
  primitive: GltfPrimitive,
): AugEnginePbrMaterialFromGltf | undefined {
  if (primitive.material === undefined) return undefined;
  const material = parsed.root.materials?.[primitive.material];
  if (!material) return undefined;
  const pbr = material.pbrMetallicRoughness;
  const factor = pbr?.baseColorFactor ?? [1, 1, 1, 1];
  const result: AugEnginePbrMaterialFromGltf = {
    baseColor: [factor[0], factor[1], factor[2]],
    metallic: pbr?.metallicFactor ?? 1,
    roughness: pbr?.roughnessFactor ?? 1,
  };
  if (material.alphaMode === "MASK") {
    result.alphaCutoff = Number.isFinite(material.alphaCutoff) ? Math.max(0, material.alphaCutoff!) : 0.5;
    result.baseColorAlpha = factor[3] ?? 1;
  }
  if (material.doubleSided === true) result.doubleSided = true;
  const textureIndex = pbr?.baseColorTexture?.index;
  if (textureIndex !== undefined) {
    const source = extractTextureSource(parsed, textureIndex);
    if (source) result.baseColorTexture = source;
  }
  const metallicRoughnessIndex = pbr?.metallicRoughnessTexture?.index;
  if (metallicRoughnessIndex !== undefined) {
    const source = extractTextureSource(parsed, metallicRoughnessIndex);
    if (source) result.metallicRoughnessTexture = source;
  }
  const normalIndex = material.normalTexture?.index;
  if (normalIndex !== undefined) {
    const source = extractTextureSource(parsed, normalIndex);
    if (source) result.normalTexture = source;
  }
  return result;
}

function extractTextureSource(
  parsed: GltfParsed,
  textureIndex: number,
): AugEngineTextureSource | undefined {
  const texture = parsed.root.textures?.[textureIndex];
  if (!texture) return undefined;
  // EXT_texture_webp: browsers decode WebP (createImageBitmap), so the
  // extension's image wins, as the spec asks of a client that supports it.
  // The core `source` is the optional PNG/JPEG fallback (absent when the
  // extension is required, as in kael-varr.webp.glb).
  const webpIndex = texture.extensions?.EXT_texture_webp?.source;
  const webpImage = webpIndex !== undefined ? parsed.root.images?.[webpIndex] : undefined;
  const image = webpImage ?? (texture.source !== undefined ? parsed.root.images?.[texture.source] : undefined);
  if (!image) return undefined;
  const result: AugEngineTextureSource = {};
  if (image.mimeType) result.mimeType = image.mimeType;
  else if (image === webpImage) result.mimeType = "image/webp";
  if (image.uri) result.uri = image.uri;
  if (image.bufferView !== undefined && parsed.bin) {
    const view = parsed.root.bufferViews?.[image.bufferView];
    if (view && view.buffer === 0) {
      const start = view.byteOffset ?? 0;
      result.bytes = parsed.bin.slice(start, start + view.byteLength);
    }
  }
  return result.bytes || result.uri ? result : undefined;
}

function requireAttr(primitive: GltfPrimitive, name: string): number {
  const idx = primitive.attributes[name];
  if (idx === undefined) throw new Error(`primitive missing required ${name}`);
  return idx;
}

// ──────────────────────────────────────────────────────────────────────
// Skin + animation extractors
// ──────────────────────────────────────────────────────────────────────

export interface AugEngineSkinFromGltf {
  /** Joint glTF node indices in skin order. */
  joints: number[];
  /** Inverse-bind matrices, packed 16 floats per joint. */
  inverseBindMatrices: Float32Array;
}

export function extractSkin(parsed: GltfParsed, skin: GltfSkin): AugEngineSkinFromGltf {
  const joints = [...skin.joints];
  let inverseBindMatrices: Float32Array;
  if (skin.inverseBindMatrices !== undefined) {
    const ibm = readAccessor(parsed, skin.inverseBindMatrices);
    if (!(ibm instanceof Float32Array)) {
      throw new Error("skin inverseBindMatrices must be float32");
    }
    inverseBindMatrices = ibm;
  } else {
    // Default: identity per joint.
    inverseBindMatrices = new Float32Array(joints.length * 16);
    for (let j = 0; j < joints.length; j++) {
      inverseBindMatrices[j * 16] = 1;
      inverseBindMatrices[j * 16 + 5] = 1;
      inverseBindMatrices[j * 16 + 10] = 1;
      inverseBindMatrices[j * 16 + 15] = 1;
    }
  }
  return { joints, inverseBindMatrices };
}

export interface AugEngineAnimationTrack {
  /** Target node index in the glTF graph. */
  node: number;
  path: "translation" | "rotation" | "scale";
  /** Keyframe times in seconds (sorted). */
  times: Float32Array;
  /** Per-keyframe values: vec3 for translation/scale, quat for rotation. */
  values: Float32Array;
  interpolation: "LINEAR" | "STEP" | "CUBICSPLINE";
}

export interface AugEngineAnimationFromGltf {
  name: string;
  duration: number;
  tracks: AugEngineAnimationTrack[];
}

export function extractAnimation(
  parsed: GltfParsed,
  anim: GltfAnimation,
): AugEngineAnimationFromGltf {
  const tracks: AugEngineAnimationTrack[] = [];
  let duration = 0;
  for (const channel of anim.channels) {
    const sampler = anim.samplers[channel.sampler];
    if (!sampler || channel.target.node === undefined) continue;
    if (channel.target.path === "weights") continue; // morph targets — phase 9 stretch
    const times = readAccessor(parsed, sampler.input) as Float32Array;
    const values = readAccessor(parsed, sampler.output) as Float32Array;
    if (times.length > 0) {
      const last = times[times.length - 1]!;
      if (last > duration) duration = last;
    }
    tracks.push({
      node: channel.target.node,
      path: channel.target.path,
      times,
      values,
      interpolation: sampler.interpolation ?? "LINEAR",
    });
  }
  return {
    name: anim.name ?? "anim",
    duration,
    tracks,
  };
}

// ──────────────────────────────────────────────────────────────────────
// Convenience top-level loader
// ──────────────────────────────────────────────────────────────────────

export interface LoadedGlb {
  parsed: GltfParsed;
  /** First available primitive, preserved for existing single-mesh consumers. */
  mesh: AugEngineMeshFromGltf | null;
  /** All renderable primitives in deterministic mesh/primitive order. */
  primitives: AugEnginePrimitiveFromGltf[];
  /** First skin, if any. Preserved for existing single-skin consumers. */
  skin: AugEngineSkinFromGltf | null;
  /** All skins in glTF skin-index order — a mesh node binds via `node.skin`. */
  skins: AugEngineSkinFromGltf[];
  /** All animations. */
  animations: AugEngineAnimationFromGltf[];
}

export interface AugEnginePrimitiveFromGltf {
  /** Flat index used by geometry refs such as `gltf:<url>#<primitiveIndex>`. */
  primitiveIndex: number;
  meshIndex: number;
  meshPrimitiveIndex: number;
  mesh: AugEngineMeshFromGltf;
}

export function loadGlb(bytes: Uint8Array | ArrayBuffer): LoadedGlb {
  const parsed = parseGlb(bytes);
  const root = parsed.root;
  const primitives: AugEnginePrimitiveFromGltf[] = [];
  for (const [meshIndex, gltfMesh] of (root.meshes ?? []).entries()) {
    for (const [meshPrimitiveIndex, primitive] of gltfMesh.primitives.entries()) {
      primitives.push({
        primitiveIndex: primitives.length,
        meshIndex,
        meshPrimitiveIndex,
        mesh: extractPrimitive(parsed, primitive),
      });
    }
  }
  const mesh = primitives[0]?.mesh ?? null;
  const skins = (root.skins ?? []).map((s) => extractSkin(parsed, s));
  const skin = skins[0] ?? null;
  const animations = (root.animations ?? []).map((a) => extractAnimation(parsed, a));
  return { parsed, mesh, primitives, skin, skins, animations };
}

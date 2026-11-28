/**
 * glTF 2.0 type definitions — minimal subset for AugEngine's loader.
 *
 * Only the fields we read are typed. Optional everything-else is left
 * as `unknown` so downstream code doesn't accidentally rely on glTF
 * features the loader hasn't shipped yet.
 *
 * Spec: https://github.com/KhronosGroup/glTF/tree/main/specification/2.0
 */

export interface GltfRoot {
  asset: { version: string; generator?: string };
  scene?: number;
  scenes?: GltfScene[];
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  skins?: GltfSkin[];
  animations?: GltfAnimation[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
  materials?: GltfMaterial[];
  textures?: GltfTexture[];
  images?: GltfImage[];
  samplers?: unknown[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

export interface GltfScene {
  name?: string;
  nodes?: number[];
}

export interface GltfNode {
  name?: string;
  children?: number[];
  mesh?: number;
  skin?: number;
  /** Local position; default [0,0,0]. */
  translation?: [number, number, number];
  /** Local rotation as quaternion (x,y,z,w); default [0,0,0,1]. */
  rotation?: [number, number, number, number];
  /** Local scale; default [1,1,1]. */
  scale?: [number, number, number];
  /** Or full local matrix (column-major 16 floats). Wins if present. */
  matrix?: number[];
}

export interface GltfMesh {
  name?: string;
  primitives: GltfPrimitive[];
}

export interface GltfPrimitive {
  attributes: {
    POSITION?: number;
    NORMAL?: number;
    TEXCOORD_0?: number;
    JOINTS_0?: number;
    WEIGHTS_0?: number;
    [k: string]: number | undefined;
  };
  indices?: number;
  material?: number;
  mode?: number; // default 4 (TRIANGLES)
  extensions?: Record<string, unknown>;
  /** Morph targets — one entry per shape key. Each entry maps an
   *  attribute name (POSITION / NORMAL / TANGENT) to its delta-accessor. */
  targets?: ReadonlyArray<{
    POSITION?: number;
    NORMAL?: number;
    TANGENT?: number;
    [k: string]: number | undefined;
  }>;
}

export interface GltfSkin {
  name?: string;
  inverseBindMatrices?: number;
  skeleton?: number;
  joints: number[];
}

export interface GltfAnimation {
  name?: string;
  channels: GltfAnimationChannel[];
  samplers: GltfAnimationSampler[];
}

export interface GltfAnimationChannel {
  sampler: number;
  target: { node?: number; path: "translation" | "rotation" | "scale" | "weights" };
}

export interface GltfAnimationSampler {
  input: number;
  output: number;
  interpolation?: "LINEAR" | "STEP" | "CUBICSPLINE";
}

export interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number; // 5120 BYTE..5126 FLOAT
  count: number;
  type: "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT2" | "MAT3" | "MAT4";
  normalized?: boolean;
  min?: number[];
  max?: number[];
}

export interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
  /** Vendor / Khronos extensions — used by EXT_meshopt_compression to
   *  declare codec mode + filter + decompressed byte length per view. */
  extensions?: Record<string, unknown>;
}

export interface GltfBuffer {
  byteLength: number;
  uri?: string; // for non-GLB; GLB embeds buffer 0 in the BIN chunk
}

export interface GltfMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number];
    baseColorTexture?: { index: number };
    metallicFactor?: number;
    roughnessFactor?: number;
    metallicRoughnessTexture?: { index: number };
  };
  normalTexture?: { index: number; scale?: number };
  emissiveFactor?: [number, number, number];
  emissiveTexture?: { index: number };
  alphaMode?: "OPAQUE" | "MASK" | "BLEND";
  /** MASK only: fragments with alpha below this are discarded (glTF default 0.5). */
  alphaCutoff?: number;
  doubleSided?: boolean;
}

export interface GltfTexture {
  source?: number;
  sampler?: number;
  extensions?: {
    /** EXT_texture_webp: the WebP image. `source` (if any) is the PNG/JPEG fallback. */
    EXT_texture_webp?: { source?: number };
    [name: string]: unknown;
  };
}

export interface GltfImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
}

// Component type → bytes per component
export const COMPONENT_BYTES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

// Type → component count
export const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

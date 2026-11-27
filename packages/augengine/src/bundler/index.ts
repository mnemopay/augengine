/**
 * Asset bundler — Phase 12 closeout. Provides the data layer the
 * `augengine bundle` CLI command consumes: scene + assets + post-processing
 * config rolled into a deployable web bundle.
 *
 * Phase 12 ships the bundle manifest format + a JS-side copy/optimize
 * stage. KTX2/Basis texture compression + Draco mesh compression are
 * declared in the manifest but actually executed by the CLI's external
 * tool calls (ktx2c, draco_encoder), since they're native binaries.
 */

import type { SceneSpec } from "../scene/types.js";
import type { LoadedGlb } from "../scene/loaders/gltf.js";

export interface BundleManifest {
  version: 1;
  /** Scene file paths (relative). Loaded in order; first = entry. */
  scenes: string[];
  /** Asset declarations. Each declares optimization to apply at bundle time. */
  assets: BundleAsset[];
  /** Optional post-processing config. */
  postProcessing?: { passes: string[]; bloomThreshold?: number; exposure?: number };
  /** Target platform — drives renderer features + asset format. */
  target: "web" | "native-godot";
  /** Optional `.prompt` files included in the bundle. */
  prompts?: string[];
}

export interface BundleAsset {
  /** Path relative to the bundle source root. */
  src: string;
  /** Output path relative to bundle dist root. */
  dst: string;
  kind: "texture" | "mesh" | "audio" | "video" | "font" | "data";
  /** Optional optimization steps; CLI executes via local tools. */
  optimize?: BundleOptimize[];
}

export type BundleOptimize =
  | { kind: "ktx2"; quality?: number }
  | { kind: "basis"; quality?: number }
  | { kind: "draco"; level?: number }
  | { kind: "gltf-transform"; meshopt?: boolean; dedupe?: boolean }
  | { kind: "audio"; codec: "opus" | "aac"; bitrate?: number };

export interface BundleResult {
  /** Manifest as written to dist root. */
  manifest: BundleManifest;
  /** Per-asset stats — bytes before + after, ratio. */
  stats: Record<string, { bytesIn: number; bytesOut: number; ratio: number }>;
  /** Total bundle size in bytes. */
  totalBytes: number;
}

export interface GltfAssetAudit {
  bytes: number;
  primitives: number;
  vertices: number;
  indices: number;
  morphTargets: number;
  skins: number;
  animations: number;
  extensionsUsed: string[];
  extensionsRequired: string[];
  compression: {
    ktx2Textures: number;
    meshopt: boolean;
    draco: boolean;
  };
  estimatedGpuBytes: {
    vertexBuffers: number;
    indexBuffers: number;
    morphDeltaBuffers: number;
    embeddedTexturePayloads: number;
    totalKnown: number;
  };
  recommendations: string[];
}

export function auditLoadedGltfAsset(loaded: LoadedGlb, bytes: number): GltfAssetAudit {
  const extensionsUsed = loaded.parsed.root.extensionsUsed ?? [];
  const extensionsRequired = loaded.parsed.root.extensionsRequired ?? [];
  let vertices = 0;
  let indices = 0;
  let morphTargets = 0;
  let vertexBuffers = 0;
  let indexBuffers = 0;
  let morphDeltaBuffers = 0;
  for (const primitive of loaded.primitives) {
    const mesh = primitive.mesh;
    vertices += mesh.vertexCount;
    indices += mesh.indexCount;
    vertexBuffers += mesh.vertices.byteLength;
    indexBuffers += mesh.indices.byteLength;
    const targetCount = mesh.morphTargets?.length ?? 0;
    morphTargets += targetCount;
    morphDeltaBuffers += mesh.vertexCount * targetCount * 3 * 4;
  }
  let embeddedTexturePayloads = 0;
  let ktx2Textures = 0;
  for (const image of loaded.parsed.root.images ?? []) {
    if (image.mimeType === "image/ktx2" || image.uri?.toLowerCase().endsWith(".ktx2")) ktx2Textures++;
    if (image.bufferView !== undefined) {
      const view = loaded.parsed.root.bufferViews?.[image.bufferView];
      embeddedTexturePayloads += view?.byteLength ?? 0;
    }
  }
  const meshopt = extensionsUsed.includes("EXT_meshopt_compression");
  const draco = extensionsUsed.includes("KHR_draco_mesh_compression") ||
    loaded.primitives.some((primitive) => Boolean(
      loaded.parsed.root.meshes?.[primitive.meshIndex]?.primitives[primitive.meshPrimitiveIndex]?.extensions?.KHR_draco_mesh_compression,
    ));
  const recommendations: string[] = [];
  if (ktx2Textures === 0 && embeddedTexturePayloads > 0) {
    recommendations.push("Convert color/data textures to KTX2/BasisU for browser GPU memory and download wins.");
  }
  if (!meshopt && !draco && vertices > 5000) {
    recommendations.push("Run gltfpack/EXT_meshopt_compression for high-vertex GLB delivery.");
  }
  if (morphTargets > 0 && loaded.animations.length === 0) {
    recommendations.push("Asset has facial shapes but no animation clip; add a deterministic proof clip or runtime driver.");
  }
  return {
    bytes,
    primitives: loaded.primitives.length,
    vertices,
    indices,
    morphTargets,
    skins: loaded.skin ? 1 : 0,
    animations: loaded.animations.length,
    extensionsUsed,
    extensionsRequired,
    compression: { ktx2Textures, meshopt, draco },
    estimatedGpuBytes: {
      vertexBuffers,
      indexBuffers,
      morphDeltaBuffers,
      embeddedTexturePayloads,
      totalKnown: vertexBuffers + indexBuffers + morphDeltaBuffers + embeddedTexturePayloads,
    },
    recommendations,
  };
}

/** Validate a manifest's structural shape. Throws on malformed input. */
export function validateManifest(m: unknown): asserts m is BundleManifest {
  if (typeof m !== "object" || m === null) throw new Error("manifest: not an object");
  const v = m as Record<string, unknown>;
  if (v.version !== 1) throw new Error(`manifest: unsupported version ${v.version}`);
  if (!Array.isArray(v.scenes)) throw new Error("manifest: scenes must be array");
  if (!Array.isArray(v.assets)) throw new Error("manifest: assets must be array");
  if (typeof v.target !== "string") throw new Error("manifest: target must be string");
}

/** Hash a SceneSpec for cache-buster file naming. */
export function sceneSpecHash(spec: SceneSpec): string {
  // Re-use the deterministic JSON canonicalizer the persistence layer ships.
  // Imported at use-site so this module doesn't ESM-cycle persistence/index.js.
  const json = JSON.stringify(spec);
  let h = 0;
  for (let i = 0; i < json.length; i++) {
    h = ((h << 5) - h + json.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(16);
}

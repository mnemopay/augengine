/**
 * MorphDeltaStorage — per-mesh GPU storage buffers carrying the
 * 52 × N position deltas (and optionally 52 × N normal deltas) the
 * vertex shader samples to apply ARKit blendshape weights.
 *
 * Phase 6 v2 closes the data path:
 *
 *   audio →
 *   augengine-audio2face.Audio2FaceProvider →
 *   BlendshapeBuffer / renderer.setBlendshapeWeights →
 *   BlendshapeUniformPool (per-entity 208-byte uniform of 52 floats) →
 *   augengine_morph_position(weights, position_deltas, vertex_idx) →
 *   vertex shader's `pos += offset`
 *
 * The uniform pool lives in `blendshapes.ts` and feeds the weights.
 * THIS module is the other half — it owns the per-mesh delta buffers
 * the WGSL helper samples. glTF morph-target import populates the
 * deltas at mesh-upload time; runtime never mutates them.
 *
 * Layout per mesh: a Float32Array of length `vertexCount × 52 × 3`,
 * laid out as `[v0_shape0_xyz, v0_shape1_xyz, ..., v1_shape0_xyz, ...]`
 * to match the indexing the WGSL helper expects:
 *
 *   idx = vertex_idx * 52 + shape_k
 *   position_deltas[idx] = vec3<f32>(...)
 *
 * Storage-buffer access pattern is sequential per-shape, so the layout
 * keeps cache-line locality across the inner k-loop the helper runs.
 */

import { ARKIT_BLENDSHAPE_COUNT } from "./blendshape-constants.js";
import type {
  BlendshapeBufferLike,
  BlendshapeDeviceLike,
} from "./blendshapes.js";

/**
 * Subset of `AugEngineMeshFromGltf.morphTargets[]` we accept — duplicated
 * here as a structural type so this module doesn't have to import the
 * glTF loader (which lives in `scene/`). Matches the shape returned by
 * `extractPrimitive()` so callers can pass `mesh.morphTargets` directly.
 */
export interface MorphTargetSource {
  /** Per-vertex position deltas, length = 3 × vertexCount. */
  positions: Float32Array;
  /** Optional per-vertex normal deltas, same length. */
  normals?: Float32Array;
}

/** Bytes per vec3 delta entry (vec3<f32> is 12; WGSL storage buffer
 *  aligns to 16). We pack as vec3 and rely on the WGSL helper to read
 *  one vec3 at a time — array<vec3<f32>> in a storage buffer is laid
 *  out as 16-byte stride per vec3 on WebGPU. */
export const MORPH_DELTA_STRIDE_BYTES = 16;

export interface MorphDeltasInput {
  /** Number of vertices in the target mesh. */
  vertexCount: number;
  /**
   * Per-vertex per-shape position deltas, in the layout
   * `[v0_shape0_xyz, v0_shape1_xyz, ..., v_{N-1}_shape{K-1}_xyz]`.
   * Length must equal `vertexCount × ARKIT_BLENDSHAPE_COUNT × 3`.
   * Shorter inputs (fewer than 52 shapes) are zero-padded.
   */
  positionDeltas: Float32Array;
  /** Optional normal deltas with identical layout. */
  normalDeltas?: Float32Array;
}

export interface MorphDeltasHandle {
  meshKey: string;
  vertexCount: number;
  positionBuffer: BlendshapeBufferLike;
  /** Always present; zero-filled when a source mesh has no normal deltas. */
  normalBuffer: BlendshapeBufferLike;
  /** Bytes uploaded — useful for the asset-budget telemetry. */
  uploadedBytes: number;
}

/**
 * Bind-group entries the morph-variant pipeline consumes (alongside the
 * existing weight uniform from `BlendshapeUniformPool`):
 *
 *   binding 0: uniform   weights[52]                  (208 B)
 *   binding 1: storage   position_deltas[52 × N]      read-only
 *   binding 2: storage   normal_deltas[52 × N]        read-only (optional)
 */
export const MORPH_POSITION_DELTAS_BINDING = 1;
export const MORPH_NORMAL_DELTAS_BINDING = 2;

const USAGE_STORAGE_COPY_DST = 0x80 | 0x8; // STORAGE | COPY_DST

export class MorphDeltaStorage {
  private readonly meshes = new Map<string, MorphDeltasHandle>();

  constructor(private readonly device: BlendshapeDeviceLike) {}

  /** Upload position + optional normal deltas for a mesh, keyed by the
   *  same `meshKey` the renderer uses for vertex/index buffers
   *  (`primitive:<shape>` / `gltf:<url>#<primIdx>` / etc). */
  upload(meshKey: string, input: MorphDeltasInput): MorphDeltasHandle {
    if (this.meshes.has(meshKey)) {
      const existing = this.meshes.get(meshKey)!;
      return existing;
    }
    if (input.vertexCount <= 0) {
      throw new Error(`MorphDeltaStorage.upload(${meshKey}): vertexCount must be positive`);
    }
    const expectedFloats = input.vertexCount * ARKIT_BLENDSHAPE_COUNT * 3;
    if (input.positionDeltas.length > expectedFloats) {
      throw new Error(
        `MorphDeltaStorage.upload(${meshKey}): positionDeltas length ${input.positionDeltas.length} exceeds vertexCount × 52 × 3 = ${expectedFloats}`,
      );
    }
    if (input.normalDeltas && input.normalDeltas.length > expectedFloats) {
      throw new Error(
        `MorphDeltaStorage.upload(${meshKey}): normalDeltas length ${input.normalDeltas.length} exceeds vertexCount × 52 × 3 = ${expectedFloats}`,
      );
    }

    // Pad to vec4 stride for the storage buffer.
    const positionBuffer = this.uploadPadded(input.positionDeltas, input.vertexCount, expectedFloats);
    // Every morph draw binds normal storage. Position-only source morphs
    // receive zero normal deltas so they deform without changing lighting.
    const normalBuffer = this.uploadPadded(
      input.normalDeltas ?? new Float32Array(0),
      input.vertexCount,
      expectedFloats,
    );

    const uploadedBytes =
      input.vertexCount * ARKIT_BLENDSHAPE_COUNT * MORPH_DELTA_STRIDE_BYTES * 2;
    const handle: MorphDeltasHandle = {
      meshKey,
      vertexCount: input.vertexCount,
      positionBuffer,
      normalBuffer,
      uploadedBytes,
    };
    this.meshes.set(meshKey, handle);
    return handle;
  }

  /** Look up an existing handle. Returns null when nothing's uploaded
   *  for this mesh — callers fall back to the morph-disabled pipeline. */
  get(meshKey: string): MorphDeltasHandle | null {
    return this.meshes.get(meshKey) ?? null;
  }

  /** Test whether morph data exists for a mesh — drives the
   *  `hasMorphTargets` flag the pipeline factory uses to select the
   *  morph-aware variant. */
  has(meshKey: string): boolean {
    return this.meshes.has(meshKey);
  }

  /** Release one mesh's morph buffers. */
  clear(meshKey: string): void {
    const handle = this.meshes.get(meshKey);
    if (!handle) return;
    handle.positionBuffer.destroy?.();
    handle.normalBuffer.destroy?.();
    this.meshes.delete(meshKey);
  }

  /** Release every mesh's morph buffers. Called on scene unload. */
  clearAll(): void {
    for (const handle of this.meshes.values()) {
      handle.positionBuffer.destroy?.();
      handle.normalBuffer.destroy?.();
    }
    this.meshes.clear();
  }

  /** Total bytes resident on the GPU for morph deltas — telemetry only. */
  totalUploadedBytes(): number {
    let total = 0;
    for (const handle of this.meshes.values()) total += handle.uploadedBytes;
    return total;
  }

  /** Number of meshes tracked. */
  size(): number {
    return this.meshes.size;
  }

  /**
   * Pad vec3 deltas into the vec4 stride storage layout. The WGSL helper
   * `augengine_morph_position` reads `array<vec3<f32>>`, but in WGSL storage
   * buffers vec3<f32> arrays are laid out at vec4 stride — every vec3
   * is followed by 4 padding bytes. We do the padding CPU-side once at
   * upload rather than per-frame.
   */
  private uploadPadded(
    src: Float32Array,
    vertexCount: number,
    expectedFloats: number,
  ): BlendshapeBufferLike {
    const slots = vertexCount * ARKIT_BLENDSHAPE_COUNT;
    const padded = new Float32Array(slots * 4); // 4 floats per slot for vec4 stride
    const inputSlots = Math.min(src.length / 3, slots);
    for (let s = 0; s < inputSlots; s++) {
      padded[s * 4 + 0] = src[s * 3 + 0] ?? 0;
      padded[s * 4 + 1] = src[s * 3 + 1] ?? 0;
      padded[s * 4 + 2] = src[s * 3 + 2] ?? 0;
      // padded[s*4+3] left at 0
    }
    const buffer = this.device.createBuffer({
      size: padded.byteLength,
      usage: USAGE_STORAGE_COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, padded);
    void expectedFloats; // documented; unused after clamp-via-inputSlots
    return buffer;
  }
}

// ──────────────────────────────────────────────────────────────────────
// Packing helpers — bridge glTF morph targets to the GPU layout
// ──────────────────────────────────────────────────────────────────────

/**
 * Pack `AugEngineMeshFromGltf.morphTargets[]` into the per-vertex-per-shape
 * layout the WGSL helper expects:
 *
 *   layout[v * 52 + k] = vec3 delta for vertex v, shape k
 *
 * The source is laid out the other way (one Float32Array per shape,
 * with vertex-major indexing inside it). We transpose + zero-pad up to
 * the 52-channel ARKit convention so meshes with fewer shapes still
 * bind cleanly.
 *
 * Returns positions + (optional) normals. Caller passes both into
 * `MorphDeltaStorage.upload({ vertexCount, positionDeltas, normalDeltas })`.
 *
 * Channel order: the first `targets.length` channels carry the source
 * shapes in glTF order. Remaining channels (up to 52) are zero. Callers
 * that want ARKit-name-keyed mapping should remap their `targets[]`
 * array before calling — this helper is index-preserving.
 */
export function packMorphTargetsForGpu(
  vertexCount: number,
  targets: ReadonlyArray<MorphTargetSource>,
): { positionDeltas: Float32Array; normalDeltas?: Float32Array } {
  if (vertexCount <= 0) {
    throw new Error("packMorphTargetsForGpu: vertexCount must be positive");
  }
  if (targets.length === 0) {
    return { positionDeltas: new Float32Array(vertexCount * ARKIT_BLENDSHAPE_COUNT * 3) };
  }
  if (targets.length > ARKIT_BLENDSHAPE_COUNT) {
    throw new Error(
      `packMorphTargetsForGpu: ${targets.length} targets exceeds ARKit max ${ARKIT_BLENDSHAPE_COUNT}`,
    );
  }

  const positionDeltas = new Float32Array(vertexCount * ARKIT_BLENDSHAPE_COUNT * 3);
  let anyNormals = false;
  for (const t of targets) {
    if (t.normals !== undefined) {
      anyNormals = true;
      break;
    }
  }
  const normalDeltas = anyNormals
    ? new Float32Array(vertexCount * ARKIT_BLENDSHAPE_COUNT * 3)
    : undefined;

  for (let k = 0; k < targets.length; k++) {
    const t = targets[k]!;
    if (t.positions.length !== vertexCount * 3) {
      throw new Error(
        `packMorphTargetsForGpu: target ${k} positions length ${t.positions.length} != vertexCount × 3 (${vertexCount * 3})`,
      );
    }
    if (t.normals !== undefined && t.normals.length !== vertexCount * 3) {
      throw new Error(
        `packMorphTargetsForGpu: target ${k} normals length ${t.normals.length} != vertexCount × 3`,
      );
    }
    for (let v = 0; v < vertexCount; v++) {
      const dstBase = (v * ARKIT_BLENDSHAPE_COUNT + k) * 3;
      positionDeltas[dstBase] = t.positions[v * 3]!;
      positionDeltas[dstBase + 1] = t.positions[v * 3 + 1]!;
      positionDeltas[dstBase + 2] = t.positions[v * 3 + 2]!;
      if (normalDeltas && t.normals) {
        normalDeltas[dstBase] = t.normals[v * 3]!;
        normalDeltas[dstBase + 1] = t.normals[v * 3 + 1]!;
        normalDeltas[dstBase + 2] = t.normals[v * 3 + 2]!;
      }
    }
  }
  return normalDeltas
    ? { positionDeltas, normalDeltas }
    : { positionDeltas };
}

/**
 * Convenience: pack + upload in one call. Reads `vertexCount` from
 * the first target's positions array.
 */
export function uploadMorphTargets(
  storage: MorphDeltaStorage,
  meshKey: string,
  targets: ReadonlyArray<MorphTargetSource>,
): void {
  if (targets.length === 0) return;
  const vertexCount = (targets[0]!.positions.length) / 3;
  const packed = packMorphTargetsForGpu(vertexCount, targets);
  storage.upload(meshKey, {
    vertexCount,
    positionDeltas: packed.positionDeltas,
    ...(packed.normalDeltas !== undefined ? { normalDeltas: packed.normalDeltas } : {}),
  });
}

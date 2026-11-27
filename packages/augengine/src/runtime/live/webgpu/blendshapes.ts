/**
 * Blendshape uniform pool + WGSL helper — the renderer half of Phase 6
 * (Audio2Face).
 *
 * Owns one 208-byte (52-float) uniform buffer per entity that carries
 * morph-target weights from the audio2face provider into the vertex
 * shader. Buffers are lazy-allocated on first `setWeights(entityId, ...)`
 * and cached for the entity's lifetime.
 *
 * The WGSL helper `BLENDSHAPE_WGSL_HELPER` is a small `fn` block the
 * pipeline factory will splice into any vertex shader compiled for a
 * mesh that carries morph-target deltas. Splicing + delta-buffer
 * authoring are NOT in v1 — this module ships the data path
 * (weights → GPU uniform) so audio2face's `BlendshapePlayer` has a
 * concrete sink. Mesh-side deltas + the actual vertex-shader splice
 * land alongside the glTF morph-target importer in a follow-up.
 */

import { ARKIT_BLENDSHAPE_COUNT } from "./blendshape-constants.js";

/** Each entity's morph weights live in a 52-float buffer (208 bytes). */
export const BLENDSHAPE_UNIFORM_SIZE_BYTES = ARKIT_BLENDSHAPE_COUNT * 4;

/** Minimal GPUDevice surface this module relies on. Lets tests pass a
 *  stub without spinning up navigator.gpu. */
export interface BlendshapeDeviceLike {
  createBuffer(desc: { size: number; usage: number; mappedAtCreation?: boolean }): BlendshapeBufferLike;
  queue: { writeBuffer(buf: BlendshapeBufferLike, offset: number, data: ArrayBufferView | ArrayBuffer): void };
}

export interface BlendshapeBufferLike {
  /** Optional destroy hook so the pool can release on `clear`. */
  destroy?(): void;
}

/** Standard GPUBufferUsage flags this module needs. Mirrors the public
 *  WebGPU constants but isolated so the module is usable in Node tests
 *  where `GPUBufferUsage` may be undefined. */
const USAGE_UNIFORM_COPY_DST = 0x40 | 0x8; // UNIFORM | COPY_DST

export class BlendshapeUniformPool {
  private readonly buffers = new Map<string, BlendshapeBufferLike>();
  /**
   * Last weights pushed per entity, so the pool exposes a sync read for
   * tests + telemetry without a GPU readback. Mirrors the buffer.
   */
  private readonly latestWeights = new Map<string, Float32Array>();

  constructor(private readonly device: BlendshapeDeviceLike) {}

  /** Write 52 weights for an entity. Allocates the buffer on first call. */
  setWeights(entityId: string, weights: Float32Array): void {
    if (weights.length !== ARKIT_BLENDSHAPE_COUNT) {
      throw new Error(
        `BlendshapeUniformPool.setWeights(${entityId}): expected ${ARKIT_BLENDSHAPE_COUNT} weights, got ${weights.length}`,
      );
    }
    let buf = this.buffers.get(entityId);
    if (!buf) {
      buf = this.device.createBuffer({
        size: BLENDSHAPE_UNIFORM_SIZE_BYTES,
        usage: USAGE_UNIFORM_COPY_DST,
      });
      this.buffers.set(entityId, buf);
    }
    this.device.queue.writeBuffer(buf, 0, weights);
    // Cache a snapshot for sync reads.
    let cached = this.latestWeights.get(entityId);
    if (!cached) {
      cached = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
      this.latestWeights.set(entityId, cached);
    }
    cached.set(weights);
  }

  /** Get the GPU buffer for an entity, lazy-allocating if needed. Useful
   *  for binding into a pipeline's bind group. */
  ensureBuffer(entityId: string): BlendshapeBufferLike {
    let buf = this.buffers.get(entityId);
    if (!buf) {
      buf = this.device.createBuffer({
        size: BLENDSHAPE_UNIFORM_SIZE_BYTES,
        usage: USAGE_UNIFORM_COPY_DST,
      });
      this.buffers.set(entityId, buf);
    }
    return buf;
  }

  /** Sync read of the last-pushed weights for an entity. Returns null
   *  when nothing has been pushed. */
  peekWeights(entityId: string): Float32Array | null {
    const w = this.latestWeights.get(entityId);
    return w ? w.slice() : null;
  }

  /** Release one entity's buffer. Calls `.destroy()` when the platform
   *  exposes it (real WebGPU does; the test stub doesn't have to). */
  clear(entityId: string): void {
    const buf = this.buffers.get(entityId);
    if (buf?.destroy) buf.destroy();
    this.buffers.delete(entityId);
    this.latestWeights.delete(entityId);
  }

  /** Release every buffer. Called on scene unload. */
  clearAll(): void {
    for (const buf of this.buffers.values()) buf.destroy?.();
    this.buffers.clear();
    this.latestWeights.clear();
  }

  /** Number of entities currently tracked. */
  size(): number {
    return this.buffers.size;
  }
}

/**
 * WGSL helper that callers splice into a vertex shader at the
 * pre-skinning offset stage. Expects:
 *
 *   - `weights: array<f32, 52>` bound to the BlendshapeUniformPool buffer
 *   - `deltas:  array<vec3<f32>, 52 * V>` storage buffer of per-vertex
 *     per-channel position offsets (52 deltas per vertex, V = vertex count)
 *   - `vertex_idx: u32` — the current vertex index
 *
 * Returns the summed offset to add to the vertex position before the
 * model matrix transforms it. Normal handling is symmetric (separate
 * deltas storage buffer); shipped here as `augengine_morph_normal_offset`.
 *
 * NOT YET WIRED into the engine's pipeline factory. Lands when the glTF
 * loader gains morph-target import. Until then this is the canonical
 * shape so downstream callers (and tests) can lock against it.
 */
export const BLENDSHAPE_WGSL_HELPER = /* wgsl */ `
const AUGENGINE_BLENDSHAPE_COUNT: u32 = 52u;

fn augengine_morph_position(
  weights: array<f32, 52>,
  position_deltas: ptr<storage, array<vec3<f32>>, read>,
  vertex_idx: u32,
) -> vec3<f32> {
  var offset = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < AUGENGINE_BLENDSHAPE_COUNT; k = k + 1u) {
    let w = weights[k];
    if (w == 0.0) { continue; }
    let idx = vertex_idx * AUGENGINE_BLENDSHAPE_COUNT + k;
    offset = offset + w * (*position_deltas)[idx];
  }
  return offset;
}

fn augengine_morph_normal(
  weights: array<f32, 52>,
  normal_deltas: ptr<storage, array<vec3<f32>>, read>,
  vertex_idx: u32,
) -> vec3<f32> {
  var offset = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < AUGENGINE_BLENDSHAPE_COUNT; k = k + 1u) {
    let w = weights[k];
    if (w == 0.0) { continue; }
    let idx = vertex_idx * AUGENGINE_BLENDSHAPE_COUNT + k;
    offset = offset + w * (*normal_deltas)[idx];
  }
  return offset;
}
`;

/** Bind group layout descriptor for the blendshape uniform — read-only
 *  in the vertex stage. Returned as a value so callers slot it into
 *  their pipeline's bind group layout list. */
export const BLENDSHAPE_BIND_GROUP_ENTRY = {
  binding: 0,
  visibility: 1, // GPUShaderStage.VERTEX
  buffer: { type: "uniform" as const },
};

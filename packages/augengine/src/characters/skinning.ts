/**
 * Skinning — per-joint world matrix → bone matrix array → vertex shader
 * (skin matrix = sum_j weight[j] * boneMatrix[j]).
 *
 * Phase 1 / 2: CPU-side matrix assembly producing a Float32Array suitable
 * for upload to a WebGPU storage buffer. Phase 9 swaps in a compute-shader
 * path for >100-character scenes.
 *
 * glTF 2.0 skin: { joints: nodeIndex[], inverseBindMatrices: mat4[] }.
 * Per joint j the bone matrix is:
 *   boneMatrix[j] = jointWorldMatrix[j] * inverseBindMatrix[j]
 */

import type { EntityId, SkinComponent } from "../scene/types.js";
import { m4Mul } from "../scene/math.js";
import { getWorldMatrix, type TransformWorldCache } from "../scene/transform.js";

/**
 * Build the bone matrix array for a skin in the order of its joints.
 * Returns a flat Float32Array (16 * jointCount). Allocation-free if `out`
 * provided.
 */
export function buildBoneMatrices(
  skin: SkinComponent,
  worldCache: TransformWorldCache,
  out?: Float32Array,
): Float32Array {
  const n = skin.jointEntities.length;
  const buf = out && out.length >= n * 16 ? out : new Float32Array(n * 16);
  const scratch = new Float32Array(16);
  for (let j = 0; j < n; j++) {
    const jointId = skin.jointEntities[j]! as EntityId;
    const worldMat = getWorldMatrix(worldCache, jointId);
    if (!worldMat) {
      // Fallback to identity bone (joint not present in transform cache).
      buf.set(IDENTITY16, j * 16);
      continue;
    }
    const ibm = skin.inverseBindMatrices.subarray(j * 16, j * 16 + 16);
    m4Mul(scratch, worldMat as unknown as Float32Array, ibm as unknown as Float32Array);
    buf.set(scratch, j * 16);
  }
  return buf;
}

const IDENTITY16 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/**
 * Linear-blend skinning of a single vertex by bone matrices + per-vertex
 * joints/weights. CPU reference path used by tests; the GPU path lives in
 * the WebGPU vertex shader and consumes the same bone matrix buffer.
 */
export function skinVertexLBS(
  position: Float32Array,
  jointIndices: Uint16Array,
  jointWeights: Float32Array,
  boneMatrices: Float32Array,
): [number, number, number] {
  const px = position[0]!, py = position[1]!, pz = position[2]!;
  let ox = 0, oy = 0, oz = 0;
  for (let k = 0; k < 4; k++) {
    const w = jointWeights[k] ?? 0;
    if (w <= 0) continue;
    const j = jointIndices[k]!;
    const bm = boneMatrices.subarray(j * 16, j * 16 + 16);
    const x = bm[0]! * px + bm[4]! * py + bm[8]! * pz + bm[12]!;
    const y = bm[1]! * px + bm[5]! * py + bm[9]! * pz + bm[13]!;
    const z = bm[2]! * px + bm[6]! * py + bm[10]! * pz + bm[14]!;
    ox += w * x;
    oy += w * y;
    oz += w * z;
  }
  return [ox, oy, oz];
}

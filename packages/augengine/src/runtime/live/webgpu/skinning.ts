/**
 * WebGPU skinning compute path — Phase 3.
 *
 * The compute shader takes:
 *   - Per-frame joint matrices (one per skin joint)
 *   - Per-vertex source positions + normals (rest pose)
 *   - Per-vertex joint indices + weights
 * and writes:
 *   - Per-vertex skinned positions + normals
 *
 * The vertex shader then reads the SKINNED buffer instead of the rest
 * buffer. This avoids per-vertex skinning math in the vertex shader,
 * which becomes a bottleneck at >100 characters.
 *
 * Phase 3 ships the WGSL source + the JS-side bone-matrix builder.
 * Wiring this into the LiveRenderer's pipeline graph happens in the
 * Phase 9 GPU refactor.
 */

import {
  type Mat4,
  type Vec3,
  type Quat,
  type EntityId,
  Tag,
} from "../../../scene/types.js";
import { m4FromTRS, m4Mul, m4Identity } from "../../../scene/math.js";
import type { Scene } from "../../../scene/scene.js";

/** WGSL — skinning compute shader. Workgroup size 64 over vertex count. */
export const SKINNING_COMPUTE_WGSL = /* wgsl */ `
struct Vertex {
  // Interleaved layout — (px, py, pz, nx, ny, nz)
  data: array<f32, 6>,
};

struct VertexAttrs {
  // (j0, j1, j2, j3) packed as u32 from u8x4 OR u16x4
  joints: vec4<u32>,
  weights: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       v_in:      array<Vertex>;
@group(0) @binding(1) var<storage, read>       v_attrs:   array<VertexAttrs>;
@group(0) @binding(2) var<storage, read>       bone_mats: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> v_out:     array<Vertex>;
@group(0) @binding(4) var<uniform>             u_count:   u32;

fn skin_matrix(idx: u32) -> mat4x4<f32> {
  let a = v_attrs[idx];
  return
    bone_mats[a.joints.x] * a.weights.x +
    bone_mats[a.joints.y] * a.weights.y +
    bone_mats[a.joints.z] * a.weights.z +
    bone_mats[a.joints.w] * a.weights.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u_count) { return; }

  let m = skin_matrix(i);

  let src = v_in[i];
  let pos = vec4<f32>(src.data[0], src.data[1], src.data[2], 1.0);
  let nrm = vec4<f32>(src.data[3], src.data[4], src.data[5], 0.0);

  let outPos = m * pos;
  let outNrm = normalize((m * nrm).xyz);

  v_out[i].data[0] = outPos.x;
  v_out[i].data[1] = outPos.y;
  v_out[i].data[2] = outPos.z;
  v_out[i].data[3] = outNrm.x;
  v_out[i].data[4] = outNrm.y;
  v_out[i].data[5] = outNrm.z;
}
`;

/**
 * UV-preserving skinning variant for textured glTF meshes. The compute pass
 * transforms position/normal data while forwarding TEXCOORD_0 unchanged so
 * the subsequent PBR fragment stage can sample the original texture.
 */
export const SKINNING_COMPUTE_WGSL_UV = SKINNING_COMPUTE_WGSL
  .replace("array<f32, 6>", "array<f32, 8>")
  .replace(
    "v_out[i].data[5] = outNrm.z;",
    "v_out[i].data[5] = outNrm.z;\n  v_out[i].data[6] = src.data[6];\n  v_out[i].data[7] = src.data[7];",
  );

// ──────────────────────────────────────────────────────────────────────
// Bone matrix builder — pure CPU, no GPU dependency. Tested independently.
const boneMulScratch = new Float32Array(16);
// ──────────────────────────────────────────────────────────────────────

/**
 * Build the bone matrix array for a Skin component using the scene's
 * world-transform cache. Result is uploaded to the compute shader's
 * `bone_mats` storage buffer.
 *
 *   bone_matrix[j] = jointWorldMatrix[j] * inverseBindMatrix[j]
 *
 * Allocation-free if `out` is provided.
 */
export function buildBoneMatricesGltf(opts: {
  scene: Scene;
  jointEntityIds: EntityId[];
  inverseBindMatrices: Float32Array;
  out?: Float32Array;
}): Float32Array {
  const { scene, jointEntityIds, inverseBindMatrices } = opts;
  const n = jointEntityIds.length;
  const buf = opts.out && opts.out.length >= n * 16
    ? opts.out
    : new Float32Array(n * 16);

  const wt = scene.worldTransforms();
  // One scratch for the joint world copy. The multiply is synchronous.
  const scratch = boneMulScratch;

  for (let j = 0; j < n; j++) {
    const jointId = jointEntityIds[j]!;
    const idx = wt.indexById.get(jointId);
    if (idx === undefined) {
      m4Identity(scratch as Mat4);
    } else {
      scratch.set(wt.worldMatrices.subarray(idx * 16, idx * 16 + 16));
    }
    const ibm = inverseBindMatrices.subarray(j * 16, j * 16 + 16) as Mat4;
    const dest = buf.subarray(j * 16, j * 16 + 16) as Mat4;
    m4Mul(dest, scratch as Mat4, ibm);
  }
  return buf;
}

/**
 * Apply a glTF animation track to the scene by setting joint local
 * transforms. Caller decides which time to sample (looped via clipTime
 * in animation.ts, or driven by external mocap data).
 */
export function applyAnimationTrack(opts: {
  scene: Scene;
  /** glTF node index → AugEngine entity id. Built once at load time. */
  nodeToEntity: Map<number, EntityId>;
  node: number;
  path: "translation" | "rotation" | "scale";
  value: Vec3 | Quat;
}): boolean {
  const { scene, nodeToEntity, node, path, value } = opts;
  const id = nodeToEntity.get(node);
  if (!id) return false;
  const entity = scene.get(id);
  if (!entity) return false;
  const tx = entity.components[Tag.Transform];
  if (!tx) return false;
  const next = { ...tx };
  if (path === "translation") next.position = value as Vec3;
  else if (path === "scale") next.scale = value as Vec3;
  else if (path === "rotation") next.rotation = value as Quat;
  scene.setLocalTransform(id, next.position, next.rotation, next.scale);
  return true;
}

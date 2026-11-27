/**
 * Skinning joint-palette correctness tests — Track 2 of the AugEngine
 * 9-track plan (2026-05-28). These exercise the CPU-side bone-matrix
 * builder consumed by the WebGPU skinning compute pass:
 *
 *     bone_matrix[j] = jointWorldMatrix[j] * inverseBindMatrix[j]
 *
 * Math conventions sanity-checked here:
 *   - All matrices are column-major Float32Array(16), consistent with the
 *     WGSL `mat4x4<f32>` storage-buffer interpretation (one matrix per
 *     16 contiguous floats, columns walking down the buffer). The renderer
 *     uploads the buffer verbatim with `device.queue.writeBuffer(bone_mats,
 *     ...)`, so any row-major confusion would surface as a sheared mesh.
 *   - At bind pose (every joint's world transform equals the joint's bind
 *     transform), bone matrices MUST be identity — otherwise the rest
 *     pose visibly deforms even without any animation playing. This is
 *     the load-bearing "did I get the inverse-bind direction right"
 *     check.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../../scene/scene.js";
import { Tag, type EntityId, type Entity } from "../../../scene/types.js";
import {
  m4Identity,
  m4Invert,
  type MutQuat,
  type MutVec3,
} from "../../../scene/math.js";
import {
  buildBoneMatricesGltf,
  SKINNING_COMPUTE_WGSL,
  SKINNING_COMPUTE_WGSL_UV,
} from "./skinning.js";

function makeScene(entities: Entity[]): Scene {
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "skinning-test" },
    entities,
  });
}

function transformEntity(
  id: string,
  position: MutVec3,
  rotation: MutQuat = [0, 0, 0, 1],
  scale: MutVec3 = [1, 1, 1],
  parent: string | null = null,
): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: {
        position,
        rotation,
        scale,
        parent: parent as EntityId | null,
      },
    },
  };
}

/** Pack an inverseBindMatrices buffer as the inverse of each joint's
 *  bind-pose world matrix — the standard glTF skin baking. */
function makeInverseBindMatrices(jointWorldMatrices: Float32Array[]): Float32Array {
  const out = new Float32Array(jointWorldMatrices.length * 16);
  const inv = new Float32Array(16);
  for (let j = 0; j < jointWorldMatrices.length; j++) {
    const inverted = m4Invert(inv, jointWorldMatrices[j]!);
    if (!inverted) throw new Error(`joint ${j}: world matrix is singular`);
    out.set(inv, j * 16);
  }
  return out;
}

describe("augengine / webgpu skinning — buildBoneMatricesGltf", () => {
  it("produces identity bone matrices at bind pose (jointWorld == bindWorld)", () => {
    // Two unparented joints at fixed world positions.
    const hip = transformEntity("hip", [0, 1.0, 0]);
    const knee = transformEntity("knee", [0, 0.5, 0]);
    const scene = makeScene([hip, knee]);

    // Bind-pose world matrices == the joints' current transforms (no
    // animation applied), so inverseBindMatrices = inverse(world).
    const wt = scene.worldTransforms();
    const bindHip = wt.worldMatrices.slice(0, 16);
    const bindKnee = wt.worldMatrices.slice(16, 32);
    const inverseBindMatrices = makeInverseBindMatrices([bindHip, bindKnee]);

    const bone = buildBoneMatricesGltf({
      scene,
      jointEntityIds: ["hip" as EntityId, "knee" as EntityId],
      inverseBindMatrices,
    });

    // Each bone matrix must be identity within float epsilon — anything
    // else means the mesh would visibly deform at rest. This is the
    // load-bearing "is row-major confusion present" check.
    const ident = m4Identity(new Float32Array(16));
    for (let j = 0; j < 2; j++) {
      const got = bone.subarray(j * 16, j * 16 + 16);
      for (let k = 0; k < 16; k++) {
        expect(got[k]).toBeCloseTo(ident[k]!, 5);
      }
    }
  });

  it("translates by the joint's delta-from-bind in column-major layout", () => {
    // One joint at bind position (0, 1, 0). At runtime we move it by
    // (+0.5, 0, 0). The bone matrix should encode a +0.5 X translation
    // in column-major position (index 12), proving the renderer uploads
    // matrices the WGSL `mat4x4<f32>` interprets correctly.
    const joint = transformEntity("j0", [0, 1.0, 0]);
    const scene = makeScene([joint]);

    const wtBind = scene.worldTransforms();
    const bindMat = wtBind.worldMatrices.slice(0, 16);
    const ibm = makeInverseBindMatrices([bindMat]);

    // Move the joint at runtime.
    scene.setLocalTransform("j0" as EntityId, [0.5, 1.0, 0], [0, 0, 0, 1], [1, 1, 1]);

    const bone = buildBoneMatricesGltf({
      scene,
      jointEntityIds: ["j0" as EntityId],
      inverseBindMatrices: ibm,
    });

    // Column-major translation row lives at indices 12/13/14.
    expect(bone[12]).toBeCloseTo(0.5, 5);
    expect(bone[13]).toBeCloseTo(0, 5);
    expect(bone[14]).toBeCloseTo(0, 5);
    // Top-left 3×3 rotation block stays identity.
    expect(bone[0]).toBeCloseTo(1, 5);
    expect(bone[5]).toBeCloseTo(1, 5);
    expect(bone[10]).toBeCloseTo(1, 5);
  });

  it("returns identity bone for joints missing from the world cache", () => {
    // No entity registered for 'ghost'. The builder must fall back to
    // identity so the WGSL skin sum is well-defined (weights still sum
    // to 1, just the missing joint contributes the rest pose).
    const scene = makeScene([transformEntity("real", [0, 0, 0])]);
    const ibm = new Float32Array(2 * 16);
    // Identity IBMs for both slots.
    ibm[0] = ibm[5] = ibm[10] = ibm[15] = 1;
    ibm[16 + 0] = ibm[16 + 5] = ibm[16 + 10] = ibm[16 + 15] = 1;

    const bone = buildBoneMatricesGltf({
      scene,
      jointEntityIds: ["real" as EntityId, "ghost" as EntityId],
      inverseBindMatrices: ibm,
    });

    // Slot 1 (ghost) -> identity matrix.
    const ghost = bone.subarray(16, 32);
    expect(ghost[0]).toBeCloseTo(1, 5);
    expect(ghost[5]).toBeCloseTo(1, 5);
    expect(ghost[10]).toBeCloseTo(1, 5);
    expect(ghost[12]).toBeCloseTo(0, 5);
  });

  it("reuses the provided output Float32Array (allocation-free path)", () => {
    const j = transformEntity("j", [0, 0, 0]);
    const scene = makeScene([j]);
    const ibm = new Float32Array(16);
    ibm[0] = ibm[5] = ibm[10] = ibm[15] = 1;
    const out = new Float32Array(16);
    const got = buildBoneMatricesGltf({
      scene,
      jointEntityIds: ["j" as EntityId],
      inverseBindMatrices: ibm,
      out,
    });
    expect(got).toBe(out);
  });
});

describe("augengine / webgpu skinning — WGSL source sanity", () => {
  it("compute shader declares the canonical 5-binding layout", () => {
    // The renderer's bind group wiring assumes bindings 0..4 with these
    // semantics. A drift in binding order would silently mis-bind the
    // bone-matrix buffer to vertex-input or vertex-output → garbled mesh.
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@group\(0\) @binding\(0\)\s+var<storage,\s*read>\s+v_in:/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@group\(0\) @binding\(1\)\s+var<storage,\s*read>\s+v_attrs:/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@group\(0\) @binding\(2\)\s+var<storage,\s*read>\s+bone_mats:/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@group\(0\) @binding\(3\)\s+var<storage,\s*read_write>\s+v_out:/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@group\(0\) @binding\(4\)\s+var<uniform>\s+u_count:/);
  });

  it("compute shader applies linear blend skinning (Σ wᵢ·Mᵢ·v)", () => {
    // Trust but verify the math kernel — the linear-blend sum + matrix
    // multiply must appear textually. Anything else (DQS, etc.) would
    // require uploading per-vertex tangents and a different attribute
    // packing.
    expect(SKINNING_COMPUTE_WGSL).toMatch(/bone_mats\[a\.joints\.x\]\s*\*\s*a\.weights\.x/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/m\s*\*\s*pos/);
    expect(SKINNING_COMPUTE_WGSL).toMatch(/@compute @workgroup_size\(64\)/);
  });

  it("UV-preserving variant forwards TEXCOORD_0 untouched", () => {
    // Track 2 + Track 3 share this pipeline — the morphed/skinned mesh
    // must still sample the right pixel on the PBR base-colour texture.
    expect(SKINNING_COMPUTE_WGSL_UV).toMatch(/array<f32,\s*8>/);
    expect(SKINNING_COMPUTE_WGSL_UV).toMatch(/data\[6\]\s*=\s*src\.data\[6\]/);
    expect(SKINNING_COMPUTE_WGSL_UV).toMatch(/data\[7\]\s*=\s*src\.data\[7\]/);
  });
});

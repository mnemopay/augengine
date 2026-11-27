/**
 * Skinned-cube integration test — Track 2 of the AugEngine 9-track plan.
 *
 * End-to-end exercise of the skin + animation pipeline (CPU side):
 *   1. Build an in-memory GLB with 2 joints + 1 rotation channel
 *      (skinned-cube-fixture.ts).
 *   2. Parse via loadGlb (the same path the live renderer uses).
 *   3. Build a GltfSceneInstance — verifies Tag.Skin lands on the mesh
 *      entity with the right joint list.
 *   4. Sample the clip at t=0 → bone matrices are identity (bind pose).
 *   5. Sample at t=1 → child-joint bone matrix encodes a 90° Y rotation,
 *      i.e. non-identity, validating that animation + skin compose.
 *
 * No GPU dependency — exercises `buildBoneMatricesGltf` directly so this
 * test runs in plain vitest. The Playwright e2e in
 * `e2e/proof-room-capture.spec.ts` covers the WebGPU compute pass.
 */

import { describe, it, expect } from "vitest";
import { loadGlb } from "../../../scene/loaders/gltf.js";
import {
  buildGltfSceneInstance,
  applyGltfAnimation,
} from "../../../scene/loaders/gltf-instance.js";
import { Scene } from "../../../scene/scene.js";
import { Tag, type EntityId } from "../../../scene/types.js";
import { buildBoneMatricesGltf } from "./skinning.js";
import { buildSkinnedCubeGlb } from "./skinned-cube-fixture.js";

describe("augengine / skinned-cube end-to-end (Track 2)", () => {
  it("the fixture round-trips through parseGlb → loadGlb with skin + animation intact", () => {
    const { bytes, meta } = buildSkinnedCubeGlb();
    const loaded = loadGlb(bytes);

    expect(loaded.skin).not.toBeNull();
    expect(loaded.skin!.joints).toEqual([1, 2]);
    expect(loaded.skin!.inverseBindMatrices.length).toBe(2 * 16);

    expect(loaded.animations).toHaveLength(1);
    expect(loaded.animations[0]!.name).toBe(meta.clipName);
    expect(loaded.animations[0]!.duration).toBeCloseTo(meta.clipDuration, 4);
    expect(loaded.animations[0]!.tracks).toHaveLength(1);
    expect(loaded.animations[0]!.tracks[0]!.path).toBe("rotation");

    expect(loaded.mesh).not.toBeNull();
    expect(loaded.mesh!.vertexCount).toBe(meta.vertexCount);
    expect(loaded.mesh!.indexCount).toBe(meta.indexCount);
    expect(loaded.mesh!.joints).toBeDefined();
    expect(loaded.mesh!.weights).toBeDefined();
  });

  it("buildGltfSceneInstance wires Tag.Skin onto the mesh entity", () => {
    const { bytes } = buildSkinnedCubeGlb();
    const loaded = loadGlb(bytes);
    const instance = buildGltfSceneInstance(loaded, {
      idPrefix: "cube",
      url: "test://cube.glb",
    });
    const meshEntity = instance.entities.find((e) => e.id === "cube_node_0");
    expect(meshEntity).toBeTruthy();
    const skin = meshEntity!.components[Tag.Skin]!;
    expect(skin.jointEntities).toEqual(["cube_node_1", "cube_node_2"]);
    expect(instance.jointEntityIds).toEqual(["cube_node_1", "cube_node_2"]);
  });

  it("bone matrices at t=0 are identity (bind pose === t=0 of the clip)", () => {
    const { bytes } = buildSkinnedCubeGlb();
    const loaded = loadGlb(bytes);
    const instance = buildGltfSceneInstance(loaded, {
      idPrefix: "cube",
      url: "test://cube.glb",
    });
    const scene = new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: instance.entities,
    });

    // Drive the clip at t=0 — first keyframe is identity rotation, so
    // the child joint stays at its bind transform and bone matrices
    // should be identity.
    applyGltfAnimation(scene, instance, loaded.animations[0]!, 0);

    const bone = buildBoneMatricesGltf({
      scene,
      jointEntityIds: instance.jointEntityIds,
      inverseBindMatrices: loaded.skin!.inverseBindMatrices,
    });

    for (let j = 0; j < instance.jointEntityIds.length; j++) {
      const m = bone.subarray(j * 16, j * 16 + 16);
      // Identity matrix diagonal.
      expect(m[0]).toBeCloseTo(1, 5);
      expect(m[5]).toBeCloseTo(1, 5);
      expect(m[10]).toBeCloseTo(1, 5);
      expect(m[15]).toBeCloseTo(1, 5);
      // Off-diagonal stays at zero.
      expect(m[1]).toBeCloseTo(0, 5);
      expect(m[4]).toBeCloseTo(0, 5);
      // Translation zero (joints sit at origin).
      expect(m[12]).toBeCloseTo(0, 5);
      expect(m[13]).toBeCloseTo(0, 5);
      expect(m[14]).toBeCloseTo(0, 5);
    }
  });

  it("bone matrices at t=1.0s encode the child joint's 90° Y rotation (animation drives skin)", () => {
    const { bytes } = buildSkinnedCubeGlb();
    const loaded = loadGlb(bytes);
    const instance = buildGltfSceneInstance(loaded, {
      idPrefix: "cube",
      url: "test://cube.glb",
    });
    const scene = new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: instance.entities,
    });

    // Run the clip non-looped so t=1 maps to the second keyframe value
    // (90° about Y, quat = (0, 0.7071068, 0, 0.7071068)).
    applyGltfAnimation(scene, instance, loaded.animations[0]!, 1.0, { loop: false });

    // Confirm the child joint's local rotation actually got written.
    const tipEntity = scene.get("cube_node_2" as EntityId)!;
    const tipRot = tipEntity.components[Tag.Transform]!.rotation;
    expect(tipRot[1]).toBeCloseTo(0.7071068, 4);
    expect(tipRot[3]).toBeCloseTo(0.7071068, 4);

    const bone = buildBoneMatricesGltf({
      scene,
      jointEntityIds: instance.jointEntityIds,
      inverseBindMatrices: loaded.skin!.inverseBindMatrices,
    });

    // Joint 0 (root) should still be identity — nothing animated it.
    const root = bone.subarray(0, 16);
    expect(root[0]).toBeCloseTo(1, 5);
    expect(root[5]).toBeCloseTo(1, 5);
    expect(root[10]).toBeCloseTo(1, 5);

    // Joint 1 (tip) must be NON-identity — this is the load-bearing
    // assertion that "the animation actually changes the skin palette
    // the WGSL compute shader will read."
    const tip = bone.subarray(16, 32);
    // 90° about Y → bone[0] = cos(90°) = 0, bone[10] = cos(90°) = 0,
    //                bone[2] = -sin(90°) = -1, bone[8] = sin(90°) = 1.
    // (Column-major: column j sits at bone[j*4..j*4+3].)
    expect(tip[0]).toBeCloseTo(0, 4);
    expect(tip[10]).toBeCloseTo(0, 4);
    // Off-diagonal sign tracks the Y-rotation handedness.
    const offDiagMagnitude = Math.hypot(tip[2]!, tip[8]!);
    expect(offDiagMagnitude).toBeCloseTo(Math.sqrt(2), 3);
  });

  it("driving the clip mid-frame produces an interpolated joint rotation (not snapped)", () => {
    // Halfway through the 1-second clip → slerp(identity, 90°, 0.5) = 45°.
    const { bytes } = buildSkinnedCubeGlb();
    const loaded = loadGlb(bytes);
    const instance = buildGltfSceneInstance(loaded, {
      idPrefix: "cube",
      url: "test://cube.glb",
    });
    const scene = new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: instance.entities,
    });
    applyGltfAnimation(scene, instance, loaded.animations[0]!, 0.5, { loop: false });
    const tipRot = scene.get("cube_node_2" as EntityId)!.components[Tag.Transform]!.rotation;
    // 45° about Y → (0, sin(22.5°), 0, cos(22.5°)).
    expect(tipRot[1]).toBeCloseTo(Math.sin(Math.PI / 8), 3);
    expect(tipRot[3]).toBeCloseTo(Math.cos(Math.PI / 8), 3);
  });
});

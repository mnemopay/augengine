import { describe, expect, it } from "vitest";
import { Scene } from "../scene.js";
import { Tag, type EntityId } from "../types.js";
import type { LoadedGlb } from "./gltf.js";
import {
  applyGltfAnimation,
  buildGltfSceneInstance,
  sampleGltfAnimationTrack,
} from "./gltf-instance.js";

function loadedFixture(): LoadedGlb {
  return {
    parsed: {
      bin: null,
      root: {
        asset: { version: "2.0" },
        nodes: [
          { name: "maya", children: [1] },
          { name: "mesh", mesh: 0, skin: 0, children: [2] },
          { name: "hips", translation: [0, 0.65, 0], children: [3] },
          { name: "head", translation: [0, 0.6, 0] },
        ],
        meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
        skins: [{ joints: [2, 3] }],
      },
    },
    mesh: null,
    primitives: [{
      primitiveIndex: 0,
      meshIndex: 0,
      meshPrimitiveIndex: 0,
      mesh: {
        vertices: new Float32Array(),
        vertexStride: 6,
        indices: new Uint32Array(),
        vertexCount: 0,
        indexCount: 0,
      },
    }],
    skin: {
      joints: [2, 3],
      inverseBindMatrices: new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]),
    },
    skins: [{
      joints: [2, 3],
      inverseBindMatrices: new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]),
    }],
    animations: [{
      name: "nod",
      duration: 1,
      tracks: [{
        node: 3,
        path: "rotation",
        interpolation: "LINEAR",
        times: new Float32Array([0, 1]),
        values: new Float32Array([0, 0, 0, 1, 0, 1, 0, 0]),
      }],
    }],
  };
}

describe("gltf scene instance", () => {
  it("builds mesh, joint, and skin entities from a loaded glb", () => {
    const instance = buildGltfSceneInstance(loadedFixture(), {
      idPrefix: "maya",
      url: "/models/maya.glb",
      position: [1, 2, 3],
    });
    const mesh = instance.entities.find((entity) => entity.id === "maya_node_1");
    expect(instance.meshEntityIds).toEqual(["maya_node_1"]);
    expect(instance.jointEntityIds).toEqual(["maya_node_2", "maya_node_3"]);
    expect(mesh?.components[Tag.Mesh]).toMatchObject({
      geometry: { kind: "gltf", url: "/models/maya.glb", primitiveIndex: 0 },
    });
    expect(mesh?.components[Tag.Skin]).toMatchObject({
      jointEntities: ["maya_node_2", "maya_node_3"],
      rootEntity: "maya_root",
    });
  });

  it("samples and applies gltf animation tracks to scene joints", () => {
    const loaded = loadedFixture();
    const instance = buildGltfSceneInstance(loaded, {
      idPrefix: "maya",
      url: "/models/maya.glb",
    });
    const scene = new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: instance.entities,
      defaultCamera: null,
    });
    applyGltfAnimation(scene, instance, loaded.animations[0]!, 0.5);
    const head = scene.get("maya_node_3" as EntityId)!;
    const rot = head.components[Tag.Transform]!.rotation;
    expect(rot[1]).toBeGreaterThan(0.6);
    expect(rot[3]).toBeGreaterThan(0.6);
  });

  it("binds each skinned mesh node to its OWN skin (multi-skin character)", () => {
    const ibm0 = new Float32Array(32).fill(0.1);
    const ibm1 = new Float32Array(32).fill(0.9);
    const loaded: LoadedGlb = {
      parsed: {
        bin: null,
        root: {
          asset: { version: "2.0" },
          nodes: [
            { name: "body", mesh: 0, skin: 0 },
            { name: "head", mesh: 1, skin: 1 },
            { name: "hips", translation: [0, 0.65, 0] },
            { name: "neck", translation: [0, 0.6, 0] },
          ],
          meshes: [
            { primitives: [{ attributes: { POSITION: 0 } }] },
            { primitives: [{ attributes: { POSITION: 0 } }] },
          ],
          skins: [{ joints: [2, 3] }, { joints: [2, 3] }],
        },
      },
      mesh: null,
      primitives: [
        { primitiveIndex: 0, meshIndex: 0, meshPrimitiveIndex: 0, mesh: { vertices: new Float32Array(), vertexStride: 6, indices: new Uint32Array(), vertexCount: 0, indexCount: 0 } },
        { primitiveIndex: 1, meshIndex: 1, meshPrimitiveIndex: 0, mesh: { vertices: new Float32Array(), vertexStride: 6, indices: new Uint32Array(), vertexCount: 0, indexCount: 0 } },
      ],
      skin: { joints: [2, 3], inverseBindMatrices: ibm0 },
      skins: [
        { joints: [2, 3], inverseBindMatrices: ibm0 },
        { joints: [2, 3], inverseBindMatrices: ibm1 },
      ],
      animations: [],
    };
    const instance = buildGltfSceneInstance(loaded, { idPrefix: "cy", url: "/c.glb" });
    const body = instance.entities.find((e) => e.id === "cy_node_0");
    const head = instance.entities.find((e) => e.id === "cy_node_1");
    expect(body?.components[Tag.Skin]?.inverseBindMatrices).toBe(ibm0);
    expect(head?.components[Tag.Skin]?.inverseBindMatrices).toBe(ibm1);
  });

  it("handles cubic spline tracks by reading key values instead of tangents", () => {
    const value = sampleGltfAnimationTrack({
      node: 0,
      path: "translation",
      interpolation: "CUBICSPLINE",
      times: new Float32Array([0, 1]),
      values: new Float32Array([
        0, 0, 0, 1, 2, 3, 0, 0, 0,
        0, 0, 0, 3, 4, 5, 0, 0, 0,
      ]),
    }, 1);
    expect(value).toEqual([3, 4, 5]);
  });
});

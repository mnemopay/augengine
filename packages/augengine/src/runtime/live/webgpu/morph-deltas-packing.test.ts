import { describe, it, expect } from "vitest";
import {
  packMorphTargetsForGpu,
  uploadMorphTargets,
  MorphDeltaStorage,
  type MorphTargetSource,
} from "./morph-deltas.js";
import { ARKIT_BLENDSHAPE_COUNT } from "./blendshape-constants.js";
import type { BlendshapeBufferLike, BlendshapeDeviceLike } from "./blendshapes.js";

function fakeDevice() {
  const buffers: number[] = [];
  const device: BlendshapeDeviceLike = {
    createBuffer(desc) {
      buffers.push(desc.size);
      const handle: BlendshapeBufferLike = { destroy() {} };
      return handle;
    },
    queue: { writeBuffer() {} },
  };
  return { device, buffers };
}

describe("packMorphTargetsForGpu", () => {
  it("returns a 52-channel zero buffer when no targets are supplied", () => {
    const out = packMorphTargetsForGpu(4, []);
    expect(out.positionDeltas.length).toBe(4 * ARKIT_BLENDSHAPE_COUNT * 3);
    expect(out.normalDeltas).toBeUndefined();
    for (const v of out.positionDeltas) expect(v).toBe(0);
  });

  it("transposes from per-shape arrays to per-vertex-per-shape layout", () => {
    // 2 verts, 2 shapes.
    const targets: MorphTargetSource[] = [
      // shape 0: vertex 0 offset (1, 0, 0); vertex 1 offset (0, 1, 0)
      { positions: new Float32Array([1, 0, 0, 0, 1, 0]) },
      // shape 1: vertex 0 offset (0, 0, 2); vertex 1 offset (0, 0, 3)
      { positions: new Float32Array([0, 0, 2, 0, 0, 3]) },
    ];
    const out = packMorphTargetsForGpu(2, targets);
    // Layout: [v0_s0_xyz, v0_s1_xyz, ..., v0_s51_xyz, v1_s0_xyz, v1_s1_xyz, ...]
    // v0 s0
    expect(out.positionDeltas[0]).toBe(1);
    expect(out.positionDeltas[1]).toBe(0);
    expect(out.positionDeltas[2]).toBe(0);
    // v0 s1
    expect(out.positionDeltas[3]).toBe(0);
    expect(out.positionDeltas[4]).toBe(0);
    expect(out.positionDeltas[5]).toBe(2);
    // v1 s0 — starts at (ARKIT_BLENDSHAPE_COUNT * 3) = 156 floats in
    const v1Base = ARKIT_BLENDSHAPE_COUNT * 3;
    expect(out.positionDeltas[v1Base + 0]).toBe(0);
    expect(out.positionDeltas[v1Base + 1]).toBe(1);
    expect(out.positionDeltas[v1Base + 2]).toBe(0);
    // v1 s1
    expect(out.positionDeltas[v1Base + 3]).toBe(0);
    expect(out.positionDeltas[v1Base + 4]).toBe(0);
    expect(out.positionDeltas[v1Base + 5]).toBe(3);
  });

  it("pads unused channels (3..51) with zeros", () => {
    const out = packMorphTargetsForGpu(1, [
      { positions: new Float32Array([1, 1, 1]) },
    ]);
    // shape 0 = (1, 1, 1); shapes 1..51 should all be zero.
    for (let k = 1; k < ARKIT_BLENDSHAPE_COUNT; k++) {
      expect(out.positionDeltas[k * 3]).toBe(0);
      expect(out.positionDeltas[k * 3 + 1]).toBe(0);
      expect(out.positionDeltas[k * 3 + 2]).toBe(0);
    }
  });

  it("includes normals when at least one target supplies them", () => {
    const out = packMorphTargetsForGpu(1, [
      { positions: new Float32Array([0, 0, 0]) }, // no normals
      {
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([0.5, 0, 0]),
      },
    ]);
    expect(out.normalDeltas).toBeDefined();
    // Target 0 has no normals → those slots stay zero.
    expect(out.normalDeltas![0]).toBe(0);
    // Target 1's normal lands at slot 1.
    expect(out.normalDeltas![3]).toBe(0.5);
  });

  it("rejects mis-sized position arrays", () => {
    expect(() =>
      packMorphTargetsForGpu(2, [{ positions: new Float32Array([1, 2, 3]) }]),
    ).toThrow(/positions length/);
  });

  it("rejects target count above 52", () => {
    const tooMany: MorphTargetSource[] = [];
    for (let i = 0; i < ARKIT_BLENDSHAPE_COUNT + 1; i++) {
      tooMany.push({ positions: new Float32Array([0, 0, 0]) });
    }
    expect(() => packMorphTargetsForGpu(1, tooMany)).toThrow(/exceeds ARKit max/);
  });

  it("rejects vertexCount <= 0", () => {
    expect(() => packMorphTargetsForGpu(0, [])).toThrow(/positive/);
  });
});

describe("uploadMorphTargets", () => {
  it("no-ops on empty targets", () => {
    const { device, buffers } = fakeDevice();
    const storage = new MorphDeltaStorage(device);
    uploadMorphTargets(storage, "primitive:cube", []);
    expect(buffers).toHaveLength(0);
    expect(storage.size()).toBe(0);
  });

  it("infers vertexCount from the first target + uploads packed deltas", () => {
    const { device, buffers } = fakeDevice();
    const storage = new MorphDeltaStorage(device);
    uploadMorphTargets(storage, "primitive:cube", [
      { positions: new Float32Array([0, 0, 0, 1, 0, 0]) },
    ]);
    // Position and zero-normal buffers, each 2 verts x 52 x 16 = 1664 B.
    expect(buffers).toHaveLength(2);
    expect(buffers[0]).toBe(2 * ARKIT_BLENDSHAPE_COUNT * 16);
    expect(storage.size()).toBe(1);
  });

  it("uploads two buffers when any target supplies normals", () => {
    const { device, buffers } = fakeDevice();
    const storage = new MorphDeltaStorage(device);
    uploadMorphTargets(storage, "k", [
      {
        positions: new Float32Array([0, 0, 0]),
        normals: new Float32Array([1, 0, 0]),
      },
    ]);
    expect(buffers).toHaveLength(2);
  });
});

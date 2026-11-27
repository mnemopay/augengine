import { describe, expect, it } from "vitest";
import { clipToUvVelocity, MOTION_MORPH_WGSL, MOTION_RIGID_WGSL, VELOCITY_NONE } from "./motion.js";
import { buildWgsl } from "./pipeline-factory.js";
import { TAA_WGSL } from "./temporal.js";

describe("motion vectors", () => {
  it("is zero for an unchanged clip position", () => {
    expect(clipToUvVelocity([0.2, -0.4, 0.5, 2], [0.2, -0.4, 0.5, 2])).toEqual([0, 0]);
  });

  it("points right/down in uv when the object moved right/down on screen", () => {
    // ndc x 0 → 0.5 moves uv.x +0.25; ndc y 0 → −0.5 moves uv.y (down) +0.25.
    const [vx, vy] = clipToUvVelocity([0.5, -0.5, 0, 1], [0, 0, 0, 1]);
    expect(vx).toBeCloseTo(0.25, 6);
    expect(vy).toBeCloseTo(0.25, 6);
  });

  it("divides by w (perspective)", () => {
    const [vx] = clipToUvVelocity([1, 0, 0, 2], [0, 0, 0, 1]);
    expect(vx).toBeCloseTo(0.25, 6);
  });

  it("uses a clear sentinel the resolve recognises", () => {
    expect(VELOCITY_NONE).toBeLessThan(-100);
    expect(TAA_WGSL).toContain("vel.x > -100.0");
  });

  it("main-pass and motion-pass clip positions are both @invariant", () => {
    // The motion pass depth-tests less-equal against the main pass, which is
    // only exact when both compute the position identically.
    const main = buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1 });
    const morph = buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1, hasMorphTargets: true });
    expect(main).toContain("@builtin(position) @invariant clipPos");
    expect(morph).toContain("@builtin(position) @invariant clipPos");
    expect(MOTION_RIGID_WGSL).toContain("@builtin(position) @invariant clipPos");
    expect(MOTION_MORPH_WGSL).toContain("@builtin(position) @invariant clipPos");
    // Same expression as the main morph vertex shader.
    expect(morph).toContain("let morphedPos = in.position + posOffset;");
    expect(MOTION_MORPH_WGSL).toContain("let morphedPos = position + off;");
  });

  it("TAA reads flags as bits, so velocity cannot disturb history/AO", () => {
    expect(TAA_WGSL).toContain("(flagBits & 1u) != 0u");
    expect(TAA_WGSL).toContain("(flagBits & 2u) != 0u");
    expect(TAA_WGSL).toContain("(flagBits & 4u) != 0u");
  });
});

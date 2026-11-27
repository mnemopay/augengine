/**
 * Animation interpolation mode tests — Track 2 of the AugEngine 9-track
 * plan (2026-05-28). Exercises the LINEAR / STEP / CUBICSPLINE modes
 * documented in the glTF 2.0 sampler spec via the canonical glTF
 * playback path (`sampleGltfAnimationTrack`) — the same sampler the
 * proof room uses to drive Maya's "maya_breathe" clip.
 *
 * Scope: glTF animation channel/sampler boundaries + bind-pose / t=0
 * equivalence. The authored-clip path (`characters/animation.ts
 * sampleTrack`) is exercised by its own existing test file and runs
 * pure LINEAR slerp — interpolation-mode support there is deferred to
 * the eventual state-machine consumer in a future track.
 */

import { describe, it, expect } from "vitest";
import { sampleGltfAnimationTrack } from "../scene/loaders/gltf-instance.js";

describe("augengine / animation — sampleGltfAnimationTrack boundary times", () => {
  // The proof room loads Maya's "maya_breathe" clip via this sampler.
  // These tests pin the t=0 / t=duration / out-of-range semantics so a
  // refactor doesn't silently drift the proof-room visual.

  const rotTrack = {
    node: 0,
    path: "rotation" as const,
    times: new Float32Array([0, 1]),
    values: new Float32Array([
      0, 0, 0, 1,
      0, 0.7071068, 0, 0.7071068,
    ]),
    interpolation: "LINEAR" as const,
  };

  it("returns the first keyframe at t=0 exactly", () => {
    const r = sampleGltfAnimationTrack(rotTrack, 0)!;
    expect(r[3]).toBeCloseTo(1, 4);
    expect(r[1]).toBeCloseTo(0, 4);
  });

  it("returns the last keyframe at t=duration exactly", () => {
    const r = sampleGltfAnimationTrack(rotTrack, 1)!;
    expect(r[1]).toBeCloseTo(0.7071068, 4);
    expect(r[3]).toBeCloseTo(0.7071068, 4);
  });

  it("clamps to the last keyframe for t > duration (caller wraps if looping)", () => {
    // glTF tracks don't loop intrinsically; the looping wrap happens in
    // applyGltfAnimation. A raw sample past the end clamps.
    const r = sampleGltfAnimationTrack(rotTrack, 5.0)!;
    expect(r[1]).toBeCloseTo(0.7071068, 4);
  });

  it("clamps to the first keyframe for t < 0 (caller wraps if looping)", () => {
    const r = sampleGltfAnimationTrack(rotTrack, -2.0)!;
    expect(r[3]).toBeCloseTo(1, 4);
    expect(r[1]).toBeCloseTo(0, 4);
  });

  it("LINEAR slerps to ~45° at the midpoint between identity and 90° about Y", () => {
    const mid = sampleGltfAnimationTrack(rotTrack, 0.5)!;
    expect(mid[1]).toBeCloseTo(Math.sin(Math.PI / 8), 3);
    expect(mid[3]).toBeCloseTo(Math.cos(Math.PI / 8), 3);
  });

  it("STEP holds the previous keyframe (no interpolation across the bracket)", () => {
    const stepTrack = { ...rotTrack, interpolation: "STEP" as const };
    const mid = sampleGltfAnimationTrack(stepTrack, 0.5)!;
    expect(mid[3]).toBeCloseTo(1, 4); // still identity at t=0.5 with STEP
    expect(mid[1]).toBeCloseTo(0, 4);
  });

  it("STEP still resolves to the keyframe value at exact keyframe times", () => {
    const stepTrack = { ...rotTrack, interpolation: "STEP" as const };
    const at1 = sampleGltfAnimationTrack(stepTrack, 1.0)!;
    expect(at1[1]).toBeCloseTo(0.7071068, 4);
  });

  it("CUBICSPLINE reads the centre value per keyframe (in/out tangents skipped)", () => {
    // glTF CUBICSPLINE packs (in-tangent, value, out-tangent) for every
    // keyframe — 3× the data of LINEAR. The renderer currently treats
    // CUBICSPLINE as the centre value at the keyframe time, matching
    // gltf-instance.test.ts:99-110.
    //
    // Two keyframes packed as 3-vec3 vec3 vec3 each. The decoy tangents
    // sit either side of the centre value (60° about Y at t=1).
    const cubicTrack = {
      node: 0,
      path: "translation" as const,
      times: new Float32Array([0, 1]),
      values: new Float32Array([
        99, 99, 99,  // in-tangent (decoy)
        0, 0, 0,     // value
        99, 99, 99,  // out-tangent (decoy)
        99, 99, 99,  // in-tangent (decoy)
        3, 4, 5,     // value
        99, 99, 99,  // out-tangent (decoy)
      ]),
      interpolation: "CUBICSPLINE" as const,
    };
    const at1 = sampleGltfAnimationTrack(cubicTrack, 1)!;
    expect(at1).toEqual([3, 4, 5]);
    const at0 = sampleGltfAnimationTrack(cubicTrack, 0)!;
    expect(at0).toEqual([0, 0, 0]);
  });

  it("returns null for an empty track (no keyframes to sample)", () => {
    const empty = { ...rotTrack, times: new Float32Array(0), values: new Float32Array(0) };
    expect(sampleGltfAnimationTrack(empty, 0)).toBeNull();
  });
});

describe("augengine / animation — bind pose vs t=0 equivalence", () => {
  it("a clip whose first keyframe is the bind rotation returns the bind quat at t=0", () => {
    // The proof-room "no animation playing" visual must equal the
    // bind-pose rendering. Concretely: if a clip's first keyframe at t=0
    // is the joint's bind rotation, sampling at t=0 must yield the same
    // quaternion — no slerp drift, no normalisation artefacts.
    //
    // Bind rotation chosen as a non-identity quaternion to catch the
    // case where the sampler accidentally returns identity regardless
    // of the keyframe value.
    const bindRot = [0, 0.3826834, 0, 0.9238795] as const; // 45° about Y
    const track = {
      node: 0,
      path: "rotation" as const,
      times: new Float32Array([0, 1]),
      values: new Float32Array([
        bindRot[0], bindRot[1], bindRot[2], bindRot[3],
        0, 0.7071068, 0, 0.7071068, // 90° about Y at t=1
      ]),
      interpolation: "LINEAR" as const,
    };
    const at0 = sampleGltfAnimationTrack(track, 0)!;
    expect(at0[0]).toBeCloseTo(bindRot[0], 4);
    expect(at0[1]).toBeCloseTo(bindRot[1], 4);
    expect(at0[2]).toBeCloseTo(bindRot[2], 4);
    expect(at0[3]).toBeCloseTo(bindRot[3], 4);
  });

  it("a single-keyframe track always yields that keyframe (degenerate clip)", () => {
    // Maya's static-pose fallback: a clip with one keyframe means "hold
    // this pose forever." The sampler must not divide by zero on the
    // (t1 - t0) denominator.
    const track = {
      node: 0,
      path: "rotation" as const,
      times: new Float32Array([0.5]),
      values: new Float32Array([0, 0.7071068, 0, 0.7071068]),
      interpolation: "LINEAR" as const,
    };
    for (const t of [-1, 0, 0.5, 1, 100]) {
      const r = sampleGltfAnimationTrack(track, t)!;
      expect(r[1]).toBeCloseTo(0.7071068, 4);
      expect(r[3]).toBeCloseTo(0.7071068, 4);
    }
  });
});

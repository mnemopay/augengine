/**
 * Phase 9 animation tests — clips, sampling, state machine, blend trees.
 */

import { describe, it, expect } from "vitest";
import {
  sampleTrack,
  clipTime,
  AnimStateMachine,
  type AnimationClip,
  type JointTrack,
  type BlendTree,
  type AnimState,
} from "./animation.js";

function makeIdentityRotTrack(): JointTrack {
  return {
    jointIndex: 0,
    times: new Float32Array([0, 1, 2]),
    rotations: new Float32Array([
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
    ]),
  };
}

function makeRotatingTrack(): JointTrack {
  // 90° around Y → quat (0, sin(45°), 0, cos(45°)) ≈ (0, 0.707, 0, 0.707)
  return {
    jointIndex: 0,
    times: new Float32Array([0, 1]),
    rotations: new Float32Array([
      0, 0, 0, 1,        // identity at t=0
      0, 0.7071068, 0, 0.7071068,  // 90° at t=1
    ]),
  };
}

describe("augengine / animation — clipTime", () => {
  it("loops time within duration", () => {
    const clip: AnimationClip = { name: "loop", duration: 2, tracks: [], loopable: true };
    expect(clipTime(clip, 0)).toBe(0);
    expect(clipTime(clip, 1.5)).toBe(1.5);
    expect(clipTime(clip, 2.5)).toBeCloseTo(0.5);
    expect(clipTime(clip, -0.5)).toBeCloseTo(1.5);
  });

  it("clamps for non-loop clips", () => {
    const clip: AnimationClip = { name: "once", duration: 2, tracks: [], loopable: false };
    expect(clipTime(clip, -1)).toBe(0);
    expect(clipTime(clip, 5)).toBe(2);
  });
});

describe("augengine / animation — sampleTrack", () => {
  it("returns identity rotation across an identity track", () => {
    const tr = makeIdentityRotTrack();
    const r = sampleTrack(tr, 0.5);
    expect(r.rot![3]).toBeCloseTo(1);
    expect(r.rot![0]).toBeCloseTo(0);
  });

  it("interpolates rotation between keyframes", () => {
    const tr = makeRotatingTrack();
    // At t=0.5 should be ~45° around Y → quat (0, sin(22.5°), 0, cos(22.5°))
    const r = sampleTrack(tr, 0.5);
    expect(r.rot![1]).toBeCloseTo(Math.sin(Math.PI / 8), 2);
    expect(r.rot![3]).toBeCloseTo(Math.cos(Math.PI / 8), 2);
  });

  it("clamps to first/last keyframe outside range", () => {
    const tr = makeRotatingTrack();
    const before = sampleTrack(tr, -1);
    expect(before.rot![3]).toBeCloseTo(1);
    const after = sampleTrack(tr, 2);
    expect(after.rot![1]).toBeCloseTo(0.7071068, 4);
  });
});

describe("augengine / animation — state machine", () => {
  function makeIdleClip(): AnimationClip {
    return {
      name: "idle",
      duration: 1,
      tracks: [makeIdentityRotTrack()],
      loopable: true,
    };
  }
  function makeRunClip(): AnimationClip {
    return {
      name: "run",
      duration: 1,
      tracks: [makeRotatingTrack()],
      loopable: true,
    };
  }

  it("starts in the first added state", () => {
    const sm = new AnimStateMachine();
    sm.addState({ name: "idle", source: { kind: "clip", clip: makeIdleClip() }, loop: true });
    sm.addState({ name: "run", source: { kind: "clip", clip: makeRunClip() }, loop: true });
    expect(sm.current).toBe("idle");
  });

  it("transitions when condition is met", () => {
    const sm = new AnimStateMachine();
    sm.addState({ name: "idle", source: { kind: "clip", clip: makeIdleClip() }, loop: true });
    sm.addState({ name: "run", source: { kind: "clip", clip: makeRunClip() }, loop: true });
    sm.addTransition({
      from: "idle",
      to: "run",
      condition: (p) => (p.speed ?? 0) > 0.5,
      blendTime: 0.3,
    });

    sm.setParam("speed", 1);
    sm.step(0.01); // triggers transition
    expect(sm.next).toBe("run");
    sm.step(0.5); // exceeds blend duration
    expect(sm.current).toBe("run");
    expect(sm.next).toBeNull();
  });

  it("evaluates blends current + next during transition", () => {
    const sm = new AnimStateMachine();
    sm.addState({ name: "idle", source: { kind: "clip", clip: makeIdleClip() }, loop: true });
    sm.addState({ name: "run", source: { kind: "clip", clip: makeRunClip() }, loop: true });
    sm.forceTransition("run", 0.4);
    sm.step(0.2); // halfway through transition

    const out = new Float32Array(4);
    sm.evaluate(out, null);
    // Halfway between identity and rotating → should be ~22.5°
    expect(out[3]!).toBeCloseTo(Math.cos(Math.PI / 16), 1);
  });
});

describe("augengine / animation — blend tree (1D)", () => {
  it("interpolates between two clips by 1D parameter", () => {
    const idle: AnimationClip = {
      name: "idle",
      duration: 1,
      tracks: [makeIdentityRotTrack()],
      loopable: true,
    };
    const run: AnimationClip = {
      name: "run",
      duration: 1,
      tracks: [makeRotatingTrack()],
      loopable: true,
    };
    const tree: BlendTree = {
      kind: "1d",
      parameter: "speed",
      children: [
        { clip: idle, position: 0 },
        { clip: run, position: 1 },
      ],
    };
    const state: AnimState = { name: "loco", source: { kind: "blend", tree } };
    const sm = new AnimStateMachine();
    sm.addState(state);
    sm.setParam("speed", 0.5);
    sm.step(0.01);
    const out = new Float32Array(4);
    sm.evaluate(out, null);
    // Halfway blend → ~22.5° around Y (with normalized quat sum)
    expect(out[1]!).toBeGreaterThan(0);
    expect(out[3]!).toBeGreaterThan(0.9);
  });
});

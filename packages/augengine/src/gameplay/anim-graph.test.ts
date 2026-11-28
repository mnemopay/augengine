import { describe, expect, it } from "vitest";
import type { AugEngineAnimationFromGltf, AugEngineAnimationTrack } from "../scene/loaders/gltf.js";
import {
  AnimGraph,
  applyAdditive,
  bindClip,
  blend1DWeights,
  blendPoses,
  boneMask,
  createAnimRig,
  createPose,
  crossFadeWeight,
  findClip,
  rootDelta,
  sampleClip,
  type AnimGraphSpec,
  type AnimRigNode,
} from "./anim-graph.js";

// Armature (Blender-style 0.01 scale) → Hips → Spine → LeftUpperArm, plus a leg off the hips.
const NODES: AnimRigNode[] = [
  { name: "Armature", scale: [0.01, 0.01, 0.01], children: [1] },
  { name: "Hips", translation: [0, 100, 0], children: [2, 4] },
  { name: "Spine", translation: [0, 10, 0], children: [3] },
  { name: "LeftUpperArm", translation: [10, 20, 0] },
  { name: "LeftUpperLeg", translation: [8, -5, 0] },
];

const yQuat = (deg: number): number[] => {
  const h = (deg * Math.PI) / 360;
  return [0, Math.sin(h), 0, Math.cos(h)];
};
const angleY = (q: ArrayLike<number>, o = 0): number => (2 * Math.atan2(q[o + 1]!, q[o + 3]!) * 180) / Math.PI;

function track(node: number, path: AugEngineAnimationTrack["path"], times: number[], values: number[], interpolation: AugEngineAnimationTrack["interpolation"] = "LINEAR"): AugEngineAnimationTrack {
  return { node, path, times: Float32Array.from(times), values: Float32Array.from(values), interpolation };
}

/** Spine yaw goes 0 → deg over the clip; hips travel `travel` (armature units) along +Z. */
function clip(name: string, duration: number, deg: number, travel = 0, bob = 0): AugEngineAnimationFromGltf {
  return {
    name,
    duration,
    tracks: [
      track(2, "rotation", [0, duration], [...yQuat(0), ...yQuat(deg)]),
      track(1, "translation", [0, duration / 2, duration], [0, 100, 0, 0, 100 + bob, travel / 2, 0, 100, travel]),
    ],
  };
}

const CLIPS = [
  clip("Armature|Idle", 2, 10),
  clip("Armature|Walk", 1, 20, 150),
  clip("Armature|Run", 0.5, 40, 200),
  clip("Attack", 0.8, 60),
  clip("Hit", 0.4, -30),
  clip("Death", 1, 90),
];

function graph(extra: Partial<AnimGraphSpec> = {}): AnimGraph {
  const spec: AnimGraphSpec = {
    states: [
      { name: "idle", motion: { clip: "Idle" } },
      { name: "move", motion: { blend1d: { param: "speed", children: [{ clip: "Walk", at: 1.5 }, { clip: "Run", at: 4 }] } }, rootMotion: true },
      { name: "attack", motion: { clip: "Attack", loop: false } },
      { name: "hit", motion: { clip: "Hit", loop: false } },
      { name: "death", motion: { clip: "Death", loop: false } },
    ],
    transitions: [
      { from: "idle", to: "move", when: [{ param: "speed", op: ">", value: 0.1 }], duration: 0.2 },
      { from: "move", to: "idle", when: [{ param: "speed", op: "<=", value: 0.1 }], duration: 0.25 },
      { from: "*", to: "attack", when: [{ trigger: "attack" }], duration: 0.1 },
      { from: "attack", to: "idle", exitTime: 0.9, duration: 0.15 },
      { from: "*", to: "hit", when: [{ trigger: "hit" }], duration: 0.05, allowSelf: true },
      { from: "hit", to: "idle", exitTime: 1, duration: 0.2 },
      { from: "*", to: "death", when: [{ flag: "dead" }], duration: 0.1 },
    ],
    events: { Walk: [{ time: 0.25, name: "footL" }, { time: 0.75, name: "footR" }], Attack: [{ time: 0.35, name: "hit" }], Idle: [{ time: 0, name: "breathe" }] },
    ...extra,
  };
  return new AnimGraph(createAnimRig(NODES), CLIPS, spec);
}

describe("rig + clip binding", () => {
  it("builds rest poses, finds the hips and the armature scale above it", () => {
    const rig = createAnimRig(NODES);
    expect(rig.rootSlot).toBe(1);
    expect(rig.rest.t[1 * 3 + 1]).toBe(100);
    // Armature scale 0.01 maps 100 armature units to 1 model metre.
    expect(rig.rootParentToModel[0]).toBeCloseTo(0.01, 6);
  });

  it("resolves Blender 'Armature|Name' clips by suffix, case-insensitively", () => {
    expect(findClip(CLIPS, "walk")?.name).toBe("Armature|Walk");
    expect(findClip(CLIPS, "Attack")?.name).toBe("Attack");
    expect(findClip(CLIPS, "nope")).toBeUndefined();
  });

  it("samples linear, step and cubic-spline tracks", () => {
    const rig = createAnimRig(NODES);
    const lin = bindClip(rig, { name: "l", duration: 1, tracks: [track(4, "translation", [0, 1], [0, 0, 0, 2, 4, 6])] });
    const step = bindClip(rig, { name: "s", duration: 1, tracks: [track(4, "translation", [0, 1], [0, 0, 0, 2, 4, 6], "STEP")] });
    // CUBICSPLINE stores in-tangent, value, out-tangent per key.
    const cub = bindClip(rig, { name: "c", duration: 1, tracks: [track(4, "translation", [0, 1], [9, 9, 9, 0, 0, 0, 9, 9, 9, 9, 9, 9, 2, 4, 6, 9, 9, 9], "CUBICSPLINE")] });
    const out = createPose(NODES.length);
    sampleClip(lin, 0.25, rig.rest, out);
    expect([...out.t.subarray(12, 15)]).toEqual([0.5, 1, 1.5]);
    sampleClip(step, 0.75, rig.rest, out);
    expect([...out.t.subarray(12, 15)]).toEqual([0, 0, 0]);
    sampleClip(cub, 0.5, rig.rest, out);
    expect([...out.t.subarray(12, 15)]).toEqual([1, 2, 3]);
    // Untouched channels come from the rest pose.
    expect(out.t[1 * 3 + 1]).toBe(100);
  });

  it("boneMask covers a bone's subtree by node name or humanoid id", () => {
    const rig = createAnimRig(NODES);
    expect([...boneMask(rig, ["spine"])]).toEqual([0, 0, 1, 1, 0]);
    expect([...boneMask(rig, ["Hips"])]).toEqual([0, 1, 1, 1, 1]);
  });
});

describe("blend math", () => {
  it("1D weights bracket the parameter and clamp outside the range", () => {
    const w = new Float32Array(3);
    const at = [0, 1.5, 4];
    blend1DWeights(at, 0.75, w);
    expect([...w]).toEqual([0.5, 0.5, 0]);
    blend1DWeights(at, 3.375, w);
    expect(w[1]).toBeCloseTo(0.25, 6);
    expect(w[2]).toBeCloseTo(0.75, 6);
    blend1DWeights(at, -1, w);
    expect([...w]).toEqual([1, 0, 0]);
    blend1DWeights(at, 9, w);
    expect([...w]).toEqual([0, 0, 1]);
    blend1DWeights(at, 1.5, w);
    expect(w[0]! + w[1]! + w[2]!).toBeCloseTo(1, 6);
  });

  it("cross-fade weight is a smoothstep from 0 to 1", () => {
    expect(crossFadeWeight(0, 0.2)).toBe(0);
    expect(crossFadeWeight(0.1, 0.2)).toBeCloseTo(0.5, 6);
    expect(crossFadeWeight(0.05, 0.2)).toBeCloseTo(0.15625, 6);
    expect(crossFadeWeight(0.3, 0.2)).toBe(1);
    expect(crossFadeWeight(0, 0)).toBe(1);
  });

  it("blendPoses halves a 90 degree yaw and respects masks", () => {
    const a = createPose(2), b = createPose(2), out = createPose(2);
    b.r.set(yQuat(90), 0);
    b.r.set(yQuat(90), 4);
    b.t.set([2, 0, 0], 0);
    blendPoses(out, a, b, 0.5, Uint8Array.from([1, 0]));
    expect(angleY(out.r, 0)).toBeCloseTo(45, 4);
    expect(out.t[0]).toBeCloseTo(1, 6);
    expect(angleY(out.r, 4)).toBeCloseTo(0, 6);
  });

  it("blendPoses takes the short way round when quaternions are in opposite hemispheres", () => {
    const a = createPose(1), b = createPose(1), out = createPose(1);
    a.r.set(yQuat(10));
    b.r.set(yQuat(30).map((x) => -x)); // same rotation, negated quaternion
    blendPoses(out, a, b, 0.5);
    expect(angleY(out.r)).toBeCloseTo(20, 3);
  });

  it("additive layer adds the clip's offset from its reference frame", () => {
    const base = createPose(1), clipP = createPose(1), ref = createPose(1), out = createPose(1);
    base.r.set(yQuat(20));
    ref.r.set(yQuat(5));
    clipP.r.set(yQuat(35)); // +30 relative to ref
    clipP.t.set([0, 1, 0]);
    applyAdditive(out, base, clipP, ref, 1);
    expect(angleY(out.r)).toBeCloseTo(50, 3);
    expect(out.t[1]).toBeCloseTo(1, 6);
    applyAdditive(out, base, clipP, ref, 0.5);
    expect(angleY(out.r)).toBeCloseTo(35, 0); // nlerp: within 0.5 degree of 15
    applyAdditive(out, base, ref, ref, 1);
    expect(angleY(out.r)).toBeCloseTo(20, 4);
  });
});

describe("state machine", () => {
  it("cross-fades idle → locomotion on a parameter, with smoothstep weights", () => {
    const g = graph();
    g.update(0.1);
    expect(g.stateName).toBe("idle");
    g.setParam("speed", 2);
    g.update(0.1); // transition starts this update; 0.1 s of a 0.2 s fade elapsed
    expect(g.stateName).toBe("move");
    expect(g.fadingFrom).toBe("idle");
    expect(g.fadeWeight).toBeCloseTo(0.5, 5);
    g.update(0.1);
    expect(g.fadingFrom).toBeNull();
    expect(g.fadeWeight).toBe(1);
    // speed 2 between walk (1.5) and run (4): 0.8 / 0.2.
    expect(g.blendWeights[0]).toBeCloseTo(0.8, 5);
    expect(g.blendWeights[1]).toBeCloseTo(0.2, 5);
  });

  it("the blended pose sits between the two clips at the cross-fade midpoint", () => {
    const g = graph();
    g.update(1); // idle at t=1 → spine yaw 5
    g.setParam("speed", 1.5); // pure walk
    g.update(0.1); // fade w = 0.5; walk at 0.1 s → yaw 2
    const yaw = angleY(g.pose.r, 2 * 4);
    // idle has advanced to 1.1 s → 5.5 degrees; walk 2 degrees; halfway ≈ 3.75
    expect(yaw).toBeCloseTo(3.75, 1);
  });

  it("phase-syncs walk and run so both children share one normalised time", () => {
    const g = graph({ entry: "move" });
    g.setParam("speed", 2.75); // halfway: effective cycle 0.5*1 + 0.5*0.5 = 0.75 s
    g.update(0.375);
    expect(g.stateTime).toBeCloseTo(0.5, 5);
    // Spine yaw = blend of walk at 50% (10) and run at 50% (20) = 15.
    expect(angleY(g.pose.r, 8)).toBeCloseTo(15, 1);
  });

  it("returns from a one-shot at its exit time and consumes the trigger", () => {
    const g = graph();
    g.setTrigger("attack");
    g.update(0.01);
    expect(g.stateName).toBe("attack");
    g.update(0.5);
    expect(g.stateName).toBe("attack"); // norm ≈ 0.64 < 0.9
    g.update(0.25); // norm ≈ 0.95 → transition evaluated next update
    g.update(0.01);
    expect(g.stateName).toBe("idle");
    g.update(0.5);
    expect(g.stateName).toBe("idle"); // trigger was consumed, no re-attack
  });

  it("any-state hit pre-empts and can restart itself; death holds", () => {
    const g = graph({ entry: "move" });
    g.setParam("speed", 3);
    g.update(0.2);
    g.setTrigger("hit");
    g.update(0.2);
    expect(g.stateName).toBe("hit");
    expect(g.stateTime).toBeCloseTo(0.5, 5);
    g.setTrigger("hit");
    g.update(0.1);
    expect(g.stateName).toBe("hit");
    expect(g.stateTime).toBeCloseTo(0.25, 5); // restarted
    g.setFlag("dead", true);
    g.update(0.1);
    expect(g.stateName).toBe("death");
    for (let i = 0; i < 40; i++) g.update(0.1);
    expect(g.stateName).toBe("death"); // no self re-entry, no exit
    expect(angleY(g.pose.r, 8)).toBeCloseTo(90, 1); // held on the last frame
  });

  it("interrupting a cross-fade fades from a frozen snapshot of the blended pose", () => {
    const g = graph();
    g.setParam("speed", 2);
    g.update(0.1);
    expect(g.fadingFrom).toBe("idle");
    g.setTrigger("attack");
    g.update(0.01);
    expect(g.stateName).toBe("attack");
    expect(g.fadingFrom).toBe("(frozen)");
  });

  it("rejects specs that reference missing clips or states", () => {
    const rig = createAnimRig(NODES);
    expect(() => new AnimGraph(rig, CLIPS, { states: [{ name: "a", motion: { clip: "Jump" } }], transitions: [] })).toThrow(/Jump/);
    expect(() => new AnimGraph(rig, CLIPS, { states: [{ name: "a", motion: { clip: "Idle" } }], transitions: [{ from: "a", to: "b" }] })).toThrow(/"b"/);
  });
});

describe("events", () => {
  it("fires clip-time events once per crossing, including several wraps in one long step", () => {
    const g = graph({ entry: "move" });
    g.setParam("speed", 1.5); // pure walk, 1 s cycle, footL 0.25 / footR 0.75
    const names: string[] = [];
    g.onEvent((e) => names.push(e.name));
    g.update(0.3);
    expect(names).toEqual(["footL"]);
    g.update(0.5);
    expect(names).toEqual(["footL", "footR"]);
    names.length = 0;
    g.update(2.0); // 0.8 → 2.8: footL@1.25, footR@1.75, footL@2.25, footR@2.75
    expect(names).toEqual(["footL", "footR", "footL", "footR"]);
  });

  it("fires an event at clip time 0 on the state's first update, and the hit frame of an attack", () => {
    const g = graph();
    const fired = g.update(0.01).map((e) => e.name);
    expect(fired).toEqual(["breathe"]);
    g.setTrigger("attack");
    const seen: string[] = [];
    for (let i = 0; i < 30; i++) for (const e of g.update(1 / 60)) seen.push(`${e.state}:${e.name}`);
    expect(seen).toEqual(["attack:hit"]);
  });
});

describe("root motion", () => {
  it("moves travel out of the hips into rootMotion, scaled into model space", () => {
    const g = graph({ entry: "move" });
    g.setParam("speed", 1.5); // walk: 150 armature units/cycle = 1.5 m
    g.update(0.25);
    expect(g.rootMotion[2]).toBeCloseTo(0.375, 4);
    expect(g.rootMotion[1]).toBe(0);
    // Hips held at the clip-start XZ so the mesh doesn't run away from its root.
    expect(g.pose.t[1 * 3 + 2]).toBeCloseTo(0, 4);
    let total = 0;
    for (let i = 0; i < 30; i++) { g.update(0.1); total += g.rootMotion[2]; }
    expect(total).toBeCloseTo(4.5, 3); // 3 s at 1.5 m/s, across 3 wraps
  });

  it("keeps the vertical hip bob in the pose", () => {
    const rig = createAnimRig(NODES);
    const g = new AnimGraph(rig, [clip("Bob", 1, 0, 100, 4)], {
      states: [{ name: "s", motion: { clip: "Bob" }, rootMotion: true }],
      transitions: [],
    });
    g.update(0.5);
    expect(g.pose.t[1 * 3 + 1]).toBeCloseTo(104, 4);
    expect(g.pose.t[1 * 3 + 2]).toBeCloseTo(0, 4);
  });

  it("rootDelta adds whole cycles when a loop wraps", () => {
    const rig = createAnimRig(NODES);
    const walk = bindClip(rig, CLIPS[1]!);
    const out = new Float32Array(3);
    rootDelta(walk, 0.9, 0.1, 1, out);
    expect(out[2]).toBeCloseTo(30, 4);
    rootDelta(walk, 0.9, 0.1, 2, out);
    expect(out[2]).toBeCloseTo(180, 4);
  });
});

describe("layers", () => {
  it("an additive upper-body layer fades in and out over a one-shot without touching the legs", () => {
    const g = graph({ layers: [{ name: "flinch", mode: "additive", mask: ["spine"] }] });
    g.update(0.5);
    const legBefore = [...g.pose.r.subarray(16, 20)];
    g.playLayer("flinch", "Hit", { fadeIn: 0.05, fadeOut: 0.1 });
    g.update(0.2);
    expect(g.layerWeight("flinch")).toBeCloseTo(1, 5);
    // Hit's spine goes 0 → -30 over 0.4 s; at 0.2 s the offset is -15 from its first frame.
    const idleYaw = (0.7 / 2) * 10;
    expect(angleY(g.pose.r, 8)).toBeCloseTo(idleYaw - 15, 0);
    expect([...g.pose.r.subarray(16, 20)]).toEqual(legBefore);
    g.update(0.15); // inside the last fadeOut window
    expect(g.layerWeight("flinch")).toBeLessThan(1);
    g.update(0.2);
    expect(g.layerWeight("flinch")).toBe(0);
  });

  it("an override layer replaces masked bones by its weight", () => {
    const g = graph({ layers: [{ name: "upper", mode: "override", mask: ["spine"] }] });
    g.playLayer("upper", "Death", { fadeIn: 0, loop: true });
    g.update(0.5); // Death spine at 0.5 s = 45
    expect(angleY(g.pose.r, 8)).toBeCloseTo(45, 2);
  });
});

describe("playback rate parameter", () => {
  it("scales a blend state's phase by speedParam", () => {
    const rig = createAnimRig(NODES);
    const g = new AnimGraph(rig, CLIPS, {
      states: [{ name: "run", motion: { blend1d: { param: "speed", children: [{ clip: "Run", at: 4 }] }, speedParam: "rate" } }],
      transitions: [],
      params: { speed: 4, rate: 2 },
    });
    g.update(0.125); // Run is 0.5 s; at 2x a quarter second is a full cycle
    expect(g.stateTime).toBeCloseTo(0.5, 5);
    g.setParam("rate", 0); // 0 means "not set" → 1x
    g.update(0.125);
    expect(g.stateTime).toBeCloseTo(0.75, 5);
  });
});

describe("any-state exceptions", () => {
  it("skips any-state transitions out of excluded states", () => {
    const rig = createAnimRig(NODES);
    const g = new AnimGraph(rig, CLIPS, {
      states: [{ name: "idle", motion: { clip: "Idle" } }, { name: "attack", motion: { clip: "Attack", loop: false } }, { name: "death", motion: { clip: "Death", loop: false } }],
      transitions: [
        { from: "*", to: "attack", when: [{ trigger: "attack" }], except: ["death"] },
        { from: "*", to: "death", when: [{ flag: "dead" }] },
      ],
    });
    g.setFlag("dead", true);
    g.update(0.1);
    expect(g.stateName).toBe("death");
    g.setTrigger("attack");
    g.update(0.1);
    expect(g.stateName).toBe("death");
  });
});

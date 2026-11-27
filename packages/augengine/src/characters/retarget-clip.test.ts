import { describe, expect, it } from "vitest";
import type { AugEngineAnimationFromGltf } from "../scene/loaders/gltf.js";
import { retargetClip, type RigNode, type RigSource } from "./retarget-clip.js";
import { mFromTRS, mMul, mPos, qAxisAngle, qRotP, vNorm, vSub, type M, type Q, type V } from "./rig-math.js";

/** A small humanoid: hips → spine → neck → head, two arms, two legs. */
function rig(names: Record<string, string>, opts: { armDown?: boolean; yaw?: number; scale?: number } = {}): RigSource {
  const s = opts.scale ?? 1;
  const armRot = (side: 1 | -1): Q => (opts.armDown ? qAxisAngle([0, 0, 1], -side * (Math.PI / 4)) : [0, 0, 0, 1]);
  const n = (key: string, t: V, children: number[] = [], rotation: Q = [0, 0, 0, 1]): RigNode => ({
    name: names[key] ?? key, translation: [t[0] * s, t[1] * s, t[2] * s], rotation, children,
  });
  const nodes: RigNode[] = [
    { name: "Armature", children: [1], rotation: opts.yaw ? qAxisAngle([0, 1, 0], opts.yaw) : [0, 0, 0, 1] }, // 0
    n("hips", [0, 1, 0], [2, 9, 12]), // 1
    n("spine", [0, 0.2, 0], [3]), // 2
    n("chest", [0, 0.2, 0], [4, 5, 7]), // 3
    n("neck", [0, 0.2, 0], [15]), // 4
    n("leftUpperArm", [0.2, 0, 0], [6], armRot(1)), // 5
    n("leftLowerArm", [0.3, 0, 0]), // 6
    n("rightUpperArm", [-0.2, 0, 0], [8], armRot(-1)), // 7
    n("rightLowerArm", [-0.3, 0, 0]), // 8
    n("leftUpperLeg", [0.1, -0.05, 0], [10]), // 9
    n("leftLowerLeg", [0, -0.45, 0], [11]), // 10
    n("leftFoot", [0, -0.45, 0]), // 11
    n("rightUpperLeg", [-0.1, -0.05, 0], [13]), // 12
    n("rightLowerLeg", [0, -0.45, 0], [14]), // 13
    n("rightFoot", [0, -0.45, 0]), // 14
    n("head", [0, 0.15, 0]), // 15
  ];
  return { nodes, joints: nodes.map((_, i) => i).slice(1) };
}

const MIXAMO: Record<string, string> = {
  hips: "mixamorig:Hips", spine: "mixamorig:Spine", chest: "mixamorig:Spine1", neck: "mixamorig:Neck", head: "mixamorig:Head",
  leftUpperArm: "mixamorig:LeftArm", leftLowerArm: "mixamorig:LeftForeArm", rightUpperArm: "mixamorig:RightArm", rightLowerArm: "mixamorig:RightForeArm",
  leftUpperLeg: "mixamorig:LeftUpLeg", leftLowerLeg: "mixamorig:LeftLeg", leftFoot: "mixamorig:LeftFoot",
  rightUpperLeg: "mixamorig:RightUpLeg", rightLowerLeg: "mixamorig:RightLeg", rightFoot: "mixamorig:RightFoot",
};
const QUAT: Record<string, string> = {
  hips: "Hips", spine: "Abdomen", chest: "Torso", neck: "Neck", head: "Head",
  leftUpperArm: "UpperArm.L", leftLowerArm: "LowerArm.L", rightUpperArm: "UpperArm.R", rightLowerArm: "LowerArm.R",
  leftUpperLeg: "UpperLeg.L", leftLowerLeg: "LowerLeg.L", leftFoot: "Foot.L",
  rightUpperLeg: "UpperLeg.R", rightLowerLeg: "LowerLeg.R", rightFoot: "Foot.R",
};

/** Clip: left upper arm rotates from rest to 90° forward-down over 1 s; hips move 1 m forward. */
function raiseArmClip(r: RigSource, armAxis: V): AugEngineAnimationFromGltf {
  const rest = (r.nodes[5]!.rotation ?? [0, 0, 0, 1]) as Q;
  const q1 = qAxisAngle(armAxis, Math.PI / 2);
  const mul = (a: Q, b: Q): Q => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
  const end = mul(rest, q1);
  const hip0 = r.nodes[1]!.translation!;
  return {
    name: "raise",
    duration: 1,
    tracks: [
      { node: 5, path: "rotation", times: new Float32Array([0, 1]), values: new Float32Array([...rest, ...end]), interpolation: "LINEAR" },
      { node: 1, path: "translation", times: new Float32Array([0, 1]), values: new Float32Array([...hip0, hip0[0]!, hip0[1]!, hip0[2]! + 1]), interpolation: "LINEAR" },
    ],
  };
}

/** World positions of every node with a clip applied at time t (test-only forward kinematics). */
function posed(r: RigSource, clip: AugEngineAnimationFromGltf, t: number): V[] {
  const over = new Map<number, { t?: V; r?: Q }>();
  for (const tr of clip.tracks) {
    const times = tr.times;
    let k = 0;
    while (k + 1 < times.length && times[k + 1]! <= t) k++;
    const k2 = Math.min(k + 1, times.length - 1);
    const u = k2 === k ? 0 : (t - times[k]!) / (times[k2]! - times[k]!);
    const c = tr.path === "rotation" ? 4 : 3;
    const v = Array.from({ length: c }, (_, i) => tr.values[k * c + i]! + (tr.values[k2 * c + i]! - tr.values[k * c + i]!) * u);
    const o = over.get(tr.node) ?? {};
    if (tr.path === "rotation") { const l = Math.hypot(...v); o.r = v.map((x) => x / l) as Q; } else o.t = v as V;
    over.set(tr.node, o);
  }
  const parent = new Array<number>(r.nodes.length).fill(-1);
  r.nodes.forEach((n, i) => (n.children ?? []).forEach((c) => { parent[c] = i; }));
  const w: M[] = [];
  const visit = (i: number) => {
    const n = r.nodes[i]!;
    const o = over.get(i);
    const local = mFromTRS(o?.t ?? ((n.translation ?? [0, 0, 0]) as V), o?.r ?? ((n.rotation ?? [0, 0, 0, 1]) as Q), (n.scale ?? [1, 1, 1]) as V);
    w[i] = parent[i]! >= 0 ? mMul(w[parent[i]!]!, local) : local;
    for (const c of n.children ?? []) visit(c);
  };
  visit(0);
  return w.map(mPos);
}

const dir = (p: V[], a: number, b: number): V => vNorm(vSub(p[b]!, p[a]!));
const close = (a: V, b: V, eps = 1e-3) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i]!, Math.round(-Math.log10(eps))));

describe("retargetClip", () => {
  it("same skeleton, different names: reproduces the source motion", () => {
    const src = rig(MIXAMO);
    const tgt = rig(QUAT);
    const clip = raiseArmClip(src, [0, 1, 0]);
    const out = retargetClip({ source: src, target: tgt, clip });
    expect(out.missing).toEqual([]);
    for (const t of [0, 0.5, 1]) {
      const ps = posed(src, clip, t);
      const pt = posed(tgt, out.clip, t);
      close(dir(pt, 5, 6), dir(ps, 5, 6));
      close(pt[1]!, ps[1]!);
    }
  });

  it("corrects rest pose: A-pose target follows a T-pose source", () => {
    const src = rig(MIXAMO); // arms out (T-pose)
    const tgt = rig(QUAT, { armDown: true }); // arms 45° down (A-pose)
    const clip = raiseArmClip(src, [0, 1, 0]);
    const out = retargetClip({ source: src, target: tgt, clip });
    for (const t of [0, 1]) close(dir(posed(tgt, out.clip, t), 5, 6), dir(posed(src, clip, t), 5, 6));
  });

  it("keeps the target's own neck/head posture (no spine correction)", () => {
    const src = rig(MIXAMO);
    // Source neck leans forward 30° in its bind pose; the target's is upright.
    src.nodes[4] = { ...src.nodes[4]!, rotation: qAxisAngle([1, 0, 0], Math.PI / 6) };
    const tgt = rig(QUAT);
    const still: AugEngineAnimationFromGltf = { name: "still", duration: 1, tracks: [] };
    const out = retargetClip({ source: src, target: tgt, clip: still });
    close(dir(posed(tgt, out.clip, 0.5), 4, 15), [0, 1, 0]);
  });

  it("maps motion into the target's facing (target turned 180°)", () => {
    const src = rig(MIXAMO);
    const tgt = rig(QUAT, { yaw: Math.PI });
    const clip = raiseArmClip(src, [0, 1, 0]);
    const out = retargetClip({ source: src, target: tgt, clip });
    const yawFlip = qAxisAngle([0, 1, 0], Math.PI);
    close(dir(posed(tgt, out.clip, 1), 5, 6), qRotP(yawFlip, dir(posed(src, clip, 1), 5, 6)));
    // Hips travel 1 m along the target's own forward (−Z after the turn).
    const p0 = posed(tgt, out.clip, 0)[1]!, p1 = posed(tgt, out.clip, 1)[1]!;
    close(vSub(p1, p0), [0, 0, -1]);
  });

  it("scales hips travel by hip height and supports in-place", () => {
    const src = rig(MIXAMO);
    const tgt = rig(QUAT, { scale: 2 });
    const clip = raiseArmClip(src, [0, 1, 0]);
    const keep = retargetClip({ source: src, target: tgt, clip });
    const d = vSub(posed(tgt, keep.clip, 1)[1]!, posed(tgt, keep.clip, 0)[1]!);
    close(d, [0, 0, 2]);
    const inPlace = retargetClip({ source: src, target: tgt, clip, rootMotion: "inPlace" });
    const d2 = vSub(posed(tgt, inPlace.clip, 1)[1]!, posed(tgt, inPlace.clip, 0)[1]!);
    close(d2, [0, 0, 0]);
  });

  it("writes evenly sampled LINEAR tracks with continuous quaternions", () => {
    const src = rig(MIXAMO);
    const out = retargetClip({ source: src, target: rig(QUAT), clip: raiseArmClip(src, [0, 1, 0]), fps: 10 });
    const rot = out.clip.tracks.find((t) => t.path === "rotation" && t.node === 5)!;
    expect(rot.times.length).toBe(11);
    for (let i = 1; i < rot.times.length; i++) {
      const dot = [0, 1, 2, 3].reduce((s, k) => s + rot.values[(i - 1) * 4 + k]! * rot.values[i * 4 + k]!, 0);
      expect(dot).toBeGreaterThan(0);
    }
    expect(out.clip.name).toBe("raise (retargeted)");
  });
});

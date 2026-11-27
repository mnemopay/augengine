/**
 * Clip retargeting between two glTF humanoid skeletons, with rest-pose
 * correction. Used by the Studio ("use this clip on that model") and by the
 * Add-character pipeline, and usable at runtime.
 *
 * World-space delta method. For every bone both rigs have (matched through
 * the canonical humanoid ids in humanoid.ts):
 *
 *   D_s(t) = R_s(t) · R_s(bind)⁻¹            source world rotation delta
 *   D_t(t) = F · D_s(t) · F⁻¹                 same delta in the target's
 *                                              facing frame (F maps the
 *                                              source's forward/left/up onto
 *                                              the target's)
 *   R_t(t) = D_t(t) · C · R_t(bind)           target world rotation
 *
 * C is the rest-pose correction for limbs: the shortest rotation taking the
 * target's bind bone direction (joint → mapped child joint) onto the
 * source's, so a T-pose clip on an A-pose rig (or the reverse) points arms
 * and legs the same way. Spine, neck and head get none (identity), so each
 * character keeps its own upright posture.
 * Target local rotations follow from the target hierarchy top-down, so
 * unmapped in-between joints (twist bones, extra spine joints) just carry
 * their rest pose. Hips translation is transferred as a displacement from
 * bind, rotated into the target frame and scaled by hip height.
 */
import type { AugEngineAnimationFromGltf, AugEngineAnimationTrack } from "../scene/loaders/gltf.js";
import { sampleGltfAnimationTrack } from "../scene/loaders/gltf-instance.js";
import type { Quat } from "../scene/types.js";
import { HUMANOID_PARENT, REQUIRED_BONES, resolveHumanoidRig, type HumanoidBone } from "./humanoid.js";
import {
  mFromTRS, mInvert, mMul, mPos, mRot, mTransformPoint,
  qContinuous, qFromBasis, qFromToP, qId, qInvP, qMulP, qNormP, qRotP,
  vCross, vNorm, vSub, vScale, vAdd, type M, type Q, type V,
} from "./rig-math.js";

/** A glTF node as it appears in the JSON (only the fields retargeting needs). */
export interface RigNode {
  name?: string;
  translation?: number[];
  rotation?: number[];
  scale?: number[];
  matrix?: number[];
  children?: number[];
}

export interface RigSource {
  nodes: readonly RigNode[];
  /** Skin joint node indices (glTF `skins[n].joints`). */
  joints: readonly number[];
}

export interface RetargetClipOptions {
  source: RigSource;
  target: RigSource;
  clip: AugEngineAnimationFromGltf;
  /** Samples per second written into the output clip (default 30). */
  fps?: number;
  /** "inPlace" removes horizontal hips travel (default "keep"). */
  rootMotion?: "keep" | "inPlace";
  /** Name of the output clip (default `${clip.name} (retargeted)`). */
  name?: string;
}

export interface RetargetClipResult {
  clip: AugEngineAnimationFromGltf;
  /** Matched bones: canonical id → [source node, target node]. */
  matched: Map<HumanoidBone, [number, number]>;
  /** Required bones missing from either rig (retarget still runs; those limbs keep rest pose). */
  missing: HumanoidBone[];
}

type Overrides = Map<number, { t?: V; r?: Q; s?: V }>;

function restTRS(n: RigNode): { t: V; r: Q; s: V } {
  if (n.matrix && n.matrix.length >= 16) {
    const m = n.matrix;
    const sx = Math.hypot(m[0]!, m[1]!, m[2]!), sy = Math.hypot(m[4]!, m[5]!, m[6]!), sz = Math.hypot(m[8]!, m[9]!, m[10]!);
    return { t: [m[12]!, m[13]!, m[14]!], r: mRot(m), s: [sx, sy, sz] };
  }
  return {
    t: (n.translation ? [...n.translation] : [0, 0, 0]) as V,
    r: (n.rotation ? [...n.rotation] : [0, 0, 0, 1]) as Q,
    s: (n.scale ? [...n.scale] : [1, 1, 1]) as V,
  };
}

/** Node order with parents before children, plus each node's parent (-1 = scene root). */
function hierarchy(nodes: readonly RigNode[]): { order: number[]; parent: number[] } {
  const parent = new Array<number>(nodes.length).fill(-1);
  nodes.forEach((n, i) => (n.children ?? []).forEach((c) => { parent[c] = i; }));
  const order: number[] = [];
  const seen = new Set<number>();
  const visit = (i: number) => {
    if (seen.has(i)) return;
    seen.add(i);
    order.push(i);
    for (const c of nodes[i]!.children ?? []) visit(c);
  };
  nodes.forEach((_, i) => { if (parent[i] === -1) visit(i); });
  nodes.forEach((_, i) => visit(i));
  return { order, parent };
}

function worldMatrices(nodes: readonly RigNode[], h: { order: number[]; parent: number[] }, over?: Overrides): M[] {
  const out = new Array<M>(nodes.length);
  for (const i of h.order) {
    const rest = restTRS(nodes[i]!);
    const o = over?.get(i);
    const local = mFromTRS(o?.t ?? rest.t, o?.r ?? rest.r, o?.s ?? rest.s);
    const p = h.parent[i]!;
    out[i] = p >= 0 ? mMul(out[p]!, local) : local;
  }
  return out;
}

function sampleClip(clip: AugEngineAnimationFromGltf, time: number): Overrides {
  const out: Overrides = new Map();
  for (const track of clip.tracks) {
    const v = sampleGltfAnimationTrack(track, time);
    if (!v) continue;
    const cur = out.get(track.node) ?? {};
    if (track.path === "translation") cur.t = [v[0], v[1], v[2]];
    else if (track.path === "rotation") cur.r = qNormP(v as Quat);
    else cur.s = [v[0], v[1], v[2]];
    out.set(track.node, cur);
  }
  return out;
}

/** Joints of a rig as {name, parent} with parent = nearest ancestor joint (index into `joints`). */
function jointTable(rig: RigSource, h: { parent: number[] }): { name: string; parent: number }[] {
  const pos = new Map(rig.joints.map((n, i) => [n, i]));
  return rig.joints.map((node) => {
    let p = h.parent[node]!;
    while (p >= 0 && !pos.has(p)) p = h.parent[p]!;
    return { name: rig.nodes[node]?.name ?? `node${node}`, parent: p >= 0 ? pos.get(p)! : -1 };
  });
}

/** Canonical bone → node index for a rig. */
export function humanoidNodes(rig: RigSource): Map<HumanoidBone, number> {
  const h = hierarchy(rig.nodes);
  const resolved = resolveHumanoidRig(jointTable(rig, h));
  const out = new Map<HumanoidBone, number>();
  for (const [ji, bone] of resolved) out.set(bone, rig.joints[ji]!);
  return out;
}

/**
 * Facing frame of a rig in bind pose: columns = (character's left, up, forward).
 * Left comes from the hips/legs (or arms), up is world +Y projected out.
 */
function facing(world: M[], bones: Map<HumanoidBone, number>): Q {
  const pairs: [HumanoidBone, HumanoidBone][] = [["leftUpperLeg", "rightUpperLeg"], ["leftUpperArm", "rightUpperArm"], ["leftShoulder", "rightShoulder"]];
  let left: V = [1, 0, 0];
  for (const [l, r] of pairs) {
    const li = bones.get(l), ri = bones.get(r);
    if (li === undefined || ri === undefined) continue;
    const d = vSub(mPos(world[li]!), mPos(world[ri]!));
    const flat: V = [d[0], 0, d[2]];
    if (Math.hypot(flat[0], flat[2]) > 1e-6) { left = vNorm(flat); break; }
  }
  const up: V = [0, 1, 0];
  const fwd = vNorm(vCross(left, up));
  return qFromBasis(left, up, fwd);
}

/** First canonical descendant of `bone` that both rigs have — the bone's "direction" end. */
function childBone(bone: HumanoidBone, has: (b: HumanoidBone) => boolean): HumanoidBone | null {
  const kids = (Object.keys(HUMANOID_PARENT) as HumanoidBone[]).filter((b) => HUMANOID_PARENT[b] === bone);
  // Prefer the chain continuation (spine→chest, upperArm→lowerArm) over side branches.
  const preferred: Partial<Record<HumanoidBone, HumanoidBone[]>> = {
    hips: ["spine", "chest", "upperChest", "neck"],
    spine: ["chest", "upperChest", "neck"],
    chest: ["upperChest", "neck"],
    upperChest: ["neck", "head"],
    neck: ["head"],
  };
  for (const b of preferred[bone] ?? kids) if (has(b)) return b;
  for (const b of kids) if (has(b)) return b;
  return null;
}

export function retargetClip(opts: RetargetClipOptions): RetargetClipResult {
  const { source, target, clip } = opts;
  const fps = opts.fps ?? 30;
  const hs = hierarchy(source.nodes);
  const ht = hierarchy(target.nodes);
  const sBones = humanoidNodes(source);
  const tBones = humanoidNodes(target);

  const matched = new Map<HumanoidBone, [number, number]>();
  for (const [b, sn] of sBones) {
    const tn = tBones.get(b);
    if (tn !== undefined) matched.set(b, [sn, tn]);
  }
  const missing = REQUIRED_BONES.filter((b) => !matched.has(b));

  const sBind = worldMatrices(source.nodes, hs);
  const tBind = worldMatrices(target.nodes, ht);
  const Fs = facing(sBind, sBones);
  const Ft = facing(tBind, tBones);
  const F = qMulP(Ft, qInvP(Fs)); // source frame → target frame
  const Finv = qInvP(F);

  // Rest-pose correction per matched limb bone. Only arms and legs: their
  // rest direction is a pose choice (T vs A). The spine, neck and head keep
  // the target's own rest orientation — a source rig's forward-leaning neck
  // must not tip the target's head down.
  const has = (b: HumanoidBone) => matched.has(b);
  const corr = new Map<HumanoidBone, Q>();
  const ordered = [...matched.keys()].sort((a, b) => depth(a) - depth(b));
  for (const b of ordered) {
    if (!LIMB_BONES.has(b)) { corr.set(b, qId()); continue; }
    const [sn, tn] = matched.get(b)!;
    const kid = childBone(b, has);
    if (!kid) {
      const p = HUMANOID_PARENT[b];
      corr.set(b, (p && corr.get(p)) || qId());
      continue;
    }
    const [sk, tk] = matched.get(kid)!;
    const dirS = qRotP(F, vSub(mPos(sBind[sk]!), mPos(sBind[sn]!)));
    const dirT = vSub(mPos(tBind[tk]!), mPos(tBind[tn]!));
    corr.set(b, qFromToP(dirT, dirS));
  }

  // Hip height ratio (hips above the lowest foot/toe in bind).
  const ground = (world: M[], bones: Map<HumanoidBone, number>) => {
    const ys = (["leftFoot", "rightFoot", "leftToes", "rightToes"] as const).map((b) => bones.get(b)).filter((i): i is number => i !== undefined).map((i) => mPos(world[i]!)[1]);
    return ys.length ? Math.min(...ys) : 0;
  };
  const sHips = sBones.get("hips");
  const tHips = tBones.get("hips");
  const sH = sHips !== undefined ? mPos(sBind[sHips]!)[1] - ground(sBind, sBones) : 1;
  const tH = tHips !== undefined ? mPos(tBind[tHips]!)[1] - ground(tBind, tBones) : 1;
  const ratio = sH > 1e-6 ? tH / sH : 1;

  const byTargetNode = new Map<number, HumanoidBone>();
  for (const [b, [, tn]] of matched) byTargetNode.set(tn, b);

  const frames = Math.max(2, Math.round(clip.duration * fps) + 1);
  const times = new Float32Array(frames);
  const rotValues = new Map<number, Float32Array>();
  for (const [, [, tn]] of matched) rotValues.set(tn, new Float32Array(frames * 4));
  const hipValues = tHips !== undefined && sHips !== undefined && matched.has("hips") ? new Float32Array(frames * 3) : null;
  const prevRot = new Map<number, Q>();

  for (let f = 0; f < frames; f++) {
    const t = frames === 1 ? 0 : (clip.duration * f) / (frames - 1);
    times[f] = t;
    const sWorld = worldMatrices(source.nodes, hs, sampleClip(clip, t));
    const tWorld = new Array<M>(target.nodes.length);
    for (const i of ht.order) {
      const rest = restTRS(target.nodes[i]!);
      const p = ht.parent[i]!;
      const parentW = p >= 0 ? tWorld[p]! : null;
      let r: Q = rest.r;
      let tr: V = rest.t;
      const bone = byTargetNode.get(i);
      if (bone) {
        const [sn] = matched.get(bone)!;
        const Ds = qMulP(mRot(sWorld[sn]!), qInvP(mRot(sBind[sn]!)));
        const Dt = qMulP(qMulP(F, Ds), Finv);
        const want = qMulP(qMulP(Dt, corr.get(bone)!), mRot(tBind[i]!));
        const parentRot = parentW ? mRot(parentW) : qId();
        r = qNormP(qMulP(qInvP(parentRot), want));
        if (bone === "hips" && hipValues) {
          let delta = qRotP(F, vSub(mPos(sWorld[sn]!), mPos(sBind[sn]!)));
          delta = vScale(delta, ratio);
          if (opts.rootMotion === "inPlace") delta = [0, delta[1], 0];
          const wantPos = vAdd(mPos(tBind[i]!), delta);
          tr = parentW ? mTransformPoint(mInvert(parentW), wantPos) : wantPos;
          hipValues.set(tr, f * 3);
        }
        const cont = qContinuous(prevRot.get(i) ?? null, r);
        prevRot.set(i, cont);
        rotValues.get(i)!.set(cont, f * 4);
      }
      const local = mFromTRS(tr, r, rest.s);
      tWorld[i] = parentW ? mMul(parentW, local) : local;
    }
  }

  const tracks: AugEngineAnimationTrack[] = [];
  for (const [node, values] of rotValues) tracks.push({ node, path: "rotation", times, values, interpolation: "LINEAR" });
  if (hipValues && tHips !== undefined) tracks.push({ node: tHips, path: "translation", times, values: hipValues, interpolation: "LINEAR" });
  return {
    clip: { name: opts.name ?? `${clip.name} (retargeted)`, duration: clip.duration, tracks },
    matched,
    missing,
  };
}

const LIMB_BONES = new Set<HumanoidBone>([
  "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
  "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
  "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
]);

function depth(b: HumanoidBone): number {
  let d = 0;
  for (let p = HUMANOID_PARENT[b]; p; p = HUMANOID_PARENT[p]) d++;
  return d;
}

/** Read a skin's joints + all nodes straight from a parsed glTF root. */
export function rigFromGltfRoot(root: { nodes?: RigNode[]; skins?: { joints: number[] }[] }, skinIndex = 0): RigSource | null {
  const skin = root.skins?.[skinIndex];
  if (!skin || !root.nodes) return null;
  return { nodes: root.nodes, joints: skin.joints };
}


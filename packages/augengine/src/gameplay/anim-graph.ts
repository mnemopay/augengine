/**
 * Animation graph: glTF clips in, a runtime state machine + blend tree out.
 *
 * Layers, from the bottom up:
 *
 *   AnimRig      the skeleton the graph drives: one slot per glTF node, with
 *                its rest TRS. Built from the glTF JSON, so it works on any
 *                skinned file (Quaternius, Mixamo, the Studio auto-rig).
 *   BoundClip    a glTF animation whose tracks are resolved to rig slots,
 *                plus clip-time events ("footstep" at 0.21 s, "hit" at 0.35 s).
 *   Motion       what a state plays: one clip, or a 1D blend tree with
 *                phase sync (walk/run at matching foot timing).
 *   AnimGraph    named states, data-driven transitions (param conditions,
 *                triggers, exit time, any-state), cross-fades, root motion,
 *                override/additive layers with bone masks, event dispatch.
 *
 * Everything is pure data + math (no GPU, no DOM), deterministic for a given
 * sequence of `update(dt)` calls, and allocation-free per frame after setup.
 * The result is applied to the scene's joint entities with `applyToScene`,
 * which the renderer's existing compute-skinning path then consumes.
 *
 * Rotation blending uses normalised lerp with hemisphere correction (the
 * usual engine choice: cheap, commutative across weights, and within a few
 * tenths of a degree of slerp for the < 90 degree differences poses have).
 */

import type { AugEngineAnimationFromGltf } from "../scene/loaders/gltf.js";
import type { EntityId, Vec3 } from "../scene/types.js";
import { Tag } from "../scene/types.js";
import type { Scene } from "../scene/scene.js";
import { humanoidBoneOf } from "../characters/humanoid.js";
import { mFromTRS, mMul, type M } from "../characters/rig-math.js";

// ──────────────────────────────────────────────────────────────────────
// Rig + pose
// ──────────────────────────────────────────────────────────────────────

/** Local TRS per rig slot: t = 3 floats, r = 4 (x, y, z, w), s = 3. */
export interface RigPose {
  t: Float32Array;
  r: Float32Array;
  s: Float32Array;
}

export interface AnimRig {
  /** glTF node index of each slot. */
  readonly nodes: readonly number[];
  readonly names: readonly string[];
  readonly slotOfNode: ReadonlyMap<number, number>;
  /** Parent slot (-1 for a root). */
  readonly parent: Int32Array;
  readonly rest: RigPose;
  /** Slot whose horizontal translation is root motion (the hips), or -1. */
  readonly rootSlot: number;
  /**
   * Rest-pose model-space matrix of the root slot's parent. Root-motion
   * deltas are authored in that space (Blender exports often put a 100x
   * scale or a -90 degree X rotation on the armature node); this maps them
   * into the model root's space.
   */
  readonly rootParentToModel: M;
}

/** The glTF node fields a rig needs. */
export interface AnimRigNode {
  name?: string;
  translation?: readonly number[];
  rotation?: readonly number[];
  scale?: readonly number[];
  matrix?: readonly number[];
  children?: readonly number[];
}

export interface CreateAnimRigOptions {
  /** Node to use as the root-motion bone. Default: the node named like hips. */
  rootMotionNode?: number;
}

export function createPose(slots: number): RigPose {
  const p: RigPose = { t: new Float32Array(slots * 3), r: new Float32Array(slots * 4), s: new Float32Array(slots * 3) };
  for (let i = 0; i < slots; i++) {
    p.r[i * 4 + 3] = 1;
    p.s[i * 3] = p.s[i * 3 + 1] = p.s[i * 3 + 2] = 1;
  }
  return p;
}

export function copyPose(out: RigPose, src: RigPose): void {
  out.t.set(src.t);
  out.r.set(src.r);
  out.s.set(src.s);
}

/** Build a rig over every node of a glTF file (the JSON `nodes` array). */
export function createAnimRig(nodes: readonly AnimRigNode[], opts: CreateAnimRigOptions = {}): AnimRig {
  const n = nodes.length;
  const parent = new Int32Array(n).fill(-1);
  nodes.forEach((node, i) => (node.children ?? []).forEach((c) => { if (c >= 0 && c < n) parent[c] = i; }));
  const rest = createPose(n);
  const names: string[] = [];
  const slotOfNode = new Map<number, number>();
  nodes.forEach((node, i) => {
    names.push(node.name ?? `node${i}`);
    slotOfNode.set(i, i);
    const trs = nodeTRS(node);
    rest.t.set(trs.t, i * 3);
    rest.r.set(trs.r, i * 4);
    rest.s.set(trs.s, i * 3);
  });
  let rootSlot = opts.rootMotionNode ?? -1;
  if (rootSlot < 0) rootSlot = names.findIndex((nm) => humanoidBoneOf(nm) === "hips");
  // Model-space rest matrix of the root bone's parent chain.
  let rootParentToModel: M = mFromTRS([0, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
  if (rootSlot >= 0) {
    const chain: number[] = [];
    for (let p = parent[rootSlot]!; p >= 0; p = parent[p]!) chain.unshift(p);
    for (const p of chain) rootParentToModel = mMul(rootParentToModel, mFromTRS(vec3At(rest.t, p), quatAt(rest.r, p), vec3At(rest.s, p)));
  }
  return { nodes: nodes.map((_, i) => i), names, slotOfNode, parent, rest, rootSlot, rootParentToModel };
}

function nodeTRS(node: AnimRigNode): { t: number[]; r: number[]; s: number[] } {
  const m = node.matrix;
  if (m && m.length >= 16) {
    const sx = Math.hypot(m[0]!, m[1]!, m[2]!) || 1, sy = Math.hypot(m[4]!, m[5]!, m[6]!) || 1, sz = Math.hypot(m[8]!, m[9]!, m[10]!) || 1;
    const r00 = m[0]! / sx, r10 = m[1]! / sx, r20 = m[2]! / sx, r01 = m[4]! / sy, r11 = m[5]! / sy, r21 = m[6]! / sy, r02 = m[8]! / sz, r12 = m[9]! / sz, r22 = m[10]! / sz;
    const tr = r00 + r11 + r22;
    let q: number[];
    if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s]; }
    else if (r00 > r11 && r00 > r22) { const s = Math.sqrt(1 + r00 - r11 - r22) * 2; q = [0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s]; }
    else if (r11 > r22) { const s = Math.sqrt(1 + r11 - r00 - r22) * 2; q = [(r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s]; }
    else { const s = Math.sqrt(1 + r22 - r00 - r11) * 2; q = [(r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s]; }
    return { t: [m[12]!, m[13]!, m[14]!], r: q, s: [sx, sy, sz] };
  }
  return {
    t: node.translation ? [...node.translation] : [0, 0, 0],
    r: node.rotation ? [...node.rotation] : [0, 0, 0, 1],
    s: node.scale ? [...node.scale] : [1, 1, 1],
  };
}

const vec3At = (a: Float32Array, i: number): [number, number, number] => [a[i * 3]!, a[i * 3 + 1]!, a[i * 3 + 2]!];
const quatAt = (a: Float32Array, i: number): [number, number, number, number] => [a[i * 4]!, a[i * 4 + 1]!, a[i * 4 + 2]!, a[i * 4 + 3]!];

/**
 * Slots in the subtree of each named bone (bone included). Names match the
 * rig's node names exactly, or by canonical humanoid id ("spine", "leftUpperArm").
 */
export function boneMask(rig: AnimRig, bones: readonly string[]): Uint8Array {
  const mask = new Uint8Array(rig.names.length);
  const roots = new Set<number>();
  for (const b of bones) {
    const lower = b.toLowerCase();
    rig.names.forEach((nm, i) => {
      if (nm.toLowerCase() === lower || humanoidBoneOf(nm)?.toLowerCase() === lower) roots.add(i);
    });
  }
  for (let i = 0; i < mask.length; i++) {
    for (let p = i; p >= 0; p = rig.parent[p]!) if (roots.has(p)) { mask[i] = 1; break; }
  }
  return mask;
}

// ──────────────────────────────────────────────────────────────────────
// Clips
// ──────────────────────────────────────────────────────────────────────

export interface AnimEventSpec {
  /** Clip time in seconds. */
  time: number;
  name: string;
}

/** A fired event. `clip` is the source clip's name, `state` the graph state or layer. */
export interface AnimEvent {
  name: string;
  clip: string;
  state: string;
  time: number;
}

interface BoundTrack {
  slot: number;
  /** 0 translation, 1 rotation, 2 scale. */
  path: 0 | 1 | 2;
  times: Float32Array;
  values: Float32Array;
  /** 0 linear, 1 step, 2 cubic spline (sampled at the key values, linear in between). */
  interp: 0 | 1 | 2;
}

export interface BoundClip {
  readonly name: string;
  readonly duration: number;
  readonly tracks: readonly BoundTrack[];
  readonly events: readonly AnimEventSpec[];
  /** Index of the root slot's translation track, or -1. */
  readonly rootTrack: number;
}

/** Resolve a glTF animation's tracks to rig slots. Tracks on nodes outside the rig are dropped. */
export function bindClip(rig: AnimRig, clip: AugEngineAnimationFromGltf, events: readonly AnimEventSpec[] = []): BoundClip {
  const tracks: BoundTrack[] = [];
  let rootTrack = -1;
  for (const tr of clip.tracks) {
    const slot = rig.slotOfNode.get(tr.node);
    if (slot === undefined || tr.times.length === 0) continue;
    const path = tr.path === "translation" ? 0 : tr.path === "rotation" ? 1 : 2;
    const interp = tr.interpolation === "STEP" ? 1 : tr.interpolation === "CUBICSPLINE" ? 2 : 0;
    if (path === 0 && slot === rig.rootSlot) rootTrack = tracks.length;
    tracks.push({ slot, path, times: tr.times, values: tr.values, interp });
  }
  const sorted = [...events].sort((a, b) => a.time - b.time);
  return { name: clip.name, duration: Math.max(clip.duration, 1e-4), tracks, events: sorted, rootTrack };
}

/** Find a clip by exact name, then by the part after the last '|' (Blender "Armature|Run"), case-insensitive. */
export function findClip(clips: readonly AugEngineAnimationFromGltf[], name: string): AugEngineAnimationFromGltf | undefined {
  const lower = name.toLowerCase();
  return clips.find((c) => c.name === name)
    ?? clips.find((c) => c.name.toLowerCase() === lower)
    ?? clips.find((c) => c.name.toLowerCase().split("|").pop() === lower);
}

/** Key interval + blend factor for time t (clamped to the key range). */
function keyAt(times: Float32Array, t: number): { lo: number; hi: number; a: number } {
  const last = times.length - 1;
  if (t <= times[0]!) return { lo: 0, hi: 0, a: 0 };
  if (t >= times[last]!) return { lo: last, hi: last, a: 0 };
  let lo = 0, hi = last;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid]! <= t) lo = mid; else hi = mid;
  }
  const t0 = times[lo]!, t1 = times[hi]!;
  return { lo, hi, a: t1 > t0 ? (t - t0) / (t1 - t0) : 0 };
}

/** Sample a clip into `out`, starting from `base` (usually the rest pose) for channels the clip does not animate. */
export function sampleClip(clip: BoundClip, time: number, base: RigPose, out: RigPose): void {
  copyPose(out, base);
  sampleClipOver(clip, time, out);
}

/** Overwrite only the channels the clip animates. */
function sampleClipOver(clip: BoundClip, time: number, out: RigPose): void {
  for (const tr of clip.tracks) {
    const { lo, hi, a: a0 } = keyAt(tr.times, time);
    const a = tr.interp === 1 ? 0 : a0;
    const stride = tr.path === 1 ? 4 : 3;
    // CUBICSPLINE stores [in-tangent, value, out-tangent] per key.
    const lb = tr.interp === 2 ? (lo * 3 + 1) * stride : lo * stride;
    const hb = tr.interp === 2 ? (hi * 3 + 1) * stride : hi * stride;
    const v = tr.values;
    if (tr.path === 1) {
      let bx = v[hb]!, by = v[hb + 1]!, bz = v[hb + 2]!, bw = v[hb + 3]!;
      const ax = v[lb]!, ay = v[lb + 1]!, az = v[lb + 2]!, aw = v[lb + 3]!;
      if (ax * bx + ay * by + az * bz + aw * bw < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; }
      let x = ax + (bx - ax) * a, y = ay + (by - ay) * a, z = az + (bz - az) * a, w = aw + (bw - aw) * a;
      const len = Math.hypot(x, y, z, w) || 1;
      x /= len; y /= len; z /= len; w /= len;
      const o = tr.slot * 4;
      out.r[o] = x; out.r[o + 1] = y; out.r[o + 2] = z; out.r[o + 3] = w;
    } else {
      const dst = tr.path === 0 ? out.t : out.s;
      const o = tr.slot * 3;
      dst[o] = v[lb]! + (v[hb]! - v[lb]!) * a;
      dst[o + 1] = v[lb + 1]! + (v[hb + 1]! - v[lb + 1]!) * a;
      dst[o + 2] = v[lb + 2]! + (v[hb + 2]! - v[lb + 2]!) * a;
    }
  }
}

/** Root-bone translation of a clip at time t (in the root's parent space), or null if the clip has no root track. */
function rootTranslation(clip: BoundClip, t: number, out: Float32Array): boolean {
  if (clip.rootTrack < 0) return false;
  const tr = clip.tracks[clip.rootTrack]!;
  const { lo, hi, a: a0 } = keyAt(tr.times, t);
  const a = tr.interp === 1 ? 0 : a0;
  const lb = tr.interp === 2 ? (lo * 3 + 1) * 3 : lo * 3;
  const hb = tr.interp === 2 ? (hi * 3 + 1) * 3 : hi * 3;
  for (let k = 0; k < 3; k++) out[k] = tr.values[lb + k]! + (tr.values[hb + k]! - tr.values[lb + k]!) * a;
  return true;
}

const rtA = new Float32Array(3), rtB = new Float32Array(3), rtC = new Float32Array(3), rtD = new Float32Array(3);

/**
 * Root displacement (root-parent space) between clip times t0 → t1, adding
 * whole-cycle travel when a looping clip wraps. `wraps` = number of times
 * the clip passed its end in between.
 */
export function rootDelta(clip: BoundClip, t0: number, t1: number, wraps: number, out: Float32Array): void {
  out[0] = out[1] = out[2] = 0;
  if (clip.rootTrack < 0) return;
  if (wraps <= 0) {
    rootTranslation(clip, t0, rtA);
    rootTranslation(clip, t1, rtB);
    for (let k = 0; k < 3; k++) out[k] = rtB[k]! - rtA[k]!;
    return;
  }
  rootTranslation(clip, 0, rtC);
  rootTranslation(clip, clip.duration, rtD);
  rootTranslation(clip, t0, rtA);
  rootTranslation(clip, t1, rtB);
  for (let k = 0; k < 3; k++) {
    const cycle = rtD[k]! - rtC[k]!;
    out[k] = (rtD[k]! - rtA[k]!) + (wraps - 1) * cycle + (rtB[k]! - rtC[k]!);
  }
}

// ──────────────────────────────────────────────────────────────────────
// Pose math
// ──────────────────────────────────────────────────────────────────────

/**
 * out = blend(a, b, w) per slot: translation/scale lerp, rotation nlerp with
 * hemisphere fix. With `mask`, slots where mask[i] = 0 keep `a`. `out` may alias `a`.
 */
export function blendPoses(out: RigPose, a: RigPose, b: RigPose, w: number, mask?: Uint8Array): void {
  const n = a.r.length / 4;
  for (let i = 0; i < n; i++) {
    const wi = mask && !mask[i] ? 0 : w;
    const o3 = i * 3, o4 = i * 4;
    if (wi <= 0) {
      if (out !== a) {
        out.t[o3] = a.t[o3]!; out.t[o3 + 1] = a.t[o3 + 1]!; out.t[o3 + 2] = a.t[o3 + 2]!;
        out.s[o3] = a.s[o3]!; out.s[o3 + 1] = a.s[o3 + 1]!; out.s[o3 + 2] = a.s[o3 + 2]!;
        out.r[o4] = a.r[o4]!; out.r[o4 + 1] = a.r[o4 + 1]!; out.r[o4 + 2] = a.r[o4 + 2]!; out.r[o4 + 3] = a.r[o4 + 3]!;
      }
      continue;
    }
    for (let k = 0; k < 3; k++) {
      out.t[o3 + k] = a.t[o3 + k]! + (b.t[o3 + k]! - a.t[o3 + k]!) * wi;
      out.s[o3 + k] = a.s[o3 + k]! + (b.s[o3 + k]! - a.s[o3 + k]!) * wi;
    }
    const ax = a.r[o4]!, ay = a.r[o4 + 1]!, az = a.r[o4 + 2]!, aw = a.r[o4 + 3]!;
    let bx = b.r[o4]!, by = b.r[o4 + 1]!, bz = b.r[o4 + 2]!, bw = b.r[o4 + 3]!;
    if (ax * bx + ay * by + az * bz + aw * bw < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; }
    let x = ax + (bx - ax) * wi, y = ay + (by - ay) * wi, z = az + (bz - az) * wi, ww = aw + (bw - aw) * wi;
    const len = Math.hypot(x, y, z, ww) || 1;
    x /= len; y /= len; z /= len; ww /= len;
    out.r[o4] = x; out.r[o4 + 1] = y; out.r[o4 + 2] = z; out.r[o4 + 3] = ww;
  }
}

/**
 * Additive layer: out = base ∘ w·(clip − ref). Rotation: base · slerp(I, ref⁻¹·clip, w);
 * translation: base + w·(clip − ref); scale untouched. `out` may alias `base`.
 */
export function applyAdditive(out: RigPose, base: RigPose, clip: RigPose, ref: RigPose, w: number, mask?: Uint8Array): void {
  if (out !== base) copyPose(out, base);
  const n = base.r.length / 4;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const o3 = i * 3, o4 = i * 4;
    for (let k = 0; k < 3; k++) out.t[o3 + k] = base.t[o3 + k]! + (clip.t[o3 + k]! - ref.t[o3 + k]!) * w;
    // d = inverse(ref) * clip   (unit quats: inverse = conjugate)
    const rx = -ref.r[o4]!, ry = -ref.r[o4 + 1]!, rz = -ref.r[o4 + 2]!, rw = ref.r[o4 + 3]!;
    const cx = clip.r[o4]!, cy = clip.r[o4 + 1]!, cz = clip.r[o4 + 2]!, cw = clip.r[o4 + 3]!;
    let dx = rw * cx + rx * cw + ry * cz - rz * cy;
    let dy = rw * cy - rx * cz + ry * cw + rz * cx;
    let dz = rw * cz + rx * cy - ry * cx + rz * cw;
    let dw = rw * cw - rx * cx - ry * cy - rz * cz;
    if (dw < 0) { dx = -dx; dy = -dy; dz = -dz; dw = -dw; }
    // nlerp(identity, d, w)
    let ex = dx * w, ey = dy * w, ez = dz * w, ew = 1 + (dw - 1) * w;
    const len = Math.hypot(ex, ey, ez, ew) || 1;
    ex /= len; ey /= len; ez /= len; ew /= len;
    // out.r = base.r * e
    const bx = base.r[o4]!, by = base.r[o4 + 1]!, bz = base.r[o4 + 2]!, bw = base.r[o4 + 3]!;
    out.r[o4] = bw * ex + bx * ew + by * ez - bz * ey;
    out.r[o4 + 1] = bw * ey - bx * ez + by * ew + bz * ex;
    out.r[o4 + 2] = bw * ez + bx * ey - by * ex + bz * ew;
    out.r[o4 + 3] = bw * ew - bx * ex - by * ey - bz * ez;
  }
}

/**
 * 1D blend weights: the two children bracketing `value` share the weight
 * linearly; outside the range the nearest child gets 1. `positions` must be
 * sorted ascending. Writes into `out` (same length).
 */
export function blend1DWeights(positions: readonly number[], value: number, out: Float32Array): void {
  out.fill(0);
  const n = positions.length;
  if (n === 0) return;
  if (value <= positions[0]!) { out[0] = 1; return; }
  if (value >= positions[n - 1]!) { out[n - 1] = 1; return; }
  for (let i = 0; i < n - 1; i++) {
    const lo = positions[i]!, hi = positions[i + 1]!;
    if (value >= lo && value <= hi) {
      const a = hi > lo ? (value - lo) / (hi - lo) : 0;
      out[i] = 1 - a;
      out[i + 1] = a;
      return;
    }
  }
}

/** Cross-fade weight of the incoming state after `elapsed` of `duration` seconds (smoothstep). */
export function crossFadeWeight(elapsed: number, duration: number): number {
  if (duration <= 0) return 1;
  const x = Math.min(1, Math.max(0, elapsed / duration));
  return x * x * (3 - 2 * x);
}

// ──────────────────────────────────────────────────────────────────────
// Graph spec (JSON-serialisable)
// ──────────────────────────────────────────────────────────────────────

export type MotionSpec =
  | {
      clip: string;
      speed?: number;
      loop?: boolean;
      /** Start (and, with speed 0, hold) at this clip time. */
      startTime?: number;
      /** Multiply playback by this parameter's value (absent/0 → 1). */
      speedParam?: string;
    }
  | {
      blend1d: { param: string; children: { clip: string; at: number; speed?: number }[] };
      speed?: number;
      /** Multiply playback by this parameter's value, e.g. to speed the run cycle up past its authored pace. */
      speedParam?: string;
    };

export interface StateSpec {
  name: string;
  motion: MotionSpec;
  /** Extract the root bone's horizontal travel as `graph.rootMotion` instead of moving the hips. */
  rootMotion?: boolean;
}

export type ConditionSpec =
  | { param: string; op: ">" | "<" | ">=" | "<=" | "==" | "!="; value: number }
  | { trigger: string }
  | { flag: string; is?: boolean };

export interface TransitionSpec {
  /** Source state, or "*" for any state. */
  from: string;
  to: string;
  /** All must hold. Empty/absent = unconditional (use with exitTime). */
  when?: ConditionSpec[];
  /** Normalised time (0..1, may exceed 1 for loops) the source must reach first. */
  exitTime?: number;
  /** Cross-fade seconds. Default 0.2. */
  duration?: number;
  /** Any-state only: allow re-entering the current state (a second hit restarts the hit). */
  allowSelf?: boolean;
  /** Any-state only: states this transition never fires from (e.g. no attack out of death). */
  except?: string[];
}

export interface LayerSpec {
  name: string;
  mode: "override" | "additive";
  /** Bones (with their subtrees) the layer touches. Absent = whole body. */
  mask?: string[];
}

export interface AnimGraphSpec {
  states: StateSpec[];
  transitions: TransitionSpec[];
  /** Initial state (default: first). */
  entry?: string;
  layers?: LayerSpec[];
  /** Events per clip name, clip time in seconds. */
  events?: Record<string, AnimEventSpec[]>;
  /** Initial parameter values. */
  params?: Record<string, number>;
}

export interface PlayLayerOptions {
  weight?: number;
  fadeIn?: number;
  fadeOut?: number;
  loop?: boolean;
  speed?: number;
}

// ──────────────────────────────────────────────────────────────────────
// Runtime
// ──────────────────────────────────────────────────────────────────────

type Motion =
  | { kind: "clip"; clip: BoundClip; speed: number; loop: boolean; startTime: number; speedParam: string | null }
  | { kind: "blend"; param: string; children: BoundClip[]; at: number[]; speeds: number[]; speed: number; weights: Float32Array; speedParam: string | null };

interface State {
  spec: StateSpec;
  motion: Motion;
  index: number;
}

/** A playing instance of a state: local time + normalised phase. */
interface Playing {
  state: State;
  /** Clip seconds for a clip motion. */
  time: number;
  /** Normalised time 0..∞ (number of cycles played). */
  norm: number;
  /** True until the first advance, so events at the start time fire. */
  fresh: boolean;
}

interface Transition {
  from: number; // -1 = any
  to: number;
  when: ConditionSpec[];
  exitTime: number | null;
  duration: number;
  allowSelf: boolean;
  except: Set<number>;
}

interface Layer {
  spec: LayerSpec;
  mask: Uint8Array | undefined;
  clip: BoundClip | null;
  time: number;
  weight: number;
  target: number;
  fadeIn: number;
  fadeOut: number;
  loop: boolean;
  speed: number;
  ref: RigPose;
  pose: RigPose;
}

export class AnimGraph {
  readonly rig: AnimRig;
  /** Output pose after the last `update`. */
  readonly pose: RigPose;
  /**
   * Root motion of the last `update`, in the model root's local space
   * (x, y, z metres before the model's own scale). Zero for states without
   * `rootMotion`. Rotate by the character's yaw to move it in the world.
   */
  readonly rootMotion: [number, number, number] = [0, 0, 0];
  readonly params: Record<string, number> = {};
  private readonly triggers = new Set<string>();
  private readonly states: State[] = [];
  private readonly byName = new Map<string, State>();
  private readonly transitions: Transition[] = [];
  private readonly clips = new Map<string, BoundClip>();
  private readonly layers: Layer[] = [];
  private current: Playing;
  private source: Playing | null = null;
  /** When a fade is interrupted, the blended pose at that instant fades out instead. */
  private frozen: RigPose | null = null;
  private fadeElapsed = 0;
  private fadeDuration = 0;
  private readonly scratchA: RigPose;
  private readonly scratchB: RigPose;
  private readonly scratchC: RigPose;
  private readonly frozenPose: RigPose;
  private readonly fired: AnimEvent[] = [];
  private readonly touched: Uint8Array;
  private readonly rd = new Float32Array(3);
  private readonly rdSum = new Float32Array(3);
  private readonly listeners: Array<(e: AnimEvent) => void> = [];
  private readonly rootInv: number[] | null;

  constructor(rig: AnimRig, clips: readonly AugEngineAnimationFromGltf[], spec: AnimGraphSpec) {
    this.rig = rig;
    const n = rig.names.length;
    this.pose = createPose(n);
    copyPose(this.pose, rig.rest);
    this.scratchA = createPose(n);
    this.scratchB = createPose(n);
    this.scratchC = createPose(n);
    this.frozenPose = createPose(n);
    this.touched = new Uint8Array(n);
    this.rootInv = invertLinear3(rig.rootParentToModel);
    Object.assign(this.params, spec.params ?? {});
    const bind = (name: string): BoundClip => {
      let b = this.clips.get(name);
      if (b) return b;
      const raw = findClip(clips, name);
      if (!raw) throw new Error(`AnimGraph: clip "${name}" not found (have: ${clips.map((c) => c.name).join(", ")})`);
      b = bindClip(rig, raw, spec.events?.[name] ?? spec.events?.[raw.name] ?? []);
      for (const t of b.tracks) this.touched[t.slot] = 1;
      this.clips.set(name, b);
      return b;
    };
    if (spec.states.length === 0) throw new Error("AnimGraph: at least one state is required");
    for (const s of spec.states) {
      if (this.byName.has(s.name)) throw new Error(`AnimGraph: duplicate state "${s.name}"`);
      let motion: Motion;
      if ("clip" in s.motion) {
        motion = {
          kind: "clip",
          clip: bind(s.motion.clip),
          speed: s.motion.speed ?? 1,
          loop: s.motion.loop ?? true,
          startTime: s.motion.startTime ?? 0,
          speedParam: s.motion.speedParam ?? null,
        };
      } else {
        const kids = [...s.motion.blend1d.children].sort((a, b) => a.at - b.at);
        if (kids.length === 0) throw new Error(`AnimGraph: blend state "${s.name}" has no children`);
        motion = {
          kind: "blend",
          param: s.motion.blend1d.param,
          children: kids.map((k) => bind(k.clip)),
          at: kids.map((k) => k.at),
          speeds: kids.map((k) => k.speed ?? 1),
          speed: s.motion.speed ?? 1,
          weights: new Float32Array(kids.length),
          speedParam: s.motion.speedParam ?? null,
        };
      }
      const st: State = { spec: s, motion, index: this.states.length };
      this.states.push(st);
      this.byName.set(s.name, st);
    }
    for (const t of spec.transitions) {
      const to = this.byName.get(t.to);
      const from = t.from === "*" ? null : this.byName.get(t.from);
      if (!to) throw new Error(`AnimGraph: transition to unknown state "${t.to}"`);
      if (t.from !== "*" && !from) throw new Error(`AnimGraph: transition from unknown state "${t.from}"`);
      this.transitions.push({
        from: from ? from.index : -1,
        to: to.index,
        when: t.when ?? [],
        exitTime: t.exitTime ?? null,
        duration: t.duration ?? 0.2,
        allowSelf: t.allowSelf ?? false,
        except: new Set((t.except ?? []).map((nm) => {
          const st = this.byName.get(nm);
          if (!st) throw new Error(`AnimGraph: transition except lists unknown state "${nm}"`);
          return st.index;
        })),
      });
    }
    // Any-state transitions are checked first (hit/death pre-empt locomotion).
    this.transitions.sort((a, b) => (a.from === -1 ? 0 : 1) - (b.from === -1 ? 0 : 1));
    for (const l of spec.layers ?? []) {
      this.layers.push({
        spec: l,
        mask: l.mask ? boneMask(rig, l.mask) : undefined,
        clip: null, time: 0, weight: 0, target: 0, fadeIn: 0.1, fadeOut: 0.15, loop: false, speed: 1,
        ref: createPose(n), pose: createPose(n),
      });
    }
    const entry = this.byName.get(spec.entry ?? spec.states[0]!.name);
    if (!entry) throw new Error(`AnimGraph: unknown entry state "${spec.entry}"`);
    this.current = this.play(entry);
    // Pre-bind clips that only layers will use, lazily via playLayer.
    this.bindLazy = bind;
  }

  private readonly bindLazy: (name: string) => BoundClip;

  // ── parameters ──────────────────────────────────────────────────────

  setParam(name: string, value: number): void { this.params[name] = value; }
  setFlag(name: string, on: boolean): void { this.params[name] = on ? 1 : 0; }
  setTrigger(name: string): void { this.triggers.add(name); }
  resetTrigger(name: string): void { this.triggers.delete(name); }

  /** Subscribe to clip events; returns unsubscribe. Events also come back from `update`. */
  onEvent(cb: (e: AnimEvent) => void): () => void {
    this.listeners.push(cb);
    return () => { const i = this.listeners.indexOf(cb); if (i >= 0) this.listeners.splice(i, 1); };
  }

  // ── inspection ──────────────────────────────────────────────────────

  get stateName(): string { return this.current.state.spec.name; }
  /** Normalised time of the current state (cycles; 1 = one full play of the clip). */
  get stateTime(): number { return this.current.norm; }
  /** The fading-out state, if a cross-fade is running. */
  get fadingFrom(): string | null { return this.frozen ? "(frozen)" : this.source?.state.spec.name ?? null; }
  /** Incoming-state weight of the running cross-fade (1 when none). */
  get fadeWeight(): number { return this.source || this.frozen ? crossFadeWeight(this.fadeElapsed, this.fadeDuration) : 1; }
  /** Blend weights of the current state's 1D tree (empty for clip states). */
  get blendWeights(): Float32Array { return this.current.state.motion.kind === "blend" ? this.current.state.motion.weights : new Float32Array(0); }
  layerWeight(name: string): number { return this.layers.find((l) => l.spec.name === name)?.weight ?? 0; }
  /** Slots any clip in the graph animates (the ones `applyToScene` writes). */
  get touchedSlots(): Uint8Array { return this.touched; }

  // ── control ─────────────────────────────────────────────────────────

  /** Jump to a state with a cross-fade, bypassing conditions. */
  crossFade(to: string, duration = 0.2): void {
    const st = this.byName.get(to);
    if (!st) throw new Error(`AnimGraph: unknown state "${to}"`);
    this.beginTransition(st, duration);
  }

  /** Play a clip on a layer (one-shot unless loop). Additive layers use the clip's first frame as reference. */
  playLayer(layer: string, clip: string, opts: PlayLayerOptions = {}): void {
    const l = this.layers.find((x) => x.spec.name === layer);
    if (!l) throw new Error(`AnimGraph: unknown layer "${layer}"`);
    const b = this.bindLazy(clip);
    l.clip = b;
    l.time = 0;
    l.target = opts.weight ?? 1;
    l.fadeIn = opts.fadeIn ?? 0.08;
    l.fadeOut = opts.fadeOut ?? 0.15;
    l.loop = opts.loop ?? false;
    l.speed = opts.speed ?? 1;
    if (l.spec.mode === "additive") sampleClip(b, 0, this.rig.rest, l.ref);
  }

  stopLayer(layer: string, fadeOut = 0.15): void {
    const l = this.layers.find((x) => x.spec.name === layer);
    if (!l) return;
    l.target = 0;
    l.fadeOut = fadeOut;
  }

  /**
   * Advance by dt seconds: evaluate transitions, advance times, fire events,
   * compute the output pose and root motion. Returns the events fired this
   * update (the array is reused; copy it to keep it).
   */
  update(dt: number): readonly AnimEvent[] {
    this.fired.length = 0;
    this.rootMotion[0] = this.rootMotion[1] = this.rootMotion[2] = 0;
    if (!(dt >= 0)) dt = 0;

    this.evaluateTransitions();

    // Advance the fading-out source (no events from it: they belong to the outgoing motion).
    if (this.source) this.advance(this.source, dt, false, null);
    this.rdSum[0] = this.rdSum[1] = this.rdSum[2] = 0;
    this.advance(this.current, dt, true, this.rdSum);

    // Pose: current state, then cross-fade from source/frozen.
    this.samplePlaying(this.current, this.scratchA);
    if (this.source || this.frozen) {
      this.fadeElapsed += dt;
      const w = crossFadeWeight(this.fadeElapsed, this.fadeDuration);
      if (this.frozen) {
        blendPoses(this.pose, this.frozen, this.scratchA, w);
      } else {
        this.samplePlaying(this.source!, this.scratchB);
        blendPoses(this.pose, this.scratchB, this.scratchA, w);
      }
      // Root motion only from the incoming state, scaled by its weight.
      for (let k = 0; k < 3; k++) this.rdSum[k] = this.rdSum[k]! * w;
      if (this.fadeElapsed >= this.fadeDuration) { this.source = null; this.frozen = null; }
    } else {
      copyPose(this.pose, this.scratchA);
    }

    // Layers.
    for (const l of this.layers) this.updateLayer(l, dt);

    // Root motion: root-parent space → model space (linear part only).
    if (this.current.state.spec.rootMotion) {
      const m = this.rig.rootParentToModel;
      const x = this.rdSum[0]!, y = this.rdSum[1]!, z = this.rdSum[2]!;
      // Horizontal only: vertical travel (a jump arc, hip bob) stays in the pose.
      this.rootMotion[0] = m[0]! * x + m[4]! * y + m[8]! * z;
      this.rootMotion[1] = 0;
      this.rootMotion[2] = m[2]! * x + m[6]! * y + m[10]! * z;
    }

    for (const e of this.fired) for (const cb of this.listeners) cb(e);
    return this.fired;
  }

  /**
   * Write the pose into the scene's joint entities (local TRS). `nodeToEntity`
   * is `GltfSceneInstance.nodeToEntity`. Only slots some clip animates are written.
   */
  applyToScene(scene: Scene, nodeToEntity: ReadonlyMap<number, EntityId>): void {
    const p = this.pose;
    for (let i = 0; i < this.touched.length; i++) {
      if (!this.touched[i]) continue;
      const id = nodeToEntity.get(this.rig.nodes[i]!);
      if (!id) continue;
      const tx = scene.get(id)?.components[Tag.Transform];
      if (!tx) continue;
      scene.setLocalTransform(
        id,
        [p.t[i * 3]!, p.t[i * 3 + 1]!, p.t[i * 3 + 2]!],
        [p.r[i * 4]!, p.r[i * 4 + 1]!, p.r[i * 4 + 2]!, p.r[i * 4 + 3]!],
        [p.s[i * 3]!, p.s[i * 3 + 1]!, p.s[i * 3 + 2]!],
      );
    }
  }

  // ── internals ───────────────────────────────────────────────────────

  private play(st: State): Playing {
    const time = st.motion.kind === "clip" ? st.motion.startTime : 0;
    const norm = st.motion.kind === "clip" ? time / st.motion.clip.duration : 0;
    return { state: st, time, norm, fresh: true };
  }

  private beginTransition(to: State, duration: number): void {
    if (duration <= 0) {
      this.source = null;
      this.frozen = null;
    } else if (this.source || this.frozen) {
      // Interrupting a fade: freeze what is on screen now and fade from that.
      copyPose(this.frozenPose, this.pose);
      this.frozen = this.frozenPose;
      this.source = null;
    } else {
      this.source = this.current;
    }
    this.fadeElapsed = 0;
    this.fadeDuration = Math.max(0, duration);
    this.current = this.play(to);
  }

  private evaluateTransitions(): void {
    const cur = this.current.state.index;
    for (const t of this.transitions) {
      if (t.from !== -1 && t.from !== cur) continue;
      if (t.from === -1 && ((t.to === cur && !t.allowSelf) || t.except.has(cur))) continue;
      if (t.exitTime !== null && this.current.norm < t.exitTime) continue;
      if (t.exitTime === null && t.when.length === 0) continue; // never fires on its own
      if (!t.when.every((c) => this.check(c))) continue;
      for (const c of t.when) if ("trigger" in c) this.triggers.delete(c.trigger);
      this.beginTransition(this.states[t.to]!, t.duration);
      return; // one transition per update
    }
  }

  private check(c: ConditionSpec): boolean {
    if ("trigger" in c) return this.triggers.has(c.trigger);
    if ("flag" in c) return ((this.params[c.flag] ?? 0) !== 0) === (c.is ?? true);
    const v = this.params[c.param] ?? 0;
    switch (c.op) {
      case ">": return v > c.value;
      case "<": return v < c.value;
      case ">=": return v >= c.value;
      case "<=": return v <= c.value;
      case "==": return v === c.value;
      case "!=": return v !== c.value;
    }
  }

  /** Advance a playing state; fire its events; accumulate root delta (root-parent space). */
  private advance(p: Playing, dt: number, events: boolean, rootOut: Float32Array | null): void {
    const m = p.state.motion;
    const rate = m.speed * (m.speedParam ? this.params[m.speedParam] || 1 : 1);
    if (m.kind === "clip") {
      const d = m.clip.duration;
      const t0 = p.time;
      const raw = t0 + dt * rate;
      let t1 = raw, wraps = 0;
      if (m.loop) {
        wraps = Math.floor(raw / d) - Math.floor(t0 / d);
        t1 = raw;
        p.time = raw;
        p.norm = raw / d;
      } else {
        t1 = Math.min(raw, d);
        p.time = t1;
        p.norm = t1 / d;
      }
      const a = m.loop ? mod(t0, d) : Math.min(t0, d);
      const b = m.loop ? mod(t1, d) : t1;
      if (events) this.fireEvents(m.clip, a, b, wraps, p.state.spec.name, p.fresh);
      p.fresh = false;
      if (rootOut && p.state.spec.rootMotion) {
        rootDelta(m.clip, a, b, wraps, this.rd);
        for (let k = 0; k < 3; k++) rootOut[k] = rootOut[k]! + this.rd[k]!;
      }
      return;
    }
    // Blend tree with phase sync: one normalised phase drives all children.
    blend1DWeights(m.at, this.params[m.param] ?? 0, m.weights);
    let effDur = 0;
    for (let i = 0; i < m.children.length; i++) effDur += m.weights[i]! * (m.children[i]!.duration / m.speeds[i]!);
    effDur = Math.max(effDur, 1e-4);
    const n0 = p.norm;
    const n1 = n0 + (dt * rate) / effDur;
    p.norm = n1;
    p.time = n1 * effDur;
    const wraps = Math.floor(n1) - Math.floor(n0);
    let dom = 0;
    for (let i = 1; i < m.weights.length; i++) if (m.weights[i]! > m.weights[dom]!) dom = i;
    for (let i = 0; i < m.children.length; i++) {
      const c = m.children[i]!;
      const w = m.weights[i]!;
      if (w <= 0) continue;
      const a = mod(n0, 1) * c.duration, b = mod(n1, 1) * c.duration;
      if (events && i === dom) this.fireEvents(c, a, b, wraps, p.state.spec.name, p.fresh);
      if (rootOut && p.state.spec.rootMotion) {
        rootDelta(c, a, b, wraps, this.rd);
        for (let k = 0; k < 3; k++) rootOut[k] = rootOut[k]! + this.rd[k]! * w;
      }
    }
    p.fresh = false;
  }

  /** Events with time in (a, b] ([a, b] on a state's first update), plus full cycles when the clip wrapped. */
  private fireEvents(clip: BoundClip, a: number, b: number, wraps: number, state: string, inclusive = false): void {
    if (clip.events.length === 0) return;
    const push = (e: AnimEventSpec) => this.fired.push({ name: e.name, clip: clip.name, state, time: e.time });
    const after = (t: number) => (inclusive ? t >= a : t > a);
    if (wraps <= 0) {
      for (const e of clip.events) if (after(e.time) && e.time <= b) push(e);
      return;
    }
    for (const e of clip.events) if (after(e.time)) push(e);
    for (let w = 1; w < wraps; w++) for (const e of clip.events) push(e);
    for (const e of clip.events) if (e.time <= b) push(e);
  }

  private samplePlaying(p: Playing, out: RigPose): void {
    const m = p.state.motion;
    if (m.kind === "clip") {
      const t = m.loop ? mod(p.time, m.clip.duration) : Math.min(p.time, m.clip.duration);
      sampleClip(m.clip, t, this.rig.rest, out);
    } else {
      // Blend children pairwise into `out` with running weight normalisation.
      let acc = 0;
      for (let i = 0; i < m.children.length; i++) {
        const w = m.weights[i]!;
        if (w <= 0) continue;
        const c = m.children[i]!;
        const t = mod(p.norm, 1) * c.duration;
        if (acc === 0) {
          sampleClip(c, t, this.rig.rest, out);
        } else {
          sampleClip(c, t, this.rig.rest, this.scratchC);
          blendPoses(out, out, this.scratchC, w / (acc + w));
        }
        acc += w;
      }
      if (acc === 0) copyPose(out, this.rig.rest);
    }
    if (p.state.spec.rootMotion && this.rig.rootSlot >= 0) this.pinRoot(p, out);
  }

  /** Keep the root bone's horizontal position at its clip-start value (motion goes to `rootMotion`). */
  private pinRoot(p: Playing, out: RigPose): void {
    const m = p.state.motion;
    const clip = m.kind === "clip" ? m.clip : m.children[0]!;
    const inv = this.rootInv;
    if (!inv || !rootTranslation(clip, 0, rtA)) return;
    const o = this.rig.rootSlot * 3;
    // Offset from the start position, mapped to model space; keep only model Y
    // (the hip bob), map that back to the root's parent space.
    const dx = out.t[o]! - rtA[0]!, dy = out.t[o + 1]! - rtA[1]!, dz = out.t[o + 2]! - rtA[2]!;
    const mm = this.rig.rootParentToModel;
    const my = mm[1]! * dx + mm[5]! * dy + mm[9]! * dz;
    out.t[o] = rtA[0]! + inv[3]! * my;
    out.t[o + 1] = rtA[1]! + inv[4]! * my;
    out.t[o + 2] = rtA[2]! + inv[5]! * my;
  }

  private updateLayer(l: Layer, dt: number): void {
    if (!l.clip) return;
    l.time += dt * l.speed;
    const d = l.clip.duration;
    const ended = !l.loop && l.time >= d;
    if (ended || l.target === 0) {
      l.weight = Math.max(0, l.weight - (l.fadeOut > 0 ? dt / l.fadeOut : 1));
    } else if (!l.loop && l.fadeOut > 0 && d - l.time < l.fadeOut) {
      // Fade out so the one-shot reaches zero exactly at its end.
      l.weight = Math.min(l.weight, Math.max(0, ((d - l.time) / l.fadeOut) * l.target));
    } else {
      l.weight = Math.min(l.target, l.weight + (l.fadeIn > 0 ? dt / l.fadeIn : 1));
    }
    if (l.weight <= 0 && (ended || l.target === 0)) {
      l.clip = null;
      l.weight = 0;
      return;
    }
    const t = l.loop ? mod(l.time, d) : Math.min(l.time, d);
    const prev = l.loop ? mod(l.time - dt * l.speed, d) : Math.min(l.time - dt * l.speed, d);
    this.fireEvents(l.clip, prev, t, l.loop && t < prev ? 1 : 0, l.spec.name);
    sampleClip(l.clip, t, this.rig.rest, l.pose);
    if (l.spec.mode === "override") blendPoses(this.pose, this.pose, l.pose, l.weight, l.mask);
    else applyAdditive(this.pose, this.pose, l.pose, l.ref, l.weight, l.mask);
  }
}

function mod(a: number, d: number): number {
  return d > 0 ? ((a % d) + d) % d : 0;
}

/** Inverse of the upper-left 3x3 of a column-major 4x4, column-major 3x3 out. */
function invertLinear3(m: M): number[] | null {
  const a = m[0]!, b = m[4]!, c = m[8]!, d = m[1]!, e = m[5]!, f = m[9]!, g = m[2]!, h = m[6]!, i = m[10]!;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-20) return null;
  const inv = 1 / det;
  // Row-major inverse entries.
  const r00 = A * inv, r01 = -(b * i - c * h) * inv, r02 = (b * f - c * e) * inv;
  const r10 = B * inv, r11 = (a * i - c * g) * inv, r12 = -(a * f - c * d) * inv;
  const r20 = C * inv, r21 = -(a * h - b * g) * inv, r22 = (a * e - b * d) * inv;
  // Column-major: [col0(r00,r10,r20), col1(...), col2(...)]
  return [r00, r10, r20, r01, r11, r21, r02, r12, r22];
}

/** Convenience: rotate a model-space root-motion vector by a yaw (radians about +Y). */
export function rotateByYaw(v: Vec3, yaw: number): [number, number, number] {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}

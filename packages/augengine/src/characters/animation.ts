/**
 * Animation system — clips + state machine + blend tree.
 *
 * Phase 9 partial. Pure-data, deterministic, no GPU dependency. Drives the
 * `AnimationMixer` component (already in scene/types). The mixer evaluates
 * clips per frame, blends results, and writes joint local rotations into
 * the scene; the WebGPU skinning compute path consumes those.
 *
 * Design:
 *   - AnimationClip: per-joint keyframe tracks (rotation + position).
 *   - StateMachine: named states, transitions with conditions, blend times.
 *   - BlendTree: 1D + 2D blend trees for parameterized motion (e.g.,
 *     locomotion blend by speed/direction).
 *
 * All math operates on quaternions + Vec3 from scene/math.ts.
 */

import type { Quat, Vec3 } from "../scene/types.js";
import { qSlerp, v3Lerp, type MutQuat, type MutVec3 } from "../scene/math.js";

// ──────────────────────────────────────────────────────────────────────
// Clips
// ──────────────────────────────────────────────────────────────────────

/** A keyframe track for a single joint. Times are in seconds. */
export interface JointTrack {
  jointIndex: number;
  /** Sorted keyframe times. */
  times: Float32Array;
  /** Per-keyframe rotations as flat (x, y, z, w) quad. */
  rotations?: Float32Array;
  /** Per-keyframe positions as flat (x, y, z) triple. */
  positions?: Float32Array;
}

export interface AnimationClip {
  name: string;
  /** Total clip duration in seconds. */
  duration: number;
  tracks: JointTrack[];
  /** Whether the clip's last keyframe equals its first (loop seam-free). */
  loopable: boolean;
}

/**
 * Sample a single joint track at time t. Linear-interp positions, slerp
 * rotations. Allocation-free if outRot/outPos are provided.
 */
export function sampleTrack(
  track: JointTrack,
  t: number,
  outRot?: MutQuat,
  outPos?: MutVec3,
): { rot?: Quat; pos?: Vec3 } {
  const times = track.times;
  if (times.length === 0) return {};

  // Binary search for the keyframe interval [i, i+1].
  let lo = 0;
  let hi = times.length - 1;
  if (t <= times[0]!) {
    lo = 0;
    hi = 0;
  } else if (t >= times[hi]!) {
    lo = hi;
  } else {
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid]! <= t) lo = mid;
      else hi = mid;
    }
  }
  const t0 = times[lo]!;
  const t1 = times[hi]!;
  const alpha = t1 > t0 ? (t - t0) / (t1 - t0) : 0;

  const result: { rot?: Quat; pos?: Vec3 } = {};

  if (track.rotations) {
    const rot = outRot ?? ([0, 0, 0, 1] as MutQuat);
    const a: Quat = [
      track.rotations[lo * 4]!,
      track.rotations[lo * 4 + 1]!,
      track.rotations[lo * 4 + 2]!,
      track.rotations[lo * 4 + 3]!,
    ];
    const b: Quat = [
      track.rotations[hi * 4]!,
      track.rotations[hi * 4 + 1]!,
      track.rotations[hi * 4 + 2]!,
      track.rotations[hi * 4 + 3]!,
    ];
    qSlerp(rot, a, b, alpha);
    result.rot = rot;
  }
  if (track.positions) {
    const pos = outPos ?? ([0, 0, 0] as MutVec3);
    const a: Vec3 = [
      track.positions[lo * 3]!,
      track.positions[lo * 3 + 1]!,
      track.positions[lo * 3 + 2]!,
    ];
    const b: Vec3 = [
      track.positions[hi * 3]!,
      track.positions[hi * 3 + 1]!,
      track.positions[hi * 3 + 2]!,
    ];
    v3Lerp(pos, a, b, alpha);
    result.pos = pos;
  }
  return result;
}

/** Wrap time into [0, duration) for looped clips, clamp otherwise. */
export function clipTime(clip: AnimationClip, t: number): number {
  if (clip.loopable) {
    const d = clip.duration;
    if (d <= 0) return 0;
    return ((t % d) + d) % d;
  }
  return Math.max(0, Math.min(clip.duration, t));
}

// ──────────────────────────────────────────────────────────────────────
// State Machine
// ──────────────────────────────────────────────────────────────────────

export interface AnimState {
  name: string;
  /** Clip(s) to play in this state. Multi-clip = blend tree. */
  source: { kind: "clip"; clip: AnimationClip } | { kind: "blend"; tree: BlendTree };
  /** Default playback speed multiplier. */
  speed?: number;
  /** Loop while in this state. */
  loop?: boolean;
}

export interface AnimTransition {
  from: string;
  to: string;
  /** Predicate evaluated against the parameter map. */
  condition: (params: Record<string, number>) => boolean;
  /** Blend time in seconds. */
  blendTime: number;
}

export class AnimStateMachine {
  states = new Map<string, AnimState>();
  transitions: AnimTransition[] = [];
  current = "";
  next: string | null = null;
  blendT = 0;
  blendDuration = 0;
  stateTime = 0;
  params: Record<string, number> = {};

  addState(state: AnimState): void {
    this.states.set(state.name, state);
    if (!this.current) this.current = state.name;
  }

  addTransition(t: AnimTransition): void {
    this.transitions.push(t);
  }

  setParam(name: string, value: number): void {
    this.params[name] = value;
  }

  /** Force a transition outside the conditional system. */
  forceTransition(to: string, blendTime = 0.2): void {
    if (!this.states.has(to)) return;
    if (to === this.current) return;
    this.next = to;
    this.blendT = 0;
    this.blendDuration = blendTime;
  }

  step(dt: number): void {
    this.stateTime += dt;
    if (this.next) {
      this.blendT += dt;
      if (this.blendT >= this.blendDuration) {
        this.current = this.next;
        this.next = null;
        this.blendT = 0;
        this.blendDuration = 0;
        this.stateTime = 0;
      }
      return;
    }
    // Evaluate transitions in declared order.
    for (const t of this.transitions) {
      if (t.from !== this.current) continue;
      if (t.condition(this.params)) {
        this.forceTransition(t.to, t.blendTime);
        return;
      }
    }
  }

  /**
   * Sample the current state's clip(s) into a per-joint output buffer.
   * Caller provides `outRotations` (4 floats per joint) and `outPositions`
   * (3 floats per joint, optional). During a transition, blend between
   * current and next.
   */
  evaluate(
    outRotations: Float32Array,
    outPositions: Float32Array | null,
  ): void {
    const cur = this.states.get(this.current);
    if (!cur) return;
    sampleState(cur, this.stateTime, outRotations, outPositions, this.params);
    if (this.next) {
      const nxt = this.states.get(this.next);
      if (!nxt) return;
      const tmpRot = new Float32Array(outRotations.length);
      const tmpPos = outPositions ? new Float32Array(outPositions.length) : null;
      sampleState(nxt, this.blendT, tmpRot, tmpPos, this.params);
      const alpha = this.blendDuration > 0 ? this.blendT / this.blendDuration : 1;
      blendInto(outRotations, tmpRot, alpha, outPositions, tmpPos);
    }
  }
}

function sampleState(
  state: AnimState,
  time: number,
  outRot: Float32Array,
  outPos: Float32Array | null,
  params: Record<string, number>,
): void {
  const speed = state.speed ?? 1.0;
  const t = time * speed;
  if (state.source.kind === "clip") {
    const clip = state.source.clip;
    const sampleT = state.loop ?? clip.loopable ? clipTime(clip, t) : Math.min(t, clip.duration);
    sampleClipInto(clip, sampleT, outRot, outPos);
  } else {
    sampleBlendTreeInto(state.source.tree, t, outRot, outPos, params);
  }
}

/** Slerp/lerp `target ← lerp(target, src, alpha)` per joint. */
function blendInto(
  targetRot: Float32Array,
  srcRot: Float32Array,
  alpha: number,
  targetPos: Float32Array | null,
  srcPos: Float32Array | null,
): void {
  const a: MutQuat = [0, 0, 0, 1];
  const b: MutQuat = [0, 0, 0, 1];
  const out: MutQuat = [0, 0, 0, 1];
  const n = targetRot.length / 4;
  for (let j = 0; j < n; j++) {
    a[0] = targetRot[j * 4]!;
    a[1] = targetRot[j * 4 + 1]!;
    a[2] = targetRot[j * 4 + 2]!;
    a[3] = targetRot[j * 4 + 3]!;
    b[0] = srcRot[j * 4]!;
    b[1] = srcRot[j * 4 + 1]!;
    b[2] = srcRot[j * 4 + 2]!;
    b[3] = srcRot[j * 4 + 3]!;
    qSlerp(out, a, b, alpha);
    targetRot[j * 4] = out[0];
    targetRot[j * 4 + 1] = out[1];
    targetRot[j * 4 + 2] = out[2];
    targetRot[j * 4 + 3] = out[3];
  }
  if (targetPos && srcPos) {
    const m = targetPos.length / 3;
    for (let j = 0; j < m; j++) {
      const ax = targetPos[j * 3]!;
      const ay = targetPos[j * 3 + 1]!;
      const az = targetPos[j * 3 + 2]!;
      const bx = srcPos[j * 3]!;
      const by = srcPos[j * 3 + 1]!;
      const bz = srcPos[j * 3 + 2]!;
      targetPos[j * 3] = ax + (bx - ax) * alpha;
      targetPos[j * 3 + 1] = ay + (by - ay) * alpha;
      targetPos[j * 3 + 2] = az + (bz - az) * alpha;
    }
  }
}

function sampleClipInto(
  clip: AnimationClip,
  t: number,
  outRot: Float32Array,
  outPos: Float32Array | null,
): void {
  const tmpRot: MutQuat = [0, 0, 0, 1];
  const tmpPos: MutVec3 = [0, 0, 0];
  for (const tr of clip.tracks) {
    const r = sampleTrack(tr, t, tmpRot, tmpPos);
    if (r.rot) {
      const j = tr.jointIndex;
      outRot[j * 4] = r.rot[0];
      outRot[j * 4 + 1] = r.rot[1];
      outRot[j * 4 + 2] = r.rot[2];
      outRot[j * 4 + 3] = r.rot[3];
    }
    if (r.pos && outPos) {
      const j = tr.jointIndex;
      outPos[j * 3] = r.pos[0];
      outPos[j * 3 + 1] = r.pos[1];
      outPos[j * 3 + 2] = r.pos[2];
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// Blend trees — 1D + 2D parameterized motion
// ──────────────────────────────────────────────────────────────────────

export interface BlendChild {
  clip: AnimationClip;
  /** Blend space position. 1D = single number, 2D = [x, y]. */
  position: number | [number, number];
  /** Per-clip speed multiplier. */
  speed?: number;
}

export interface BlendTree {
  kind: "1d" | "2d";
  /** Parameter name(s) read from state machine params. 1D = string, 2D = [x, y]. */
  parameter: string | [string, string];
  children: BlendChild[];
}

function sampleBlendTreeInto(
  tree: BlendTree,
  t: number,
  outRot: Float32Array,
  outPos: Float32Array | null,
  params: Record<string, number>,
): void {
  // Compute weights for each child based on parameter distance.
  const weights = computeBlendWeights(tree, params);
  // Zero output, then accumulate weighted contributions.
  outRot.fill(0);
  if (outPos) outPos.fill(0);

  const tmpRot = new Float32Array(outRot.length);
  const tmpPos = outPos ? new Float32Array(outPos.length) : null;

  for (let i = 0; i < tree.children.length; i++) {
    const w = weights[i]!;
    if (w < 1e-4) continue;
    const child = tree.children[i]!;
    const sampleT = clipTime(child.clip, t * (child.speed ?? 1));
    sampleClipInto(child.clip, sampleT, tmpRot, tmpPos);

    // Accumulate. For rotations we'd ideally do a chain of slerp; for phase 9
    // we use a normalized-quaternion sum which is fast and visually fine when
    // weights are well-conditioned (sum to 1, no opposing extremes).
    for (let k = 0; k < tmpRot.length; k++) outRot[k]! += tmpRot[k]! * w;
    if (outPos && tmpPos) {
      for (let k = 0; k < tmpPos.length; k++) outPos[k]! += tmpPos[k]! * w;
    }
  }

  // Renormalize quaternions per-joint after accumulation.
  for (let j = 0; j < outRot.length / 4; j++) {
    const x = outRot[j * 4]!;
    const y = outRot[j * 4 + 1]!;
    const z = outRot[j * 4 + 2]!;
    const w = outRot[j * 4 + 3]!;
    const len = Math.hypot(x, y, z, w);
    if (len < 1e-9) {
      outRot[j * 4] = 0;
      outRot[j * 4 + 1] = 0;
      outRot[j * 4 + 2] = 0;
      outRot[j * 4 + 3] = 1;
    } else {
      const inv = 1 / len;
      outRot[j * 4] = x * inv;
      outRot[j * 4 + 1] = y * inv;
      outRot[j * 4 + 2] = z * inv;
      outRot[j * 4 + 3] = w * inv;
    }
  }
}

function computeBlendWeights(
  tree: BlendTree,
  params: Record<string, number>,
): number[] {
  const out = new Array(tree.children.length).fill(0);
  if (tree.kind === "1d") {
    const param = tree.parameter as string;
    const v = params[param] ?? 0;
    // Find bracketing pair.
    let bestLo = -1;
    let bestHi = -1;
    let loP = -Infinity;
    let hiP = Infinity;
    for (let i = 0; i < tree.children.length; i++) {
      const p = tree.children[i]!.position as number;
      if (p <= v && p > loP) { loP = p; bestLo = i; }
      if (p >= v && p < hiP) { hiP = p; bestHi = i; }
    }
    if (bestLo === -1 && bestHi === -1) return out;
    if (bestLo === -1) { out[bestHi] = 1; return out; }
    if (bestHi === -1) { out[bestLo] = 1; return out; }
    if (bestLo === bestHi) { out[bestLo] = 1; return out; }
    const range = hiP - loP;
    const alpha = range > 0 ? (v - loP) / range : 0;
    out[bestLo] = 1 - alpha;
    out[bestHi] = alpha;
    return out;
  }
  // 2D — Gradient/Cartesian inverse-distance weighting.
  const [xParam, yParam] = tree.parameter as [string, string];
  const px = params[xParam] ?? 0;
  const py = params[yParam] ?? 0;
  const dists: number[] = [];
  let total = 0;
  for (const c of tree.children) {
    const [cx, cy] = c.position as [number, number];
    const d = Math.hypot(px - cx, py - cy);
    const w = 1 / (d * d + 1e-4);
    dists.push(w);
    total += w;
  }
  if (total > 0) {
    for (let i = 0; i < dists.length; i++) out[i] = dists[i]! / total;
  }
  return out;
}

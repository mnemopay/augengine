/**
 * Runtime IK on scene entities: an exact world-space two-bone solver (legs,
 * arms) and a look-at (head, eyes), as plain components you put on any
 * entity, plus `applyIkComponents(scene)` to run them after animation each
 * frame.
 *
 *   ik_two_bone: { root, mid, tip, mode: "target" | "ground", target?, targetPosition?,
 *                  pole?, groundOffset?, weight? }
 *   look_at:     { bone, target?, targetPosition?, axis?, maxAngleDeg?, weight? }
 *
 * "ground" mode plants the tip (foot) at `heightAt(x, z) + groundOffset`
 * under its current x/z — flat ground at y = 0 unless the caller passes a
 * terrain sampler. Bones are assumed to have uniform scale; IK only writes
 * rotations.
 *
 * The older characters/ik.ts `solveTwoBoneIK` is an approximation kept for
 * existing callers; this module is the exact one.
 */
import type { Scene } from "../scene/scene.js";
import { Tag, type EntityId, type Quat, type Vec3 } from "../scene/types.js";
import {
  mPos, mRot, qAngle, qAxisAngle, qFromToP, qId, qInvP, qMulP, qNormP, qRotP, qSlerpP,
  vAdd, vCross, vDot, vLen, vNorm, vScale, vSub, type M, type Q, type V,
} from "./rig-math.js";

export const IK_TWO_BONE = "ik_two_bone";
export const LOOK_AT = "look_at";

export interface IkTwoBoneComponent {
  root: EntityId;
  mid: EntityId;
  tip: EntityId;
  mode: "target" | "ground";
  /** Entity whose world position is the goal ("target" mode). */
  target?: EntityId;
  /** Fixed world-space goal ("target" mode, used when `target` is unset). */
  targetPosition?: Vec3;
  /** World point the bend (knee/elbow) should face; default keeps the current bend. */
  pole?: Vec3;
  /** Height of the tip joint above the ground in "ground" mode (ankle height). */
  groundOffset?: number;
  /** 0..1 blend with the animated pose (default 1). */
  weight?: number;
}

export interface LookAtComponent {
  bone: EntityId;
  target?: EntityId;
  targetPosition?: Vec3;
  /** Bone-local axis that should point at the target (default +Z). */
  axis?: Vec3;
  /** Largest rotation away from the animated pose (default 70°). */
  maxAngleDeg?: number;
  weight?: number;
}

export interface TwoBoneSolution {
  /** World-space rotation to apply to the root joint (about the root). */
  rootDelta: Q;
  /** World-space rotation to apply to the mid joint after the root delta. */
  midDelta: Q;
  /** False when the goal was out of reach and the chain was stretched toward it. */
  reached: boolean;
  /** Where the tip ends up. */
  tip: V;
}

/**
 * Exact two-bone IK in world space. `a`, `b`, `c` = root, mid, tip joint
 * positions; bone lengths are preserved; the bend lies in the plane through
 * the goal and the pole.
 */
export function solveTwoBoneWorld(a: Vec3, b: Vec3, c: Vec3, goal: Vec3, pole?: Vec3): TwoBoneSolution {
  const lab = vLen(vSub(b, a));
  const lbc = vLen(vSub(c, b));
  const toGoal = vSub(goal, a);
  let dist = vLen(toGoal);
  let reached = true;
  const maxR = lab + lbc - 1e-5;
  const minR = Math.abs(lab - lbc) + 1e-5;
  if (dist > maxR) { dist = maxR; reached = false; }
  if (dist < minR) { dist = minR; reached = false; }
  const dir = vLen(toGoal) > 1e-9 ? vNorm(toGoal) : vNorm(vSub(c, a));

  // Bend direction: pole (or the current knee) projected off the a→goal axis.
  const bendRef = vSub(pole ?? vAdd(b, vSub(b, vScale(vAdd(a, c), 0.5))), a);
  let bend = vSub(bendRef, vScale(dir, vDot(bendRef, dir)));
  if (vLen(bend) < 1e-9) {
    bend = vCross(dir, Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]);
  }
  bend = vNorm(bend);

  const cosA = Math.max(-1, Math.min(1, (lab * lab + dist * dist - lbc * lbc) / (2 * lab * dist)));
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const newB = vAdd(a, vAdd(vScale(dir, lab * cosA), vScale(bend, lab * sinA)));
  const newC = vAdd(a, vScale(dir, dist));

  const rootDelta = qFromToP(vSub(b, a), vSub(newB, a));
  // Where the tip went after rotating only the root, then swing the mid onto newC.
  const c1 = vAdd(newB, qRotP(rootDelta, vSub(c, b)));
  const midDelta = qFromToP(vSub(c1, newB), vSub(newC, newB));
  return { rootDelta, midDelta, reached, tip: newC };
}

/** World rotation that turns `axisWorld` (from `from`) toward `goal`, clamped to `maxAngle`. */
export function lookAtDelta(from: Vec3, axisWorld: Vec3, goal: Vec3, maxAngle = (70 * Math.PI) / 180): Q {
  const want = vNorm(vSub(goal, from));
  if (vLen(want) < 1e-9) return qId();
  let q = qFromToP(axisWorld, want);
  const ang = qAngle(q);
  if (ang > maxAngle) q = qSlerpP(qId(), q, maxAngle / ang);
  return q;
}

interface WorldView {
  m: (id: EntityId) => M | null;
}

function world(scene: Scene): WorldView {
  const wt = scene.worldTransforms();
  return {
    m: (id) => {
      const i = wt.indexById.get(id);
      return i === undefined ? null : Array.from(wt.worldMatrices.subarray(i * 16, i * 16 + 16));
    },
  };
}

function parentWorldRot(scene: Scene, id: EntityId): Q {
  const parent = scene.get(id)?.components[Tag.Transform]?.parent;
  if (!parent) return qId();
  const m = world(scene).m(parent);
  return m ? mRot(m) : qId();
}

/** Set an entity's WORLD rotation by rewriting its local rotation (position/scale kept). */
function setWorldRotation(scene: Scene, id: EntityId, worldRot: Q, weight: number): void {
  const t = scene.get(id)?.components[Tag.Transform];
  if (!t) return;
  let local = qNormP(qMulP(qInvP(parentWorldRot(scene, id)), worldRot));
  if (weight < 1) local = qSlerpP(t.rotation as Quat, local, Math.max(0, weight));
  scene.setLocalTransform(id, t.position, local, t.scale);
}

function goalOf(scene: Scene, target: EntityId | undefined, fixed: Vec3 | undefined): V | null {
  if (target) {
    const m = world(scene).m(target);
    if (m) return mPos(m);
  }
  return fixed ? [fixed[0], fixed[1], fixed[2]] : null;
}

export interface ApplyIkOptions {
  /** Ground height under (x, z) for "ground" mode (default flat at 0). */
  heightAt?: (x: number, z: number) => number;
}

/** True when `child` sits somewhere under `ancestor` in the transform hierarchy. */
export function isDescendant(scene: Scene, child: EntityId, ancestor: EntityId): boolean {
  let p = scene.get(child)?.components[Tag.Transform]?.parent ?? null;
  for (let guard = 0; p && guard < 256; guard++) {
    if (p === ancestor) return true;
    p = scene.get(p)?.components[Tag.Transform]?.parent ?? null;
  }
  return false;
}

/**
 * A two-bone chain must be connected (tip under mid under root): some rigs
 * (Quaternius, many game-export rigs) keep feet/hands as separate IK
 * controls, and rotating the knee can't move those.
 */
export function twoBoneChainProblem(scene: Scene, ik: Pick<IkTwoBoneComponent, "root" | "mid" | "tip">): string | null {
  for (const id of [ik.root, ik.mid, ik.tip]) if (!scene.get(id)) return `joint ${id} not found`;
  if (!isDescendant(scene, ik.mid, ik.root)) return `${ik.mid} is not under ${ik.root}`;
  if (!isDescendant(scene, ik.tip, ik.mid)) return `${ik.tip} is not under ${ik.mid} (separate IK control?)`;
  return null;
}

/** Solve one two-bone chain on the scene. Returns false if a joint is missing or the chain isn't connected. */
export function applyTwoBone(scene: Scene, ik: IkTwoBoneComponent, opts: ApplyIkOptions = {}): boolean {
  if (twoBoneChainProblem(scene, ik)) return false;
  const w = world(scene);
  const ma = w.m(ik.root), mb = w.m(ik.mid), mc = w.m(ik.tip);
  if (!ma || !mb || !mc) return false;
  const a = mPos(ma), b = mPos(mb), c = mPos(mc);
  let goal: V | null;
  if (ik.mode === "ground") {
    const h = (opts.heightAt ?? (() => 0))(c[0], c[2]);
    goal = [c[0], h + (ik.groundOffset ?? 0.08), c[2]];
  } else {
    goal = goalOf(scene, ik.target, ik.targetPosition);
  }
  if (!goal) return false;
  const sol = solveTwoBoneWorld(a, b, c, goal, ik.pole);
  const weight = ik.weight ?? 1;
  const rootRot = qMulP(sol.rootDelta, mRot(ma));
  setWorldRotation(scene, ik.root, rootRot, weight);
  // The mid's world rotation was carried by the root delta; apply the mid delta on top.
  const mb2 = world(scene).m(ik.mid);
  if (!mb2) return false;
  const midRot = qMulP(weight >= 1 ? sol.midDelta : qSlerpP(qId(), sol.midDelta, weight), mRot(mb2));
  setWorldRotation(scene, ik.mid, midRot, 1);
  return true;
}

export function applyLookAt(scene: Scene, la: LookAtComponent): boolean {
  const m = world(scene).m(la.bone);
  if (!m) return false;
  const goal = goalOf(scene, la.target, la.targetPosition);
  if (!goal) return false;
  const rot = mRot(m);
  const axis = qRotP(rot, la.axis ?? [0, 0, 1]);
  const delta = lookAtDelta(mPos(m), axis, goal, ((la.maxAngleDeg ?? 70) * Math.PI) / 180);
  setWorldRotation(scene, la.bone, qMulP(delta, rot), la.weight ?? 1);
  return true;
}

/** Run every ik_two_bone and look_at component in the scene (call after animation). */
export function applyIkComponents(scene: Scene, opts: ApplyIkOptions = {}): number {
  let solved = 0;
  for (const e of [...scene.entities.values()]) {
    const c = e.components as Record<string, unknown>;
    const ik = c[IK_TWO_BONE] as IkTwoBoneComponent | IkTwoBoneComponent[] | undefined;
    for (const chain of ik ? (Array.isArray(ik) ? ik : [ik]) : []) if (applyTwoBone(scene, chain, opts)) solved++;
    const la = c[LOOK_AT] as LookAtComponent | LookAtComponent[] | undefined;
    for (const one of la ? (Array.isArray(la) ? la : [la]) : []) if (applyLookAt(scene, one)) solved++;
  }
  return solved;
}

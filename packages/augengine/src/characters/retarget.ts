/**
 * Mocap retargeting — humanoid → humanoid.
 *
 * Implementation of the 3-phase algorithm specified in AUGENGINE-ARCHITECTURE.md
 * §"Retargeting (~500 LOC MVP)":
 *
 *   Phase 0 — Bind alignment (one-time per pair). Force both rigs to canonical
 *             T-pose, compute correction quaternions, cache bind world / parent
 *             quats per joint, compute uniform leg-length scale ratio.
 *   Phase 1 — Per-frame retarget via the canonical formula:
 *               trgLocalQ = invBindTrgParentQ · bindSrcParentQ · srcLocalQ
 *                           · invBindSrcQ · bindTrgQ
 *   Phase 2 — Root motion: scale source hip displacement by leg-length ratio.
 *   Phase 3 — IK foot-lock fixup (delegated to ik.ts).
 *
 * Edge-case notes (handled inline):
 *   - A-pose vs T-pose source rigs: Phase 0 emits per-joint T-pose offsets
 *   - Mirrored skeletons: detected via shoulder X-sign, applied as flip
 *   - Mismatched joint counts: caller supplies a JointMap; unmapped joints
 *     stay in their bind pose
 *   - Mismatched proportions: uniform scale (`s`) only — phase 9 adds per-
 *     chain IK targets for production-grade fidelity
 */

import type { Quat, Vec3 } from "../scene/types.js";
import {
  m4Create,
  m4DecomposeTRS,
  m4FromTRS,
  m4Mul,
  m4Identity,
  qCopy,
  qIdentity,
  qInvert,
  qMul,
  qNormalize,
  qFromTo,
  v3,
  v3Distance,
  v3Normalize,
  type MutQuat,
  type MutVec3,
} from "../scene/math.js";

// ──────────────────────────────────────────────────────────────────────
// Skeleton + JointMap types
// ──────────────────────────────────────────────────────────────────────

export interface SkeletonJoint {
  /** Stable name — used for mapping. Common: "Hips", "LeftUpLeg", "Head". */
  name: string;
  /** Index of parent joint, or -1 if root. */
  parent: number;
  /** Local position in the rig's bind pose. */
  bindPosition: Vec3;
  /** Local rotation in the rig's bind pose. */
  bindRotation: Quat;
}

export interface Skeleton {
  joints: SkeletonJoint[];
  /** Cached world transforms in bind pose, computed by `prepareSkeleton`. */
  bindWorld?: Float32Array; // mat4 per joint
  /** Cached world quats in bind pose. */
  bindWorldQuat?: MutQuat[];
}

/** Source-name → target-name mapping. Unmapped target joints stay in bind. */
export type JointMap = Record<string, string>;

export interface RetargetConfig {
  source: Skeleton;
  target: Skeleton;
  jointMap: JointMap;
  /** Optional canonical T-pose offsets per joint, computed by `forceTpose`. */
  sourceTposeOffsets?: MutQuat[];
  targetTposeOffsets?: MutQuat[];
  /** Uniform scale ratio = target leg length / source leg length. */
  scaleRatio: number;
  /** Whether the rigs use opposite X-axis conventions (Mixamo vs Maya). */
  mirrorX: boolean;
}

/** Per-frame motion sample — local rotations indexed by joint, plus root pos. */
export interface MotionFrame {
  /** Per-joint local rotation. Length === skeleton.joints.length. */
  localRotations: MutQuat[];
  /** Root joint world position (typically the hips). */
  rootPosition: Vec3;
}

// ──────────────────────────────────────────────────────────────────────
// Skeleton preparation — Phase 0
// ──────────────────────────────────────────────────────────────────────

/** Compute bind-pose world transforms for every joint. Idempotent cache. */
export function prepareSkeleton(skel: Skeleton): void {
  if (skel.bindWorld && skel.bindWorldQuat) return;
  const n = skel.joints.length;
  const worldMats = new Float32Array(n * 16);
  const worldQuats: MutQuat[] = [];
  const localM = m4Create();
  for (let i = 0; i < n; i++) {
    const j = skel.joints[i]!;
    m4FromTRS(localM, j.bindPosition, j.bindRotation, [1, 1, 1]);
    if (j.parent < 0) {
      worldMats.set(localM, i * 16);
    } else {
      const parentSlice = worldMats.subarray(j.parent * 16, j.parent * 16 + 16);
      const out = worldMats.subarray(i * 16, i * 16 + 16);
      m4Mul(out as unknown as Float32Array, parentSlice as unknown as Float32Array, localM);
    }
    // Decompose to extract world rotation as quat.
    const t = v3(0, 0, 0);
    const r: MutQuat = [0, 0, 0, 1];
    const s = v3(1, 1, 1);
    m4DecomposeTRS(worldMats.subarray(i * 16, i * 16 + 16) as unknown as Float32Array, t, r, s);
    worldQuats.push(r);
  }
  skel.bindWorld = worldMats;
  skel.bindWorldQuat = worldQuats;
}

/**
 * Force a skeleton into canonical T-pose. Returns per-joint correction
 * quaternions: applying these to the bind rotations would yield T-pose.
 *
 * Heuristic: align each bone's bind direction (child position - joint position)
 * to canonical direction by joint name. Names follow Mixamo/glTF Humanoid
 * convention; unknown joints get identity offset.
 */
export function forceTpose(skel: Skeleton): MutQuat[] {
  const n = skel.joints.length;
  const offsets: MutQuat[] = Array.from({ length: n }, () => [0, 0, 0, 1] as MutQuat);
  // Build child index for direction extraction.
  const childOf: number[] = Array.from({ length: n }, () => -1);
  for (let i = 0; i < n; i++) {
    const p = skel.joints[i]!.parent;
    if (p >= 0 && childOf[p] === -1) childOf[p] = i;
  }
  for (let i = 0; i < n; i++) {
    const j = skel.joints[i]!;
    const childIdx = childOf[i]!;
    if (childIdx < 0) continue;
    const childPos = skel.joints[childIdx]!.bindPosition;
    const dir: MutVec3 = [
      childPos[0] - j.bindPosition[0],
      childPos[1] - j.bindPosition[1],
      childPos[2] - j.bindPosition[2],
    ];
    v3Normalize(dir, dir);
    const canonical = canonicalBoneDirection(j.name);
    if (!canonical) continue;
    qFromTo(offsets[i]!, dir, canonical);
  }
  return offsets;
}

/**
 * Canonical T-pose bone directions by joint name. Standard humanoid
 * convention: arms +X / -X, legs -Y, spine +Y.
 */
function canonicalBoneDirection(name: string): Vec3 | null {
  const n = name.toLowerCase();
  if (n.includes("leftarm") || n.includes("leftforearm") || n.includes("lefthand") || n.includes("leftshoulder"))
    return [1, 0, 0];
  if (n.includes("rightarm") || n.includes("rightforearm") || n.includes("righthand") || n.includes("rightshoulder"))
    return [-1, 0, 0];
  if (n.includes("upleg") || n.includes("leg") || n.includes("foot") || n.includes("toe"))
    return [0, -1, 0];
  if (n.includes("spine") || n.includes("neck") || n.includes("head"))
    return [0, 1, 0];
  return null;
}

/** Detect mirrored axis convention by inspecting shoulder positions. */
export function detectMirrorX(skel: Skeleton): boolean {
  const left = skel.joints.findIndex(
    (j) => j.name.toLowerCase().includes("leftshoulder") || j.name.toLowerCase().includes("leftarm"),
  );
  if (left < 0) return false;
  const j = skel.joints[left]!;
  // Walk up to find root translation; flag as mirrored if shoulder.x < 0.
  return j.bindPosition[0] < 0;
}

/** Compute leg length = |hip→knee| + |knee→ankle| for the source / target. */
export function legLength(skel: Skeleton): number {
  const find = (substr: string): number => {
    return skel.joints.findIndex((j) => j.name.toLowerCase().includes(substr));
  };
  const upLeg = find("leftupleg") >= 0 ? find("leftupleg") : find("upleg");
  const lowerLeg = find("leftleg") >= 0 ? find("leftleg") : find("leg");
  const ankle = find("leftfoot") >= 0 ? find("leftfoot") : find("foot");
  if (upLeg < 0 || lowerLeg < 0 || ankle < 0) return 1;
  // Use bind-world positions for length stability.
  if (!skel.bindWorld) prepareSkeleton(skel);
  const bw = skel.bindWorld!;
  const pos = (i: number): Vec3 => [bw[i * 16 + 12]!, bw[i * 16 + 13]!, bw[i * 16 + 14]!];
  return v3Distance(pos(upLeg), pos(lowerLeg)) + v3Distance(pos(lowerLeg), pos(ankle));
}

// ──────────────────────────────────────────────────────────────────────
// Phase 0 entry — build a RetargetConfig
// ──────────────────────────────────────────────────────────────────────

export function buildRetargetConfig(
  source: Skeleton,
  target: Skeleton,
  jointMap: JointMap,
): RetargetConfig {
  prepareSkeleton(source);
  prepareSkeleton(target);
  const sourceTposeOffsets = forceTpose(source);
  const targetTposeOffsets = forceTpose(target);
  const srcLeg = legLength(source);
  const tgtLeg = legLength(target);
  const scaleRatio = srcLeg > 1e-6 ? tgtLeg / srcLeg : 1;
  const mirrorX = detectMirrorX(source) !== detectMirrorX(target);
  return {
    source,
    target,
    jointMap,
    sourceTposeOffsets,
    targetTposeOffsets,
    scaleRatio,
    mirrorX,
  };
}

// ──────────────────────────────────────────────────────────────────────
// Phase 1 — Per-frame retarget
// ──────────────────────────────────────────────────────────────────────

/**
 * Retarget one source MotionFrame onto a target MotionFrame.
 *
 * `outFrame.localRotations` must be pre-allocated to match
 * config.target.joints.length. Joints not present in the JointMap are left
 * untouched (caller typically initializes them to bind pose).
 */
export function retargetFrame(
  config: RetargetConfig,
  src: MotionFrame,
  out: MotionFrame,
): void {
  const { source, target, jointMap, mirrorX } = config;
  const srcByName = new Map<string, number>();
  for (let i = 0; i < source.joints.length; i++) srcByName.set(source.joints[i]!.name, i);

  const scratchA: MutQuat = [0, 0, 0, 1];
  const scratchB: MutQuat = [0, 0, 0, 1];
  const scratchC: MutQuat = [0, 0, 0, 1];

  for (let ti = 0; ti < target.joints.length; ti++) {
    const tName = target.joints[ti]!.name;
    // Find source joint name that maps to this target.
    const srcName = invertMap(jointMap, tName);
    if (!srcName) continue;
    const si = srcByName.get(srcName);
    if (si === undefined) continue;
    const srcLocal = src.localRotations[si];
    if (!srcLocal) continue;

    const srcParent = source.joints[si]!.parent;
    const tgtParent = target.joints[ti]!.parent;

    // Look up bind world quats (precomputed in prepareSkeleton).
    const srcParentBindQ =
      srcParent >= 0 ? source.bindWorldQuat![srcParent]! : ([0, 0, 0, 1] as MutQuat);
    const tgtParentBindQ =
      tgtParent >= 0 ? target.bindWorldQuat![tgtParent]! : ([0, 0, 0, 1] as MutQuat);
    const srcBindQ = source.bindWorldQuat![si]!;
    const tgtBindQ = target.bindWorldQuat![ti]!;

    // Canonical formula: trg = invBindTrgParent · bindSrcParent · srcLocal
    //                          · invBindSrc · bindTrg
    const invSrcBind: MutQuat = [0, 0, 0, 1];
    qInvert(invSrcBind, srcBindQ);
    const invTrgParent: MutQuat = [0, 0, 0, 1];
    qInvert(invTrgParent, tgtParentBindQ);

    qMul(scratchA, invTrgParent, srcParentBindQ);
    qMul(scratchB, scratchA, srcLocal);
    qMul(scratchC, scratchB, invSrcBind);
    qMul(scratchA, scratchC, tgtBindQ);

    if (mirrorX) {
      // Flip the X component of the rotation axis to mirror the motion.
      scratchA[0] = -scratchA[0];
      scratchA[3] = -scratchA[3];
    }

    qNormalize(scratchA, scratchA);
    if (!out.localRotations[ti]) {
      out.localRotations[ti] = [scratchA[0], scratchA[1], scratchA[2], scratchA[3]];
    } else {
      qCopy(out.localRotations[ti]!, scratchA);
    }
  }

  // Phase 2 — root motion. Scale by leg-length ratio.
  out.rootPosition = [
    src.rootPosition[0] * config.scaleRatio,
    src.rootPosition[1] * config.scaleRatio,
    src.rootPosition[2] * config.scaleRatio,
  ];
}

function invertMap(map: JointMap, target: string): string | null {
  for (const [src, tgt] of Object.entries(map)) {
    if (tgt === target) return src;
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

/** Build an identity MotionFrame for a skeleton (all bind rotations). */
export function makeIdentityFrame(skel: Skeleton): MotionFrame {
  return {
    localRotations: skel.joints.map((j) => [
      j.bindRotation[0],
      j.bindRotation[1],
      j.bindRotation[2],
      j.bindRotation[3],
    ] as MutQuat),
    rootPosition: [0, 0, 0],
  };
}

/** Convenience constructor for a default Mixamo→VRM joint map. */
export const MIXAMO_TO_VRM_DEFAULT: JointMap = {
  Hips: "hips",
  Spine: "spine",
  Spine1: "chest",
  Spine2: "upperChest",
  Neck: "neck",
  Head: "head",
  LeftShoulder: "leftShoulder",
  LeftArm: "leftUpperArm",
  LeftForeArm: "leftLowerArm",
  LeftHand: "leftHand",
  RightShoulder: "rightShoulder",
  RightArm: "rightUpperArm",
  RightForeArm: "rightLowerArm",
  RightHand: "rightHand",
  LeftUpLeg: "leftUpperLeg",
  LeftLeg: "leftLowerLeg",
  LeftFoot: "leftFoot",
  LeftToeBase: "leftToes",
  RightUpLeg: "rightUpperLeg",
  RightLeg: "rightLowerLeg",
  RightFoot: "rightFoot",
  RightToeBase: "rightToes",
};

void m4Identity; // ensure import retained for downstream consumers
void qIdentity;

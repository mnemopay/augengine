/**
 * Per-frame MediaPipe → joint-rotation fitting.
 *
 * Phase 3 closure — converts a per-frame MediaPipe landmarks dump into
 * `localRotations` for the synthetic MediaPipe Skeleton. Each joint with
 * a child reference computes the rotation that aims its bind-pose forward
 * vector at its current MediaPipe-derived forward vector.
 *
 * Approach: bone-aim. For each joint j with a child reference c:
 *   bindForward_j = normalize(childBindWorld_c - jointBindWorld_j)
 *   liveForward_j = normalize(mediaPipeWorld_c - mediaPipeWorld_j)
 *   worldRot_j    = qFromTo(bindForward, liveForward)
 *   localRot_j    = inv(parentWorldRot) · worldRot_j
 *
 * Limits:
 *   - Single-DOF (no twist around bone axis). Most upper-body NPC work
 *     is fine without it; roll requires a "twin reference" (e.g. shoulder
 *     line for spine roll) which is out of scope here.
 *   - Joints without a child reference (terminal joints like Head, Hand,
 *     Foot tips) keep identity local rotation.
 */

import type { Quat, Vec3 } from "../scene/types.js";
import {
  qFromTo,
  qMul,
  qInvert,
  qNormalize,
  v3Sub,
  v3Normalize,
  type MutQuat,
  type MutVec3,
} from "../scene/math.js";
import {
  MEDIAPIPE_SKELETON_JOINTS,
  type MediaPipeFrame,
} from "./mediapipe.js";
import type { MotionFrame, Skeleton } from "./retarget.js";

// ──────────────────────────────────────────────────────────────────────
// Child-reference table — for each joint, which MediaPipe landmark
// represents its child end-effector. Joints not listed here are terminal
// (Head, Hand, Foot tips) or computed (Hips, Spine, Chest, Neck).
// ──────────────────────────────────────────────────────────────────────

/**
 * jointName → MediaPipe landmark index used as the bone's "tip" reference.
 * Head/hands/toes are terminal; their parent already pointed at them, so
 * they get identity rotation here.
 */
const CHILD_LANDMARK: Record<string, number> = {
  // Spine + neck use computed positions (handled below as "chest_world", etc).
  // Limbs aim at the next MediaPipe landmark down the chain.
  LeftShoulder: 13, // → LEFT_ELBOW
  LeftArm: 15,      // → LEFT_WRIST
  LeftForeArm: 19,  // → LEFT_INDEX (proxy for hand direction)
  RightShoulder: 14,
  RightArm: 16,
  RightForeArm: 20,
  LeftUpLeg: 25,    // → LEFT_KNEE
  LeftLeg: 27,      // → LEFT_ANKLE
  LeftFoot: 31,     // → LEFT_FOOT_INDEX
  RightUpLeg: 26,
  RightLeg: 28,
  RightFoot: 32,
};

// ──────────────────────────────────────────────────────────────────────
// Computed joint world positions for spine chain
// ──────────────────────────────────────────────────────────────────────

interface Synthetic {
  hipMid: Vec3;
  spine: Vec3;
  chest: Vec3;
  shoulderMid: Vec3;
}

function syntheticPositions(frame: MediaPipeFrame): Synthetic {
  const lm = frame.landmarks;
  const hipMid: Vec3 = [
    (lm[23]!.x + lm[24]!.x) * 0.5,
    (lm[23]!.y + lm[24]!.y) * 0.5,
    (lm[23]!.z + lm[24]!.z) * 0.5,
  ];
  const shoulderMid: Vec3 = [
    (lm[11]!.x + lm[12]!.x) * 0.5,
    (lm[11]!.y + lm[12]!.y) * 0.5,
    (lm[11]!.z + lm[12]!.z) * 0.5,
  ];
  const spine: Vec3 = [
    hipMid[0] + (shoulderMid[0] - hipMid[0]) * 0.3,
    hipMid[1] + (shoulderMid[1] - hipMid[1]) * 0.3,
    hipMid[2] + (shoulderMid[2] - hipMid[2]) * 0.3,
  ];
  const chest: Vec3 = [
    hipMid[0] + (shoulderMid[0] - hipMid[0]) * 0.6,
    hipMid[1] + (shoulderMid[1] - hipMid[1]) * 0.6,
    hipMid[2] + (shoulderMid[2] - hipMid[2]) * 0.6,
  ];
  return { hipMid, spine, chest, shoulderMid };
}

function jointWorldFromFrame(
  jointName: string,
  frame: MediaPipeFrame,
  syn: Synthetic,
): Vec3 | null {
  const lm = frame.landmarks;
  switch (jointName) {
    case "Hips": return syn.hipMid;
    case "Spine": return syn.spine;
    case "Chest": return syn.chest;
    case "Neck": return syn.shoulderMid;
    case "Head": return [lm[0]!.x, lm[0]!.y, lm[0]!.z]; // NOSE
    case "LeftShoulder": return [lm[11]!.x, lm[11]!.y, lm[11]!.z];
    case "LeftArm": return [lm[13]!.x, lm[13]!.y, lm[13]!.z];
    case "LeftForeArm": return [lm[15]!.x, lm[15]!.y, lm[15]!.z];
    case "LeftHand": return [lm[19]!.x, lm[19]!.y, lm[19]!.z];
    case "RightShoulder": return [lm[12]!.x, lm[12]!.y, lm[12]!.z];
    case "RightArm": return [lm[14]!.x, lm[14]!.y, lm[14]!.z];
    case "RightForeArm": return [lm[16]!.x, lm[16]!.y, lm[16]!.z];
    case "RightHand": return [lm[20]!.x, lm[20]!.y, lm[20]!.z];
    case "LeftUpLeg": return [lm[23]!.x, lm[23]!.y, lm[23]!.z];
    case "LeftLeg": return [lm[25]!.x, lm[25]!.y, lm[25]!.z];
    case "LeftFoot": return [lm[27]!.x, lm[27]!.y, lm[27]!.z];
    case "LeftToeBase": return [lm[31]!.x, lm[31]!.y, lm[31]!.z];
    case "RightUpLeg": return [lm[24]!.x, lm[24]!.y, lm[24]!.z];
    case "RightLeg": return [lm[26]!.x, lm[26]!.y, lm[26]!.z];
    case "RightFoot": return [lm[28]!.x, lm[28]!.y, lm[28]!.z];
    case "RightToeBase": return [lm[32]!.x, lm[32]!.y, lm[32]!.z];
    default: return null;
  }
}

function jointWorldFromSkeleton(
  skel: Skeleton,
  jointIdx: number,
  worldOut: Vec3[],
): Vec3 {
  // Cache or compute. For a Skeleton without bind-world cache, just walk parents.
  if (worldOut[jointIdx]) return worldOut[jointIdx]!;
  const j = skel.joints[jointIdx]!;
  if (j.parent < 0) {
    worldOut[jointIdx] = j.bindPosition;
    return j.bindPosition;
  }
  const parent = jointWorldFromSkeleton(skel, j.parent, worldOut);
  const w: Vec3 = [
    parent[0] + j.bindPosition[0],
    parent[1] + j.bindPosition[1],
    parent[2] + j.bindPosition[2],
  ];
  worldOut[jointIdx] = w;
  return w;
}

// ──────────────────────────────────────────────────────────────────────
// Main fit
// ──────────────────────────────────────────────────────────────────────

/**
 * Fit a MediaPipe frame into per-joint local rotations on the supplied
 * synthetic Skeleton. Returns a MotionFrame consumable by retargetFrame.
 *
 * `bindFrame` is the same neutral-pose frame used to construct the skeleton
 * (typically frame 0 of a T-pose calibration capture). It defines the
 * "rest direction" each bone aims at when no rotation is applied.
 */
export function fitMediaPipeFrame(
  skeleton: Skeleton,
  bindFrame: MediaPipeFrame,
  liveFrame: MediaPipeFrame,
): MotionFrame {
  const bindSyn = syntheticPositions(bindFrame);
  const liveSyn = syntheticPositions(liveFrame);

  const localRots: MutQuat[] = skeleton.joints.map(
    () => [0, 0, 0, 1] as MutQuat,
  );
  const worldRots: MutQuat[] = skeleton.joints.map(
    () => [0, 0, 0, 1] as MutQuat,
  );

  // Walk parent-first so parent world rotation is finalized before child reads it.
  for (let i = 0; i < skeleton.joints.length; i++) {
    const j = skeleton.joints[i]!;

    // Determine the child reference point for this joint's aim. Spine
    // chain joints aim at the next link (Hips→Spine→Chest→Neck→Head),
    // limbs aim at their CHILD_LANDMARK entry.
    let childName: string | null = null;
    switch (j.name) {
      case "Hips": childName = "Spine"; break;
      case "Spine": childName = "Chest"; break;
      case "Chest": childName = "Neck"; break;
      case "Neck": childName = "Head"; break;
      default:
        if (j.name in CHILD_LANDMARK) {
          // Find the child joint by landmark.
          const childLm = CHILD_LANDMARK[j.name];
          const childJoint = skeleton.joints.find(
            (jj) =>
              jj.parent === i ||
              (jj.name === inferChildName(j.name)),
          );
          childName = childJoint?.name ?? null;
          void childLm;
        }
    }

    if (!childName) {
      // Terminal joint — keep identity local rotation.
      // worldRot_i = parent's world rot (no change).
      if (j.parent >= 0) {
        worldRots[i] = [...worldRots[j.parent]!] as MutQuat;
      }
      continue;
    }

    const jointBindWorld = jointWorldFromFrame(j.name, bindFrame, bindSyn);
    const childBindWorld = jointWorldFromFrame(childName, bindFrame, bindSyn);
    const jointLiveWorld = jointWorldFromFrame(j.name, liveFrame, liveSyn);
    const childLiveWorld = jointWorldFromFrame(childName, liveFrame, liveSyn);
    if (!jointBindWorld || !childBindWorld || !jointLiveWorld || !childLiveWorld) {
      if (j.parent >= 0) worldRots[i] = [...worldRots[j.parent]!] as MutQuat;
      continue;
    }

    const bindDir: MutVec3 = [0, 0, 0];
    const liveDir: MutVec3 = [0, 0, 0];
    v3Sub(bindDir, childBindWorld, jointBindWorld);
    v3Sub(liveDir, childLiveWorld, jointLiveWorld);
    v3Normalize(bindDir, bindDir);
    v3Normalize(liveDir, liveDir);

    // World rotation that aims bindDir at liveDir.
    qFromTo(worldRots[i]!, bindDir, liveDir);

    // Local rotation = inv(parentWorld) * worldRot_i
    if (j.parent >= 0) {
      const invParent: MutQuat = [0, 0, 0, 1];
      qInvert(invParent, worldRots[j.parent]!);
      qMul(localRots[i]!, invParent, worldRots[i]!);
      qNormalize(localRots[i]!, localRots[i]!);
    } else {
      localRots[i] = [...worldRots[i]!] as MutQuat;
    }
  }

  return {
    localRotations: localRots,
    rootPosition: liveSyn.hipMid,
  };
}

/** Best-effort child-joint name lookup. */
function inferChildName(parentName: string): string {
  const map: Record<string, string> = {
    LeftShoulder: "LeftArm",
    LeftArm: "LeftForeArm",
    LeftForeArm: "LeftHand",
    RightShoulder: "RightArm",
    RightArm: "RightForeArm",
    RightForeArm: "RightHand",
    LeftUpLeg: "LeftLeg",
    LeftLeg: "LeftFoot",
    LeftFoot: "LeftToeBase",
    RightUpLeg: "RightLeg",
    RightLeg: "RightFoot",
    RightFoot: "RightToeBase",
  };
  return map[parentName] ?? "";
}

void MEDIAPIPE_SKELETON_JOINTS;
void jointWorldFromSkeleton;

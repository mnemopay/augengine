/**
 * Mixamo / mocap retarget surface.
 *
 * Mixamo Y-Bot is the slice skeleton (Patrol / Auditor / Sealbreaker
 * share it). This module is the one import path Studio's "Import"
 * button uses when the dropped glTF has Mixamo joint names.
 */

import {
  MIXAMO_TO_VRM_DEFAULT,
  buildRetargetConfig,
  makeIdentityFrame,
  prepareSkeleton,
  retargetFrame,
  type JointMap,
  type MotionFrame,
  type RetargetConfig,
  type Skeleton,
} from "../characters/retarget.js";

/** Mixamo Y-Bot joint names in typical FBX/glTF export order. */
export const MIXAMO_YBOT_JOINTS: readonly string[] = [
  "Hips",
  "Spine",
  "Spine1",
  "Spine2",
  "Neck",
  "Head",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "LeftUpLeg",
  "LeftLeg",
  "LeftFoot",
  "LeftToeBase",
  "RightUpLeg",
  "RightLeg",
  "RightFoot",
  "RightToeBase",
];

export const MIXAMO_TO_ENGINE: JointMap = MIXAMO_TO_VRM_DEFAULT;

export function isMixamoSkeleton(skel: Skeleton): boolean {
  const names = new Set(skel.joints.map((j) => j.name));
  const hits = MIXAMO_YBOT_JOINTS.filter((n) => names.has(n)).length;
  return hits >= 8;
}

export function retargetMixamoToEngine(
  source: Skeleton,
  target: Skeleton,
  frame: MotionFrame,
  jointMap: JointMap = MIXAMO_TO_ENGINE,
): MotionFrame {
  prepareSkeleton(source);
  prepareSkeleton(target);
  const config: RetargetConfig = buildRetargetConfig(source, target, jointMap);
  const out = makeIdentityFrame(target);
  retargetFrame(config, frame, out);
  return out;
}

export function describeMixamoHandoff(skel: Skeleton): string {
  if (isMixamoSkeleton(skel)) {
    return `Mixamo Y-Bot (${skel.joints.length} joints) — retarget via MIXAMO_TO_ENGINE.`;
  }
  return `${skel.joints.length} joints — supply a JointMap; unmapped joints stay in bind.`;
}

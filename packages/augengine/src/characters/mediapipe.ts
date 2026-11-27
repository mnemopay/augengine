/**
 * MediaPipe Pose → glTF humanoid joint mapping.
 *
 * Phase 3 — drives the DeleMocapShort native render. The dele-mocap
 * pipeline produces `landmarks.json` with MediaPipe Pose's 33 keypoints
 * (3D world coords per frame). To drive a rigged glTF humanoid, we need:
 *
 *   1. A static mapping of MediaPipe landmark index → glTF joint name
 *   2. Per-frame: landmark positions → joint local rotations (via the
 *      retargeting algorithm in characters/retarget.ts, or a simpler
 *      look-at-style approach for visible skeletons)
 *
 * This file ships #1 + a helper that converts a MediaPipe landmarks frame
 * into a `MotionFrame` consumable by `retargetFrame()`.
 */

import type { Quat, Vec3 } from "../scene/types.js";
import {
  type MotionFrame,
  type Skeleton,
  MIXAMO_TO_VRM_DEFAULT,
} from "./retarget.js";
import { qIdentity, type MutQuat } from "../scene/math.js";

/**
 * MediaPipe Pose 33-landmark schema.
 * https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
 */
export const MEDIAPIPE_POSE_LANDMARKS = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

/**
 * One frame of MediaPipe landmarks. Matches the shape produced by the
 * dele-mocap process.py output (one entry per frame in `landmarks.json`).
 */
export interface MediaPipeFrame {
  /** 33 landmarks; each is (x, y, z) in world coords (meters from hip). */
  landmarks: Array<{ x: number; y: number; z: number; visibility?: number }>;
}

/**
 * Map MediaPipe landmark indices to a synthetic Skeleton joint hierarchy.
 * MediaPipe doesn't model spine joints between hips and shoulders —
 * we infer a single "Spine" midpoint between hip-mid and shoulder-mid.
 *
 * Joint order matches the glTF skin's `joints` array convention (parent-first).
 */
export const MEDIAPIPE_SKELETON_JOINTS: ReadonlyArray<{
  name: string;
  parent: number;
  /** MediaPipe landmark index this joint derives its world position from.
   *  -1 means computed from neighbors (hip-mid, shoulder-mid, spine, etc). */
  landmark: number;
}> = [
  { name: "Hips",          parent: -1, landmark: -1 }, // 0  (mid of LEFT_HIP + RIGHT_HIP)
  { name: "Spine",          parent: 0,  landmark: -1 }, // 1
  { name: "Chest",          parent: 1,  landmark: -1 }, // 2  (mid of shoulders, ~25% up from hip)
  { name: "Neck",           parent: 2,  landmark: -1 }, // 3  (mid of shoulders)
  { name: "Head",           parent: 3,  landmark: 0  }, // 4  (NOSE)
  { name: "LeftShoulder",   parent: 2,  landmark: 11 }, // 5
  { name: "LeftArm",        parent: 5,  landmark: 13 }, // 6  (LEFT_ELBOW)
  { name: "LeftForeArm",    parent: 6,  landmark: 15 }, // 7  (LEFT_WRIST)
  { name: "LeftHand",       parent: 7,  landmark: 19 }, // 8  (LEFT_INDEX)
  { name: "RightShoulder",  parent: 2,  landmark: 12 }, // 9
  { name: "RightArm",       parent: 9,  landmark: 14 }, // 10 (RIGHT_ELBOW)
  { name: "RightForeArm",   parent: 10, landmark: 16 }, // 11 (RIGHT_WRIST)
  { name: "RightHand",      parent: 11, landmark: 20 }, // 12 (RIGHT_INDEX)
  { name: "LeftUpLeg",      parent: 0,  landmark: 23 }, // 13 (LEFT_HIP)
  { name: "LeftLeg",        parent: 13, landmark: 25 }, // 14 (LEFT_KNEE)
  { name: "LeftFoot",       parent: 14, landmark: 27 }, // 15 (LEFT_ANKLE)
  { name: "LeftToeBase",    parent: 15, landmark: 31 }, // 16 (LEFT_FOOT_INDEX)
  { name: "RightUpLeg",     parent: 0,  landmark: 24 }, // 17
  { name: "RightLeg",       parent: 17, landmark: 26 }, // 18
  { name: "RightFoot",      parent: 18, landmark: 28 }, // 19
  { name: "RightToeBase",   parent: 19, landmark: 32 }, // 20
];

/**
 * Build a synthetic Skeleton in MediaPipe's bind pose. The bind pose comes
 * from frame 0 of a "neutral" pose — typically the actor standing in a
 * T-pose for one second of capture. Caller passes that frame here.
 */
export function makeMediaPipeSkeleton(bindFrame: MediaPipeFrame): Skeleton {
  const lm = bindFrame.landmarks;
  const lpos = (i: number): Vec3 => [lm[i]!.x, lm[i]!.y, lm[i]!.z];

  // Hip-mid + shoulder-mid + spine interpolations.
  const hipMid: Vec3 = midpoint(lpos(23), lpos(24));
  const shoulderMid: Vec3 = midpoint(lpos(11), lpos(12));
  const chest: Vec3 = lerp3(hipMid, shoulderMid, 0.6);
  const spine: Vec3 = lerp3(hipMid, shoulderMid, 0.3);

  const joints = MEDIAPIPE_SKELETON_JOINTS.map((spec, idx) => {
    let pos: Vec3;
    if (spec.landmark >= 0) {
      pos = lpos(spec.landmark);
    } else if (idx === 0) pos = hipMid;
    else if (idx === 1) pos = spine;
    else if (idx === 2) pos = chest;
    else if (idx === 3) pos = shoulderMid;
    else pos = [0, 0, 0];
    // Skeleton positions in retarget.ts are LOCAL (relative to parent),
    // not world. Compute local = world - parentWorld.
    const parentWorld = spec.parent >= 0
      ? worldOf(MEDIAPIPE_SKELETON_JOINTS[spec.parent]!, lm, hipMid, spine, chest, shoulderMid)
      : [0, 0, 0] as Vec3;
    const localPos: Vec3 = [
      pos[0] - parentWorld[0],
      pos[1] - parentWorld[1],
      pos[2] - parentWorld[2],
    ];
    return {
      name: spec.name,
      parent: spec.parent,
      bindPosition: localPos,
      bindRotation: [0, 0, 0, 1] as Quat,
    };
  });

  return { joints };
}

function worldOf(
  spec: { name: string; parent: number; landmark: number },
  lm: MediaPipeFrame["landmarks"],
  hipMid: Vec3,
  spine: Vec3,
  chest: Vec3,
  shoulderMid: Vec3,
): Vec3 {
  if (spec.landmark >= 0) {
    const l = lm[spec.landmark]!;
    return [l.x, l.y, l.z];
  }
  if (spec.name === "Hips") return hipMid;
  if (spec.name === "Spine") return spine;
  if (spec.name === "Chest") return chest;
  if (spec.name === "Neck") return shoulderMid;
  return [0, 0, 0];
}

/**
 * Convert a MediaPipe frame to a AugEngine MotionFrame. Returns hip world
 * position + per-joint identity rotations (joint world positions are
 * available via the skeleton, but rotations require a per-frame fitting
 * step — that's the next-session work).
 *
 * For phase 3 MVP: this produces a "stick-figure" frame where joints
 * snap to MediaPipe positions but rotations stay at bind. That's enough
 * to verify the load + render pipeline works end-to-end.
 */
export function mediaPipeFrameToMotionFrame(
  skeleton: Skeleton,
  frame: MediaPipeFrame,
): MotionFrame {
  const lm = frame.landmarks;
  const hipMid: Vec3 = midpoint(
    [lm[23]!.x, lm[23]!.y, lm[23]!.z],
    [lm[24]!.x, lm[24]!.y, lm[24]!.z],
  );
  return {
    localRotations: skeleton.joints.map(() => qIdentity([0, 0, 0, 1] as MutQuat)),
    rootPosition: hipMid,
  };
}

/**
 * Default mapping: MediaPipe joint names → glTF humanoid VRM joint names.
 * Combine with MIXAMO_TO_VRM_DEFAULT in retarget.ts for an end-to-end map
 * MediaPipe → VRM.
 */
export const MEDIAPIPE_TO_VRM: Record<string, string> = {
  Hips: "hips",
  Spine: "spine",
  Chest: "chest",
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

/** MediaPipe → Mixamo (alternate target rig). Mixamo→VRM lives in retarget.ts. */
export const MEDIAPIPE_TO_MIXAMO: Record<string, string> = Object.fromEntries(
  Object.keys(MIXAMO_TO_VRM_DEFAULT).map((mixamoName) => [mixamoName, mixamoName]),
);

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
}
function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

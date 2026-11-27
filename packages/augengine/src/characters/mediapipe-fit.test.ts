/**
 * Phase 3 fit tests — synthetic MediaPipe frame → joint rotation MotionFrame.
 */

import { describe, it, expect } from "vitest";
import { fitMediaPipeFrame } from "./mediapipe-fit.js";
import { makeMediaPipeSkeleton, type MediaPipeFrame } from "./mediapipe.js";

/** Build a T-pose neutral frame (arms out horizontally). */
function tPoseFrame(): MediaPipeFrame {
  // Coordinates roughly in meters from hip-mid origin, +Y up, +X right, +Z forward.
  const z = 0;
  const lm = (x: number, y: number, zz = z) => ({ x, y, z: zz });
  // 33 entries to match MediaPipe Pose. We only fill the ones the fitter reads;
  // the rest are zeros (won't affect the synthetic Skeleton built from this).
  const arr = new Array(33).fill(0).map(() => lm(0, 0));
  // Face — nose
  arr[0] = lm(0, 1.65);
  // Shoulders — 1m wide at y=1.45
  arr[11] = lm(0.5, 1.45);
  arr[12] = lm(-0.5, 1.45);
  // Elbows — straight out at y=1.45
  arr[13] = lm(0.85, 1.45);
  arr[14] = lm(-0.85, 1.45);
  // Wrists — further out
  arr[15] = lm(1.2, 1.45);
  arr[16] = lm(-1.2, 1.45);
  // Index fingers
  arr[19] = lm(1.32, 1.45);
  arr[20] = lm(-1.32, 1.45);
  // Hips — y=0.95
  arr[23] = lm(0.18, 0.95);
  arr[24] = lm(-0.18, 0.95);
  // Knees — y=0.5
  arr[25] = lm(0.18, 0.5);
  arr[26] = lm(-0.18, 0.5);
  // Ankles — y=0.05
  arr[27] = lm(0.18, 0.05);
  arr[28] = lm(-0.18, 0.05);
  // Foot indices — y=0.0, slightly forward
  arr[31] = lm(0.18, 0.0, 0.15);
  arr[32] = lm(-0.18, 0.0, 0.15);
  return { landmarks: arr };
}

/** Same as t-pose but with right arm rotated 90° down (wrist below shoulder). */
function rightArmDownFrame(): MediaPipeFrame {
  const f = tPoseFrame();
  // Right shoulder stays. Right elbow + wrist + index drop straight down.
  f.landmarks[14] = { x: -0.5, y: 1.10, z: 0 };
  f.landmarks[16] = { x: -0.5, y: 0.75, z: 0 };
  f.landmarks[20] = { x: -0.5, y: 0.63, z: 0 };
  return f;
}

describe("augengine / phase 3 — fitMediaPipeFrame", () => {
  it("identity for the bind frame", () => {
    const bind = tPoseFrame();
    const skel = makeMediaPipeSkeleton(bind);
    const motion = fitMediaPipeFrame(skel, bind, bind);
    // Every local rotation should be ~identity (w ≈ 1).
    for (const q of motion.localRotations) {
      expect(Math.abs(q[3])).toBeGreaterThan(0.999);
    }
  });

  it("non-trivial rotation for a moved limb", () => {
    const bind = tPoseFrame();
    const live = rightArmDownFrame();
    const skel = makeMediaPipeSkeleton(bind);
    const motion = fitMediaPipeFrame(skel, bind, live);
    // RightShoulder is index 9 in MEDIAPIPE_SKELETON_JOINTS — its rotation
    // should NOT be identity now (the arm dropped).
    const rs = motion.localRotations[9]!;
    const isIdentity = Math.abs(rs[0]) < 1e-3 && Math.abs(rs[1]) < 1e-3 && Math.abs(rs[2]) < 1e-3;
    expect(isIdentity).toBe(false);
  });

  it("root position tracks hip-mid across frames", () => {
    const bind = tPoseFrame();
    const live = tPoseFrame();
    // Shift the live hips by +0.5 in X.
    live.landmarks[23] = { x: 0.68, y: 0.95, z: 0 };
    live.landmarks[24] = { x: 0.32, y: 0.95, z: 0 };
    const skel = makeMediaPipeSkeleton(bind);
    const motion = fitMediaPipeFrame(skel, bind, live);
    expect(motion.rootPosition[0]).toBeCloseTo(0.5, 2);
  });
});

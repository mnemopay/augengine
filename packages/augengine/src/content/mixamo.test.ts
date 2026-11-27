import { describe, it, expect } from "vitest";
import { makeIdentityFrame, type Skeleton } from "../characters/retarget.js";
import {
  MIXAMO_YBOT_JOINTS,
  MIXAMO_TO_ENGINE,
  describeMixamoHandoff,
  isMixamoSkeleton,
  retargetMixamoToEngine,
} from "./mixamo.js";

function ybot(): Skeleton {
  return {
    joints: MIXAMO_YBOT_JOINTS.map((name, i) => ({
      name,
      parent: i === 0 ? -1 : i === 1 ? 0 : i - 1,
      bindPosition: [0, i === 0 ? 0 : 0.1, 0] as const,
      bindRotation: [0, 0, 0, 1] as const,
    })),
  };
}

describe("Mixamo retarget", () => {
  it("detects a Y-Bot joint set", () => {
    expect(isMixamoSkeleton(ybot())).toBe(true);
    expect(isMixamoSkeleton({ joints: [{ name: "root", parent: -1, bindPosition: [0, 0, 0], bindRotation: [0, 0, 0, 1] }] })).toBe(false);
    expect(describeMixamoHandoff(ybot())).toMatch(/Mixamo Y-Bot/);
  });

  it("maps Mixamo names onto the engine humanoid", () => {
    expect(MIXAMO_TO_ENGINE.Hips).toBe("hips");
    expect(MIXAMO_TO_ENGINE.LeftArm).toBe("leftUpperArm");
  });

  it("retargets a bind frame without throwing", () => {
    const src = ybot();
    const tgt: Skeleton = {
      joints: Object.values(MIXAMO_TO_ENGINE).map((name, i) => ({
        name,
        parent: i === 0 ? -1 : 0,
        bindPosition: [0, 0.1 * i, 0] as const,
        bindRotation: [0, 0, 0, 1] as const,
      })),
    };
    const out = retargetMixamoToEngine(src, tgt, makeIdentityFrame(src));
    expect(out.localRotations).toHaveLength(tgt.joints.length);
  });
});

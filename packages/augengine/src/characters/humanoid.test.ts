import { describe, expect, it } from "vitest";
import { arkitCoverage, humanoidBoneOf, mapHumanoidBones, missingHumanoidBones, resolveHumanoidRig } from "./humanoid.js";

describe("humanoidBoneOf", () => {
  it.each([
    ["mixamorig:Hips", "hips"],
    ["mixamorig:LeftUpLeg", "leftUpperLeg"],
    ["mixamorig1:RightForeArm", "rightLowerArm"],
    ["mixamorig:LeftShoulder", "leftShoulder"],
    ["mixamorig:Spine1", "chest"],
    ["J_Bip_L_UpperLeg", "leftUpperLeg"],
    ["J_Bip_R_Hand", "rightHand"],
    ["J_Bip_C_Head", "head"],
    ["leftUpperArm", "leftUpperArm"],
    ["thigh_l", "leftUpperLeg"],
    ["calf_r", "rightLowerLeg"],
    ["upperarm_l", "leftUpperArm"],
    ["clavicle_r", "rightShoulder"],
    ["pelvis", "hips"],
    ["UpperLeg.L", "leftUpperLeg"],
    ["LowerArm.R", "rightLowerArm"],
    ["Foot.L", "leftFoot"],
    ["DEF-thigh.L", "leftUpperLeg"],
    ["LeftUpperLeg", "leftUpperLeg"],
    ["Head", "head"],
  ])("%s → %s", (name, bone) => {
    expect(humanoidBoneOf(name)).toBe(bone);
  });

  it("rejects non-humanoid and side-less limb names", () => {
    expect(humanoidBoneOf("Weapon")).toBeNull();
    expect(humanoidBoneOf("PT.L")).toBeNull();
    expect(humanoidBoneOf("UpperArm")).toBeNull();
    expect(humanoidBoneOf("Leg1.L")).toBeNull();
  });
});

describe("resolveHumanoidRig", () => {
  it("assigns the spine by structure (Quaternius Abdomen/Torso/Chest)", () => {
    const joints = [
      { name: "Root", parent: -1 },
      { name: "Hips", parent: 0 },
      { name: "Abdomen", parent: 1 },
      { name: "Torso", parent: 2 },
      { name: "Chest", parent: 3 },
      { name: "Neck", parent: 4 },
      { name: "Head", parent: 5 },
      { name: "Shoulder.L", parent: 4 },
    ];
    const r = resolveHumanoidRig(joints);
    expect(r.get(2)).toBe("spine");
    expect(r.get(3)).toBe("chest");
    expect(r.get(4)).toBe("upperChest");
    expect(r.get(7)).toBe("leftShoulder");
    expect(r.get(0)).toBeUndefined();
  });

  it("handles a single spine joint", () => {
    const r = resolveHumanoidRig([
      { name: "hips", parent: -1 },
      { name: "spine", parent: 0 },
      { name: "neck", parent: 1 },
    ]);
    expect(r.get(1)).toBe("spine");
  });
});

describe("mapHumanoidBones / coverage", () => {
  it("maps Mixamo onto VRoid names", () => {
    const m = mapHumanoidBones(["mixamorig:Hips", "mixamorig:LeftArm", "mixamorig:Tail"], ["J_Bip_C_Hips", "J_Bip_L_UpperArm"]);
    expect(m).toEqual({ "mixamorig:Hips": "J_Bip_C_Hips", "mixamorig:LeftArm": "J_Bip_L_UpperArm" });
  });

  it("reports missing required bones", () => {
    expect(missingHumanoidBones(["Hips", "Head"])).toContain("leftUpperLeg");
  });

  it("counts ARKit-52 morphs regardless of case and prefixes", () => {
    const c = arkitCoverage(["Face.jawOpen", "EYEBLINKLEFT", "smile"]);
    expect(c.matched).toEqual(["eyeBlinkLeft", "jawOpen"]);
    expect(c.missing).toHaveLength(50);
  });
});

/**
 * Humanoid bone naming — one canonical id per standard bone (VRM 1.0
 * humanoid names) and an alias resolver that recognises the rigs people
 * actually export: Mixamo ("mixamorig:LeftUpLeg"), VRM/VRoid ("J_Bip_L_UpperLeg"
 * and the VRM spec names), Unreal ("thigh_l", "calf_l"), Blender/Rigify
 * ("thigh.L", "DEF-thigh.L", "UpperLeg.L" as in Quaternius packs) and the
 * AugEngine auto-rig ("LeftUpperLeg").
 *
 * The resolver works on a normalised name (prefix stripped, lower-case,
 * separators removed, side extracted) so new spellings mostly "just work".
 */

export const HUMANOID_BONES = [
  "hips", "spine", "chest", "upperChest", "neck", "head",
  "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
  "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
  "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
] as const;

export type HumanoidBone = (typeof HUMANOID_BONES)[number];

/** Parent of each canonical bone in the canonical hierarchy (null = root). */
export const HUMANOID_PARENT: Record<HumanoidBone, HumanoidBone | null> = {
  hips: null,
  spine: "hips",
  chest: "spine",
  upperChest: "chest",
  neck: "upperChest",
  head: "neck",
  leftShoulder: "upperChest",
  leftUpperArm: "leftShoulder",
  leftLowerArm: "leftUpperArm",
  leftHand: "leftLowerArm",
  rightShoulder: "upperChest",
  rightUpperArm: "rightShoulder",
  rightLowerArm: "rightUpperArm",
  rightHand: "rightLowerArm",
  leftUpperLeg: "hips",
  leftLowerLeg: "leftUpperLeg",
  leftFoot: "leftLowerLeg",
  leftToes: "leftFoot",
  rightUpperLeg: "hips",
  rightLowerLeg: "rightUpperLeg",
  rightFoot: "rightLowerLeg",
  rightToes: "rightFoot",
};

type Side = "left" | "right" | null;

/** Side-less body-part words → canonical stem (combined with side below). */
const PART_ALIASES: [string[], string][] = [
  [["hips", "hip", "pelvis", "root_hips", "bip01pelvis"], "hips"],
  [["spine", "spine01", "spine0", "abdomen", "waist", "lowerback"], "spine"],
  [["spine1", "spine02", "chest", "torso", "spine2body"], "chest"],
  [["spine2", "spine03", "upperchest", "chestupper"], "upperChest"],
  [["neck", "neck01", "neck1"], "neck"],
  [["head"], "head"],
  [["shoulder", "clavicle", "collar", "collarbone"], "Shoulder"],
  [["arm", "upperarm", "uparm", "armupper"], "UpperArm"],
  [["forearm", "lowerarm", "elbow", "armlower", "forarm"], "LowerArm"],
  [["hand", "wrist"], "Hand"],
  [["upleg", "upperleg", "thigh", "legupper", "femur"], "UpperLeg"],
  [["leg", "lowerleg", "calf", "shin", "knee", "leglower"], "LowerLeg"],
  [["foot", "ankle"], "Foot"],
  [["toebase", "toe", "toes", "ball"], "Toes"],
];

const PREFIXES = [/^mixamorig\d*[:_]/i, /^j_bip_[lrc]_/i, /^def[-_.]/i, /^bip0?1[_ ]?/i, /^armature[|/:]/i, /^characterarmature[|/:]/i, /^cc_base_/i];

function sideOf(raw: string): { side: Side; rest: string } {
  let s = raw;
  // VRoid: J_Bip_L_UpperLeg — the side lives in the stripped prefix.
  const vroid = /^j_bip_([lrc])_/i.exec(s);
  if (vroid) {
    const c = vroid[1]!.toLowerCase();
    return { side: c === "l" ? "left" : c === "r" ? "right" : null, rest: s.slice(vroid[0].length) };
  }
  for (const p of PREFIXES) s = s.replace(p, "");
  const lower = s.toLowerCase();
  const patterns: [RegExp, Side][] = [
    [/^left[_ .-]?/, "left"], [/^right[_ .-]?/, "right"],
    [/[_ .-]?left$/, "left"], [/[_ .-]?right$/, "right"],
    [/^l[_ .-]/, "left"], [/^r[_ .-]/, "right"],
    [/[_ .-]l$/, "left"], [/[_ .-]r$/, "right"],
  ];
  for (const [re, side] of patterns) {
    if (re.test(lower)) return { side, rest: lower.replace(re, "") };
  }
  return { side: null, rest: lower };
}

const squash = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Canonical humanoid bone for a joint name, or null when it isn't a standard bone. */
export function humanoidBoneOf(name: string): HumanoidBone | null {
  const exact = HUMANOID_BONES.find((b) => b.toLowerCase() === name.toLowerCase());
  if (exact) return exact;
  const { side, rest } = sideOf(name.trim());
  const key = squash(rest);
  if (!key) return null;
  for (const [aliases, stem] of PART_ALIASES) {
    if (!aliases.includes(key)) continue;
    const sided = stem[0] === stem[0]!.toUpperCase();
    if (sided) {
      if (!side) return null;
      return `${side}${stem}` as HumanoidBone;
    }
    if (side) return null;
    return stem as HumanoidBone;
  }
  return null;
}

/**
 * Map joint names of a source rig onto a target rig through the canonical
 * humanoid ids. Returns source name → target name for every bone both rigs
 * have. When two joints of one rig resolve to the same bone, the first
 * (closest to the root in file order) wins.
 */
export function mapHumanoidBones(sourceNames: readonly string[], targetNames: readonly string[]): Record<string, string> {
  const byBone = (names: readonly string[]) => {
    const m = new Map<HumanoidBone, string>();
    for (const n of names) {
      const b = humanoidBoneOf(n);
      if (b && !m.has(b)) m.set(b, n);
    }
    return m;
  };
  const src = byBone(sourceNames);
  const tgt = byBone(targetNames);
  const out: Record<string, string> = {};
  for (const [bone, sName] of src) {
    const tName = tgt.get(bone);
    if (tName) out[sName] = tName;
  }
  return out;
}

/**
 * Resolve a whole rig: name aliases first, then the spine by structure. Rigs
 * disagree on what to call the bones between the hips and the neck
 * (Mixamo Spine/Spine1/Spine2, Quaternius Abdomen/Torso/Chest, UE
 * spine_01..05), so the joints on the path hips → neck are re-assigned by
 * order: the first is `spine`, the second `chest`, the last (if 3+) `upperChest`.
 * `parent` is the index of the nearest ancestor joint (-1 for none).
 */
export function resolveHumanoidRig(joints: readonly { name: string; parent: number }[]): Map<number, HumanoidBone> {
  const out = new Map<number, HumanoidBone>();
  const taken = new Set<HumanoidBone>();
  joints.forEach((j, i) => {
    const b = humanoidBoneOf(j.name);
    if (b && !taken.has(b)) {
      out.set(i, b);
      taken.add(b);
    }
  });
  const indexOf = (bone: HumanoidBone) => [...out.entries()].find(([, b]) => b === bone)?.[0];
  const hips = indexOf("hips");
  const top = indexOf("neck") ?? indexOf("head");
  if (hips === undefined || top === undefined) return out;
  const path: number[] = [];
  let p = joints[top]!.parent;
  for (let guard = 0; p >= 0 && p !== hips && guard < 64; p = joints[p]!.parent, guard++) path.unshift(p);
  // Neck not under the hips, or directly on them: keep the name-based result.
  if (p !== hips || path.length === 0) return out;
  for (const i of path) {
    const b = out.get(i);
    if (b === "spine" || b === "chest" || b === "upperChest") out.delete(i);
  }
  for (const b of ["spine", "chest", "upperChest"] as const) {
    for (const [i, bb] of out) if (bb === b) out.delete(i);
  }
  const roles: HumanoidBone[] = path.length === 1 ? ["spine"] : path.length === 2 ? ["spine", "chest"] : ["spine", "chest", ...Array(path.length - 3).fill(null), "upperChest"];
  path.forEach((i, k) => {
    const role = roles[k];
    if (role) out.set(i, role);
  });
  return out;
}

/** Which canonical bones a rig is missing (handy for import warnings). */
export function missingHumanoidBones(names: readonly string[], required: readonly HumanoidBone[] = REQUIRED_BONES): HumanoidBone[] {
  const have = new Set(names.map(humanoidBoneOf).filter(Boolean) as HumanoidBone[]);
  return required.filter((b) => !have.has(b));
}

/** Bones a rig needs before clips can be retargeted onto it meaningfully. */
export const REQUIRED_BONES: readonly HumanoidBone[] = [
  "hips", "spine", "head",
  "leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm",
  "leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg",
];

/** Apple ARKit's 52 face blendshape names — the naming AugEngine lip-sync expects. */
export const ARKIT_52 = [
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
  "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
  "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
  "eyeWideLeft", "eyeWideRight",
  "jawForward", "jawLeft", "jawOpen", "jawRight",
  "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
  "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
  "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
  "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
  "noseSneerLeft", "noseSneerRight", "tongueOut",
] as const;

/** How many of the ARKit-52 shapes a list of morph-target names covers (case-insensitive). */
export function arkitCoverage(morphNames: readonly string[]): { matched: string[]; missing: string[] } {
  const have = new Set(morphNames.map((n) => n.replace(/^.*[.:|]/, "").toLowerCase()));
  const matched = ARKIT_52.filter((n) => have.has(n.toLowerCase()));
  return { matched, missing: ARKIT_52.filter((n) => !have.has(n.toLowerCase())) };
}

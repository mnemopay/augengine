import { skinVertexLBS } from "../characters/skinning.js";
import { m4Create, m4FromTRS, m4Invert, m4Mul, type MutQuat } from "../scene/math.js";
import type { Mat4, Quat, Vec3 } from "../scene/types.js";
import { loadGlb, type AugEngineAnimationFromGltf } from "../scene/loaders/gltf.js";
import { sampleGltfAnimationTrack } from "../scene/loaders/gltf-instance.js";
import { BUTTON } from "./input.js";
import type { Fighter } from "./sim.js";

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

export type RigId = "mara" | "holt";
export type RigClip =
  | "idle"
  | "walk"
  | "crouch"
  | "air"
  | "punch"
  | "kick"
  | "block"
  | "hit"
  | "down"
  | "throw";

const LOOP_CLIPS = new Set<RigClip>(["idle", "walk"]);

interface Profile {
  hipsY: number;
  shoulderZ: number;
  torsoX: number;
  knot: number;
}

const PROFILES: Record<RigId, Profile> = {
  mara: { hipsY: 50, shoulderZ: 5, torsoX: 7, knot: 5 },
  holt: { hipsY: 48, shoulderZ: 8, torsoX: 10, knot: 2.5 },
};

interface JointDef {
  name: string;
  parent: number;
  t: Vec3;
}

interface BoxDef {
  center: Vec3;
  half: Vec3;
  joint: number;
  part: number;
  marker?: "hand" | "head" | "foot";
}

export interface RigMarkers {
  hand: number;
  head: number;
  foot: number;
}

interface RigCache {
  clips: Map<string, AugEngineAnimationFromGltf>;
  bind: Float32Array;
  joints: Uint16Array;
  weights: Float32Array;
  indices: Uint32Array;
  parts: Uint8Array;
  parents: number[];
  localT: Vec3[];
  ibm: Float32Array;
  markers: RigMarkers;
}

const cache = new Map<RigId, RigCache>();

/** Which clip the tick is in. One-shots clamp. Idle and walk loop on poseFrame/60. */
export function clipForFighter(fighter: Fighter): RigClip {
  if (fighter.phase === "knockdown") return "down";
  if (fighter.phase === "blockstun") return "block";
  if (fighter.phase === "hitstun") return "hit";
  if (fighter.phase === "throw" || fighter.phase === "thrown") return "throw";
  if (fighter.phase === "attack") {
    const id = fighter.move?.id;
    if (id === "shin" || id === "spire" || id === "reaping") return "kick";
    return "punch";
  }
  if (fighter.y > 0) return "air";
  if (fighter.crouching) return "crouch";
  if ((fighter.held & (BUTTON.LEFT | BUTTON.RIGHT)) !== 0) return "walk";
  return "idle";
}

export function rigMarkers(id: RigId = "mara"): RigMarkers {
  return ensure(id).markers;
}

export function rigSurface(id: RigId = "mara"): { indices: Uint32Array; parts: Uint8Array } {
  const rig = ensure(id);
  return { indices: rig.indices, parts: rig.parts };
}

/** Skinned vertex positions in mesh space. Time is poseFrame/60, not wall clock. */
export function sampleSkinned(id: RigId, clipName: RigClip, poseFrame: number): Float32Array {
  const rig = ensure(id);
  const clip = rig.clips.get(clipName);
  if (!clip) throw new Error(`${id} has no ${clipName} clip`);
  let time = poseFrame / 60;
  if (LOOP_CLIPS.has(clipName) && clip.duration > 0) {
    time = ((time % clip.duration) + clip.duration) % clip.duration;
  }
  const localT = rig.localT.map((t) => [t[0], t[1], t[2]] as [number, number, number]);
  const localR: Quat[] = rig.localT.map(() => [0, 0, 0, 1]);
  for (const track of clip.tracks) {
    const joint = track.node - 1;
    if (joint < 0 || joint >= localT.length) continue;
    const sampled = sampleGltfAnimationTrack(track, time);
    if (!sampled) continue;
    if (track.path === "translation") {
      localT[joint] = [sampled[0] ?? 0, sampled[1] ?? 0, sampled[2] ?? 0];
    } else if (track.path === "rotation") {
      localR[joint] = [sampled[0] ?? 0, sampled[1] ?? 0, sampled[2] ?? 0, sampled[3] ?? 1];
    }
  }
  const worlds = jointWorlds(rig.parents, localT, localR);
  const bones = new Float32Array(worlds.length * 16);
  const scratch = m4Create();
  for (let j = 0; j < worlds.length; j++) {
    const world = worlds[j];
    if (!world) continue;
    const ibm = rig.ibm.subarray(j * 16, j * 16 + 16);
    m4Mul(scratch, world, ibm);
    bones.set(scratch, j * 16);
  }
  const out = new Float32Array(rig.bind.length);
  const pos = new Float32Array(3);
  const j4 = new Uint16Array(4);
  const w4 = new Float32Array(4);
  for (let v = 0; v < rig.bind.length / 3; v++) {
    pos[0] = rig.bind[v * 3]!;
    pos[1] = rig.bind[v * 3 + 1]!;
    pos[2] = rig.bind[v * 3 + 2]!;
    j4.set(rig.joints.subarray(v * 4, v * 4 + 4));
    w4.set(rig.weights.subarray(v * 4, v * 4 + 4));
    const skinned = skinVertexLBS(pos, j4, w4, bones);
    out[v * 3] = skinned[0];
    out[v * 3 + 1] = skinned[1];
    out[v * 3 + 2] = skinned[2];
  }
  return out;
}

function ensure(id: RigId): RigCache {
  const hit = cache.get(id);
  if (hit) return hit;
  const loaded = loadGlb(buildRigGlb(id));
  const skin = loaded.skin;
  const mesh = loaded.mesh;
  if (!skin || !mesh?.joints || !mesh.weights) throw new Error(`${id} rig is not skinned`);
  const nodes = loaded.parsed.root.nodes ?? [];
  const parents = skin.joints.map(() => -1);
  const localT: Vec3[] = [];
  for (let s = 0; s < skin.joints.length; s++) {
    const nodeIndex = skin.joints[s];
    if (nodeIndex === undefined) throw new Error("skin joint missing");
    const node = nodes[nodeIndex];
    const t = node?.translation ?? [0, 0, 0];
    localT.push([t[0] ?? 0, t[1] ?? 0, t[2] ?? 0]);
    for (const child of node?.children ?? []) {
      const childSkin = skin.joints.indexOf(child);
      if (childSkin >= 0) parents[childSkin] = s;
    }
  }
  const markers = markersFor(id);
  const built: RigCache = {
    clips: new Map(loaded.animations.map((clip) => [clip.name, clip])),
    bind: positionsOf(mesh.vertices, mesh.vertexStride),
    joints: mesh.joints,
    weights: mesh.weights,
    indices: mesh.indices,
    parts: partsFor(id),
    parents,
    localT,
    ibm: skin.inverseBindMatrices,
    markers,
  };
  cache.set(id, built);
  return built;
}

function positionsOf(vertices: Float32Array, stride: number): Float32Array {
  const count = vertices.length / stride;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    out[i * 3] = vertices[i * stride]!;
    out[i * 3 + 1] = vertices[i * stride + 1]!;
    out[i * 3 + 2] = vertices[i * stride + 2]!;
  }
  return out;
}

function jointWorlds(parents: number[], localT: ReadonlyArray<Vec3>, localR: readonly Quat[]): Mat4[] {
  const worlds: Mat4[] = [];
  for (let i = 0; i < parents.length; i++) {
    const t = localT[i] ?? [0, 0, 0];
    const r = localR[i] ?? [0, 0, 0, 1];
    const local = m4FromTRS(m4Create(), t, r, [1, 1, 1]);
    const parent = parents[i] ?? -1;
    if (parent < 0) {
      worlds.push(local);
      continue;
    }
    const parentWorld = worlds[parent];
    if (!parentWorld) throw new Error("joint parent is not posed yet");
    worlds.push(m4Mul(m4Create(), parentWorld, local));
  }
  return worlds;
}

function zQuat(degrees: number): MutQuat {
  const half = (degrees * Math.PI) / 360;
  return [0, 0, Math.sin(half), Math.cos(half)];
}

interface TrackDef {
  joint: number;
  path: "translation" | "rotation";
  times: number[];
  values: number[];
}

function buildRigGlb(id: RigId): Uint8Array {
  const profile = PROFILES[id];
  const joints = jointDefs(profile);
  const worlds = jointWorlds(
    joints.map((joint) => joint.parent),
    joints.map((joint) => joint.t),
    joints.map(() => [0, 0, 0, 1]),
  );
  const ibm = new Float32Array(joints.length * 16);
  for (let i = 0; i < worlds.length; i++) {
    const world = worlds[i];
    if (!world) throw new Error("missing bind world");
    const inverse = m4Invert(m4Create(), world);
    if (!inverse) throw new Error("bind pose is singular");
    ibm.set(inverse, i * 16);
  }
  const at = (index: number): Vec3 => {
    const world = worlds[index];
    if (!world) throw new Error("missing joint");
    return [world[12] ?? 0, world[13] ?? 0, world[14] ?? 0];
  };
  const mid = (a: number, b: number): Vec3 => {
    const pa = at(a);
    const pb = at(b);
    return [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
  };
  const boxes: BoxDef[] = [
    { center: at(0), half: [profile.torsoX, 6, 7], joint: 0, part: 0 },
    { center: mid(1, 1), half: [profile.torsoX, 12, 7], joint: 1, part: 0 },
    { center: at(2), half: [8, 9, 7], joint: 2, part: 1, marker: "head" },
    { center: [at(2)[0] - profile.knot, at(2)[1] + 3, at(2)[2]], half: [profile.knot * 0.7, 4, 4], joint: 2, part: 2 },
    { center: mid(3, 4), half: [4, 8, 4], joint: 3, part: 0 },
    { center: mid(4, 5), half: [3.5, 7, 3.5], joint: 4, part: 0 },
    { center: at(5), half: [4, 4, 4], joint: 5, part: 3, marker: "hand" },
    { center: mid(6, 7), half: [4, 8, 4], joint: 6, part: 0 },
    { center: mid(7, 8), half: [3.5, 7, 3.5], joint: 7, part: 0 },
    { center: at(8), half: [4, 4, 4], joint: 8, part: 3 },
    { center: mid(9, 10), half: [4, 9, 4], joint: 9, part: 0 },
    { center: mid(10, 11), half: [3.5, 9, 3.5], joint: 10, part: 0 },
    { center: at(11), half: [6, 3, 3], joint: 11, part: 3, marker: "foot" },
    { center: mid(12, 13), half: [4, 9, 4], joint: 12, part: 0 },
    { center: mid(13, 14), half: [3.5, 9, 3.5], joint: 13, part: 0 },
    { center: at(14), half: [6, 3, 3], joint: 14, part: 3 },
  ];
  // Spine box should sit on the spine, not a zero-length midpoint.
  const spine = at(1);
  boxes[1] = { center: [spine[0], spine[1] + 4, spine[2]], half: [profile.torsoX, 12, 7], joint: 1, part: 0 };

  const positions: number[] = [];
  const jointIndex: number[] = [];
  const weights: number[] = [];
  const indices: number[] = [];
  const parts: number[] = [];
  const markers: RigMarkers = { hand: 0, head: 0, foot: 0 };
  for (const box of boxes) {
    const start = positions.length / 3;
    if (box.marker) markers[box.marker] = start;
    const [cx, cy, cz] = box.center;
    const [hx, hy, hz] = box.half;
    const corners: Vec3[] = [];
    for (const z of [-hz, hz]) {
      for (const y of [-hy, hy]) {
        for (const x of [-hx, hx]) corners.push([cx + x, cy + y, cz + z]);
      }
    }
    for (const corner of corners) {
      positions.push(corner[0], corner[1], corner[2]);
      jointIndex.push(box.joint, 0, 0, 0);
      weights.push(1, 0, 0, 0);
      parts.push(box.part);
    }
    const faces = [
      [0, 1, 3, 2],
      [4, 6, 7, 5],
      [0, 4, 5, 1],
      [2, 3, 7, 6],
      [0, 2, 6, 4],
      [1, 5, 7, 3],
    ];
    for (const face of faces) {
      const [a, b, c, d] = face;
      if (a === undefined || b === undefined || c === undefined || d === undefined) continue;
      indices.push(start + a, start + b, start + c, start + a, start + c, start + d);
    }
  }
  rememberSurface(id, new Uint8Array(parts), markers);

  const tracks = clipTracks(profile, (name) => {
    const index = joints.findIndex((joint) => joint.name === name);
    if (index < 0) throw new Error(`rig has no ${name}`);
    return index;
  });
  return packGlb(joints, ibm, positions, jointIndex, weights, indices, tracks);
}

const surfaceNotes = new Map<RigId, { parts: Uint8Array; markers: RigMarkers }>();

function rememberSurface(id: RigId, parts: Uint8Array, markers: RigMarkers): void {
  surfaceNotes.set(id, { parts, markers });
}

function partsFor(id: RigId): Uint8Array {
  const note = surfaceNotes.get(id);
  if (!note) throw new Error(`${id} surface was not built`);
  return note.parts;
}

function markersFor(id: RigId): RigMarkers {
  const note = surfaceNotes.get(id);
  if (!note) throw new Error(`${id} markers were not built`);
  return note.markers;
}

function jointDefs(profile: Profile): JointDef[] {
  const z = profile.shoulderZ;
  return [
    { name: "hips", parent: -1, t: [0, profile.hipsY, 0] },
    { name: "spine", parent: 0, t: [0, 18, 0] },
    { name: "head", parent: 1, t: [0, 16, 0] },
    { name: "shoulder", parent: 1, t: [3, 10, z] },
    { name: "elbow", parent: 3, t: [0, -16, 0] },
    { name: "hand", parent: 4, t: [0, -14, 0] },
    { name: "shoulderFar", parent: 1, t: [3, 10, -z] },
    { name: "elbowFar", parent: 6, t: [0, -16, 0] },
    { name: "handFar", parent: 7, t: [0, -14, 0] },
    { name: "thighNear", parent: 0, t: [0, -4, z] },
    { name: "kneeNear", parent: 9, t: [0, -20, 0] },
    { name: "footNear", parent: 10, t: [4, -22, 0] },
    { name: "thighFar", parent: 0, t: [0, -4, -z] },
    { name: "kneeFar", parent: 12, t: [0, -20, 0] },
    { name: "footFar", parent: 13, t: [4, -22, 0] },
  ];
}

function clipTracks(profile: Profile, joint: (name: string) => number): Record<RigClip, TrackDef[]> {
  const hipsDown: Vec3 = [0, profile.hipsY - 24, 0];
  const identity = [0, 0, 0, 1];
  const breathe = zQuat(7);
  const walkA = zQuat(28);
  const walkB = zQuat(-28);
  const knee = zQuat(22);
  return {
    idle: [{ joint: joint("spine"), path: "rotation", times: [0, 0.5, 1], values: [...identity, ...breathe, ...identity] }],
    walk: [
      { joint: joint("thighNear"), path: "rotation", times: [0, 0.5, 1], values: [...walkA, ...walkB, ...walkA] },
      { joint: joint("thighFar"), path: "rotation", times: [0, 0.5, 1], values: [...walkB, ...walkA, ...walkB] },
      { joint: joint("kneeNear"), path: "rotation", times: [0, 0.5, 1], values: [...identity, ...knee, ...identity] },
      { joint: joint("kneeFar"), path: "rotation", times: [0, 0.5, 1], values: [...knee, ...identity, ...knee] },
    ],
    crouch: [
      { joint: joint("hips"), path: "translation", times: [0], values: [...hipsDown] },
      { joint: joint("thighNear"), path: "rotation", times: [0], values: [...zQuat(35)] },
      { joint: joint("thighFar"), path: "rotation", times: [0], values: [...zQuat(35)] },
      { joint: joint("kneeNear"), path: "rotation", times: [0], values: [...zQuat(-40)] },
      { joint: joint("kneeFar"), path: "rotation", times: [0], values: [...zQuat(-40)] },
    ],
    air: [
      { joint: joint("thighNear"), path: "rotation", times: [0], values: [...zQuat(40)] },
      { joint: joint("thighFar"), path: "rotation", times: [0], values: [...zQuat(40)] },
      { joint: joint("kneeNear"), path: "rotation", times: [0], values: [...zQuat(-50)] },
      { joint: joint("kneeFar"), path: "rotation", times: [0], values: [...zQuat(-50)] },
    ],
    punch: [
      { joint: joint("shoulder"), path: "rotation", times: [0, 8 / 60, 20 / 60], values: [...identity, ...zQuat(58), ...identity] },
      { joint: joint("elbow"), path: "rotation", times: [0, 8 / 60, 20 / 60], values: [...identity, ...zQuat(36), ...identity] },
    ],
    kick: [
      { joint: joint("thighNear"), path: "rotation", times: [0, 8 / 60, 20 / 60], values: [...identity, ...zQuat(62), ...identity] },
      { joint: joint("kneeNear"), path: "rotation", times: [0, 8 / 60, 20 / 60], values: [...identity, ...zQuat(-20), ...identity] },
    ],
    block: [
      { joint: joint("shoulder"), path: "rotation", times: [0], values: [...zQuat(80)] },
      { joint: joint("elbow"), path: "rotation", times: [0], values: [...zQuat(-30)] },
    ],
    hit: [{ joint: joint("spine"), path: "rotation", times: [0], values: [...zQuat(-18)] }],
    down: [
      { joint: joint("hips"), path: "translation", times: [0], values: [18, 14, 0] },
      { joint: joint("hips"), path: "rotation", times: [0], values: [...zQuat(-80)] },
    ],
    throw: [
      { joint: joint("spine"), path: "rotation", times: [0], values: [...zQuat(16)] },
      { joint: joint("shoulder"), path: "rotation", times: [0], values: [...zQuat(40)] },
    ],
  };
}

interface PackedView {
  byteOffset: number;
  byteLength: number;
}

function packGlb(
  joints: JointDef[],
  ibm: Float32Array,
  positions: number[],
  jointIndex: number[],
  weights: number[],
  indices: number[],
  tracks: Record<RigClip, TrackDef[]>,
): Uint8Array {
  const binParts: Uint8Array[] = [];
  let cursor = 0;
  const push = (bytes: Uint8Array, align: number): PackedView => {
    const pad = (align - (cursor % align)) % align;
    if (pad) {
      binParts.push(new Uint8Array(pad));
      cursor += pad;
    }
    const view = { byteOffset: cursor, byteLength: bytes.length };
    binParts.push(bytes);
    cursor += bytes.length;
    return view;
  };
  const f32 = (values: ArrayLike<number>): PackedView => push(new Uint8Array(Float32Array.from(values).buffer), 4);
  const u16 = (values: ArrayLike<number>): PackedView => push(new Uint8Array(Uint16Array.from(values).buffer), 2);

  const positionView = f32(positions);
  const indexView = u16(indices);
  const jointsView = u16(jointIndex);
  const weightsView = f32(weights);
  const ibmView = f32(ibm);

  const accessors: object[] = [
    { bufferView: 0, componentType: 5126, count: positions.length / 3, type: "VEC3" },
    { bufferView: 1, componentType: 5123, count: indices.length, type: "SCALAR" },
    { bufferView: 2, componentType: 5123, count: jointIndex.length / 4, type: "VEC4" },
    { bufferView: 3, componentType: 5126, count: weights.length / 4, type: "VEC4" },
    { bufferView: 4, componentType: 5126, count: ibm.length / 16, type: "MAT4" },
  ];
  const bufferViews: object[] = [positionView, indexView, jointsView, weightsView, ibmView].map((view) => ({
    buffer: 0,
    byteOffset: view.byteOffset,
    byteLength: view.byteLength,
  }));

  const clipNames = Object.keys(tracks) as RigClip[];
  const animations = clipNames.map((name) => {
    const channels: object[] = [];
    const samplers: object[] = [];
    for (const track of tracks[name]) {
      const timeView = f32(track.times);
      const valueView = f32(track.values);
      const timeAccessor = accessors.length;
      accessors.push({ bufferView: bufferViews.length, componentType: 5126, count: track.times.length, type: "SCALAR" });
      bufferViews.push({ buffer: 0, byteOffset: timeView.byteOffset, byteLength: timeView.byteLength });
      const valueAccessor = accessors.length;
      accessors.push({
        bufferView: bufferViews.length,
        componentType: 5126,
        count: track.values.length / (track.path === "rotation" ? 4 : 3),
        type: track.path === "rotation" ? "VEC4" : "VEC3",
      });
      bufferViews.push({ buffer: 0, byteOffset: valueView.byteOffset, byteLength: valueView.byteLength });
      const sampler = samplers.length;
      samplers.push({ input: timeAccessor, output: valueAccessor, interpolation: "LINEAR" });
      channels.push({ sampler, target: { node: track.joint + 1, path: track.path } });
    }
    return { name, channels, samplers };
  });

  const bin = new Uint8Array(cursor);
  let at = 0;
  for (const part of binParts) {
    bin.set(part, at);
    at += part.length;
  }
  const binPad = (4 - (bin.length % 4)) % 4;
  const binChunk = bin.length + binPad;

  const nodes: object[] = [
    { name: "body", mesh: 0, skin: 0 },
    ...joints.map((joint, jointIndex) => {
      const children = joints.flatMap((other, index) => (other.parent === jointIndex ? [index + 1] : []));
      const node: { name: string; translation: number[]; children?: number[] } = {
        name: joint.name,
        translation: [joint.t[0], joint.t[1], joint.t[2]],
      };
      if (children.length > 0) node.children = children;
      return node;
    }),
  ];

  const json = {
    asset: { version: "2.0", generator: "augengine-fighter-rig" },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes,
    meshes: [
      {
        name: "body",
        primitives: [
          {
            attributes: { POSITION: 0, JOINTS_0: 2, WEIGHTS_0: 3 },
            indices: 1,
          },
        ],
      },
    ],
    skins: [{ joints: joints.map((_, index) => index + 1), inverseBindMatrices: 4 }],
    animations,
    accessors,
    bufferViews,
    buffers: [{ byteLength: binChunk }],
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const jsonChunk = jsonBytes.length + jsonPad;
  const total = 12 + 8 + jsonChunk + 8 + binChunk;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, GLB_VERSION, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonChunk, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(jsonBytes, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBytes.length + i] = 0x20;
  const binAt = 20 + jsonChunk;
  view.setUint32(binAt, binChunk, true);
  view.setUint32(binAt + 4, CHUNK_BIN, true);
  out.set(bin, binAt + 8);
  return out;
}

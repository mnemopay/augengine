/**
 * Proof-room head.
 *
 * One mesh, about 21 cm tall, with the 52 blendshape slots the renderer
 * already uploads. Deltas are written on the slots `cherryToArkit` actually
 * fills. Those slots are not the audio2face name order (jaw is 23 here,
 * not 17; blink is 8, not 0). Empty slots stay zero.
 *
 * The face points +Z. The proof-room wide camera is at [0, 1.7, 5.6]
 * looking down −Z, so the nose points at that camera. The line is an
 * authored performance of "I kept the light." It is not a captured actor.
 */

import { ARKIT_BLENDSHAPE_COUNT } from "../runtime/live/webgpu/blendshape-constants.js";
import type { MeshData } from "../runtime/live/webgpu/primitives.js";
import type { MorphTargetSource } from "../runtime/live/webgpu/morph-deltas.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../scene/types.js";
import {
  CHERRY_VISEMES,
  PROCEDURAL_CHANNELS,
  cherryToArkit,
  emptyCherryWeights,
  type CherryViseme,
  type CherryWeightFrame,
  type ProceduralChannel,
} from "./cherry-visemes.js";

export const PROOF_HEAD_ENTITY_ID = "proof.head";
export const PROOF_HEAD_GENERATOR = "proof-head";
export const PROOF_HEAD_SEED = 1;
/** `meshKeyForEntity` joins procedural geometry as `proc:<generator>:<seed>`. */
export const PROOF_HEAD_MESH_KEY = `proc:${PROOF_HEAD_GENERATOR}:${PROOF_HEAD_SEED}`;
/** Interleaved position, normal, uv. `uploadMesh` must be told this stride. */
export const PROOF_HEAD_VERTEX_STRIDE = 8 as const;

export const PROOF_HEAD_LINE_TEXT = "I kept the light.";
export const PROOF_HEAD_LINE_DURATION = 3.6;
/** Open vowel of "light". Jaw slot is high. */
export const PROOF_HEAD_LINE_SPEAK_S = 1.28;
/** Swallowed /k/ in "kept". Jaw moves, and not all the way. */
export const PROOF_HEAD_LINE_SWALLOW_S = 0.4;
/** Brow arrives after the mouth has already closed. */
export const PROOF_HEAD_LINE_BROW_S = 2.05;
/** Left lid closes. The right lid only dips. */
export const PROOF_HEAD_LINE_BLINK_S = 2.42;

/** Right-foreground placement under the existing proof-room camera. */
export const PROOF_HEAD_POSITION: Vec3 = [0.55, 1.46, 4.35];

export interface ProofHeadSlots {
  jawOpen: number;
  mouthClose: number;
  mouthFunnel: number;
  mouthPucker: number;
  mouthShrugUpper: number;
  tongueOut: number;
  browInnerUp: number;
  browOuterUpL: number;
  browOuterUpR: number;
  browDown: number;
  eyeLookIn: number;
  eyeLookOut: number;
  blinkL: number;
  blinkR: number;
}

export interface ProofHeadRegions {
  lowerLip: number[];
  upperLip: number[];
  chin: number[];
  browInner: number[];
  browOuterL: number[];
  browOuterR: number[];
  lidL: number[];
  lidR: number[];
  eyeL: number[];
  eyeR: number[];
  tongue: number[];
}

export interface ProofHeadMesh extends MeshData {
  regions: ProofHeadRegions;
  slots: ProofHeadSlots;
}

const LAT = 24;
const LON = 32;
const NOSE_PHI = Math.PI / 2;
const SKIN: Rgb = [0.73, 0.51, 0.4];
const LIP: Rgb = [0.62, 0.28, 0.3];
const CAVITY: Rgb = [0.12, 0.05, 0.05];
const TONGUE: Rgb = [0.55, 0.22, 0.24];

type Rgb = readonly [number, number, number];

interface Sculpt {
  p: [number, number, number];
  n: [number, number, number];
  mouth: number;
  dPhi: number;
}

function wrapPi(d: number): number {
  let x = d;
  while (x > Math.PI) x -= Math.PI * 2;
  while (x < -Math.PI) x += Math.PI * 2;
  return x;
}

function sculpt(theta: number, phi: number): Sculpt {
  const nx = Math.sin(theta) * Math.cos(phi);
  const ny = Math.cos(theta);
  const nz = Math.sin(theta) * Math.sin(phi);
  const dPhi = wrapPi(phi - NOSE_PHI);
  let x = nx * 0.078;
  let y = ny * 0.105;
  let z = nz * 0.086;

  const nose =
    Math.exp(-(dPhi * dPhi) / (0.12 * 0.12)) *
    Math.exp(-((theta - 1.62) ** 2) / (0.22 * 0.22));
  z += nose * 0.034;
  y -= nose * 0.004;

  const socket = (d: number) =>
    Math.exp(-((theta - 1.309) ** 2) / (0.08 * 0.08)) *
    Math.exp(-((dPhi - d) ** 2) / (0.1 * 0.1));
  z -= (socket(-0.4) + socket(0.4)) * 0.012;

  const brow =
    Math.exp(-((theta - 1.16) ** 2) / (0.05 * 0.05)) *
    Math.exp(-(dPhi * dPhi) / (0.45 * 0.45));
  z += brow * 0.01;
  y += brow * 0.004;

  const mouth =
    Math.exp(-((theta - 1.963) ** 2) / (0.04 * 0.04)) *
    Math.exp(-(dPhi * dPhi) / (0.07 * 0.07));
  z -= mouth * 0.018;

  const lip =
    Math.exp(-((theta - 1.963) ** 2) / (0.1 * 0.1)) *
    Math.exp(-(dPhi * dPhi) / (0.16 * 0.16));
  z += lip * 0.008 * (1 - mouth);

  const chin =
    Math.exp(-((theta - 2.28) ** 2) / (0.12 * 0.12)) *
    Math.exp(-(dPhi * dPhi) / (0.2 * 0.2));
  z += chin * 0.012;
  y -= chin * 0.006;

  return { p: [x, y, z], n: [nx, ny, nz], mouth, dPhi };
}

function visemeIndices(viseme: CherryViseme): number[] {
  const frame = emptyCherryWeights();
  frame.visemes.sil = 0;
  frame.visemes[viseme] = 1;
  return indicesOf(frame);
}

function channelIndex(channel: ProceduralChannel): number {
  const frame = emptyCherryWeights();
  frame.visemes.sil = 0;
  frame.procedural[channel] = 1;
  const found = indicesOf(frame);
  if (found.length !== 1) {
    throw new Error(`Cherry channel ${channel} maps to ${found.length} slots`);
  }
  return found[0]!;
}

function indicesOf(frame: CherryWeightFrame): number[] {
  const weights = cherryToArkit(frame);
  const found: number[] = [];
  for (let i = 0; i < weights.length; i++) {
    if ((weights[i] ?? 0) > 0) found.push(i);
  }
  return found;
}

function onlySlot(name: string, indices: readonly number[]): number {
  if (indices.length !== 1) {
    throw new Error(`${name} maps to [${indices.join(", ")}]`);
  }
  return indices[0]!;
}

function extraSlot(name: string, indices: readonly number[], known: number): number {
  const rest = indices.filter((index) => index !== known);
  if (rest.length !== 1) {
    throw new Error(`${name} maps to [${indices.join(", ")}]`);
  }
  return rest[0]!;
}

/** Slots Cherry writes today. Built from `cherryToArkit`, not from the audio2face names. */
export function proofHeadCherrySlots(): ProofHeadSlots {
  const jawOpen = onlySlot("aa", visemeIndices("aa"));
  const slots: ProofHeadSlots = {
    jawOpen,
    mouthClose: extraSlot("PP", visemeIndices("PP"), jawOpen),
    mouthFunnel: onlySlot("FF", visemeIndices("FF")),
    mouthPucker: onlySlot("I", visemeIndices("I")),
    mouthShrugUpper: extraSlot("E", visemeIndices("E"), jawOpen),
    tongueOut: extraSlot("TH", visemeIndices("TH"), jawOpen),
    browInnerUp: channelIndex("browInnerUp"),
    browOuterUpL: channelIndex("browOuterUpL"),
    browOuterUpR: channelIndex("browOuterUpR"),
    browDown: channelIndex("browDown"),
    eyeLookIn: channelIndex("eyeLookIn"),
    eyeLookOut: channelIndex("eyeLookOut"),
    blinkL: channelIndex("blinkL"),
    blinkR: channelIndex("blinkR"),
  };
  const used = Object.values(slots);
  if (new Set(used).size !== used.length) {
    throw new Error(`Cherry face slots collide: ${used.join(", ")}`);
  }
  return slots;
}

interface LineKey {
  t: number;
  frame: CherryWeightFrame;
}

function lineKey(
  t: number,
  visemes: Partial<Record<CherryViseme, number>>,
  procedural: Partial<Record<ProceduralChannel, number>> = {},
): LineKey {
  const frame = emptyCherryWeights();
  frame.visemes.sil = 0;
  for (const viseme of CHERRY_VISEMES) {
    const weight = visemes[viseme];
    if (weight !== undefined) frame.visemes[viseme] = weight;
  }
  for (const channel of PROCEDURAL_CHANNELS) {
    const weight = procedural[channel];
    if (weight !== undefined) frame.procedural[channel] = weight;
  }
  return { t, frame };
}

const LINE_KEYS: readonly LineKey[] = [
  lineKey(0, { sil: 1 }),
  lineKey(0.18, { I: 0.7 }),
  lineKey(PROOF_HEAD_LINE_SWALLOW_S, { kk: 0.35 }),
  lineKey(0.62, { E: 0.8 }),
  lineKey(0.82, { PP: 0.92 }),
  lineKey(1.02, { TH: 0.75 }),
  lineKey(PROOF_HEAD_LINE_SPEAK_S, { aa: 0.9 }),
  lineKey(1.48, { I: 0.4 }),
  lineKey(1.68, { sil: 1 }),
  lineKey(1.85, { sil: 1 }, { browInnerUp: 0.12 }),
  lineKey(PROOF_HEAD_LINE_BROW_S, { sil: 1 }, { browInnerUp: 0.7, browOuterUpL: 0.28, browOuterUpR: 0.06 }),
  lineKey(2.28, { sil: 1 }, { browInnerUp: 0.35 }),
  lineKey(PROOF_HEAD_LINE_BLINK_S, { sil: 1 }, { blinkL: 0.95, blinkR: 0.28, browInnerUp: 0.12 }),
  lineKey(2.68, { sil: 1 }),
  lineKey(2.95, { sil: 1 }, { eyeLookIn: 0.4 }),
  lineKey(3.25, { sil: 1 }, { eyeLookIn: 0.08 }),
  lineKey(PROOF_HEAD_LINE_DURATION, { sil: 1 }),
];

function copyFrame(frame: CherryWeightFrame): CherryWeightFrame {
  const out = emptyCherryWeights();
  for (const viseme of CHERRY_VISEMES) out.visemes[viseme] = frame.visemes[viseme] ?? 0;
  for (const channel of PROCEDURAL_CHANNELS) out.procedural[channel] = frame.procedural[channel] ?? 0;
  return out;
}

function lerpFrame(a: CherryWeightFrame, b: CherryWeightFrame, u: number): CherryWeightFrame {
  const out = emptyCherryWeights();
  for (const viseme of CHERRY_VISEMES) {
    out.visemes[viseme] = (a.visemes[viseme] ?? 0) * (1 - u) + (b.visemes[viseme] ?? 0) * u;
  }
  for (const channel of PROCEDURAL_CHANNELS) {
    out.procedural[channel] = (a.procedural[channel] ?? 0) * (1 - u) + (b.procedural[channel] ?? 0) * u;
  }
  return out;
}

/** Cherry frame of the recorded line. `tSeconds` loops. */
export function proofHeadLineFrame(tSeconds: number): CherryWeightFrame {
  let t = tSeconds % PROOF_HEAD_LINE_DURATION;
  if (t < 0) t += PROOF_HEAD_LINE_DURATION;
  const first = LINE_KEYS[0]!;
  if (t <= first.t) return copyFrame(first.frame);
  const last = LINE_KEYS[LINE_KEYS.length - 1]!;
  if (t >= last.t) return copyFrame(last.frame);
  for (let i = 0; i < LINE_KEYS.length - 1; i++) {
    const a = LINE_KEYS[i]!;
    const b = LINE_KEYS[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      return lerpFrame(a.frame, b.frame, span === 0 ? 0 : (t - a.t) / span);
    }
  }
  return copyFrame(last.frame);
}

/** 52 weights for `setBlendshapeWeights`. Writes `out` when it is long enough. */
export function proofHeadLineWeights(tSeconds: number, out?: Float32Array): Float32Array {
  const weights = cherryToArkit(proofHeadLineFrame(tSeconds));
  if (!out) return weights;
  if (out.length !== ARKIT_BLENDSHAPE_COUNT) {
    throw new Error(`proofHeadLineWeights: out length ${out.length} != ${ARKIT_BLENDSHAPE_COUNT}`);
  }
  out.set(weights);
  return out;
}

/**
 * Rest pose plus `sum(weight[k] * delta[k])`, the same sum the morph
 * vertex shader applies before the model matrix.
 */
export function deformProofHead(mesh: MeshData, weights: ArrayLike<number>): Float32Array {
  const count = mesh.vertexCount;
  const out = new Float32Array(count * 3);
  const targets = mesh.morphTargets ?? [];
  for (let v = 0; v < count; v++) {
    let x = mesh.vertices[v * PROOF_HEAD_VERTEX_STRIDE] ?? 0;
    let y = mesh.vertices[v * PROOF_HEAD_VERTEX_STRIDE + 1] ?? 0;
    let z = mesh.vertices[v * PROOF_HEAD_VERTEX_STRIDE + 2] ?? 0;
    for (let k = 0; k < targets.length; k++) {
      const weight = weights[k] ?? 0;
      if (weight === 0) continue;
      const delta = targets[k]!.positions;
      x += weight * (delta[v * 3] ?? 0);
      y += weight * (delta[v * 3 + 1] ?? 0);
      z += weight * (delta[v * 3 + 2] ?? 0);
    }
    out[v * 3] = x;
    out[v * 3 + 1] = y;
    out[v * 3 + 2] = z;
  }
  return out;
}

export function buildProofHeadMesh(): ProofHeadMesh {
  const slots = proofHeadCherrySlots();
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const regions: ProofHeadRegions = {
    lowerLip: [],
    upperLip: [],
    chin: [],
    browInner: [],
    browOuterL: [],
    browOuterR: [],
    lidL: [],
    lidR: [],
    eyeL: [],
    eyeR: [],
    tongue: [],
  };

  const pushVert = (
    px: number,
    py: number,
    pz: number,
    nx: number,
    ny: number,
    nz: number,
    u: number,
    v: number,
    rgb: Rgb,
  ): number => {
    const index = positions.length / 3;
    positions.push(px, py, pz);
    normals.push(nx, ny, nz);
    uvs.push(u, v);
    colors.push(rgb[0], rgb[1], rgb[2], 1);
    return index;
  };

  const addQuad = (i0: number, i1: number, i2: number, i3: number): void => {
    indices.push(i0, i1, i2, i0, i2, i3);
  };

  for (let y = 0; y <= LAT; y++) {
    const theta = (y / LAT) * Math.PI;
    for (let x = 0; x <= LON; x++) {
      const phi = (x / LON) * Math.PI * 2;
      const s = sculpt(theta, phi);
      let rgb = SKIN;
      if (s.mouth > 0.45) rgb = CAVITY;
      else if (Math.abs(theta - 1.963) < 0.18 && Math.abs(s.dPhi) < 0.55) rgb = LIP;
      pushVert(s.p[0], s.p[1], s.p[2], s.n[0], s.n[1], s.n[2], x / LON, y / LAT, rgb);
      if (s.mouth > 0.45) continue;
      const front = Math.abs(s.dPhi) < 0.5;
      if (front && theta >= 2.02 && theta <= 2.16) regions.lowerLip.push((positions.length / 3) - 1);
      else if (front && theta >= 1.75 && theta <= 1.9) regions.upperLip.push((positions.length / 3) - 1);
      else if (front && theta >= 2.16 && theta <= 2.45) regions.chin.push((positions.length / 3) - 1);
      if (theta >= 1.02 && theta <= 1.3) {
        const absPhi = Math.abs(s.dPhi);
        if (absPhi <= 0.22) regions.browInner.push((positions.length / 3) - 1);
        else if (s.dPhi <= -0.3 && s.dPhi >= -0.7) regions.browOuterL.push((positions.length / 3) - 1);
        else if (s.dPhi >= 0.3 && s.dPhi <= 0.7) regions.browOuterR.push((positions.length / 3) - 1);
      }
    }
  }
  for (let y = 0; y < LAT; y++) {
    for (let x = 0; x < LON; x++) {
      const i0 = y * (LON + 1) + x;
      const i1 = i0 + 1;
      const i2 = i0 + (LON + 1) + 1;
      const i3 = i0 + (LON + 1);
      addQuad(i0, i1, i2, i3);
    }
  }

  const addEye = (center: [number, number, number]): number[] => {
    const lat = 4;
    const lon = 8;
    const radius = 0.011;
    const ids: number[] = [];
    for (let y = 0; y <= lat; y++) {
      const theta = (y / lat) * Math.PI;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      for (let x = 0; x <= lon; x++) {
        const phi = (x / lon) * Math.PI * 2;
        const nx = sinT * Math.cos(phi);
        const ny = cosT;
        const nz = sinT * Math.sin(phi);
        let rgb: Rgb = [0.92, 0.91, 0.88];
        if (nz > 0.55 && sinT < 0.75) rgb = [0.22, 0.16, 0.12];
        if (nz > 0.82 && sinT < 0.45) rgb = [0.04, 0.03, 0.03];
        ids.push(pushVert(
          center[0] + nx * radius,
          center[1] + ny * radius,
          center[2] + nz * radius,
          nx,
          ny,
          nz,
          x / lon,
          y / lat,
          rgb,
        ));
      }
    }
    for (let y = 0; y < lat; y++) {
      for (let x = 0; x < lon; x++) {
        const i0 = ids[y * (lon + 1) + x]!;
        const i1 = ids[y * (lon + 1) + x + 1]!;
        const i2 = ids[(y + 1) * (lon + 1) + x + 1]!;
        const i3 = ids[(y + 1) * (lon + 1) + x]!;
        addQuad(i0, i1, i2, i3);
      }
    }
    return ids;
  };

  const eyeCenter = (side: 1 | -1): [number, number, number] => {
    const s = sculpt(1.309, NOSE_PHI - side * 0.4);
    return [s.p[0], s.p[1], s.p[2] + 0.012];
  };

  const leftEye = eyeCenter(1);
  const rightEye = eyeCenter(-1);
  regions.eyeL = addEye(leftEye);
  regions.eyeR = addEye(rightEye);

  const addLid = (eye: [number, number, number]): number[] => {
    const cols = 6;
    const rows = 4;
    const ids: number[] = [];
    for (let r = 0; r < rows; r++) {
      const v = r / (rows - 1);
      const y = eye[1] + 0.004 + v * 0.014;
      for (let c = 0; c < cols; c++) {
        const u = c / (cols - 1);
        const x = eye[0] + (u - 0.5) * 0.026;
        const z = eye[2] + 0.008 + 0.003 * Math.sin(u * Math.PI);
        ids.push(pushVert(x, y, z, 0, 0.15, 0.99, u, v, SKIN));
      }
    }
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const i0 = ids[r * cols + c]!;
        const i1 = ids[r * cols + c + 1]!;
        const i2 = ids[(r + 1) * cols + c + 1]!;
        const i3 = ids[(r + 1) * cols + c]!;
        addQuad(i0, i1, i2, i3);
      }
    }
    return ids;
  };
  regions.lidL = addLid(leftEye);
  regions.lidR = addLid(rightEye);

  const tongueOrigin = sculpt(1.963, NOSE_PHI);
  const tongueIds: number[] = [];
  const tongueCols = 5;
  const tongueRows = 3;
  for (let r = 0; r < tongueRows; r++) {
    for (let c = 0; c < tongueCols; c++) {
      const u = c / (tongueCols - 1);
      const v = r / (tongueRows - 1);
      const x = (u - 0.5) * 0.028;
      const y = (v - 0.5) * 0.012;
      const z = 0.003 * Math.sin(u * Math.PI);
      tongueIds.push(pushVert(
        tongueOrigin.p[0] + x,
        tongueOrigin.p[1] - 0.006 + y,
        tongueOrigin.p[2] - 0.01 + z,
        0,
        0.25,
        0.97,
        u,
        v,
        TONGUE,
      ));
    }
  }
  for (let r = 0; r < tongueRows - 1; r++) {
    for (let c = 0; c < tongueCols - 1; c++) {
      const i0 = tongueIds[r * tongueCols + c]!;
      const i1 = tongueIds[r * tongueCols + c + 1]!;
      const i2 = tongueIds[(r + 1) * tongueCols + c + 1]!;
      const i3 = tongueIds[(r + 1) * tongueCols + c]!;
      addQuad(i0, i1, i2, i3);
    }
  }
  regions.tongue = tongueIds;

  for (const [name, ids] of Object.entries(regions)) {
    if (ids.length < 3) throw new Error(`proof head region ${name} has ${ids.length} vertices`);
  }

  const vertexCount = positions.length / 3;
  const vertices = new Float32Array(vertexCount * PROOF_HEAD_VERTEX_STRIDE);
  for (let i = 0; i < vertexCount; i++) {
    const o = i * PROOF_HEAD_VERTEX_STRIDE;
    vertices[o] = positions[i * 3] ?? 0;
    vertices[o + 1] = positions[i * 3 + 1] ?? 0;
    vertices[o + 2] = positions[i * 3 + 2] ?? 0;
    vertices[o + 3] = normals[i * 3] ?? 0;
    vertices[o + 4] = normals[i * 3 + 1] ?? 0;
    vertices[o + 5] = normals[i * 3 + 2] ?? 0;
    vertices[o + 6] = uvs[i * 2] ?? 0;
    vertices[o + 7] = uvs[i * 2 + 1] ?? 0;
  }

  const morphTargets: MorphTargetSource[] = [];
  for (let k = 0; k < ARKIT_BLENDSHAPE_COUNT; k++) {
    morphTargets.push({ positions: new Float32Array(vertexCount * 3) });
  }
  const write = (shape: number, index: number, dx: number, dy: number, dz: number): void => {
    const delta = morphTargets[shape]!.positions;
    const o = index * 3;
    delta[o] = (delta[o] ?? 0) + dx;
    delta[o + 1] = (delta[o + 1] ?? 0) + dy;
    delta[o + 2] = (delta[o + 2] ?? 0) + dz;
  };
  const xOf = (index: number): number => positions[index * 3] ?? 0;

  for (const index of regions.lowerLip) write(slots.jawOpen, index, 0, -0.018, 0.004);
  for (const index of regions.chin) write(slots.jawOpen, index, 0, -0.03, 0.012);
  for (const index of regions.tongue) write(slots.jawOpen, index, 0, -0.01, 0.002);
  for (const index of regions.lowerLip) write(slots.mouthClose, index, 0, 0.012, 0.002);
  for (const index of regions.upperLip) write(slots.mouthClose, index, 0, -0.005, 0.001);
  for (const index of [...regions.lowerLip, ...regions.upperLip]) {
    const x = xOf(index);
    const pull = Math.min(1, Math.abs(x) / 0.03);
    const inward = x === 0 ? 0 : -Math.sign(x) * pull;
    write(slots.mouthFunnel, index, inward * 0.01, 0, 0.008);
    write(slots.mouthPucker, index, inward * 0.016, 0, 0.014);
  }
  for (const index of regions.upperLip) write(slots.mouthShrugUpper, index, 0, 0.008, 0.002);
  for (const index of regions.tongue) write(slots.tongueOut, index, 0, 0.004, 0.022);
  for (const index of regions.browInner) write(slots.browInnerUp, index, 0, 0.008, 0.002);
  for (const index of regions.browOuterL) write(slots.browOuterUpL, index, 0, 0.007, 0);
  for (const index of regions.browOuterR) write(slots.browOuterUpR, index, 0, 0.007, 0);
  for (const index of [...regions.browInner, ...regions.browOuterL, ...regions.browOuterR]) {
    write(slots.browDown, index, 0, -0.006, 0);
  }
  for (const index of regions.lidL) write(slots.blinkL, index, 0, -0.012, 0.003);
  for (const index of regions.lidR) write(slots.blinkR, index, 0, -0.012, 0.003);
  const gaze = 0.0035;
  for (const index of regions.eyeL) {
    write(slots.eyeLookIn, index, -gaze, 0, 0);
    write(slots.eyeLookOut, index, gaze, 0, 0);
  }
  for (const index of regions.eyeR) {
    write(slots.eyeLookIn, index, gaze, 0, 0);
    write(slots.eyeLookOut, index, -gaze, 0, 0);
  }

  return {
    vertices,
    indices: new Uint32Array(indices),
    vertexCount,
    indexCount: indices.length,
    colors: new Float32Array(colors),
    morphTargets,
    regions,
    slots,
  };
}

export function proofHeadEntity(): Entity {
  return {
    id: PROOF_HEAD_ENTITY_ID as EntityId,
    name: "Proof head",
    components: {
      [Tag.Transform]: {
        position: PROOF_HEAD_POSITION,
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Mesh]: {
        geometry: { kind: "procedural", generator: PROOF_HEAD_GENERATOR, seed: PROOF_HEAD_SEED },
        material: { kind: "pbr", baseColor: [0.73, 0.51, 0.4], metallic: 0.02, roughness: 0.55 },
        visible: true,
      },
    },
  };
}

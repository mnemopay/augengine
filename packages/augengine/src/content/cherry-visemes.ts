/**
 * Cherry Lip-Sync v1 morph set.
 *
 * 12 visemes (Cherry ONNX, Apache-2.0/MIT) + 8 procedural channels
 * (4 brow, 2 gaze, 2 blink) = 20 morphs. ARKit 52 is a projection
 * target, not the authoring set.
 *
 * Taxonomy labels (PP/FF/aa/…) are unprotected; do not bundle Meta
 * Oculus Lipsync binaries.
 */

import { ARKIT_BLENDSHAPE_COUNT } from "../runtime/live/webgpu/blendshape-constants.js";

export const CHERRY_VISEMES = [
  "sil",
  "PP",
  "FF",
  "TH",
  "DD",
  "kk",
  "CH",
  "SS",
  "aa",
  "E",
  "I",
  "O",
] as const;

export type CherryViseme = (typeof CHERRY_VISEMES)[number];

export const PROCEDURAL_CHANNELS = [
  "browInnerUp",
  "browOuterUpL",
  "browOuterUpR",
  "browDown",
  "eyeLookIn",
  "eyeLookOut",
  "blinkL",
  "blinkR",
] as const;

export type ProceduralChannel = (typeof PROCEDURAL_CHANNELS)[number];

export const CHERRY_MORPH_COUNT = CHERRY_VISEMES.length + PROCEDURAL_CHANNELS.length; // 20

/**
 * Cherry viseme index → ARKit 52 index. Unmapped visemes stay at 0.
 * Indices match `augengine-audio2face` ARKit shape order:
 *  jawOpen=23, mouthClose=24, mouthPucker=32, mouthFunnel=31,
 *  mouthLowerDownLeft=28, tongueOut=51, mouthShrugUpper=35,
 *  mouthPressLeft=30, viseme vowels onto jaw + mouth.
 */
const CHERRY_TO_ARKIT: ReadonlyArray<readonly number[]> = [
  [],          // sil
  [23, 24],    // PP — closed lips, slight jaw
  [31],        // FF — funnel
  [51, 23],    // TH — tongue + jaw
  [23],        // DD
  [23],        // kk
  [31, 23],    // CH
  [31],        // SS
  [23],        // aa — jaw open
  [23, 35],    // E
  [32],        // I — pucker-ish
  [32, 31],    // O
];

const PROCEDURAL_TO_ARKIT: Record<ProceduralChannel, number> = {
  browInnerUp: 1,
  browOuterUpL: 2,
  browOuterUpR: 3,
  browDown: 4,
  eyeLookIn: 14,
  eyeLookOut: 15,
  blinkL: 8,
  blinkR: 9,
};

export interface CherryWeightFrame {
  visemes: Record<CherryViseme, number>;
  procedural: Record<ProceduralChannel, number>;
}

export function emptyCherryWeights(): CherryWeightFrame {
  const visemes = {} as Record<CherryViseme, number>;
  for (const v of CHERRY_VISEMES) visemes[v] = v === "sil" ? 1 : 0;
  const procedural = {} as Record<ProceduralChannel, number>;
  for (const c of PROCEDURAL_CHANNELS) procedural[c] = 0;
  return { visemes, procedural };
}

/** Pack 20 floats: visemes[0..11] then procedural[0..7]. */
export function packCherryWeights(frame: CherryWeightFrame): Float32Array {
  const out = new Float32Array(CHERRY_MORPH_COUNT);
  CHERRY_VISEMES.forEach((v, i) => {
    out[i] = frame.visemes[v] ?? 0;
  });
  PROCEDURAL_CHANNELS.forEach((c, i) => {
    out[CHERRY_VISEMES.length + i] = frame.procedural[c] ?? 0;
  });
  return out;
}

/** Project Cherry 20 onto ARKit 52 for the morph-uniform pool. */
export function cherryToArkit(frame: CherryWeightFrame): Float32Array {
  const out = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
  CHERRY_VISEMES.forEach((v, i) => {
    const w = frame.visemes[v] ?? 0;
    if (w <= 0) return;
    for (const idx of CHERRY_TO_ARKIT[i] ?? []) {
      const current = out[idx] ?? 0;
      out[idx] = Math.min(1, current + w);
    }
  });
  for (const c of PROCEDURAL_CHANNELS) {
    const w = frame.procedural[c] ?? 0;
    if (w <= 0) continue;
    const idx = PROCEDURAL_TO_ARKIT[c];
    out[idx] = Math.min(1, (out[idx] ?? 0) + w);
  }
  return out;
}

/**
 * Idle face for the proof-room close-up when no audio is playing:
 * slow blink, slight gaze drift, residual "sil".
 */
export function proceduralIdleFace(tSeconds: number): CherryWeightFrame {
  const frame = emptyCherryWeights();
  const blinkWave = Math.sin(tSeconds * 1.7);
  const blink = blinkWave > 0.96 ? 0.9 : 0;
  frame.procedural.blinkL = blink;
  frame.procedural.blinkR = blink;
  frame.procedural.eyeLookIn = 0.12 + 0.08 * Math.sin(tSeconds * 0.35);
  frame.procedural.eyeLookOut = 0.08 + 0.06 * Math.sin(tSeconds * 0.29 + 1.1);
  frame.procedural.browInnerUp = 0.06 + 0.04 * Math.sin(tSeconds * 0.22);
  frame.visemes.sil = 0.72;
  frame.visemes.aa = 0.14 + 0.12 * Math.max(0, Math.sin(tSeconds * 5.6));
  frame.visemes.E = 0.08 + 0.06 * Math.max(0, Math.sin(tSeconds * 4.1 + 0.7));
  return frame;
}

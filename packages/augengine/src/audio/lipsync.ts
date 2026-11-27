/**
 * Native lip-sync — formant + energy band analysis from audio FFT.
 *
 * Path A from the architecture doc: ~5ms/frame, "VTubing quality." Maps
 * raw mic / track audio to the Oculus 15-viseme set via Mahalanobis-ish
 * distance against tabulated formant prototypes.
 *
 * Quality ceiling: cannot distinguish p/b/m (no audio difference for
 * closed-lip), guesses on whispers, fails on sung vowels. Acceptable for
 * NPCs and avatars; phase-6 stretch ports NVIDIA Audio2Face-3D for
 * premium lip-sync.
 */

/** 15 Oculus visemes. */
export type Viseme =
  | "sil"
  | "PP"
  | "FF"
  | "TH"
  | "DD"
  | "kk"
  | "CH"
  | "SS"
  | "nn"
  | "RR"
  | "aa"
  | "E"
  | "I"
  | "O"
  | "U";

export const ALL_VISEMES: readonly Viseme[] = [
  "sil",
  "PP",
  "FF",
  "TH",
  "DD",
  "kk",
  "CH",
  "SS",
  "nn",
  "RR",
  "aa",
  "E",
  "I",
  "O",
  "U",
];

/**
 * Approximate formant prototypes (F1, F2 in Hz) for the vowel-bearing
 * visemes. Consonant visemes are gated by spectral-tilt + energy heuristics
 * because their formant structure is unstable.
 */
const VOWEL_PROTOS: Partial<Record<Viseme, [number, number]>> = {
  aa: [800, 1200],
  E: [550, 1700],
  I: [350, 2200],
  O: [500, 900],
  U: [350, 800],
};

export interface VisemeWeights {
  /** Sum is approximately 1; a "sil" weight close to 1 means silence. */
  weights: Record<Viseme, number>;
  /** Dominant viseme name (max-weight). */
  dominant: Viseme;
  /** Energy 0..1 — drives jaw open + voicing. */
  energy: number;
}

const ZERO_WEIGHTS: Record<Viseme, number> = {
  sil: 1, PP: 0, FF: 0, TH: 0, DD: 0, kk: 0, CH: 0, SS: 0, nn: 0, RR: 0,
  aa: 0, E: 0, I: 0, O: 0, U: 0,
};

/**
 * Estimate viseme weights from an FFT magnitude buffer (linear, NOT dB).
 * Caller computes the FFT in an AudioWorklet (raw samples → FFT → magnitudes)
 * and hands the result here. Keeping this synchronous and pure lets callers
 * run it inside an AudioWorklet for sample-accurate visemes.
 */
export function estimateVisemes(
  fftMag: Float32Array,
  sampleRate: number,
  options: { silenceThreshold?: number } = {},
): VisemeWeights {
  const energy = avgEnergy(fftMag);
  if (energy < (options.silenceThreshold ?? 0.005)) {
    return { weights: { ...ZERO_WEIGHTS }, dominant: "sil", energy };
  }
  // Find the two strongest peaks below 4kHz — proxies for F1, F2.
  const f1 = findPeakBetween(fftMag, sampleRate, 200, 1100);
  const f2 = findPeakBetween(fftMag, sampleRate, 800, 3500);

  const weights: Record<Viseme, number> = { ...ZERO_WEIGHTS };
  weights.sil = 0;

  // Vowel similarity by squared distance in formant space.
  let total = 0;
  const vowelScores: Partial<Record<Viseme, number>> = {};
  for (const [name, [pf1, pf2]] of Object.entries(VOWEL_PROTOS) as Array<[Viseme, [number, number]]>) {
    const d2 =
      ((f1 - pf1) / 200) ** 2 + ((f2 - pf2) / 400) ** 2;
    const score = Math.exp(-0.5 * d2);
    vowelScores[name] = score;
    total += score;
  }
  if (total > 0) {
    for (const [name, score] of Object.entries(vowelScores) as Array<[Viseme, number]>) {
      weights[name] = score / total;
    }
  }

  // Consonant heuristics — coarse for phase 1.
  const tilt = spectralTilt(fftMag);
  if (tilt < -0.3) {
    // Lots of high-frequency energy → fricative.
    weights.SS = Math.min(1, -tilt) * 0.5;
  }

  // Normalize to sum 1.
  const sum = Object.values(weights).reduce((a, b) => a + b, 0);
  if (sum > 0) {
    for (const k of Object.keys(weights)) {
      weights[k as Viseme] /= sum;
    }
  }

  let dominant: Viseme = "aa";
  let maxW = -1;
  for (const v of ALL_VISEMES) {
    const w = weights[v];
    if (w > maxW) {
      maxW = w;
      dominant = v;
    }
  }
  return { weights, dominant, energy: Math.min(1, energy * 5) };
}

function avgEnergy(mag: Float32Array): number {
  let s = 0;
  for (let i = 0; i < mag.length; i++) s += mag[i]! * mag[i]!;
  return Math.sqrt(s / mag.length);
}

function findPeakBetween(
  mag: Float32Array,
  sampleRate: number,
  loHz: number,
  hiHz: number,
): number {
  const binHz = sampleRate / 2 / mag.length;
  const lo = Math.max(0, Math.floor(loHz / binHz));
  const hi = Math.min(mag.length - 1, Math.ceil(hiHz / binHz));
  let bestI = lo;
  let bestM = -Infinity;
  for (let i = lo; i <= hi; i++) {
    if (mag[i]! > bestM) {
      bestM = mag[i]!;
      bestI = i;
    }
  }
  // Parabolic refinement for sub-bin precision.
  if (bestI > 0 && bestI < mag.length - 1) {
    const a = mag[bestI - 1]!,
      b = mag[bestI]!,
      c = mag[bestI + 1]!;
    const denom = a - 2 * b + c;
    if (Math.abs(denom) > 1e-9) {
      const offset = 0.5 * (a - c) / denom;
      return (bestI + offset) * binHz;
    }
  }
  return bestI * binHz;
}

function spectralTilt(mag: Float32Array): number {
  // Linear regression of log-magnitude vs bin index — slope is "tilt."
  const n = mag.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 1; i < n; i++) {
    const x = i;
    const y = Math.log(mag[i]! + 1e-12);
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
  }
  const m = n - 1;
  const num = m * sumXY - sumX * sumY;
  const den = m * sumX2 - sumX * sumX;
  if (Math.abs(den) < 1e-9) return 0;
  return num / den;
}

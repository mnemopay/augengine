/**
 * Physical camera exposure + white balance math (pure, testable).
 *
 * Exposure follows Lagarde & de Rousiers, "Moving Frostbite to Physically
 * Based Rendering 3.0" (SIGGRAPH 2014 course notes), §5.1:
 *   EV100 from a camera:            EV100 = log2(N² / t · 100 / S)
 *   EV100 from average luminance:   EV100 = log2(L_avg · S / K), S = 100, K = 12.5
 *   exposure (saturation-based):    H = 1 / (1.2 · 2^EV100)
 * The renderer PRE-exposes: every lit shader multiplies its output by H, so the
 * rgba16float target holds values near 1 even when the sun is 100 klux, and
 * the tonemapper runs on the exposed image.
 *
 * Auto-exposure measures a 64-bin histogram of log2 luminance of that
 * pre-exposed target (auto-exposure.ts), divides the pre-exposure back out,
 * averages log luminance between two percentiles (ignoring the darkest pixels
 * and the sun/specular highlights), and adapts toward the resulting EV100
 * with separate brightening/darkening speeds (eye adaptation).
 *
 * White balance: a von Kries diagonal gain in linear sRGB that maps the
 * colour of a black body at the chosen temperature to neutral. Planckian
 * chromaticity from Kim et al. (2002), "Design of Advanced Color Temperature
 * Control System for HDTV Applications" (cubic spline, 1667–25000 K).
 */

import type { Vec3 } from "../../../scene/types.js";
import { xyYToLinearSrgb, luminance } from "./sky-model.js";

/** Reflected-light meter calibration constant (K) and ISO used for EV100. */
export const METER_K = 12.5;

export function ev100FromCamera(aperture: number, shutterSeconds: number, iso: number): number {
  return Math.log2((aperture * aperture) / shutterSeconds * (100 / iso));
}

export function ev100FromAverageLuminance(avgLuminance: number): number {
  return Math.log2(Math.max(avgLuminance, 1e-8) * (100 / METER_K));
}

/** Saturation-based exposure: scene luminance × H = display-referred value. */
export function exposureFromEv100(ev100: number): number {
  return 1 / (1.2 * Math.pow(2, ev100));
}

/** Histogram layout shared with the WGSL in auto-exposure.ts. */
export interface HistogramLayout {
  bins: number;
  /** log2 luminance (of the PRE-EXPOSED target) at the low edge of bin 1. Bin 0 = below range / black. */
  minLog2: number;
  /** log2 luminance at the high edge of the last bin. */
  maxLog2: number;
}

export const DEFAULT_HISTOGRAM: HistogramLayout = { bins: 64, minLog2: -12, maxLog2: 6 };

/** Bin index a pre-exposed luminance falls in (CPU mirror of the shader). */
export function histogramBin(lum: number, h: HistogramLayout = DEFAULT_HISTOGRAM): number {
  if (!(lum > 1e-9)) return 0;
  const t = (Math.log2(lum) - h.minLog2) / (h.maxLog2 - h.minLog2);
  return Math.min(h.bins - 1, Math.max(1, Math.floor(t * (h.bins - 1)) + 1));
}

/** log2 luminance at the centre of a bin (bins ≥ 1). */
export function binCenterLog2(bin: number, h: HistogramLayout = DEFAULT_HISTOGRAM): number {
  return h.minLog2 + ((bin - 1 + 0.5) / (h.bins - 1)) * (h.maxLog2 - h.minLog2);
}

/**
 * Average log2 luminance of the pixels between two cumulative fractions
 * (e.g. 0.5 … 0.95 drops the darker half and the brightest 5%). Bin 0 (black /
 * sky-less void) is excluded. Returns null for an empty histogram.
 */
export function histogramAverageLog2(counts: ArrayLike<number>, lowFrac: number, highFrac: number, h: HistogramLayout = DEFAULT_HISTOGRAM): number | null {
  let total = 0;
  for (let i = 1; i < h.bins; i++) total += counts[i] ?? 0;
  if (total <= 0) return null;
  const lo = total * Math.min(lowFrac, highFrac);
  const hi = total * Math.max(lowFrac, highFrac);
  let cum = 0, sum = 0, weight = 0;
  for (let i = 1; i < h.bins; i++) {
    const c = counts[i] ?? 0;
    const a = Math.max(cum, lo), b = Math.min(cum + c, hi);
    if (b > a) {
      sum += (b - a) * binCenterLog2(i, h);
      weight += b - a;
    }
    cum += c;
  }
  if (weight <= 0) {
    // Window fell inside one bin edge — use the bin at the low fraction.
    let c2 = 0;
    for (let i = 1; i < h.bins; i++) { c2 += counts[i] ?? 0; if (c2 >= lo) return binCenterLog2(i, h); }
    return null;
  }
  return sum / weight;
}

export interface AutoExposureSettings {
  /** Bias added to the metered EV100 (positive = darker image). Default 0. */
  compensation: number;
  minEv100: number;
  maxEv100: number;
  /** EV per second when the scene gets brighter (EV rises). */
  speedUp: number;
  /** EV per second when the scene gets darker. */
  speedDown: number;
  lowPercent: number;
  highPercent: number;
}

/**
 * Defaults tuned on the smoke-game captures (2026-11-27): metering the 40–90th
 * percentile (centre-weighted, auto-exposure.ts) with −0.5 EV keeps a bright
 * low-sun sky from underexposing the ground; the K = 12.5 meter alone put
 * backlit dawn foregrounds ~2 stops under a typical photograph.
 */
export const DEFAULT_AUTO_EXPOSURE: AutoExposureSettings = {
  compensation: -0.5,
  minEv100: -2,
  maxEv100: 17,
  speedUp: 3,
  speedDown: 1.5,
  lowPercent: 0.4,
  highPercent: 0.9,
};

/**
 * Target EV100 from a histogram of the PRE-EXPOSED target that was rendered
 * with exposure `preExposure` (the luminance it measured is scene × H).
 */
export function targetEv100FromHistogram(counts: ArrayLike<number>, preExposure: number, s: AutoExposureSettings = DEFAULT_AUTO_EXPOSURE, h: HistogramLayout = DEFAULT_HISTOGRAM): number | null {
  const avgLog = histogramAverageLog2(counts, s.lowPercent, s.highPercent, h);
  if (avgLog === null) return null;
  const sceneLum = Math.pow(2, avgLog) / Math.max(preExposure, 1e-20);
  const ev = ev100FromAverageLuminance(sceneLum) + s.compensation;
  return Math.min(s.maxEv100, Math.max(s.minEv100, ev));
}

/**
 * Move `current` toward `target` at a bounded EV/second rate, with an
 * exponential ease for small differences (no overshoot).
 */
export function adaptEv100(current: number, target: number, dt: number, s: Pick<AutoExposureSettings, "speedUp" | "speedDown"> = DEFAULT_AUTO_EXPOSURE): number {
  if (!Number.isFinite(current)) return target;
  const diff = target - current;
  const speed = diff > 0 ? s.speedUp : s.speedDown;
  const ease = diff * (1 - Math.exp(-Math.max(dt, 0) * speed));
  const maxStep = speed * Math.max(dt, 0);
  const step = Math.sign(ease) * Math.min(Math.abs(ease), Math.max(maxStep, 0));
  return current + step;
}

// ─── White balance ─────────────────────────────────────────────────────

/** CIE 1931 xy chromaticity of a black body (Kim et al. 2002), 1667–25000 K. */
export function planckianXy(kelvin: number): [number, number] {
  const T = Math.min(25000, Math.max(1667, kelvin));
  const x = T <= 4000
    ? -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.17991
    : -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.24039;
  let y: number;
  if (T <= 2222) y = -1.1063814 * x ** 3 - 1.3481102 * x ** 2 + 2.18555832 * x - 0.20219683;
  else if (T <= 4000) y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867;
  else y = 3.081758 * x ** 3 - 5.8733867 * x ** 2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

/** Linear-sRGB colour of a black body, normalised to luminance 1. */
export function kelvinToLinearSrgb(kelvin: number): Vec3 {
  const [x, y] = planckianXy(kelvin);
  const c = xyYToLinearSrgb(x, y, 1);
  const cc: Vec3 = [Math.max(c[0], 1e-4), Math.max(c[1], 1e-4), Math.max(c[2], 1e-4)];
  const l = luminance(cc);
  return [cc[0] / l, cc[1] / l, cc[2] / l];
}

/** The temperature whose gains are exactly neutral (≈ D65 on the Planckian locus). */
export const NEUTRAL_KELVIN = 6504;

/**
 * Diagonal white-balance gains that render a surface lit by a `kelvin` black
 * body as neutral, normalised so (a) NEUTRAL_KELVIN gives exactly [1, 1, 1]
 * and (b) the gain preserves luminance of a neutral input.
 */
export function whiteBalanceGains(kelvin: number): Vec3 {
  const c = kelvinToLinearSrgb(kelvin);
  const ref = kelvinToLinearSrgb(NEUTRAL_KELVIN);
  const g: Vec3 = [ref[0] / c[0], ref[1] / c[1], ref[2] / c[2]];
  const l = luminance(g);
  return [g[0] / l, g[1] / l, g[2] / l];
}

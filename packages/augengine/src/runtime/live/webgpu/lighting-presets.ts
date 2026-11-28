/**
 * Quality presets for the natural-light tier (renderer.setLightingPreset).
 *
 * Medium is the default and the Intel UHD target (≥ 50 fps median on the
 * smoke-game Loading Range at 1440×900; measured numbers per preset live in
 * docs/LIGHTING.md and results/renderer-benchmark). Each step up spends GPU
 * time where it is most visible outdoors: environment resolution, then
 * shadow resolution/cascades.
 */

import type { ShadowQuality } from "./lights.js";

export type LightingPreset = "low" | "medium" | "high" | "ultra";

export interface LightingPresetSettings {
  /** Environment cube face size (specular reflections). */
  envSize: number;
  /** GGX samples per prefiltered texel (bake time only). */
  envSamples: number;
  shadowQuality: ShadowQuality;
  cascades: 1 | 2;
  /** Diameter (m) of the near shadow region in front of the camera. */
  shadowRange: number;
}

export const LIGHTING_PRESETS: Readonly<Record<LightingPreset, LightingPresetSettings>> = {
  low: { envSize: 32, envSamples: 16, shadowQuality: "low", cascades: 1, shadowRange: 24 },
  medium: { envSize: 64, envSamples: 32, shadowQuality: "low", cascades: 1, shadowRange: 32 },
  high: { envSize: 128, envSamples: 48, shadowQuality: "high", cascades: 2, shadowRange: 24 },
  ultra: { envSize: 256, envSamples: 64, shadowQuality: "high", cascades: 2, shadowRange: 28 },
};

export const LIGHTING_PRESET_NAMES: readonly LightingPreset[] = ["low", "medium", "high", "ultra"];

export function isLightingPreset(v: unknown): v is LightingPreset {
  return v === "low" || v === "medium" || v === "high" || v === "ultra";
}

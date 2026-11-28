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
import type { GiSettings } from "./gi-probes.js";

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
  /** Dynamic GI probes (gi-probes.ts); null = off (sky SH9 ambient only). */
  gi: Partial<GiSettings> | null;
  /** Volumetric sun-shaft ray-march steps (volumetric.ts); 0 = off. */
  volumetricSteps: number;
}

export const LIGHTING_PRESETS: Readonly<Record<LightingPreset, LightingPresetSettings>> = {
  low: { envSize: 32, envSamples: 16, shadowQuality: "low", cascades: 1, shadowRange: 24, gi: null, volumetricSteps: 0 },
  medium: { envSize: 64, envSamples: 32, shadowQuality: "soft", cascades: 1, shadowRange: 32, gi: { columns: 12, rays: 32, stride: 2, spacing: 4 }, volumetricSteps: 0 },
  high: { envSize: 128, envSamples: 48, shadowQuality: "soft-high", cascades: 2, shadowRange: 24, gi: { columns: 16, rays: 32, stride: 1, spacing: 3 }, volumetricSteps: 12 },
  ultra: { envSize: 256, envSamples: 64, shadowQuality: "soft-high", cascades: 2, shadowRange: 28, gi: { columns: 16, rays: 64, stride: 1, spacing: 2.5 }, volumetricSteps: 24 },
};

export const LIGHTING_PRESET_NAMES: readonly LightingPreset[] = ["low", "medium", "high", "ultra"];

export function isLightingPreset(v: unknown): v is LightingPreset {
  return v === "low" || v === "medium" || v === "high" || v === "ultra";
}

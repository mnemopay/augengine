/**
 * Combustion state → flame emitter look.
 *
 * This maps a burning body's simulated state (temperature, how fast the
 * reaction is consuming fuel) onto particle-emitter knobs, so the flame on
 * screen grows, brightens and dies with the actual chemistry rather than
 * running as a fixed decorative loop.
 *
 * IMPORTANT about what this is NOT: the mapping is authored art direction, not
 * physics. Real flame luminosity comes from soot radiation and a temperature⁴
 * term this does not evaluate; flame height comes from a buoyant plume this
 * does not solve. Nothing here is calibrated against measured flame data. It
 * is a presentation layer whose only claim is that it responds monotonically
 * to the simulated state.
 *
 * Deliberately decoupled: this takes plain numbers, not a `ReactiveComponent`,
 * so the renderer never imports the chemistry module and this stays trivially
 * testable. Callers pass `temperature(body)` and the body's reaction rate.
 */

import { FIRE_PRESET, type ParticleEmitterOptions, type ParticleEmitterSpec } from "./particle-pass.js";

export interface CombustionFlameInput {
  /** Body temperature in kelvin. */
  temperatureK: number;
  /**
   * Fuel consumption rate, mol/s. Drives emission rate and flame size. Pass 0
   * when the reaction has stalled (for example the oxidiser is exhausted) and
   * the flame goes out even if the body is still hot.
   */
  reactionMolesPerSecond: number;
  /** Temperature at or above which the body is treated as burning. */
  ignitionK: number;
  /**
   * Temperature at which the look saturates — the hottest the art direction
   * goes. Above this, intensity and size stop climbing.
   */
  saturationK?: number;
  /** Fuel rate at which emission reaches the preset rate. */
  referenceMolesPerSecond?: number;
  /** Emitter world position. */
  origin?: ParticleEmitterSpec["origin"];
}

export interface CombustionFlameLook {
  /** True when the body is hot enough AND still consuming fuel. */
  burning: boolean;
  /**
   * 0 at ignition, 1 at saturation. Exposed so a caller can drive other
   * things (light intensity, audio) off the same curve.
   */
  heat: number;
  /** Patch to feed `renderer.updateParticleEmitter` / `addParticleEmitter`. */
  options: ParticleEmitterOptions;
}

const DEFAULT_SATURATION_K = 1800;
const DEFAULT_REFERENCE_MOL_PER_S = 0.02;

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Derive emitter options from combustion state.
 *
 * A body that is below `ignitionK`, or that has stopped reacting, yields
 * `burning: false` and `active: false` — live particles then finish their
 * lifetime and the flame fades out instead of vanishing on a single frame.
 */
export function flameFromCombustion(input: CombustionFlameInput): CombustionFlameLook {
  const saturationK = input.saturationK ?? DEFAULT_SATURATION_K;
  const referenceRate = input.referenceMolesPerSecond ?? DEFAULT_REFERENCE_MOL_PER_S;
  const rate = Number.isFinite(input.reactionMolesPerSecond)
    ? Math.max(0, input.reactionMolesPerSecond)
    : 0;
  const temperatureK = Number.isFinite(input.temperatureK) ? input.temperatureK : 0;

  // Guard a degenerate authoring span (saturation at or below ignition) so the
  // heat curve cannot divide by zero and produce NaN uniforms.
  const span = saturationK - input.ignitionK;
  const heat = span > 0 ? clamp01((temperatureK - input.ignitionK) / span) : temperatureK >= input.ignitionK ? 1 : 0;
  const burning = temperatureK >= input.ignitionK && rate > 0;

  // Fuel throughput relative to the authored reference, softened by a square
  // root so a 10× fuel spike does not become a 10× particle count.
  const fuel = referenceRate > 0 ? Math.sqrt(clamp01(rate / referenceRate)) : 0;
  // Keep a floor so a barely-burning body still shows a visible pilot flame.
  const drive = burning ? Math.max(0.15, fuel) : 0;

  const options: ParticleEmitterOptions = {
    active: burning,
    emissionRate: FIRE_PRESET.emissionRate * drive,
    // Hotter flame: brighter core, faster rise, slightly larger tongues.
    intensity: FIRE_PRESET.intensity * (0.55 + 0.75 * heat),
    buoyancy: FIRE_PRESET.buoyancy * (0.6 + 0.65 * heat),
    sizeRange: [
      FIRE_PRESET.sizeRange[0] * (0.7 + 0.5 * drive),
      FIRE_PRESET.sizeRange[1] * (0.7 + 0.5 * drive),
    ],
    // A cooler flame cools faster along each particle's life, so it turns over
    // to smoke-dark sooner; a hot flame stays yellow-white for longer.
    coolRate: FIRE_PRESET.coolRate * (1.5 - 0.5 * heat),
  };
  if (input.origin) options.origin = input.origin;

  return { burning, heat, options };
}

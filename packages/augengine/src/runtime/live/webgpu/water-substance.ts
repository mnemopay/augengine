/**
 * Chemistry water state → shallow-water field.
 *
 * `src/chemistry/` already models water as a thermodynamic substance with
 * phase change: a `ReactiveComponent` carries moles of `water`, `ice` and
 * `steam`, and `temperature(body)` resolves its temperature from enthalpy.
 * Until now none of that reached the screen. This module is the bridge, the
 * same role `combustion-flame.ts` plays for fire.
 *
 * It splits cleanly into two kinds of claim, and the split matters:
 *
 *  1. QUANTITATIVE, and defensible as physics. Moles → volume via the molar
 *     mass the chemistry module itself declares and the density of liquid
 *     water; volume → still-water level via the field's own footprint. The
 *     solver conserves volume exactly, so the level you fill to is the level
 *     the simulation keeps. A test round-trips moles → volume → level →
 *     simulated volume → moles.
 *
 *  2. AUTHORED, and only directionally physical. The friction coefficient
 *     falls as temperature rises because water's viscosity does; the look
 *     shifts when the body freezes. Neither is fitted to measured data. These
 *     respond monotonically to the simulated state and nothing more.
 *
 * Deliberately decoupled: this takes plain numbers, not a `ReactiveComponent`,
 * so the renderer never imports the chemistry module and this stays trivially
 * testable. Callers pass `temperature(body)` and the body's `amountsMol`.
 */

import type { ShallowWaterSpec } from "./shallow-water.js";
import { OCEAN_WATER_LOOK, type WaterLook } from "./water-surface.js";

/**
 * Molar mass of water, kg/mol. Must equal `SUBSTANCES.water.molarMassKg` in
 * `src/chemistry/materials.ts`; `water-substance.test.ts` imports both and
 * asserts they match, so the two cannot drift apart.
 */
export const WATER_MOLAR_MASS_KG = 0.018015;

/**
 * Density of liquid water, kg/m³, at ~25 °C and 1 atm.
 *
 * The chemistry module carries no density, so this is ours. It is treated as
 * constant: thermal expansion (≈ 4 % across 0–100 °C) and the density
 * difference of ice are both ignored, which is why `waterVolumeM3FromMoles`
 * is documented as a liquid-water conversion only.
 */
export const WATER_DENSITY_KG_PER_M3 = 997;

/** Freezing point at 1 atm, K. Mirrors `WATER_FREEZING_K` in the chemistry module. */
export const WATER_FREEZING_K = 273.15;

/** Boiling point at 1 atm, K. Mirrors `WATER_BOILING_K` in the chemistry module. */
export const WATER_BOILING_K = 373.15;

/** Volume of liquid water for a given amount, m³. */
export function waterVolumeM3FromMoles(moles: number): number {
  if (!Number.isFinite(moles) || moles <= 0) return 0;
  return (moles * WATER_MOLAR_MASS_KG) / WATER_DENSITY_KG_PER_M3;
}

/** Amount of liquid water in a given volume, mol. Inverse of the above. */
export function waterMolesFromVolumeM3(volumeM3: number): number {
  if (!Number.isFinite(volumeM3) || volumeM3 <= 0) return 0;
  return (volumeM3 * WATER_DENSITY_KG_PER_M3) / WATER_MOLAR_MASS_KG;
}

/** Footprint of a shallow-water grid, m². */
export function waterFootprintM2(spec: Pick<ShallowWaterSpec, "width" | "height" | "cellSize">): number {
  return spec.width * spec.height * spec.cellSize * spec.cellSize;
}

/**
 * Still-water level that holds `volumeM3` over a flat bed at `bedLevel`.
 *
 * Flat bed only. Over a sloped or uneven bed the relationship between level
 * and volume is not linear and this will overfill — solve for the level
 * numerically against the actual bed in that case.
 */
export function surfaceLevelForVolume(
  volumeM3: number,
  spec: Pick<ShallowWaterSpec, "width" | "height" | "cellSize">,
  bedLevel = 0,
): number {
  const area = waterFootprintM2(spec);
  if (!(area > 0) || !Number.isFinite(volumeM3) || volumeM3 <= 0) return bedLevel;
  return bedLevel + volumeM3 / area;
}

export type WaterPhase = "liquid" | "frozen" | "boiling";

export interface WaterSubstanceInput {
  /** Body temperature in kelvin — pass `temperature(body)`. */
  temperatureK: number;
  /** Moles of the `water` (liquid) substance in the body. */
  liquidMoles: number;
  /** Moles of `ice`. Used only to decide the phase, not to add volume. */
  iceMoles?: number;
  /** Moles of `steam`. Used only to decide the phase, not to add volume. */
  vaporMoles?: number;
  /** Grid this body fills. Supplies the footprint for the level calculation. */
  grid: Pick<ShallowWaterSpec, "width" | "height" | "cellSize">;
  /** Flat bed elevation the water sits on, world Y. */
  bedLevel?: number;
  /** Friction at the freezing point, 1/s. The warm end is a fraction of this. */
  referenceDamping?: number;
  /** Liquid appearance restored after thawing; defaults to the ocean look. */
  liquidLook?: Readonly<WaterLook>;
}

export interface WaterSubstanceState {
  phase: WaterPhase;
  /** Liquid volume, m³. Quantitative. */
  volumeM3: number;
  /** Still-water level for that volume over a flat bed, world Y. Quantitative. */
  surfaceLevel: number;
  /**
   * Fraction of the body's water that is liquid, 0..1. 0 means the field
   * should not be drawn as liquid at all.
   */
  liquidFraction: number;
  /** Patch for `renderer.addWaterSurface` / `updateWaterSurface`. */
  options: Partial<ShallowWaterSpec> & Partial<WaterLook>;
}

const DEFAULT_REFERENCE_DAMPING = 0.3;

/**
 * Relative dynamic viscosity of liquid water against its value at 0 °C,
 * approximately 1 at freezing falling to ~0.17 at boiling.
 *
 * A two-point exponential fit through those endpoints — NOT the Vogel or
 * Andrade correlation, and not fitted to a table. Its only guaranteed property
 * is that it decreases monotonically with temperature, which is the direction
 * real water goes. Used to scale the solver's lumped linear friction, which is
 * itself not a viscosity. Do not present the result as a viscous simulation.
 */
export function relativeWaterViscosity(temperatureK: number): number {
  const t = Number.isFinite(temperatureK) ? temperatureK : WATER_FREEZING_K;
  const clamped = Math.min(Math.max(t, WATER_FREEZING_K), WATER_BOILING_K);
  const span = (clamped - WATER_FREEZING_K) / (WATER_BOILING_K - WATER_FREEZING_K);
  // exp(-1.77) ≈ 0.17 at the boiling end.
  return Math.exp(-1.77 * span);
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Derive a water-surface patch from a chemistry body's water state.
 *
 * Phase handling, and what it does NOT do:
 *  - `frozen` (at or below freezing, or ice outnumbering liquid) pauses the
 *    solver — ice does not flow — and shifts the look to a flatter, whiter,
 *    more reflective surface. It is still a water surface with a different
 *    look. No ice geometry, no crystal structure, no thickness, no cracking.
 *  - `boiling` (at or above the boiling point) does NOT render steam. Nothing
 *    in this engine turns vapour into pixels. The flag is reported so a caller
 *    can attach a particle emitter itself (see `water-spray.ts`).
 *  - Latent heat, nucleation and the kinetics of the phase change all live in
 *    `src/chemistry/system.ts`. This module only reads the outcome.
 */
export function waterFromSubstance(input: WaterSubstanceInput): WaterSubstanceState {
  const liquidMoles = Number.isFinite(input.liquidMoles) ? Math.max(0, input.liquidMoles) : 0;
  const iceMoles = Number.isFinite(input.iceMoles ?? 0) ? Math.max(0, input.iceMoles ?? 0) : 0;
  const vaporMoles = Number.isFinite(input.vaporMoles ?? 0) ? Math.max(0, input.vaporMoles ?? 0) : 0;
  const temperatureK = Number.isFinite(input.temperatureK) ? input.temperatureK : WATER_FREEZING_K;
  const bedLevel = Number.isFinite(input.bedLevel ?? 0) ? (input.bedLevel ?? 0) : 0;
  const referenceDamping = Number.isFinite(input.referenceDamping ?? DEFAULT_REFERENCE_DAMPING)
    ? (input.referenceDamping ?? DEFAULT_REFERENCE_DAMPING)
    : DEFAULT_REFERENCE_DAMPING;

  const totalWater = liquidMoles + iceMoles + vaporMoles;
  const liquidFraction = totalWater > 0 ? clamp01(liquidMoles / totalWater) : 0;
  const volumeM3 = waterVolumeM3FromMoles(liquidMoles);
  const surfaceLevel = surfaceLevelForVolume(volumeM3, input.grid, bedLevel);

  const frozen = temperatureK <= WATER_FREEZING_K || iceMoles > liquidMoles;
  const boiling = temperatureK >= WATER_BOILING_K;
  const phase: WaterPhase = frozen ? "frozen" : boiling ? "boiling" : "liquid";

  const options: Partial<ShallowWaterSpec> & Partial<WaterLook> = {
    surfaceLevel,
    // Ice holds its shape: the solver stops rather than the surface freezing
    // in place mid-wave at whatever amplitude it happened to have.
    running: !frozen,
    // Warmer water is thinner, so waves ring for longer before friction eats
    // them. Directional, not calibrated.
    damping: Math.max(0, referenceDamping) * relativeWaterViscosity(temperatureK),
  };
  // Include both phase appearances so an incremental update can thaw an icy
  // surface without retaining the old ice colours and Fresnel coefficient.
  const liquidLook = input.liquidLook ?? OCEAN_WATER_LOOK;
  options.fresnelF0 = liquidLook.fresnelF0;
  options.foamStrength = liquidLook.foamStrength;
  options.shallowTint = [...liquidLook.shallowTint];
  options.deepTint = [...liquidLook.deepTint];
  options.extinctionPerM = [...liquidLook.extinctionPerM];
  if (frozen) {
    // Ice is a dielectric with a higher F0 than water and it scatters, so it
    // reads brighter and flatter. Authored, not measured.
    options.fresnelF0 = 0.08;
    options.foamStrength = 0.1;
    options.shallowTint = [0.86, 0.93, 0.98];
    options.deepTint = [0.42, 0.55, 0.66];
    options.extinctionPerM = [0.35, 0.3, 0.26];
  }
  return { phase, volumeM3, surfaceLevel, liquidFraction, options };
}

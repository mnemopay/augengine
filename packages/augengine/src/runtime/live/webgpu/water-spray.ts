/**
 * Water state → particle emitter look, for spray and steam.
 *
 * A height-field solver cannot represent water that leaves the surface: every
 * column is single-valued in Y, so a droplet is not expressible. Spray
 * therefore has to be particles, and the engine already has a GPU particle
 * pass. This module writes NO new simulation and NO new WGSL — it derives
 * `ParticleEmitterOptions` for the existing pass, exactly the way
 * `combustion-flame.ts` derives a flame from combustion state.
 *
 * IMPORTANT about what this is NOT:
 *  - The mapping is authored art direction, not physics. Real spray comes from
 *    surface instability and air entrainment this does not solve. Droplet size
 *    distribution, breakup and evaporation are not modelled.
 *  - It is NOT automatically wired to the GPU water field. The solver's state
 *    lives in a GPU storage buffer, and sampling it per frame to find impact
 *    sites would need a readback that stalls the frame. The caller supplies
 *    the numbers — from `ShallowWaterField` (the CPU reference), from its own
 *    gameplay events, or from a deliberate `readFieldForDebug()` in tooling.
 *  - Particles do not fall back into the field. Spray leaving the water does
 *    not remove volume from it, and spray landing does not add any.
 *
 * Deliberately decoupled: plain numbers in, emitter options out. No GPU
 * device, no solver instance, no scene.
 */

import {
  SMOKE_PRESET,
  type ParticleEmitterOptions,
  type ParticleEmitterSpec,
} from "./particle-pass.js";

/**
 * Droplet spray: white, ballistic, short-lived, alpha-blended.
 *
 * Built from SMOKE_PRESET because that preset is already the alpha-blended
 * flat-colour variant; the knobs below replace everything that made it smoke.
 * Gravity is real standard gravity, so the arcs are at least ballistically
 * correct even though the emission is authored.
 */
export const SPRAY_PRESET: Readonly<ParticleEmitterSpec> = Object.freeze({
  ...SMOKE_PRESET,
  capacity: 4096,
  emissionRate: 600,
  spawnExtent: [0.18, 0.02, 0.18] as [number, number, number],
  velocityMin: [-0.8, 1.1, -0.8] as [number, number, number],
  velocityMax: [0.8, 3.2, 0.8] as [number, number, number],
  gravity: [0, -9.80665, 0] as [number, number, number],
  // Droplets are denser than air, so no buoyant rise at all.
  buoyancy: 0,
  drag: 0.9,
  turbulence: 0.6,
  turbulenceFrequency: 2.1,
  lifetime: [0.3, 0.8] as [number, number],
  sizeRange: [0.02, 0.07] as [number, number],
  coolRate: 1,
  growth: 0.2,
  tint: [0.92, 0.96, 1] as [number, number, number],
  intensity: 1.25,
  alphaScale: 0.55,
  softness: 1.4,
  seed: 0x27d4eb2f,
});

/**
 * Steam above boiling water: slow, buoyant, long-lived, nearly white.
 *
 * Reported by `waterFromSubstance()` as `phase: "boiling"`; nothing attaches
 * this automatically, because the engine makes no claim to simulate vapour
 * transport.
 */
export const STEAM_PRESET: Readonly<ParticleEmitterSpec> = Object.freeze({
  ...SMOKE_PRESET,
  capacity: 2048,
  emissionRate: 180,
  spawnExtent: [0.3, 0.02, 0.3] as [number, number, number],
  velocityMin: [-0.12, 0.35, -0.12] as [number, number, number],
  velocityMax: [0.12, 0.8, 0.12] as [number, number, number],
  gravity: [0, 0, 0] as [number, number, number],
  buoyancy: 1.1,
  drag: 0.65,
  turbulence: 0.5,
  turbulenceFrequency: 0.9,
  lifetime: [1.4, 3] as [number, number],
  sizeRange: [0.3, 0.75] as [number, number],
  growth: 2.2,
  tint: [0.88, 0.9, 0.92] as [number, number, number],
  intensity: 1,
  alphaScale: 0.18,
  softness: 2.6,
  seed: 0x165667b1,
});

export interface WaterSprayInput {
  /**
   * Flow or impact speed at the emission site, m/s. For the CPU reference
   * field this is `hypot(velocityX, velocityZ)` at the cell; for a gameplay
   * event it is whatever hit the water.
   */
  speedMetersPerSecond: number;
  /** Water depth at the site, m. Spray needs water to come from. */
  depthMeters: number;
  /** Speed at or above which the spray saturates at the preset rate. */
  referenceSpeed?: number;
  /** Depth below which no spray is produced at all, m. */
  minDepth?: number;
  /** Emitter world position — normally the surface point, not the bed. */
  origin?: ParticleEmitterSpec["origin"];
}

export interface WaterSprayLook {
  /** True when there is enough water AND enough speed to throw droplets. */
  spraying: boolean;
  /**
   * 0 at the threshold, 1 at the reference speed. Exposed so a caller can
   * drive audio or camera shake off the same curve.
   */
  strength: number;
  /** Patch to feed `renderer.addParticleEmitter` / `updateParticleEmitter`. */
  options: ParticleEmitterOptions;
}

const DEFAULT_REFERENCE_SPEED = 2.5;
const DEFAULT_MIN_DEPTH = 0.02;

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Derive spray emitter options from local water state.
 *
 * Below `minDepth`, or at zero speed, the result is `spraying: false` and
 * `active: false` — live droplets then finish their lifetime and the spray
 * fades instead of vanishing on a single frame, the same contract
 * `flameFromCombustion` uses.
 */
export function sprayFromWaterState(input: WaterSprayInput): WaterSprayLook {
  const referenceSpeed = Math.max(1e-3, input.referenceSpeed ?? DEFAULT_REFERENCE_SPEED);
  const minDepth = Math.max(0, input.minDepth ?? DEFAULT_MIN_DEPTH);
  const speed = Number.isFinite(input.speedMetersPerSecond)
    ? Math.max(0, input.speedMetersPerSecond)
    : 0;
  const depth = Number.isFinite(input.depthMeters) ? Math.max(0, input.depthMeters) : 0;

  const strength = clamp01(speed / referenceSpeed);
  const spraying = depth > minDepth && strength > 0;
  // Softened by a square root so a 10× faster impact is not a 10× particle
  // count, and floored so a barely-breaking ripple still shows something.
  const drive = spraying ? Math.max(0.12, Math.sqrt(strength)) : 0;

  const options: ParticleEmitterOptions = {
    active: spraying,
    emissionRate: SPRAY_PRESET.emissionRate * drive,
    // Faster water throws droplets higher and further.
    velocityMin: [
      SPRAY_PRESET.velocityMin[0] * (0.5 + 0.9 * drive),
      SPRAY_PRESET.velocityMin[1] * (0.4 + 1.1 * drive),
      SPRAY_PRESET.velocityMin[2] * (0.5 + 0.9 * drive),
    ],
    velocityMax: [
      SPRAY_PRESET.velocityMax[0] * (0.5 + 0.9 * drive),
      SPRAY_PRESET.velocityMax[1] * (0.4 + 1.1 * drive),
      SPRAY_PRESET.velocityMax[2] * (0.5 + 0.9 * drive),
    ],
    // Harder impacts atomise further, so the droplets get smaller, not bigger.
    sizeRange: [
      SPRAY_PRESET.sizeRange[0] * (1.25 - 0.45 * drive),
      SPRAY_PRESET.sizeRange[1] * (1.25 - 0.45 * drive),
    ],
    alphaScale: SPRAY_PRESET.alphaScale * (0.45 + 0.55 * drive),
  };
  if (input.origin) options.origin = input.origin;

  return { spraying, strength, options };
}

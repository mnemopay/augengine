/**
 * Shallow-water solver — a real height-field fluid simulation on the GPU.
 *
 * This is NOT a sum of Gerstner waves. There is no authored wave spectrum and
 * no time-parameterised displacement function. The state is a grid of water
 * columns and face velocities, and every frame it integrates the shallow-water
 * equations, so disturbances actually propagate at sqrt(g·h), reflect off the
 * domain walls, interfere with each other, shoal as they run into shallower
 * bed, and decay under friction. Drop a splash in and a ring expands; the ring
 * hits a wall and comes back.
 *
 * ── The equations actually integrated ────────────────────────────────────
 * State per cell: water column depth `h` (m), bed elevation `bed` (m), and the
 * face velocities `velocityX` / `velocityZ` (m/s) on that cell's +X and +Z
 * faces. Surface elevation is η = bed + h.
 *
 *   face momentum:  ∂u/∂t = -g ∂η/∂x  −  (linear friction)
 *   continuity:     ∂h/∂t = -∇·(h u)
 *
 * The momentum equation is the NON-ADVECTIVE form: the u·∇u term is dropped.
 * That is a deliberate, load-bearing simplification and it is the reason this
 * solver is unconditionally well-behaved at interactive step sizes. What it
 * costs is listed under "what this cannot represent" below.
 *
 * ── Why this discretisation ──────────────────────────────────────────────
 * Velocities live on cell faces (a staggered / MAC arrangement), not at cell
 * centres. Two consequences matter:
 *
 *  1. VOLUME IS CONSERVED EXACTLY, by construction rather than by tuning.
 *     Each interior face computes one discharge `q = h_face · u`, which is
 *     subtracted from the cell on one side and added to the cell on the other.
 *     Summed over the grid the interior terms telescope away and the boundary
 *     faces are walls with q = 0, so Σh is invariant to floating-point
 *     rounding only. `shallow-water.test.ts` asserts this to 1e-9 relative
 *     over thousands of steps. The ONE thing that breaks it is the dry-cell
 *     clamp (h may not go negative); the CPU reference therefore accounts for
 *     every cubic metre the clamp invents in `clampedVolumeM3`, so the leak is
 *     reported rather than hidden.
 *
 *  2. THE TWO SUB-PASSES ARE RACE-FREE WITHOUT PING-PONG BUFFERS. The flux
 *     pass reads `h` (never writes it) and writes velocity + discharge. The
 *     height pass reads discharge (never writes it) and writes `h`. Neither
 *     pass reads a value a sibling invocation is writing, so both run in place
 *     on a single storage buffer. Storing the discharge — not just the
 *     velocity — is what buys this: the height pass needs no neighbour `h`.
 *
 * The pressure gradient is taken on η and not on h. That is what makes still
 * water over a sloped bed stay still: a lake at rest has uniform η, so every
 * gradient is zero, so nothing moves. Using h there is the classic bug that
 * makes a sloped beach spontaneously generate a current, and there is a test
 * pinning it.
 *
 * Friction integrates exactly as `u *= exp(-damping·dt)` instead of
 * `u *= (1 - damping·dt)`, so it cannot overshoot into negative velocity at a
 * large step — the same discipline the particle pass uses for drag.
 *
 * ── Timestep and CFL ─────────────────────────────────────────────────────
 * Explicit integration of a wave equation is stable only below the Courant
 * limit. For this 2-D leapfrog arrangement that limit is
 *
 *     dt ≤ dx / (√2 · c)   where c = sqrt(g · h_max)
 *
 * `maxStableTimeStep()` applies a safety factor (default 0.5) to that, and
 * `ShallowWaterSim` SUBSTEPS the host's frame dt through a `FixedStepClock`
 * built on the CFL step — the engine's existing bounded fixed-step machinery,
 * not a second timing path. If the frame dt needs more substeps than the
 * budget allows, the clock drops the excess into `droppedSeconds` and
 * `getStats()` reports it. Overload is reported, never hidden in dt.
 *
 * ── What this CANNOT represent (do not claim any of it) ──────────────────
 *  - A height field is single-valued in Y. No overturning or breaking waves,
 *    no tubes, no water that folds over itself.
 *  - No separated droplets, no spray, no air entrainment. Nothing leaves the
 *    surface. Spray must be driven as particles (see water-spray.ts).
 *  - No vertical structure: no pressure as a function of depth, no stratified
 *    flow, no undertow. One velocity per column, period.
 *  - Momentum advection is omitted, so no hydraulic jumps, no bores, no
 *    steepening of a wave front into a shock.
 *  - No surface tension, no capillary waves, no viscosity beyond the lumped
 *    linear friction term.
 *  - No coupling to rigid bodies. Nothing floats and nothing displaces water
 *    unless the caller calls `splash()` itself.
 *  - Determinism: identical on one device for an identical dt sequence, but
 *    float results are not bit-identical across GPU vendors. Not a lockstep
 *    multiplayer input.
 */

import { FixedStepClock } from "../../fixed-step.js";

// ──────────────────────────────────────────────────────────────────────
// Layout — mirrored by the WGSL `WaterCell` struct below.
// ──────────────────────────────────────────────────────────────────────

/**
 * Floats per cell. All six members are f32 so WGSL alignment is 4 and the
 * struct size is exactly 24 bytes with no padding.
 */
export const WATER_CELL_STRIDE_FLOATS = 6;

/** Bytes per cell (`WATER_CELL_STRIDE_FLOATS * 4`). */
export const WATER_CELL_STRIDE_BYTES = 24;

/** Float index of each field inside one cell's stride. */
export const WATER_CELL_FIELD_OFFSETS = Object.freeze({
  /** Water column depth above the bed, metres. Never negative. */
  height: 0,
  /** Bed (floor) elevation in world Y, metres. Static. */
  bed: 1,
  /** Velocity on this cell's +X face, m/s. Zero on the wall face. */
  velocityX: 2,
  /** Velocity on this cell's +Z face, m/s. Zero on the wall face. */
  velocityZ: 3,
  /** Discharge h_face·u on the +X face, m²/s. */
  fluxX: 4,
  /** Discharge h_face·u on the +Z face, m²/s. */
  fluxZ: 5,
});

/** Bytes of the per-field simulation uniform. 2 × vec4. */
export const WATER_SIM_PARAMS_BYTES = 32;

/** Bytes of the impulse uniform. 2 × vec4. */
export const WATER_IMPULSE_PARAMS_BYTES = 32;

/** Compute workgroup is 8×8 = 64 invocations, matching the particle pass. */
export const WATER_WORKGROUP_SIZE = 8;

/** Hard cap per axis. 512×512 × 24 B = 6.3 MB, ~20 % of the 32 MB GPU ceiling. */
export const MAX_WATER_GRID = 512;

/** Recommended grid for the Intel-UHD playable floor. 128×128 × 24 B = 393 KB. */
export const PLAYABLE_WATER_GRID = 128;

/** Standard gravity, m/s². */
export const WATER_GRAVITY = 9.80665;

/**
 * Courant safety factor applied on top of the theoretical 2-D leapfrog limit
 * `dx / (√2 · c)`. 1.0 would sit exactly on the limit; 0.5 leaves margin for
 * the depth rising above `surfaceLevel` after a splash.
 */
export const DEFAULT_CFL_SAFETY = 0.5;

// ──────────────────────────────────────────────────────────────────────
// Field description
// ──────────────────────────────────────────────────────────────────────

export interface ShallowWaterSpec {
  /** Cells along X. Clamped to [2, MAX_WATER_GRID]. */
  width: number;
  /** Cells along Z. Clamped to [1, MAX_WATER_GRID]. A height of 1 is valid and 1-D. */
  height: number;
  /** Grid spacing in metres. Uniform in both axes. */
  cellSize: number;
  /** World X/Z of cell (0,0). The renderer reads grid placement from here, so
   *  solver and surface can never disagree about where the water is. */
  origin: [number, number];
  /** World Y the still surface is filled to. `h = max(surfaceLevel - bed, 0)`. */
  surfaceLevel: number;
  /**
   * World Y of a flat bed, used when no `bed` callback is supplied. Must sit
   * below `surfaceLevel` or the field starts dry and draws nothing — which is
   * exactly the trap the default preset used to fall into.
   */
  bedLevel: number;
  /** Gravity magnitude, m/s². Drives the wave speed sqrt(g·h). */
  gravity: number;
  /** Linear friction coefficient, 1/s. `u *= exp(-damping·dt)` per substep. */
  damping: number;
  /** Faces whose mean depth is at or below this are treated as dry, metres. */
  dryDepth: number;
  /** Maximum CFL substeps per host frame before the clock starts dropping time. */
  maxSubsteps: number;
  /** Safety factor on the Courant limit. See DEFAULT_CFL_SAFETY. */
  cflSafety: number;
  /** When false the field holds its state and no substeps are taken. */
  running: boolean;
}

export type ShallowWaterOptions = Partial<ShallowWaterSpec> & {
  /**
   * Bed elevation per cell, in world Y. Evaluated once at seed time on the
   * CPU, so an arbitrary callback is fine — it never reaches the GPU.
   * Defaults to a flat bed at 0.
   */
  bed?: (i: number, j: number) => number;
};

/** A 32 m square pool, 1.5 m deep, flat bed at y = -1.5, light friction. */
export const POOL_PRESET: Readonly<ShallowWaterSpec> = Object.freeze({
  width: PLAYABLE_WATER_GRID,
  height: PLAYABLE_WATER_GRID,
  cellSize: 0.25,
  origin: [-16, -16] as [number, number],
  surfaceLevel: 0,
  bedLevel: -1.5,
  gravity: WATER_GRAVITY,
  damping: 0.12,
  dryDepth: 1e-5,
  maxSubsteps: 8,
  cflSafety: DEFAULT_CFL_SAFETY,
  running: true,
});

/**
 * A shoreline: same grid, but the caller is expected to pass a sloping `bed`
 * so part of the domain is dry. Friction is higher because shallow water over
 * a rough bed loses energy faster, and the preset is deeper so the open end
 * still carries a usable wave.
 */
export const SHORE_PRESET: Readonly<ShallowWaterSpec> = Object.freeze({
  ...POOL_PRESET,
  damping: 0.25,
  surfaceLevel: 0,
});

// ──────────────────────────────────────────────────────────────────────
// Pure helpers — unit-testable without a GPU device.
// ──────────────────────────────────────────────────────────────────────

function clampNumber(value: number, lo: number, hi: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return value < lo ? lo : value > hi ? hi : value;
}

/**
 * Merge options over a preset and clamp every field to a range the solver can
 * integrate. Non-finite input falls back to the base value rather than
 * reaching the GPU and poisoning the whole field with NaN.
 */
export function resolveShallowWaterSpec(
  options: Partial<ShallowWaterSpec> = {},
  base: Readonly<ShallowWaterSpec> = POOL_PRESET,
): ShallowWaterSpec {
  const originX = options.origin?.[0];
  const originZ = options.origin?.[1];
  return {
    // Width needs at least 2 cells to have one interior face; height may be 1
    // (a 1-D channel), which is what the wave-speed test uses.
    width: Math.floor(clampNumber(options.width ?? base.width, 2, MAX_WATER_GRID, base.width)),
    height: Math.floor(clampNumber(options.height ?? base.height, 1, MAX_WATER_GRID, base.height)),
    cellSize: clampNumber(options.cellSize ?? base.cellSize, 1e-3, 1e4, base.cellSize),
    origin: [
      Number.isFinite(originX) ? (originX as number) : base.origin[0],
      Number.isFinite(originZ) ? (originZ as number) : base.origin[1],
    ],
    surfaceLevel: clampNumber(options.surfaceLevel ?? base.surfaceLevel, -1e6, 1e6, base.surfaceLevel),
    bedLevel: clampNumber(options.bedLevel ?? base.bedLevel, -1e6, 1e6, base.bedLevel),
    // Zero gravity is legal: it freezes wave propagation without destabilising
    // anything, which is a useful debugging state.
    gravity: clampNumber(options.gravity ?? base.gravity, 0, 1e4, base.gravity),
    damping: clampNumber(options.damping ?? base.damping, 0, 1e3, base.damping),
    dryDepth: clampNumber(options.dryDepth ?? base.dryDepth, 0, 1e3, base.dryDepth),
    maxSubsteps: Math.floor(clampNumber(options.maxSubsteps ?? base.maxSubsteps, 1, 64, base.maxSubsteps)),
    cflSafety: clampNumber(options.cflSafety ?? base.cflSafety, 1e-3, 1, base.cflSafety),
    running: options.running ?? base.running,
  };
}

/** Linearised shallow-water wave celerity, sqrt(g·h), m/s. */
export function shallowWaveSpeed(gravity: number, depth: number): number {
  const g = Math.max(0, gravity);
  const h = Math.max(0, depth);
  return Math.sqrt(g * h);
}

/**
 * Largest stable timestep for this grid, seconds.
 *
 * The 2-D leapfrog Courant limit is `dx / (√2 · c)`; `safety` scales it down.
 * A motionless or zero-gravity field has c = 0 and therefore no limit, so this
 * returns `Infinity` and the caller must supply its own ceiling.
 */
export function maxStableTimeStep(
  cellSize: number,
  gravity: number,
  maxDepth: number,
  safety: number = DEFAULT_CFL_SAFETY,
): number {
  const c = shallowWaveSpeed(gravity, maxDepth);
  if (!(c > 0) || !(cellSize > 0)) return Number.POSITIVE_INFINITY;
  return (safety * cellSize) / (Math.SQRT2 * c);
}

/** Cell count for a spec. */
export function waterCellCount(spec: Pick<ShallowWaterSpec, "width" | "height">): number {
  return spec.width * spec.height;
}

/** GPU storage bytes the cell buffer occupies. */
export function waterGpuBytes(spec: Pick<ShallowWaterSpec, "width" | "height">): number {
  return waterCellCount(spec) * WATER_CELL_STRIDE_BYTES;
}

/** Compute workgroups per axis needed to cover `cells` along that axis. */
export function waterWorkgroupCount(cells: number): number {
  return Math.ceil(Math.max(0, cells) / WATER_WORKGROUP_SIZE);
}

/**
 * Initial field contents: a lake filled to `surfaceLevel` over `bed`, at rest.
 *
 * Depth is `max(surfaceLevel - bed, 0)`, so anywhere the bed rises above the
 * fill level starts dry — that is how a shoreline exists at all.
 */
export function seedWaterField(
  spec: ShallowWaterSpec,
  bed?: (i: number, j: number) => number,
): Float32Array {
  const out = new Float32Array(waterCellCount(spec) * WATER_CELL_STRIDE_FLOATS);
  for (let j = 0; j < spec.height; j++) {
    for (let i = 0; i < spec.width; i++) {
      const base = (j * spec.width + i) * WATER_CELL_STRIDE_FLOATS;
      const raw = bed ? bed(i, j) : spec.bedLevel;
      const bedY = Number.isFinite(raw) ? raw : 0;
      out[base + WATER_CELL_FIELD_OFFSETS.bed] = bedY;
      out[base + WATER_CELL_FIELD_OFFSETS.height] = Math.max(spec.surfaceLevel - bedY, 0);
    }
  }
  return out;
}

/**
 * A bed that slopes linearly from `nearY` at i = 0 to `farY` at i = width-1.
 * Pass as `bed` to get a beach: the stretch where the bed climbs above
 * `surfaceLevel` starts dry and waves run up to it.
 */
export function slopedBed(
  width: number,
  nearY: number,
  farY: number,
): (i: number, j: number) => number {
  const span = Math.max(1, width - 1);
  return (i: number) => nearY + ((farY - nearY) * i) / span;
}

/**
 * Pack the simulation uniform. `out` must hold `WATER_SIM_PARAMS_BYTES / 4`
 * floats. Mirrored by the WGSL `WaterSimParams` struct:
 *   0 dt · 1 gravity · 2 cellSize · 3 dampingFactor
 *   4 width · 5 height · 6 dryDepth · 7 alignment padding (asserted zero)
 *
 * `dampingFactor` is `exp(-damping·dt)` — precomputed on the CPU so the shader
 * does not recompute an identical transcendental per cell per substep, and so
 * there is exactly one place friction is defined.
 */
export function packWaterSimParams(
  out: Float32Array,
  spec: ShallowWaterSpec,
  dt: number,
): Float32Array {
  const needed = WATER_SIM_PARAMS_BYTES / 4;
  if (out.length < needed) {
    throw new Error(`packWaterSimParams needs ${needed} floats, got ${out.length}`);
  }
  out.fill(0);
  out[0] = dt;
  out[1] = spec.gravity;
  out[2] = spec.cellSize;
  out[3] = Math.exp(-spec.damping * dt);
  out[4] = spec.width;
  out[5] = spec.height;
  out[6] = spec.dryDepth;
  // out[7] is padding forced by vec4 alignment and carries nothing. It is
  // asserted zero in the tests so a future field cannot be smuggled in here
  // and silently ignored by the shader.
  return out;
}

/**
 * Pack the impulse uniform. `out` must hold `WATER_IMPULSE_PARAMS_BYTES / 4`
 * floats. Mirrored by the WGSL `WaterImpulseParams` struct:
 *   0 worldX · 1 worldZ · 2 radius · 3 amplitude
 *   4 width · 5 height · 6 cellSize · 7 originX
 *
 * `originZ` does not fit, so the host converts the splash centre to
 * grid-relative metres before packing: slot 0/1 are already origin-subtracted.
 */
export function packWaterImpulseParams(
  out: Float32Array,
  spec: ShallowWaterSpec,
  worldX: number,
  worldZ: number,
  radius: number,
  amplitude: number,
): Float32Array {
  const needed = WATER_IMPULSE_PARAMS_BYTES / 4;
  if (out.length < needed) {
    throw new Error(`packWaterImpulseParams needs ${needed} floats, got ${out.length}`);
  }
  out.fill(0);
  out[0] = worldX - spec.origin[0];
  out[1] = worldZ - spec.origin[1];
  out[2] = Math.max(radius, 0);
  out[3] = amplitude;
  out[4] = spec.width;
  out[5] = spec.height;
  out[6] = spec.cellSize;
  out[7] = 0;
  return out;
}

/**
 * Volume a `splash()` of this radius and amplitude adds, m³ — the analytic
 * integral of the same smoothstep falloff the shader applies, evaluated over
 * the grid. Positive amplitude adds water and negative removes it, so a
 * conservation test can subtract the known contribution instead of treating
 * the splash as an unexplained leak.
 *
 * This counts whole cells inside the radius, so it matches the discrete
 * solver, not the continuous integral.
 */
export function splashVolumeM3(
  spec: ShallowWaterSpec,
  worldX: number,
  worldZ: number,
  radius: number,
  amplitude: number,
): number {
  if (!(radius > 0)) return 0;
  const cellArea = spec.cellSize * spec.cellSize;
  const cx = worldX - spec.origin[0];
  const cz = worldZ - spec.origin[1];
  let total = 0;
  for (let j = 0; j < spec.height; j++) {
    for (let i = 0; i < spec.width; i++) {
      const dxm = i * spec.cellSize - cx;
      const dzm = j * spec.cellSize - cz;
      total += splashFalloff(Math.hypot(dxm, dzm), radius) * amplitude * cellArea;
    }
  }
  return total;
}

/**
 * Radial falloff of a splash: 1 at the centre, 0 at and beyond `radius`, with
 * a smoothstep in between so the injected bump has no corner that would ring
 * as grid-frequency noise. Mirrored exactly by `splash_falloff` in the WGSL.
 */
export function splashFalloff(distance: number, radius: number): number {
  if (!(radius > 0)) return 0;
  const t = 1 - Math.min(Math.max(distance / radius, 0), 1);
  return t * t * (3 - 2 * t);
}

/**
 * Whether a face between two water columns carries flow this step.
 *
 * Both-dry is the obvious case. The case that actually matters is
 * wet-against-dry: a lake lying against a beach must NOT push itself uphill.
 * A face whose dry side has a bed at or above the wet side's SURFACE is
 * therefore a wall. Once a wave lifts that surface above the dry bed, the face
 * opens and the water runs up the beach — which is why run-up works at all.
 *
 * Without this gate, still water over a sloped bed generates a spurious
 * outward current from the shoreline on the very first step: the dry cell's
 * elevation is its bed, which sits above the lake surface, and the solver
 * reads that as a pressure gradient. There is a test pinning the at-rest case.
 *
 * Gating a face sets its discharge to zero, which still telescopes across the
 * grid, so volume conservation is unaffected by this rule.
 *
 * Mirrored by `face_is_open()` in the WGSL.
 */
export function waterFaceIsOpen(
  depthA: number,
  bedA: number,
  depthB: number,
  bedB: number,
  dryDepth: number,
): boolean {
  const wetA = depthA > dryDepth;
  const wetB = depthB > dryDepth;
  if (wetA && wetB) return true;
  if (wetA && depthA + bedA > bedB) return true;
  if (wetB && depthB + bedB > bedA) return true;
  return false;
}

/**
 * Surface normal from the discrete elevation field by central differences.
 *
 * Deliberately derived from the SAME η the vertex stage displaces to, so the
 * normal cannot drift out of sync with the geometry the way a separately
 * authored normal map does. Edge cells fall back to a one-sided difference.
 *
 * `sample(i, j)` must return η = bed + h at that cell. Returns a unit vector.
 */
export function waterSurfaceNormal(
  sample: (i: number, j: number) => number,
  i: number,
  j: number,
  width: number,
  height: number,
  cellSize: number,
): [number, number, number] {
  const iLo = Math.max(i - 1, 0);
  const iHi = Math.min(i + 1, width - 1);
  const jLo = Math.max(j - 1, 0);
  const jHi = Math.min(j + 1, height - 1);
  const spanX = (iHi - iLo) * cellSize;
  const spanZ = (jHi - jLo) * cellSize;
  const dEtaDx = spanX > 0 ? (sample(iHi, j) - sample(iLo, j)) / spanX : 0;
  const dEtaDz = spanZ > 0 ? (sample(i, jHi) - sample(i, jLo)) / spanZ : 0;
  // The surface is η(x, z), so its tangents are (1, dη/dx, 0) and
  // (0, dη/dz, 1); their cross product is (-dη/dx, 1, -dη/dz).
  const length = Math.hypot(dEtaDx, 1, dEtaDz);
  return [-dEtaDx / length, 1 / length, -dEtaDz / length];
}

/**
 * Beer–Lambert transmittance through `pathLengthM` of water with extinction
 * `extinctionPerM` (1/m), per colour channel.
 *
 * Strictly decreasing in path length and in extinction, and always in (0, 1].
 * This is the term that makes deep water dark and shallow water clear.
 */
export function waterTransmittance(extinctionPerM: number, pathLengthM: number): number {
  const k = Math.max(0, extinctionPerM);
  const d = Math.max(0, pathLengthM);
  return Math.exp(-k * d);
}

/**
 * Path length a view ray travels through a water column of depth `depthM`
 * before it reaches the bed, metres.
 *
 * Looking straight down the path is the depth; at a grazing angle it is longer
 * by 1/|cos|. `viewDotUp` is the view direction's component along +Y. The
 * divisor is floored so a perfectly horizontal view cannot produce an infinite
 * path (and therefore a black band) at the horizon.
 */
export function waterPathLength(depthM: number, viewDotUp: number, minCosine = 0.08): number {
  const d = Math.max(0, depthM);
  const c = Math.max(Math.abs(viewDotUp), Math.max(minCosine, 1e-6));
  return d / c;
}

/**
 * Schlick Fresnel reflectance. `F0` for an air→water interface is ~0.02, which
 * is why water is near-transparent looking down and near-mirror at a grazing
 * angle — the single most important cue that a surface is water at all.
 */
export function fresnelSchlick(cosTheta: number, f0: number): number {
  const c = Math.min(Math.max(cosTheta, 0), 1);
  const m = 1 - c;
  return f0 + (1 - f0) * m * m * m * m * m;
}

// ──────────────────────────────────────────────────────────────────────
// CPU reference solver
// ──────────────────────────────────────────────────────────────────────

export interface ShallowWaterStepReport {
  /** Substeps taken. */
  steps: number;
  /** Seconds the substep budget could not absorb. Reported, never hidden. */
  droppedSeconds: number;
}

/**
 * CPU reference implementation of the solver. This is the executable
 * specification: the WGSL below mirrors it statement for statement, and the
 * real-hardware probe steps both from the same initial condition and compares
 * the fields cell by cell. That comparison is the only thing that keeps the
 * two from drifting, because Node cannot execute WGSL.
 *
 * It is also what the physics invariant tests run against — volume
 * conservation, wave celerity, still-water equilibrium over a slope, symmetry
 * and long-run finiteness are all asserted here.
 *
 * Fast enough for small grids and for tooling. It is NOT the frame path: a
 * 128×128 field is 16 384 cells × 2 passes × substeps of serial JavaScript.
 */
export class ShallowWaterField {
  readonly spec: ShallowWaterSpec;
  /** Flat cell array, `WATER_CELL_STRIDE_FLOATS` floats per cell, row-major. */
  readonly cells: Float32Array;
  /**
   * Total volume the dry clamp has invented (m³) across every step so far.
   * Non-zero means a cell tried to go negative and conservation is off by
   * exactly this much. Positive = water created.
   */
  clampedVolumeM3 = 0;
  /** Volume deliberately injected by `splash()` (m³). Signed. */
  injectedVolumeM3 = 0;
  private readonly clock: FixedStepClock;

  constructor(options: ShallowWaterOptions = {}, base: Readonly<ShallowWaterSpec> = POOL_PRESET) {
    this.spec = resolveShallowWaterSpec(options, base);
    this.cells = seedWaterField(this.spec, options.bed);
    this.clock = new FixedStepClock(this.stableStep(), this.spec.maxSubsteps);
  }

  /**
   * CFL substep for this field, seconds. Derived from the deepest column
   * present at construction plus headroom for a splash, and capped so a
   * zero-gravity or empty field still produces a finite step.
   */
  private stableStep(): number {
    const limit = maxStableTimeStep(
      this.spec.cellSize,
      this.spec.gravity,
      this.maxDepth() * 2 + 0.25,
      this.spec.cflSafety,
    );
    return Number.isFinite(limit) ? Math.min(limit, 1 / 60) : 1 / 60;
  }

  /** The substep size the clock runs at, seconds. */
  get substepSeconds(): number {
    return this.clock.dt;
  }

  private index(i: number, j: number): number {
    return (j * this.spec.width + i) * WATER_CELL_STRIDE_FLOATS;
  }

  /** Water column depth at a cell, metres. */
  depthAt(i: number, j: number): number {
    return this.cells[this.index(i, j) + WATER_CELL_FIELD_OFFSETS.height]!;
  }

  /** Bed elevation at a cell, world Y. */
  bedAt(i: number, j: number): number {
    return this.cells[this.index(i, j) + WATER_CELL_FIELD_OFFSETS.bed]!;
  }

  /** Surface elevation η = bed + h at a cell, world Y. */
  elevationAt(i: number, j: number): number {
    const base = this.index(i, j);
    return (
      this.cells[base + WATER_CELL_FIELD_OFFSETS.bed]! +
      this.cells[base + WATER_CELL_FIELD_OFFSETS.height]!
    );
  }

  /** Unit surface normal at a cell, by central difference of η. */
  normalAt(i: number, j: number): [number, number, number] {
    return waterSurfaceNormal(
      (a, b) => this.elevationAt(a, b),
      i,
      j,
      this.spec.width,
      this.spec.height,
      this.spec.cellSize,
    );
  }

  /** Deepest column currently in the field, metres. */
  maxDepth(): number {
    let max = 0;
    for (let c = 0; c < this.cells.length; c += WATER_CELL_STRIDE_FLOATS) {
      const h = this.cells[c + WATER_CELL_FIELD_OFFSETS.height]!;
      if (h > max) max = h;
    }
    return max;
  }

  /** Total water volume, m³. The quantity the solver conserves exactly. */
  totalVolumeM3(): number {
    const cellArea = this.spec.cellSize * this.spec.cellSize;
    let sum = 0;
    for (let c = 0; c < this.cells.length; c += WATER_CELL_STRIDE_FLOATS) {
      sum += this.cells[c + WATER_CELL_FIELD_OFFSETS.height]!;
    }
    return sum * cellArea;
  }

  /**
   * Sum of |η - mean η| weighted by cell area — a proxy for wave energy used
   * to assert that friction removes energy while volume stays put. Not a
   * physical energy in joules; do not report it as one.
   */
  disturbanceM3(): number {
    const n = waterCellCount(this.spec);
    if (n === 0) return 0;
    let mean = 0;
    for (let j = 0; j < this.spec.height; j++) {
      for (let i = 0; i < this.spec.width; i++) mean += this.elevationAt(i, j);
    }
    mean /= n;
    const cellArea = this.spec.cellSize * this.spec.cellSize;
    let sum = 0;
    for (let j = 0; j < this.spec.height; j++) {
      for (let i = 0; i < this.spec.width; i++) sum += Math.abs(this.elevationAt(i, j) - mean);
    }
    return sum * cellArea;
  }

  /** Count of non-finite floats anywhere in the field. Must stay 0. */
  nonFiniteCount(): number {
    let count = 0;
    for (let k = 0; k < this.cells.length; k++) {
      if (!Number.isFinite(this.cells[k]!)) count++;
    }
    return count;
  }

  /**
   * Inject a radial bump. Returns the volume actually added (m³), which also
   * accumulates into `injectedVolumeM3`.
   *
   * A splash is a mass source by design — a raindrop really does add water.
   * The returned figure is what a conservation check must subtract.
   */
  splash(worldX: number, worldZ: number, radius: number, amplitude: number): number {
    if (!(radius > 0) || !Number.isFinite(amplitude)) return 0;
    const cx = worldX - this.spec.origin[0];
    const cz = worldZ - this.spec.origin[1];
    const cellArea = this.spec.cellSize * this.spec.cellSize;
    let added = 0;
    for (let j = 0; j < this.spec.height; j++) {
      for (let i = 0; i < this.spec.width; i++) {
        const dxm = i * this.spec.cellSize - cx;
        const dzm = j * this.spec.cellSize - cz;
        const weight = splashFalloff(Math.hypot(dxm, dzm), radius);
        if (weight <= 0) continue;
        const base = this.index(i, j) + WATER_CELL_FIELD_OFFSETS.height;
        const before = this.cells[base]!;
        const after = Math.max(before + weight * amplitude, 0);
        this.cells[base] = after;
        added += (after - before) * cellArea;
      }
    }
    this.injectedVolumeM3 += added;
    return added;
  }

  /**
   * One CFL substep. Public so tests can step the scheme at an exact dt
   * instead of through the accumulator.
   */
  stepOnce(dt: number): void {
    if (!Number.isFinite(dt) || dt <= 0) return;
    const { width, height, cellSize, gravity, dryDepth } = this.spec;
    const dampingFactor = Math.exp(-this.spec.damping * dt);
    const O = WATER_CELL_FIELD_OFFSETS;

    // ── Pass 1: face momentum + discharge. Reads h, writes velocity + flux.
    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        const base = this.index(i, j);
        const depthHere = this.cells[base + O.height]!;
        const bedHere = this.cells[base + O.bed]!;
        const etaHere = bedHere + depthHere;
        if (i + 1 < width) {
          const other = this.index(i + 1, j);
          const depthNext = this.cells[other + O.height]!;
          const bedNext = this.cells[other + O.bed]!;
          if (!waterFaceIsOpen(depthHere, bedHere, depthNext, bedNext, dryDepth)) {
            // A closed face carries no water and therefore no momentum.
            // Zeroing the discharge keeps the telescoping sum intact, so
            // conservation is unaffected by this gate.
            this.cells[base + O.velocityX] = 0;
            this.cells[base + O.fluxX] = 0;
          } else {
            const etaNext = bedNext + depthNext;
            const faceDepth = Math.max(0.5 * (depthHere + depthNext), 0);
            const u =
              (this.cells[base + O.velocityX]! - (gravity * dt * (etaNext - etaHere)) / cellSize) *
              dampingFactor;
            this.cells[base + O.velocityX] = u;
            this.cells[base + O.fluxX] = faceDepth * u;
          }
        } else {
          // Domain wall: reflecting boundary, no through-flow.
          this.cells[base + O.velocityX] = 0;
          this.cells[base + O.fluxX] = 0;
        }
        if (j + 1 < height) {
          const other = this.index(i, j + 1);
          const depthNext = this.cells[other + O.height]!;
          const bedNext = this.cells[other + O.bed]!;
          if (!waterFaceIsOpen(depthHere, bedHere, depthNext, bedNext, dryDepth)) {
            this.cells[base + O.velocityZ] = 0;
            this.cells[base + O.fluxZ] = 0;
          } else {
            const etaNext = bedNext + depthNext;
            const faceDepth = Math.max(0.5 * (depthHere + depthNext), 0);
            const w =
              (this.cells[base + O.velocityZ]! - (gravity * dt * (etaNext - etaHere)) / cellSize) *
              dampingFactor;
            this.cells[base + O.velocityZ] = w;
            this.cells[base + O.fluxZ] = faceDepth * w;
          }
        } else {
          this.cells[base + O.velocityZ] = 0;
          this.cells[base + O.fluxZ] = 0;
        }
      }
    }

    // ── Pass 2: continuity. Reads flux, writes h. No neighbour h is read, so
    // this is race-free in place on the GPU.
    const scale = dt / cellSize;
    const cellArea = cellSize * cellSize;
    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        const base = this.index(i, j);
        const outX = this.cells[base + O.fluxX]!;
        const inX = i > 0 ? this.cells[this.index(i - 1, j) + O.fluxX]! : 0;
        const outZ = this.cells[base + O.fluxZ]!;
        const inZ = j > 0 ? this.cells[this.index(i, j - 1) + O.fluxZ]! : 0;
        let next = this.cells[base + O.height]! - scale * (outX - inX + outZ - inZ);
        if (next < 0) {
          // The clamp is the only mass source in the scheme. Account for it.
          this.clampedVolumeM3 += -next * cellArea;
          next = 0;
        }
        this.cells[base + O.height] = next;
      }
    }
  }

  /**
   * Advance by a host frame dt, substepping through the shared
   * `FixedStepClock` at the CFL step. Returns the substeps taken and the
   * seconds the budget refused.
   */
  advance(seconds: number): ShallowWaterStepReport {
    if (!Number.isFinite(seconds) || seconds <= 0 || !this.spec.running) {
      return { steps: 0, droppedSeconds: 0 };
    }
    const before = this.clock.droppedSeconds;
    const steps = this.clock.advance(seconds, (dt) => this.stepOnce(dt));
    return { steps, droppedSeconds: this.clock.droppedSeconds - before };
  }

  /** Cumulative seconds the substep budget has refused since construction. */
  get droppedSeconds(): number {
    return this.clock.droppedSeconds;
  }
}

// ──────────────────────────────────────────────────────────────────────
// WGSL
// ──────────────────────────────────────────────────────────────────────

/**
 * Shared struct declarations. Spliced into the solver module here and into the
 * surface draw module in `water-surface.ts`, so there is exactly one
 * declaration of the cell layout across every shader that touches it.
 */
export const WATER_CELL_STRUCT_WGSL = /* wgsl */ `
struct WaterCell {
  height: f32,
  bed: f32,
  velocityX: f32,
  velocityZ: f32,
  fluxX: f32,
  fluxZ: f32,
}
`;

/**
 * The solver. Three entry points on one module:
 *   `flux_main`    — face momentum + discharge (reads h, writes u/q)
 *   `height_main`  — continuity (reads q, writes h)
 *   `impulse_main` — splash injection
 *
 * Identifier note: none of `active`, `half`, `filter`, `sample`, `common`,
 * `binding` or `typedef` appear as names here. `active` in particular is a
 * reserved WGSL keyword that Tint rejects outright, and that failure only
 * shows up on a real adapter.
 */
export const WATER_SIM_WGSL = /* wgsl */ `
${WATER_CELL_STRUCT_WGSL}
struct WaterSimParams {
  // x = dt, y = gravity, z = cellSize, w = exp(-damping*dt)
  timing: vec4<f32>,
  // x = grid width, y = grid height, z = dry depth, w = alignment padding
  grid: vec4<f32>,
}

struct WaterImpulseParams {
  // x = centre X relative to grid origin, y = centre Z relative to origin,
  // z = radius (m), w = amplitude (m)
  impulse: vec4<f32>,
  // x = grid width, y = grid height, z = cellSize, w = alignment padding
  grid: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<WaterCell>;
@group(0) @binding(1) var<uniform> params: WaterSimParams;

@group(1) @binding(0) var<uniform> impulseParams: WaterImpulseParams;

fn cell_index(i: u32, j: u32, width: u32) -> u32 {
  return j * width + i;
}

// Mirrors waterFaceIsOpen() in the TS. A lake lying against a beach must not
// push itself uphill, so a face whose dry side has a bed at or above the wet
// side's surface is a wall until a wave lifts that surface above it.
fn face_is_open(depthA: f32, bedA: f32, depthB: f32, bedB: f32, dryDepth: f32) -> bool {
  let wetA = depthA > dryDepth;
  let wetB = depthB > dryDepth;
  if (wetA && wetB) {
    return true;
  }
  if (wetA && depthA + bedA > bedB) {
    return true;
  }
  if (wetB && depthB + bedB > bedA) {
    return true;
  }
  return false;
}

// ── Pass 1: face momentum and discharge ──────────────────────────────
// Reads h across the grid, writes this cell's two face velocities and the
// matching discharges. Never writes h, so no invocation can read a value a
// sibling is writing.
@compute @workgroup_size(${WATER_WORKGROUP_SIZE}, ${WATER_WORKGROUP_SIZE})
fn flux_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(params.grid.x);
  let gridHeight = u32(params.grid.y);
  let i = gid.x;
  let j = gid.y;
  if (i >= width || j >= gridHeight) {
    return;
  }
  let dt = params.timing.x;
  let gravity = params.timing.y;
  let cellSize = params.timing.z;
  let dampingFactor = params.timing.w;
  let dryDepth = params.grid.z;

  let here = cell_index(i, j, width);
  let depthHere = cells[here].height;
  let bedHere = cells[here].bed;
  let etaHere = bedHere + depthHere;

  // +X face.
  if (i + 1u < width) {
    let other = cell_index(i + 1u, j, width);
    let depthNext = cells[other].height;
    let bedNext = cells[other].bed;
    if (!face_is_open(depthHere, bedHere, depthNext, bedNext, dryDepth)) {
      cells[here].velocityX = 0.0;
      cells[here].fluxX = 0.0;
    } else {
      let etaNext = bedNext + depthNext;
      let faceDepth = max(0.5 * (depthHere + depthNext), 0.0);
      let u = (cells[here].velocityX - gravity * dt * (etaNext - etaHere) / cellSize) * dampingFactor;
      cells[here].velocityX = u;
      cells[here].fluxX = faceDepth * u;
    }
  } else {
    // Reflecting wall.
    cells[here].velocityX = 0.0;
    cells[here].fluxX = 0.0;
  }

  // +Z face.
  if (j + 1u < gridHeight) {
    let other = cell_index(i, j + 1u, width);
    let depthNext = cells[other].height;
    let bedNext = cells[other].bed;
    if (!face_is_open(depthHere, bedHere, depthNext, bedNext, dryDepth)) {
      cells[here].velocityZ = 0.0;
      cells[here].fluxZ = 0.0;
    } else {
      let etaNext = bedNext + depthNext;
      let faceDepth = max(0.5 * (depthHere + depthNext), 0.0);
      let w = (cells[here].velocityZ - gravity * dt * (etaNext - etaHere) / cellSize) * dampingFactor;
      cells[here].velocityZ = w;
      cells[here].fluxZ = faceDepth * w;
    }
  } else {
    cells[here].velocityZ = 0.0;
    cells[here].fluxZ = 0.0;
  }
}

// ── Pass 2: continuity ───────────────────────────────────────────────
// Reads the discharges written by pass 1 and writes h. Each interior face's
// discharge leaves one cell and enters the other, so the grid sum telescopes
// and total volume is conserved to float rounding.
@compute @workgroup_size(${WATER_WORKGROUP_SIZE}, ${WATER_WORKGROUP_SIZE})
fn height_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(params.grid.x);
  let gridHeight = u32(params.grid.y);
  let i = gid.x;
  let j = gid.y;
  if (i >= width || j >= gridHeight) {
    return;
  }
  let dt = params.timing.x;
  let cellSize = params.timing.z;
  let here = cell_index(i, j, width);

  let outX = cells[here].fluxX;
  var inX = 0.0;
  if (i > 0u) {
    inX = cells[cell_index(i - 1u, j, width)].fluxX;
  }
  let outZ = cells[here].fluxZ;
  var inZ = 0.0;
  if (j > 0u) {
    inZ = cells[cell_index(i, j - 1u, width)].fluxZ;
  }

  // max() is the dry clamp. It is the one place the scheme can create water,
  // and unlike the CPU reference the GPU does not account for how much —
  // tracking it would need an atomic. Read clampedVolumeM3 off the CPU
  // reference when the exact figure matters.
  let next = cells[here].height - (dt / cellSize) * (outX - inX + outZ - inZ);
  cells[here].height = max(next, 0.0);
}

// Mirrors splashFalloff() in the TS: 1 at the centre, 0 at the radius,
// smoothstep between, so the injected bump has no corner to ring on.
fn splash_falloff(distance: f32, radius: f32) -> f32 {
  if (radius <= 0.0) {
    return 0.0;
  }
  let t = 1.0 - clamp(distance / radius, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// ── Impulse injection ────────────────────────────────────────────────
@compute @workgroup_size(${WATER_WORKGROUP_SIZE}, ${WATER_WORKGROUP_SIZE})
fn impulse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(impulseParams.grid.x);
  let gridHeight = u32(impulseParams.grid.y);
  let i = gid.x;
  let j = gid.y;
  if (i >= width || j >= gridHeight) {
    return;
  }
  let cellSize = impulseParams.grid.z;
  let offsetX = f32(i) * cellSize - impulseParams.impulse.x;
  let offsetZ = f32(j) * cellSize - impulseParams.impulse.y;
  let weight = splash_falloff(length(vec2<f32>(offsetX, offsetZ)), impulseParams.impulse.z);
  if (weight <= 0.0) {
    return;
  }
  let here = cell_index(i, j, width);
  cells[here].height = max(cells[here].height + weight * impulseParams.impulse.w, 0.0);
}
`;

// ──────────────────────────────────────────────────────────────────────
// GPU solver
// ──────────────────────────────────────────────────────────────────────

export interface ShallowWaterFieldStats {
  /** Substeps dispatched on the last `simulate()` call. */
  lastSteps: number;
  /** Cumulative seconds the CFL substep budget has refused. */
  droppedSeconds: number;
  /** The CFL substep this field runs at, seconds. */
  substepSeconds: number;
  /** Cell-buffer bytes. */
  gpuBytes: number;
}

interface FieldSlot {
  id: string;
  spec: ShallowWaterSpec;
  bed?: ShallowWaterOptions["bed"];
  cellBuffer: GPUBuffer;
  simParamsBuffer: GPUBuffer;
  impulseParamsBuffer: GPUBuffer;
  simBindGroup: GPUBindGroup;
  impulseBindGroup: GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  clock: FixedStepClock;
  lastSteps: number;
  pendingImpulses: { x: number; z: number; radius: number; amplitude: number }[];
}

/**
 * Owns the three compute pipelines and one cell buffer per water field.
 *
 * Not a scene system. `WaterSurfacePass` drives it and exposes the cell buffer
 * to the surface draw, so the thing simulated and the thing drawn are the same
 * memory — there is no copy of the state for rendering to fall behind.
 */
export class ShallowWaterSim {
  private readonly device: GPUDevice;
  private readonly cellLayout: GPUBindGroupLayout;
  private readonly impulseLayout: GPUBindGroupLayout;
  private readonly fluxPipeline: GPUComputePipeline;
  private readonly heightPipeline: GPUComputePipeline;
  private readonly impulsePipeline: GPUComputePipeline;
  private readonly fields = new Map<string, FieldSlot>();
  private readonly simScratch = new Float32Array(WATER_SIM_PARAMS_BYTES / 4);
  private readonly impulseScratch = new Float32Array(WATER_IMPULSE_PARAMS_BYTES / 4);

  constructor(device: GPUDevice) {
    this.device = device;
    this.cellLayout = device.createBindGroupLayout({
      label: "water/cell-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.impulseLayout = device.createBindGroupLayout({
      label: "water/impulse-layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }],
    });
    const module = device.createShaderModule({ label: "water/sim-shader", code: WATER_SIM_WGSL });
    const layout = device.createPipelineLayout({
      label: "water/sim-pipeline-layout",
      bindGroupLayouts: [this.cellLayout, this.impulseLayout],
    });
    this.fluxPipeline = device.createComputePipeline({
      label: "water/flux",
      layout,
      compute: { module, entryPoint: "flux_main" },
    });
    this.heightPipeline = device.createComputePipeline({
      label: "water/height",
      layout,
      compute: { module, entryPoint: "height_main" },
    });
    this.impulsePipeline = device.createComputePipeline({
      label: "water/impulse",
      layout,
      compute: { module, entryPoint: "impulse_main" },
    });
  }

  /**
   * Allocate a field. Re-attaching the same id replaces it, so this is
   * idempotent. Returns the resolved (clamped) spec.
   */
  attach(
    id: string,
    options: ShallowWaterOptions = {},
    base: Readonly<ShallowWaterSpec> = POOL_PRESET,
  ): ShallowWaterSpec {
    this.detach(id);
    const spec = resolveShallowWaterSpec(options, base);
    const seed = seedWaterField(spec, options.bed);
    let maxDepth = 0;
    for (let c = 0; c < seed.length; c += WATER_CELL_STRIDE_FLOATS) {
      const h = seed[c + WATER_CELL_FIELD_OFFSETS.height]!;
      if (h > maxDepth) maxDepth = h;
    }
    // Headroom: a splash raises the local column above the fill level, which
    // raises the wave speed, which tightens the Courant limit. Size the
    // substep for twice the still depth so a splash cannot push the field
    // past its own stability limit.
    const limit = maxStableTimeStep(spec.cellSize, spec.gravity, maxDepth * 2 + 0.25, spec.cflSafety);
    const substep = Number.isFinite(limit) ? Math.min(limit, 1 / 60) : 1 / 60;

    const cellBuffer = this.device.createBuffer({
      label: `water/cells[${id}]`,
      size: waterGpuBytes(spec),
      // COPY_SRC so `readFieldForDebug` can verify the solver against the CPU
      // reference on real hardware. Free when unused.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Cast as in sky.ts / particle-pass.ts: writeBuffer's signature pins the
    // backing store to ArrayBuffer while Float32Array is generic over
    // ArrayBufferLike. Runtime-compatible; tsc will not infer the variance.
    this.device.queue.writeBuffer(cellBuffer, 0, seed as unknown as BufferSource);

    const simParamsBuffer = this.device.createBuffer({
      label: `water/sim-params[${id}]`,
      size: WATER_SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const impulseParamsBuffer = this.device.createBuffer({
      label: `water/impulse-params[${id}]`,
      size: WATER_IMPULSE_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.fields.set(id, {
      id,
      spec,
      bed: options.bed,
      cellBuffer,
      simParamsBuffer,
      impulseParamsBuffer,
      simBindGroup: this.device.createBindGroup({
        label: `water/sim-bg[${id}]`,
        layout: this.cellLayout,
        entries: [
          { binding: 0, resource: { buffer: cellBuffer } },
          { binding: 1, resource: { buffer: simParamsBuffer } },
        ],
      }),
      impulseBindGroup: this.device.createBindGroup({
        label: `water/impulse-bg[${id}]`,
        layout: this.impulseLayout,
        entries: [{ binding: 0, resource: { buffer: impulseParamsBuffer } }],
      }),
      workgroupsX: waterWorkgroupCount(spec.width),
      workgroupsY: waterWorkgroupCount(spec.height),
      clock: new FixedStepClock(substep, spec.maxSubsteps),
      lastSteps: 0,
      pendingImpulses: [],
    });
    return spec;
  }

  /** Patch field knobs. Grid geometry changes reallocate and restart the field. */
  update(id: string, patch: ShallowWaterOptions): ShallowWaterSpec | null {
    const slot = this.fields.get(id);
    if (!slot) return null;
    const next = resolveShallowWaterSpec({ ...slot.spec, ...patch }, slot.spec);
    const geometryChanged =
      next.width !== slot.spec.width ||
      next.height !== slot.spec.height ||
      next.cellSize !== slot.spec.cellSize ||
      next.gravity !== slot.spec.gravity ||
      next.cflSafety !== slot.spec.cflSafety ||
      next.maxSubsteps !== slot.spec.maxSubsteps ||
      next.surfaceLevel !== slot.spec.surfaceLevel ||
      next.bedLevel !== slot.spec.bedLevel ||
      (patch.bed !== undefined && patch.bed !== slot.bed);
    if (geometryChanged) {
      const bed = patch.bed ?? slot.bed;
      return this.attach(id, { ...next, ...(bed ? { bed } : {}) }, next);
    }
    slot.spec = next;
    return next;
  }

  getSpec(id: string): ShallowWaterSpec | null {
    return this.fields.get(id)?.spec ?? null;
  }

  has(id: string): boolean {
    return this.fields.has(id);
  }

  get fieldCount(): number {
    return this.fields.size;
  }

  ids(): string[] {
    return [...this.fields.keys()];
  }

  /** The cell buffer backing a field — the surface draw binds this directly. */
  cellBufferFor(id: string): GPUBuffer | null {
    return this.fields.get(id)?.cellBuffer ?? null;
  }

  /** Total cell-buffer bytes across all fields (excludes the small uniforms). */
  gpuBytes(): number {
    let bytes = 0;
    for (const slot of this.fields.values()) bytes += waterGpuBytes(slot.spec);
    return bytes;
  }

  getStats(id: string): ShallowWaterFieldStats | null {
    const slot = this.fields.get(id);
    if (!slot) return null;
    return {
      lastSteps: slot.lastSteps,
      droppedSeconds: slot.clock.droppedSeconds,
      substepSeconds: slot.clock.dt,
      gpuBytes: waterGpuBytes(slot.spec),
    };
  }

  /**
   * Queue a splash. It is applied at the start of the next `simulate()` so the
   * injection and the integration land on the same encoder in the right order.
   * A splash adds water; `splashVolumeM3()` gives the exact amount.
   */
  splash(id: string, worldX: number, worldZ: number, radius: number, amplitude: number): boolean {
    const slot = this.fields.get(id);
    if (!slot) return false;
    if (![worldX, worldZ, radius, amplitude].every(Number.isFinite)) return false;
    if (!(radius > 0)) return false;
    slot.pendingImpulses.push({ x: worldX, z: worldZ, radius, amplitude });
    return true;
  }

  /**
   * Step every field. Encodes one compute pass into the caller's encoder so it
   * orders correctly against a render pass recorded on the same encoder.
   *
   * WebGPU orders dispatches within a compute pass and makes an earlier
   * dispatch's storage writes visible to a later one, which is what lets the
   * flux and height passes alternate inside a single pass.
   */
  simulate(encoder: GPUCommandEncoder, dt: number): void {
    if (this.fields.size === 0) return;
    const frame = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.25) : 0;
    const pass = encoder.beginComputePass({ label: "water/compute-pass" });
    for (const slot of this.fields.values()) {
      // Impulses apply even when the field is paused, so a splash into a
      // stopped field is still visible once it resumes.
      for (const impulse of slot.pendingImpulses) {
        packWaterImpulseParams(
          this.impulseScratch,
          slot.spec,
          impulse.x,
          impulse.z,
          impulse.radius,
          impulse.amplitude,
        );
        this.device.queue.writeBuffer(slot.impulseParamsBuffer, 0, this.impulseScratch);
        pass.setPipeline(this.impulsePipeline);
        pass.setBindGroup(0, slot.simBindGroup);
        pass.setBindGroup(1, slot.impulseBindGroup);
        pass.dispatchWorkgroups(slot.workgroupsX, slot.workgroupsY);
      }
      slot.pendingImpulses.length = 0;

      if (!slot.spec.running || frame === 0) {
        slot.lastSteps = 0;
        continue;
      }
      // One uniform write per frame: every substep uses the same dt, so the
      // packed params are identical across them.
      packWaterSimParams(this.simScratch, slot.spec, slot.clock.dt);
      this.device.queue.writeBuffer(slot.simParamsBuffer, 0, this.simScratch);
      // The clock decides how many substeps the frame earns and books the
      // remainder as dropped. The callback only counts: the dispatches are
      // encoded below so pipeline switches are not interleaved per step.
      let steps = 0;
      slot.clock.advance(frame, () => {
        steps++;
      });
      slot.lastSteps = steps;
      for (let s = 0; s < steps; s++) {
        pass.setPipeline(this.fluxPipeline);
        pass.setBindGroup(0, slot.simBindGroup);
        pass.setBindGroup(1, slot.impulseBindGroup);
        pass.dispatchWorkgroups(slot.workgroupsX, slot.workgroupsY);
        pass.setPipeline(this.heightPipeline);
        pass.dispatchWorkgroups(slot.workgroupsX, slot.workgroupsY);
      }
    }
    pass.end();
  }

  /**
   * Copy a field's cells back to the CPU. For tests and tooling ONLY: it
   * allocates a staging buffer and awaits `mapAsync`, which stalls until the
   * GPU drains. Never call this on the frame path.
   *
   * Returns one `WATER_CELL_STRIDE_FLOATS`-wide record per cell, row-major;
   * read fields with `WATER_CELL_FIELD_OFFSETS`. Resolves null for an unknown id.
   */
  async readFieldForDebug(id: string): Promise<Float32Array | null> {
    const slot = this.fields.get(id);
    if (!slot) return null;
    const bytes = waterGpuBytes(slot.spec);
    const staging = this.device.createBuffer({
      label: `water/debug-readback[${id}]`,
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "water/debug-copy" });
    encoder.copyBufferToBuffer(slot.cellBuffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    } finally {
      staging.destroy();
    }
  }

  /** Release a field's GPU buffers. Safe on an unknown id. */
  detach(id: string): void {
    const slot = this.fields.get(id);
    if (!slot) return;
    slot.cellBuffer.destroy();
    slot.simParamsBuffer.destroy();
    slot.impulseParamsBuffer.destroy();
    this.fields.delete(id);
  }

  destroy(): void {
    for (const id of [...this.fields.keys()]) this.detach(id);
  }
}

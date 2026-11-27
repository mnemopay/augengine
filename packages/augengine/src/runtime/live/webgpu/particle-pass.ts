/**
 * GPU particle pass — WGSL compute simulation + camera-facing billboard draw.
 *
 * Why this exists: the repo had two particle *simulations* with no draw path
 * (`physics/particles.ts` ships WGSL + a CPU ring buffer; `@mnemopay/augengine-vfx`
 * ships a compute dispatch) and one CPU emitter that spawns a scene entity per
 * particle (`runtime/systems/vfx-system.ts`). None of them put a pixel on the
 * screen. This module closes that gap: particles live entirely in a GPU storage
 * buffer, are stepped by a compute shader, and are drawn as billboards encoded
 * INTO the live renderer's existing main render pass.
 *
 * Pass placement (see `WebGpuLiveRenderer.render`):
 *   skin compute → particle compute → [sky → opaque geometry → PARTICLES] → post
 *
 * Particles draw last inside the main pass with `depthWriteEnabled: false` and
 * `depthCompare: "less"`, so opaque geometry correctly occludes them while they
 * never occlude each other. Output is premultiplied HDR into the renderer's
 * `rgba16float` target, so an `intensity > 1` flame is picked up by the existing
 * bloom-extract pass in `post-processing.ts` — that is what makes fire read as
 * hot rather than as orange paint.
 *
 * Emission is GPU-side and allocation-free. Each particle owns a time slot:
 * particle `i` of a `capacity`-sized pool first spawns at `i / emissionRate`
 * seconds and thereafter respawns on a `capacity / emissionRate` period. That
 * yields a steady rate with no atomics, no CPU re-seed loop and no
 * `Math.random()` on the frame path. When `lifetime > capacity / emissionRate`
 * the pool saturates and the effective rate is clamped to
 * `capacity / lifetime` — pool-limited, which is the honest behaviour.
 *
 * Determinism: the RNG is a hash of (particle index, frame counter, emitter
 * seed), so a replay that feeds the same dt sequence reproduces the same
 * simulation ON THE SAME DEVICE. Float results are not bit-identical across
 * GPU vendors, so this is NOT cross-device deterministic and must not be used
 * as a lockstep multiplayer input.
 *
 * NOT implemented here (do not document these as working):
 *   - Depth-buffer soft particles. Fading against scene depth needs the depth
 *     texture as a sampled input, which it cannot be while it is the pass's
 *     depth attachment. "Soft" below means a radial alpha falloff only.
 *   - Back-to-front sorting. Additive blending is order-independent, so the
 *     fire preset is correct; the alpha-blended smoke preset will show
 *     intra-emitter ordering artifacts.
 *   - Lighting. Particles are emissive only; they neither receive nor cast light.
 *   - Collision. Particles do not interact with scene geometry or physics.
 *
 * GPU cost: `capacity * 48` bytes per emitter. `PLAYABLE_PARTICLE_CAPACITY`
 * (16384 → 786 KB) sits at ~2.5 % of the 32 MB GPU ceiling in
 * `content/budget.ts`; `MAX_PARTICLE_CAPACITY` (65536 → 3.1 MB) is the hard cap.
 */

import type { Mat4, Vec3 } from "../../../scene/types.js";

// ──────────────────────────────────────────────────────────────────────
// Layout constants — mirrored by the WGSL `Particle` struct below.
// ──────────────────────────────────────────────────────────────────────

/**
 * Bytes per particle. WGSL std430: `vec3f` aligns to 16, so the struct is
 *   position(0,12) life(12,4) velocity(16,12) maxLife(28,4)
 *   seed(32,4) size(36,4) temperature(40,4) respawn(44,4)
 * = 48 bytes, already a multiple of the 16-byte struct alignment.
 */
export const PARTICLE_STRIDE_BYTES = 48;

/** Floats per particle (`PARTICLE_STRIDE_BYTES / 4`). */
export const PARTICLE_STRIDE_FLOATS = 12;

/** Float index of each field inside one particle's stride. */
export const PARTICLE_FIELD_OFFSETS = Object.freeze({
  position: 0,
  life: 3,
  velocity: 4,
  maxLife: 7,
  seed: 8,
  size: 9,
  temperature: 10,
  respawn: 11,
});

/** Bytes of the per-emitter simulation uniform. 10 × vec4. */
export const SIM_PARAMS_BYTES = 160;

/** Bytes of the per-emitter draw uniform. mat4 + 4 × vec4. */
export const DRAW_PARAMS_BYTES = 128;

/** Compute workgroup size — a safe default across desktop and mobile adapters. */
export const PARTICLE_WORKGROUP_SIZE = 64;

/** Vertices per billboard (two triangles, no vertex or index buffer). */
export const PARTICLE_VERTS_PER_QUAD = 6;

/** Hard per-emitter pool cap. 65536 × 48 B = 3.1 MB. */
export const MAX_PARTICLE_CAPACITY = 65_536;

/** Recommended pool size for the Intel-UHD playable floor. 16384 × 48 B = 786 KB. */
export const PLAYABLE_PARTICLE_CAPACITY = 16_384;

// ──────────────────────────────────────────────────────────────────────
// Emitter description
// ──────────────────────────────────────────────────────────────────────

/** How the fragment shader turns `temperature` into colour. */
export type ParticleColorMode = "fire" | "flat" | "ember";

/** Blend mode of the billboard draw. */
export type ParticleBlend = "additive" | "alpha";

export interface ParticleEmitterSpec {
  /** Pool size. Clamped to [1, MAX_PARTICLE_CAPACITY]. Fixed until re-attached. */
  capacity: number;
  /** Particles per second. Effective rate is capped at `capacity / lifetime[1]`. */
  emissionRate: number;
  /** World-space emitter centre. */
  origin: Vec3;
  /** Half-extents of the spawn box around `origin`, in metres. */
  spawnExtent: Vec3;
  /** Per-axis minimum spawn velocity (m/s). */
  velocityMin: Vec3;
  /** Per-axis maximum spawn velocity (m/s). */
  velocityMax: Vec3;
  /** Constant acceleration (m/s²). Fire uses a near-zero or slightly negative y. */
  gravity: Vec3;
  /** Constant wind acceleration (m/s²). */
  wind: Vec3;
  /**
   * Upward acceleration scaled by the particle's current temperature (m/s²).
   * This is what makes flame climb fast while hot and stall as it cools. It is
   * an authored look parameter, not a buoyancy derived from gas density.
   */
  buoyancy: number;
  /** Exponential velocity damping per second. `v *= exp(-drag * dt)`. */
  drag: number;
  /** Strength of the divergence-free swirl field (m/s²). */
  turbulence: number;
  /** Spatial frequency of the swirl field (1/m). */
  turbulenceFrequency: number;
  /** Lifetime range in seconds, `[min, max]`. Both must be > 0. */
  lifetime: [number, number];
  /** Billboard diameter range in metres at spawn, `[min, max]`. */
  sizeRange: [number, number];
  /**
   * Cooling exponent. `temperature = lifeRatio ^ coolRate`, so 1 cools
   * linearly over life and > 1 cools faster at the start.
   */
  coolRate: number;
  /** Fractional size growth over the particle's life. 0.8 = +80 % by death. */
  growth: number;
  /** Linear RGB tint. Multiplies the fire ramp, or is used directly in "flat". */
  tint: Vec3;
  /** HDR output multiplier. > 1 drives the existing bloom pass. */
  intensity: number;
  /** Overall alpha multiplier, 0..1. */
  alphaScale: number;
  /** Radial falloff exponent. Higher = a tighter, harder core. */
  softness: number;
  colorMode: ParticleColorMode;
  blend: ParticleBlend;
  /** RNG seed. Two emitters with the same seed and dt sequence match exactly. */
  seed: number;
  /** When false, live particles finish their life and nothing respawns. */
  active: boolean;
}

export type ParticleEmitterOptions = Partial<ParticleEmitterSpec>;

/**
 * Candle / torch flame. Additive, HDR-hot, strong temperature-driven rise,
 * mild swirl. `intensity` is deliberately > 1 so bloom catches the core.
 */
export const FIRE_PRESET: Readonly<ParticleEmitterSpec> = Object.freeze({
  capacity: 4096,
  emissionRate: 900,
  origin: [0, 0, 0] as Vec3,
  spawnExtent: [0.09, 0.02, 0.09] as Vec3,
  velocityMin: [-0.25, 0.55, -0.25] as Vec3,
  velocityMax: [0.25, 1.35, 0.25] as Vec3,
  gravity: [0, 0, 0] as Vec3,
  wind: [0, 0, 0] as Vec3,
  buoyancy: 3.1,
  drag: 1.5,
  turbulence: 1.7,
  turbulenceFrequency: 2.3,
  lifetime: [0.45, 1.05] as [number, number],
  sizeRange: [0.1, 0.22] as [number, number],
  coolRate: 1.45,
  growth: 0.55,
  tint: [1, 1, 1] as Vec3,
  intensity: 2.6,
  alphaScale: 0.85,
  softness: 1.6,
  colorMode: "fire" as ParticleColorMode,
  blend: "additive" as ParticleBlend,
  seed: 0x9e3779b9,
  active: true,
});

/**
 * Grey smoke plume. Alpha-blended, cool, slow, wide, long-lived. Draw this
 * emitter alongside FIRE_PRESET for a plume above a flame.
 */
export const SMOKE_PRESET: Readonly<ParticleEmitterSpec> = Object.freeze({
  ...FIRE_PRESET,
  capacity: 2048,
  emissionRate: 120,
  spawnExtent: [0.14, 0.05, 0.14] as Vec3,
  velocityMin: [-0.18, 0.5, -0.18] as Vec3,
  velocityMax: [0.18, 0.95, 0.18] as Vec3,
  buoyancy: 0.75,
  drag: 0.8,
  turbulence: 0.95,
  turbulenceFrequency: 1.1,
  lifetime: [1.6, 3.2] as [number, number],
  sizeRange: [0.22, 0.5] as [number, number],
  coolRate: 1,
  growth: 1.8,
  tint: [0.19, 0.18, 0.185] as Vec3,
  intensity: 1,
  alphaScale: 0.3,
  softness: 2.4,
  colorMode: "flat" as ParticleColorMode,
  blend: "alpha" as ParticleBlend,
  seed: 0x85ebca6b,
});

/** Sparks thrown off a fire: ballistic, tiny, additive, long-lived. */
export const EMBER_PRESET: Readonly<ParticleEmitterSpec> = Object.freeze({
  ...FIRE_PRESET,
  capacity: 1024,
  emissionRate: 70,
  spawnExtent: [0.07, 0.02, 0.07] as Vec3,
  velocityMin: [-0.9, 1.5, -0.9] as Vec3,
  velocityMax: [0.9, 3.4, 0.9] as Vec3,
  gravity: [0, -2.4, 0] as Vec3,
  buoyancy: 0.9,
  drag: 0.35,
  turbulence: 1.2,
  turbulenceFrequency: 1.7,
  lifetime: [0.8, 1.9] as [number, number],
  sizeRange: [0.016, 0.04] as [number, number],
  coolRate: 2.1,
  growth: -0.4,
  tint: [1, 0.62, 0.26] as Vec3,
  intensity: 4.2,
  alphaScale: 1,
  softness: 1.1,
  colorMode: "ember" as ParticleColorMode,
  seed: 0xc2b2ae35,
});

// ──────────────────────────────────────────────────────────────────────
// Pure helpers — unit-testable without a GPU device.
// ──────────────────────────────────────────────────────────────────────

function clampNumber(value: number, lo: number, hi: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return value < lo ? lo : value > hi ? hi : value;
}

function resolveVec3(value: Vec3 | undefined, fallback: Vec3): Vec3 {
  if (!value) return fallback;
  const out: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const v = value[i];
    out[i] = Number.isFinite(v) ? (v as number) : fallback[i]!;
  }
  return out;
}

/**
 * Merge caller options over a preset and clamp every field into a range the
 * shaders can handle. Invalid or non-finite input falls back to the base value
 * rather than reaching the GPU and producing NaN positions.
 */
export function resolveEmitterSpec(
  options: ParticleEmitterOptions = {},
  base: Readonly<ParticleEmitterSpec> = FIRE_PRESET,
): ParticleEmitterSpec {
  const lifetimeLo = clampNumber(options.lifetime?.[0] ?? base.lifetime[0], 1e-3, 120, base.lifetime[0]);
  const lifetimeHi = clampNumber(options.lifetime?.[1] ?? base.lifetime[1], 1e-3, 120, base.lifetime[1]);
  const sizeLo = clampNumber(options.sizeRange?.[0] ?? base.sizeRange[0], 1e-4, 100, base.sizeRange[0]);
  const sizeHi = clampNumber(options.sizeRange?.[1] ?? base.sizeRange[1], 1e-4, 100, base.sizeRange[1]);
  const seed = options.seed ?? base.seed;
  return {
    capacity: Math.max(1, Math.min(MAX_PARTICLE_CAPACITY, Math.floor(clampNumber(options.capacity ?? base.capacity, 1, MAX_PARTICLE_CAPACITY, base.capacity)))),
    emissionRate: clampNumber(options.emissionRate ?? base.emissionRate, 0, 1e6, base.emissionRate),
    origin: resolveVec3(options.origin, base.origin),
    spawnExtent: resolveVec3(options.spawnExtent, base.spawnExtent),
    velocityMin: resolveVec3(options.velocityMin, base.velocityMin),
    velocityMax: resolveVec3(options.velocityMax, base.velocityMax),
    gravity: resolveVec3(options.gravity, base.gravity),
    wind: resolveVec3(options.wind, base.wind),
    buoyancy: clampNumber(options.buoyancy ?? base.buoyancy, -1e4, 1e4, base.buoyancy),
    drag: clampNumber(options.drag ?? base.drag, 0, 100, base.drag),
    turbulence: clampNumber(options.turbulence ?? base.turbulence, 0, 1e3, base.turbulence),
    turbulenceFrequency: clampNumber(options.turbulenceFrequency ?? base.turbulenceFrequency, 0, 1e3, base.turbulenceFrequency),
    // Sorted so a caller passing [max, min] gets a usable range instead of a
    // negative span that would make every lifetime collapse to the max.
    lifetime: [Math.min(lifetimeLo, lifetimeHi), Math.max(lifetimeLo, lifetimeHi)],
    sizeRange: [Math.min(sizeLo, sizeHi), Math.max(sizeLo, sizeHi)],
    coolRate: clampNumber(options.coolRate ?? base.coolRate, 0.01, 32, base.coolRate),
    growth: clampNumber(options.growth ?? base.growth, -0.99, 32, base.growth),
    tint: resolveVec3(options.tint, base.tint),
    intensity: clampNumber(options.intensity ?? base.intensity, 0, 1e4, base.intensity),
    alphaScale: clampNumber(options.alphaScale ?? base.alphaScale, 0, 1, base.alphaScale),
    softness: clampNumber(options.softness ?? base.softness, 0.05, 32, base.softness),
    colorMode: options.colorMode ?? base.colorMode,
    blend: options.blend ?? base.blend,
    // Masked to 24 bits so the value survives the f32 uniform exactly — f32
    // only represents integers below 2^24 without rounding.
    seed: Number.isFinite(seed) ? Math.abs(Math.floor(seed)) % 0x1000000 : base.seed % 0x1000000,
    active: options.active ?? base.active,
  };
}

/** GPU storage bytes a pool of `capacity` particles occupies. */
export function particleGpuBytes(capacity: number): number {
  return Math.max(0, Math.floor(capacity)) * PARTICLE_STRIDE_BYTES;
}

/** Compute workgroups needed to cover `capacity` particles. */
export function particleWorkgroupCount(capacity: number): number {
  return Math.ceil(Math.max(0, capacity) / PARTICLE_WORKGROUP_SIZE);
}

/**
 * Steady-state alive-particle count. `emissionRate * meanLifetime`, clamped to
 * the pool size — above that the emitter is pool-limited, not rate-limited.
 */
export function expectedAliveCount(spec: ParticleEmitterSpec): number {
  const meanLifetime = (spec.lifetime[0] + spec.lifetime[1]) / 2;
  return Math.min(spec.capacity, spec.emissionRate * meanLifetime);
}

/**
 * Initial pool contents: every particle dormant, staggered so particle `i`
 * first spawns at `i / emissionRate` seconds. Without the stagger the whole
 * pool would spawn on frame 1 and then pulse once per period.
 */
export function seedParticlePool(spec: ParticleEmitterSpec): Float32Array {
  const out = new Float32Array(spec.capacity * PARTICLE_STRIDE_FLOATS);
  const rate = spec.emissionRate > 0 ? spec.emissionRate : 1;
  for (let i = 0; i < spec.capacity; i++) {
    const base = i * PARTICLE_STRIDE_FLOATS;
    // life = 0 marks the slot dead; respawn counts down to its first spawn.
    out[base + PARTICLE_FIELD_OFFSETS.respawn] = i / rate;
  }
  return out;
}

/**
 * Camera right/up basis for billboarding, read out of a column-major view
 * matrix. `m4LookAt` writes the camera's rotation transposed into the upper
 * 3×3, so the world-space basis vectors are the matrix ROWS: right is
 * (m0, m4, m8) and up is (m1, m5, m9).
 */
export function cameraBasisFromView(view: Mat4): { right: Vec3; up: Vec3 } {
  return {
    right: [view[0] ?? 1, view[4] ?? 0, view[8] ?? 0],
    up: [view[1] ?? 0, view[5] ?? 1, view[9] ?? 0],
  };
}

const COLOR_MODE_CODES: Readonly<Record<ParticleColorMode, number>> = Object.freeze({
  fire: 0,
  flat: 1,
  ember: 2,
});

/** Numeric code the fragment shader branches on. */
export function colorModeCode(mode: ParticleColorMode): number {
  return COLOR_MODE_CODES[mode] ?? 0;
}

/**
 * Pack the simulation uniform. `out` must hold `SIM_PARAMS_BYTES / 4` floats.
 * Layout (float indices) is mirrored by the WGSL `SimParams` struct:
 *   0 dt · 1 time · 2 period · 3 activeFlag
 *   4..6 origin · 8..10 spawnExtent · 12..14 velocityMin · 16..18 velocityMax
 *   20..22 gravity · 23 buoyancy · 24..26 wind · 27 drag
 *   28 lifeMin · 29 lifeMax · 30 sizeMin · 31 sizeMax
 *   32 turbulence · 33 turbulenceFrequency · 34 coolRate · 35 reserved
 *   36 frame · 37 seed
 *
 * `growth` is deliberately absent: size growth is a per-draw derived value and
 * lives in the draw uniform, so the simulation never compounds it frame to frame.
 */
export function packSimParams(
  out: Float32Array,
  spec: ParticleEmitterSpec,
  dt: number,
  time: number,
  frame: number,
): Float32Array {
  if (out.length < SIM_PARAMS_BYTES / 4) {
    throw new Error(`packSimParams needs ${SIM_PARAMS_BYTES / 4} floats, got ${out.length}`);
  }
  out.fill(0);
  // A zero or negative emission rate must not divide by zero; a huge period
  // simply means "never respawn", which is the right reading of rate 0.
  const period = spec.emissionRate > 0 ? spec.capacity / spec.emissionRate : 1e9;
  out[0] = dt;
  out[1] = time;
  out[2] = period;
  out[3] = spec.active ? 1 : 0;
  out.set(spec.origin, 4);
  out.set(spec.spawnExtent, 8);
  out.set(spec.velocityMin, 12);
  out.set(spec.velocityMax, 16);
  out.set(spec.gravity, 20);
  out[23] = spec.buoyancy;
  out.set(spec.wind, 24);
  out[27] = spec.drag;
  out[28] = spec.lifetime[0];
  out[29] = spec.lifetime[1];
  out[30] = spec.sizeRange[0];
  out[31] = spec.sizeRange[1];
  out[32] = spec.turbulence;
  out[33] = spec.turbulenceFrequency;
  out[34] = spec.coolRate;
  // out[35] reserved — growth belongs to the draw uniform, see packDrawParams.
  out[36] = frame;
  out[37] = spec.seed;
  return out;
}

/**
 * Pack the draw uniform. `out` must hold `DRAW_PARAMS_BYTES / 4` floats.
 *   0..15 viewProj · 16..18 camRight · 20..22 camUp
 *   24..26 tint · 27 intensity
 *   28 softness · 29 alphaScale · 30 colorMode · 31 growth
 */
export function packDrawParams(
  out: Float32Array,
  viewProj: Mat4,
  right: Vec3,
  up: Vec3,
  spec: ParticleEmitterSpec,
): Float32Array {
  if (out.length < DRAW_PARAMS_BYTES / 4) {
    throw new Error(`packDrawParams needs ${DRAW_PARAMS_BYTES / 4} floats, got ${out.length}`);
  }
  out.fill(0);
  out.set(viewProj as unknown as ArrayLike<number>, 0);
  out.set(right, 16);
  out.set(up, 20);
  out.set(spec.tint, 24);
  out[27] = spec.intensity;
  out[28] = spec.softness;
  out[29] = spec.alphaScale;
  out[30] = colorModeCode(spec.colorMode);
  out[31] = spec.growth;
  return out;
}

/** Premultiplied blend state for each supported mode. */
export function particleBlendState(blend: ParticleBlend): GPUBlendState {
  if (blend === "additive") {
    return {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    };
  }
  return {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
}

// ──────────────────────────────────────────────────────────────────────
// WGSL
// ──────────────────────────────────────────────────────────────────────

/** Shared struct declarations, spliced into both the compute and draw modules. */
const PARTICLE_STRUCTS_WGSL = /* wgsl */ `
struct Particle {
  position: vec3<f32>,
  life: f32,
  velocity: vec3<f32>,
  maxLife: f32,
  seed: f32,
  size: f32,
  temperature: f32,
  respawn: f32,
}
`;

export const PARTICLE_SIM_WGSL = /* wgsl */ `
${PARTICLE_STRUCTS_WGSL}
struct SimParams {
  // x = dt, y = elapsed time, z = respawn period, w = active flag
  timing: vec4<f32>,
  origin: vec4<f32>,
  spawnExtent: vec4<f32>,
  velocityMin: vec4<f32>,
  velocityMax: vec4<f32>,
  // xyz = gravity, w = buoyancy
  gravity: vec4<f32>,
  // xyz = wind, w = drag
  wind: vec4<f32>,
  // x = lifeMin, y = lifeMax, z = sizeMin, w = sizeMax
  ranges: vec4<f32>,
  // x = turbulence, y = turbulence frequency, z = cool rate, w = reserved
  shaping: vec4<f32>,
  // x = frame counter, y = emitter seed
  misc: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

// Integer hash (Wang/PCG-style mix). Deterministic and cheap; good enough
// decorrelation for spawn jitter.
fn hash_u32(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

fn rand01(state: ptr<function, u32>) -> f32 {
  (*state) = hash_u32(*state);
  // Top 24 bits → [0, 1). 24 bits is exactly f32's mantissa width.
  return f32((*state) >> 8u) * (1.0 / 16777216.0);
}

// ABC flow: div = d/dx(sin z + cos y) + d/dy(sin x + cos z) + d/dz(sin y + cos x) = 0.
// Exactly divergence-free, so particles swirl without bunching into clumps or
// tearing voids the way a naive per-axis noise field does.
fn abc_flow(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    sin(p.z) + cos(p.y),
    sin(p.x) + cos(p.z),
    sin(p.y) + cos(p.x),
  );
}

fn spawn(index: u32, carryOver: f32) -> Particle {
  var state = hash_u32(index ^ hash_u32(u32(params.misc.x) ^ u32(params.misc.y)));
  var p: Particle;
  let jitter = vec3<f32>(
    rand01(&state) * 2.0 - 1.0,
    rand01(&state) * 2.0 - 1.0,
    rand01(&state) * 2.0 - 1.0,
  );
  p.position = params.origin.xyz + jitter * params.spawnExtent.xyz;
  p.velocity = vec3<f32>(
    mix(params.velocityMin.x, params.velocityMax.x, rand01(&state)),
    mix(params.velocityMin.y, params.velocityMax.y, rand01(&state)),
    mix(params.velocityMin.z, params.velocityMax.z, rand01(&state)),
  );
  let maxLife = mix(params.ranges.x, params.ranges.y, rand01(&state));
  p.maxLife = maxLife;
  // carryOver is the (negative) overshoot past the scheduled spawn time, so a
  // particle that was due mid-frame is already that far into its life. Without
  // it the emitter quantises to the frame rate and visibly pulses.
  p.life = max(maxLife + carryOver, 1e-4);
  p.size = mix(params.ranges.z, params.ranges.w, rand01(&state));
  p.temperature = 1.0;
  p.seed = f32(state >> 8u);
  p.respawn = 0.0;
  return p;
}

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= arrayLength(&particles)) {
    return;
  }
  let dt = params.timing.x;
  let period = params.timing.z;
  // NOT named "active": that is a reserved WGSL keyword and fails parsing.
  let isActive = params.timing.w > 0.5;
  var p = particles[index];

  if (p.life > 0.0) {
    // Temperature is DERIVED from remaining life, not integrated, so it can
    // never drift or go negative after a long run.
    let lifeRatio = clamp(p.life / max(p.maxLife, 1e-4), 0.0, 1.0);
    p.temperature = pow(lifeRatio, params.shaping.z);

    p.velocity = p.velocity + params.gravity.xyz * dt;
    p.velocity = p.velocity + params.wind.xyz * dt;
    p.velocity.y = p.velocity.y + params.gravity.w * p.temperature * dt;
    let swirl = abc_flow(
      p.position * params.shaping.y + vec3<f32>(params.timing.y * 0.7)
    );
    p.velocity = p.velocity + swirl * params.shaping.x * dt;
    // Exponential drag integrates exactly, so it stays stable at any dt
    // instead of overshooting into negative velocity like (1 - drag*dt).
    p.velocity = p.velocity * exp(-params.wind.w * dt);
    p.position = p.position + p.velocity * dt;

    p.life = p.life - dt;
    if (p.life <= 0.0) {
      p.life = 0.0;
      // Wait out the remainder of this slot's period. Clamped at 0, so when
      // lifetime exceeds the period the pool saturates and the effective rate
      // becomes capacity/lifetime rather than the requested rate.
      p.respawn = max(period - p.maxLife, 0.0);
    }
  } else {
    p.respawn = p.respawn - dt;
    if (isActive && p.respawn <= 0.0) {
      p = spawn(index, p.respawn);
    }
  }

  particles[index] = p;
}
`;

export const PARTICLE_DRAW_WGSL = /* wgsl */ `
${PARTICLE_STRUCTS_WGSL}
struct DrawParams {
  viewProj: mat4x4<f32>,
  camRight: vec4<f32>,
  camUp: vec4<f32>,
  // xyz = tint, w = intensity
  tint: vec4<f32>,
  // x = softness, y = alphaScale, z = colorMode, w = growth
  shaping: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: DrawParams;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) offset: vec2<f32>,
  @location(1) temperature: f32,
  @location(2) fade: f32,
}

// Blackbody-ish flame ramp: dark smoke → deep red → orange → yellow → white.
fn fire_ramp(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  var c = mix(vec3<f32>(0.06, 0.05, 0.05), vec3<f32>(0.85, 0.12, 0.02), smoothstep(0.0, 0.25, x));
  c = mix(c, vec3<f32>(1.0, 0.45, 0.06), smoothstep(0.22, 0.52, x));
  c = mix(c, vec3<f32>(1.0, 0.85, 0.35), smoothstep(0.5, 0.8, x));
  c = mix(c, vec3<f32>(1.0, 0.97, 0.88), smoothstep(0.82, 1.0, x));
  return c;
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VsOut {
  var out: VsOut;
  let p = particles[ii];
  if (p.life <= 0.0) {
    // Park dead particles past the far plane (z/w > 1) so the rasteriser
    // clips them outright. Cheaper than a zero-area quad.
    out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.offset = vec2<f32>(0.0, 0.0);
    out.temperature = 0.0;
    out.fade = 0.0;
    return out;
  }

  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0, 1.0),
  );
  let corner = corners[vi];
  let lifeRatio = clamp(p.life / max(p.maxLife, 1e-4), 0.0, 1.0);
  let ageRatio = 1.0 - lifeRatio;
  // Growth is derived from age every frame rather than integrated into
  // p.size, so it cannot compound. Floor at 1% of spawn size: a negative
  // growth (embers shrinking) must never invert the quad.
  let grown = p.size * max(1.0 + params.shaping.w * ageRatio, 0.01);
  let halfSize = grown * 0.5;
  let world = p.position
    + params.camRight.xyz * corner.x * halfSize
    + params.camUp.xyz * corner.y * halfSize;

  out.clip = params.viewProj * vec4<f32>(world, 1.0);
  out.offset = corner;
  out.temperature = p.temperature;
  // Ease in over the first 12 % of life and out over the last 40 %, so
  // particles never pop into or out of existence.
  out.fade = smoothstep(0.0, 0.12, ageRatio) * smoothstep(0.0, 0.4, lifeRatio);
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let r = length(in.offset);
  if (r >= 1.0) {
    discard;
  }
  let falloff = pow(1.0 - r, params.shaping.x);
  let mode = params.shaping.z;
  var rgb: vec3<f32>;
  if (mode < 0.5) {
    rgb = fire_ramp(in.temperature) * params.tint.rgb;
  } else if (mode < 1.5) {
    rgb = params.tint.rgb;
  } else {
    rgb = params.tint.rgb * in.temperature;
  }
  let a = falloff * in.fade * params.shaping.y;
  // Premultiplied output — matches both blend states in particleBlendState().
  return vec4<f32>(rgb * params.tint.w * a, a);
}
`;

// ──────────────────────────────────────────────────────────────────────
// Pass
// ──────────────────────────────────────────────────────────────────────

export interface ParticlePassOptions {
  /** Colour target format. Must match the main render pass (renderer: rgba16float). */
  targetFormat: GPUTextureFormat;
  /** MSAA sample count. Must match the main render pass. */
  sampleCount: 1 | 4;
  /** Depth attachment format of the main render pass. */
  depthFormat?: GPUTextureFormat;
}

interface EmitterSlot {
  id: string;
  spec: ParticleEmitterSpec;
  particleBuffer: GPUBuffer;
  simParamsBuffer: GPUBuffer;
  drawParamsBuffer: GPUBuffer;
  computeBindGroup: GPUBindGroup;
  drawBindGroup: GPUBindGroup;
  workgroupCount: number;
}

/**
 * Owns the particle compute pipeline, one render pipeline per blend mode, and
 * a GPU pool per emitter. Not a scene system — the live renderer drives it.
 */
export class ParticlePass {
  private readonly device: GPUDevice;
  private readonly targetFormat: GPUTextureFormat;
  private readonly sampleCount: 1 | 4;
  private readonly depthFormat: GPUTextureFormat;
  private readonly computePipeline: GPUComputePipeline;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly drawLayout: GPUBindGroupLayout;
  private readonly drawModule: GPUShaderModule;
  private readonly drawPipelines = new Map<ParticleBlend, GPURenderPipeline>();
  private readonly emitters = new Map<string, EmitterSlot>();
  private readonly simScratch = new Float32Array(SIM_PARAMS_BYTES / 4);
  private readonly drawScratch = new Float32Array(DRAW_PARAMS_BYTES / 4);
  private elapsed = 0;
  private frame = 0;

  constructor(device: GPUDevice, options: ParticlePassOptions) {
    this.device = device;
    this.targetFormat = options.targetFormat;
    this.sampleCount = options.sampleCount;
    this.depthFormat = options.depthFormat ?? "depth24plus";

    this.computeLayout = device.createBindGroupLayout({
      label: "particles/compute-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    this.computePipeline = device.createComputePipeline({
      label: "particles/sim",
      layout: device.createPipelineLayout({
        label: "particles/compute-pipeline-layout",
        bindGroupLayouts: [this.computeLayout],
      }),
      compute: {
        module: device.createShaderModule({ label: "particles/sim-shader", code: PARTICLE_SIM_WGSL }),
        entryPoint: "main",
      },
    });

    // Vertex-stage storage reads must be read-only; WebGPU forbids
    // read_write storage outside compute.
    this.drawLayout = device.createBindGroupLayout({
      label: "particles/draw-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.drawModule = device.createShaderModule({
      label: "particles/draw-shader",
      code: PARTICLE_DRAW_WGSL,
    });
  }

  /** Lazily compile the render pipeline for a blend mode. */
  private pipelineFor(blend: ParticleBlend): GPURenderPipeline {
    const cached = this.drawPipelines.get(blend);
    if (cached) return cached;
    const pipeline = this.device.createRenderPipeline({
      label: `particles/draw-${blend}`,
      layout: this.device.createPipelineLayout({
        label: `particles/draw-pipeline-layout-${blend}`,
        bindGroupLayouts: [this.drawLayout],
      }),
      vertex: { module: this.drawModule, entryPoint: "vs_main" },
      fragment: {
        module: this.drawModule,
        entryPoint: "fs_main",
        targets: [{ format: this.targetFormat, blend: particleBlendState(blend) }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // Test against opaque geometry so the world occludes particles, but
      // never write depth — particles must not occlude each other or anything
      // drawn after them.
      depthStencil: {
        format: this.depthFormat,
        depthWriteEnabled: false,
        depthCompare: "less",
      },
      multisample: { count: this.sampleCount },
    });
    this.drawPipelines.set(blend, pipeline);
    return pipeline;
  }

  /**
   * Allocate an emitter. Re-attaching the same id replaces it, so this is
   * idempotent. Returns the resolved (clamped) spec.
   */
  attach(id: string, options: ParticleEmitterOptions = {}, base: Readonly<ParticleEmitterSpec> = FIRE_PRESET): ParticleEmitterSpec {
    this.detach(id);
    const spec = resolveEmitterSpec(options, base);
    const particleBuffer = this.device.createBuffer({
      label: `particles/pool[${id}]`,
      size: particleGpuBytes(spec.capacity),
      // COPY_SRC exists so `readPoolForDebug` can verify the simulation on
      // real hardware. It costs nothing when unused.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Cast as in sky.ts: writeBuffer's signature pins the backing store to
    // ArrayBuffer, while Float32Array is generic over ArrayBufferLike.
    // Runtime-compatible; tsc just will not infer the variance.
    this.device.queue.writeBuffer(particleBuffer, 0, seedParticlePool(spec) as unknown as BufferSource);

    // One uniform buffer per emitter. Sharing a single uniform across
    // dispatches would make every emitter read whichever write landed last.
    const simParamsBuffer = this.device.createBuffer({
      label: `particles/sim-params[${id}]`,
      size: SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const drawParamsBuffer = this.device.createBuffer({
      label: `particles/draw-params[${id}]`,
      size: DRAW_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.emitters.set(id, {
      id,
      spec,
      particleBuffer,
      simParamsBuffer,
      drawParamsBuffer,
      computeBindGroup: this.device.createBindGroup({
        label: `particles/compute-bg[${id}]`,
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: particleBuffer } },
          { binding: 1, resource: { buffer: simParamsBuffer } },
        ],
      }),
      drawBindGroup: this.device.createBindGroup({
        label: `particles/draw-bg[${id}]`,
        layout: this.drawLayout,
        entries: [
          { binding: 0, resource: { buffer: particleBuffer } },
          { binding: 1, resource: { buffer: drawParamsBuffer } },
        ],
      }),
      workgroupCount: particleWorkgroupCount(spec.capacity),
    });
    // Compile eagerly so a bad blend mode fails at attach, not mid-frame.
    this.pipelineFor(spec.blend);
    return spec;
  }

  /**
   * Patch emitter knobs in place. A changed `capacity` reallocates the pool
   * and therefore restarts the emitter — everything else is free.
   */
  update(id: string, patch: ParticleEmitterOptions): ParticleEmitterSpec | null {
    const slot = this.emitters.get(id);
    if (!slot) return null;
    const next = resolveEmitterSpec({ ...slot.spec, ...patch }, slot.spec);
    if (next.capacity !== slot.spec.capacity) return this.attach(id, next, next);
    slot.spec = next;
    this.pipelineFor(next.blend);
    return next;
  }

  /** Read back the resolved spec for an emitter. */
  getSpec(id: string): ParticleEmitterSpec | null {
    return this.emitters.get(id)?.spec ?? null;
  }

  has(id: string): boolean {
    return this.emitters.has(id);
  }

  /** Release an emitter's GPU buffers. Safe on an unknown id. */
  detach(id: string): void {
    const slot = this.emitters.get(id);
    if (!slot) return;
    slot.particleBuffer.destroy();
    slot.simParamsBuffer.destroy();
    slot.drawParamsBuffer.destroy();
    this.emitters.delete(id);
  }

  get emitterCount(): number {
    return this.emitters.size;
  }

  /** Total particle-pool bytes across all emitters (excludes the small uniforms). */
  gpuBytes(): number {
    let bytes = 0;
    for (const slot of this.emitters.values()) bytes += particleGpuBytes(slot.spec.capacity);
    return bytes;
  }

  /**
   * Steady-state alive estimate across emitters. Derived from rate × lifetime,
   * NOT read back from the GPU — reading the real count would need a
   * round-trip the frame path cannot afford. Label it an estimate when shown.
   */
  estimatedAliveCount(): number {
    let alive = 0;
    for (const slot of this.emitters.values()) {
      if (slot.spec.active) alive += expectedAliveCount(slot.spec);
    }
    return Math.round(alive);
  }

  /**
   * Step every emitter. Encodes one compute pass into the caller's encoder, so
   * it orders correctly against a render pass recorded on the same encoder.
   */
  simulate(encoder: GPUCommandEncoder, dt: number): void {
    if (this.emitters.size === 0) return;
    // Clamp dt so a tab-switch stall cannot teleport particles across the map.
    const step = Number.isFinite(dt) ? Math.min(Math.max(dt, 0), 0.1) : 0;
    if (step === 0) return;
    this.elapsed += step;
    this.frame += 1;

    const pass = encoder.beginComputePass({ label: "particles/compute-pass" });
    pass.setPipeline(this.computePipeline);
    for (const slot of this.emitters.values()) {
      packSimParams(this.simScratch, slot.spec, step, this.elapsed, this.frame);
      this.device.queue.writeBuffer(slot.simParamsBuffer, 0, this.simScratch);
      pass.setBindGroup(0, slot.computeBindGroup);
      pass.dispatchWorkgroups(slot.workgroupCount);
    }
    pass.end();
  }

  /**
   * Draw every emitter into an already-open render pass. Call AFTER the opaque
   * geometry draws so the depth test can occlude particles behind the world.
   * `view` supplies the billboard basis; `viewProj` the projection.
   */
  encode(pass: GPURenderPassEncoder, view: Mat4, viewProj: Mat4): void {
    if (this.emitters.size === 0) return;
    const { right, up } = cameraBasisFromView(view);
    let boundBlend: ParticleBlend | null = null;
    // Alpha-blended emitters first, then additive. Additive is
    // order-independent, so this is the only ordering that matters.
    const ordered = [...this.emitters.values()].sort(
      (a, b) => (a.spec.blend === "alpha" ? 0 : 1) - (b.spec.blend === "alpha" ? 0 : 1),
    );
    for (const slot of ordered) {
      packDrawParams(this.drawScratch, viewProj, right, up, slot.spec);
      this.device.queue.writeBuffer(slot.drawParamsBuffer, 0, this.drawScratch);
      if (boundBlend !== slot.spec.blend) {
        pass.setPipeline(this.pipelineFor(slot.spec.blend));
        boundBlend = slot.spec.blend;
      }
      pass.setBindGroup(0, slot.drawBindGroup);
      pass.draw(PARTICLE_VERTS_PER_QUAD, slot.spec.capacity);
    }
  }

  /**
   * Copy an emitter's particle pool back to the CPU. For tests and tooling
   * ONLY: it allocates a staging buffer and awaits `mapAsync`, which stalls
   * until the GPU drains. Never call this on the frame path.
   *
   * Returns one `PARTICLE_STRIDE_FLOATS`-wide record per pool slot; read
   * fields with `PARTICLE_FIELD_OFFSETS`. Resolves null for an unknown id.
   */
  async readPoolForDebug(id: string): Promise<Float32Array | null> {
    const slot = this.emitters.get(id);
    if (!slot) return null;
    const bytes = particleGpuBytes(slot.spec.capacity);
    const staging = this.device.createBuffer({
      label: `particles/debug-readback[${id}]`,
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "particles/debug-copy" });
    encoder.copyBufferToBuffer(slot.particleBuffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      // Copy out before unmapping — the mapped range is detached on unmap.
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    } finally {
      staging.destroy();
    }
  }

  destroy(): void {
    for (const id of [...this.emitters.keys()]) this.detach(id);
    this.drawPipelines.clear();
  }
}

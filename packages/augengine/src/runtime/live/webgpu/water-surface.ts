/**
 * Water surface draw — renders the shallow-water solver's height field.
 *
 * The geometry IS the simulation. The vertex stage reads the same storage
 * buffer the compute passes in `shallow-water.ts` just wrote, so there is no
 * second copy of the surface for the renderer to fall out of step with, and no
 * authored wave function running alongside the solver. If the solver is
 * paused, the surface is still; if a splash propagates, the geometry moves.
 *
 * Normals come from a central difference of the SAME elevation the vertex
 * stage displaces to, computed by WGSL that mirrors `waterSurfaceNormal()`
 * statement for statement. A surface cannot be lit by a slope it does not
 * have — which is the failure mode of shading a height field with a separately
 * authored normal map.
 *
 * ── Shading, and exactly what each term is ───────────────────────────────
 *  - FRESNEL (Schlick, F0 = 0.02 for an air→water interface) blends between
 *    reflection and refraction. This is the cue that makes a surface read as
 *    water: near-transparent looking down, near-mirror at a grazing angle.
 *  - REFRACTION samples a copy of the scene colour taken after the opaque pass
 *    and before this draw, offset along the surface normal. It is a real
 *    screen-space refraction of the actual rendered scene, not a fake.
 *  - ABSORPTION is Beer–Lambert over the path length a view ray travels
 *    through the water column: `exp(-extinction · h / |cos|)`, per channel.
 *    Depth comes from the solver's own `h`, not from a depth buffer, so it is
 *    the true simulated water depth. Deep water darkens and shifts toward
 *    `deepTint`; shallow water stays clear. Monotonic in depth by construction
 *    (`waterTransmittance` has a test pinning it).
 *  - REFLECTION is the analytic sky gradient evaluated along the reflected ray
 *    plus a Blinn-Phong sun highlight whose peak is deliberately > 1 so the
 *    existing bloom-extract pass (threshold 0.9) picks it up as glare.
 *  - FOAM whitens the surface where the solver says the water is shallow and
 *    where it says the flow is fast. Both inputs are simulated quantities.
 *
 * ── NOT implemented here (do not document these as working) ─────────────
 *  - Reflection of scene geometry. The reflection term is sky + sun only.
 *    Nothing standing in the water appears in its reflection. Planar or
 *    screen-space reflection of geometry is not built.
 *  - Caustics. No light is focused through the surface onto the bed.
 *  - Underwater rendering. The camera is assumed to be above the surface;
 *    there is no fog, no colour shift and no surface seen from below (the draw
 *    is two-sided, but the shading is not).
 *  - Wet-sand darkening, shoreline displacement, or any change to the terrain
 *    the water runs over.
 *  - Tessellation or LOD. Every quad of the grid is drawn every frame.
 *
 * ── Pass placement ──────────────────────────────────────────────────────
 * Refraction needs the opaque scene colour as a SAMPLED texture, and a texture
 * cannot be sampled while it is the colour attachment of the open pass. So
 * when a water surface exists the renderer splits its single main pass in two:
 *
 *   [sky → opaque] → copy resolved HDR → [WATER → particles] → post
 *
 * The copy is taken in the same frame, so the refracted background is never a
 * frame stale. The cost is one extra render pass and one full-resolution
 * texture copy per frame, paid only when water is on screen. See
 * `WebGpuLiveRenderer.render`.
 */

import {
  ShallowWaterSim,
  POOL_PRESET,
  WATER_CELL_STRUCT_WGSL,
  WATER_CELL_STRIDE_FLOATS,
  waterCellCount,
  type ShallowWaterOptions,
  type ShallowWaterSpec,
  type ShallowWaterFieldStats,
} from "./shallow-water.js";
import type { Mat4, Vec3 } from "../../../scene/types.js";

// ──────────────────────────────────────────────────────────────────────
// Look description
// ──────────────────────────────────────────────────────────────────────

export interface WaterLook {
  /** Linear RGB filter applied to light that has passed through the water. */
  shallowTint: Vec3;
  /** Linear RGB the water in-scatters toward as the path length grows. */
  deepTint: Vec3;
  /** Per-channel extinction coefficient, 1/m. Red extinguishes fastest in water. */
  extinctionPerM: Vec3;
  /** Schlick F0. 0.02 is the physical air→water value; higher looks metallic. */
  fresnelF0: number;
  /** Blinn-Phong exponent of the sun highlight. Higher = tighter glint. */
  specularPower: number;
  /** HDR multiplier on the sun highlight. > 1 drives the existing bloom pass. */
  specularIntensity: number;
  /** Screen-space refraction offset in UV units at 1 m of water depth. */
  refractionStrength: number;
  /** Depth over which the surface fades out at the shoreline, metres. */
  shorelineFadeM: number;
  /** Overall foam whiteness, 0..1. */
  foamStrength: number;
  /** Flow speed at which flow-driven foam saturates, m/s. */
  flowFoamSpeed: number;
  /** Floor on |cos| in the path-length divisor; stops a horizon-grazing view
   *  producing an unbounded path and a black band. */
  minViewCosine: number;
  /** Linear RGB at the top of the reflected sky gradient. */
  skyZenith: Vec3;
  /** Linear RGB at the horizon of the reflected sky gradient. */
  skyHorizon: Vec3;
}

export type WaterSurfaceOptions = ShallowWaterOptions & Partial<WaterLook>;

/**
 * Open-water look: blue-green, strongly absorbing in red, hot specular.
 *
 * Extinction is in the right ballpark for coastal sea water (red absorbs an
 * order of magnitude faster than blue) but it is NOT fitted to measured
 * spectral absorption data. It is art direction with the correct sign.
 */
export const OCEAN_WATER_LOOK: Readonly<WaterLook> = Object.freeze({
  shallowTint: [0.72, 0.92, 0.95] as Vec3,
  deepTint: [0.014, 0.075, 0.115] as Vec3,
  extinctionPerM: [0.75, 0.22, 0.12] as Vec3,
  fresnelF0: 0.02,
  specularPower: 220,
  specularIntensity: 6,
  refractionStrength: 0.035,
  shorelineFadeM: 0.14,
  foamStrength: 0.65,
  flowFoamSpeed: 1.6,
  minViewCosine: 0.08,
  skyZenith: [0.09, 0.2, 0.44] as Vec3,
  skyHorizon: [0.42, 0.52, 0.62] as Vec3,
});

/** Clear inland water: far less absorbing, so the bed stays visible. */
export const POOL_WATER_LOOK: Readonly<WaterLook> = Object.freeze({
  ...OCEAN_WATER_LOOK,
  shallowTint: [0.86, 0.96, 0.98] as Vec3,
  deepTint: [0.05, 0.19, 0.22] as Vec3,
  extinctionPerM: [0.3, 0.1, 0.06] as Vec3,
  foamStrength: 0.4,
});

// ──────────────────────────────────────────────────────────────────────
// Layout constants
// ──────────────────────────────────────────────────────────────────────

/** Bytes of the per-surface draw uniform. mat4 + 9 × vec4. */
export const WATER_DRAW_PARAMS_BYTES = 208;

/** Vertices per grid quad (two triangles, no vertex or index buffer). */
export const WATER_VERTS_PER_QUAD = 6;

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
 * Merge look options over a preset and clamp every field. Non-finite input
 * falls back rather than reaching the shader and producing NaN pixels.
 */
export function resolveWaterLook(
  options: Partial<WaterLook> = {},
  base: Readonly<WaterLook> = OCEAN_WATER_LOOK,
): WaterLook {
  return {
    shallowTint: resolveVec3(options.shallowTint, base.shallowTint),
    deepTint: resolveVec3(options.deepTint, base.deepTint),
    extinctionPerM: resolveVec3(options.extinctionPerM, base.extinctionPerM),
    fresnelF0: clampNumber(options.fresnelF0 ?? base.fresnelF0, 0, 1, base.fresnelF0),
    specularPower: clampNumber(options.specularPower ?? base.specularPower, 1, 4096, base.specularPower),
    specularIntensity: clampNumber(options.specularIntensity ?? base.specularIntensity, 0, 1e4, base.specularIntensity),
    refractionStrength: clampNumber(options.refractionStrength ?? base.refractionStrength, 0, 0.5, base.refractionStrength),
    shorelineFadeM: clampNumber(options.shorelineFadeM ?? base.shorelineFadeM, 1e-4, 100, base.shorelineFadeM),
    foamStrength: clampNumber(options.foamStrength ?? base.foamStrength, 0, 1, base.foamStrength),
    flowFoamSpeed: clampNumber(options.flowFoamSpeed ?? base.flowFoamSpeed, 1e-3, 1e3, base.flowFoamSpeed),
    minViewCosine: clampNumber(options.minViewCosine ?? base.minViewCosine, 1e-3, 1, base.minViewCosine),
    skyZenith: resolveVec3(options.skyZenith, base.skyZenith),
    skyHorizon: resolveVec3(options.skyHorizon, base.skyHorizon),
  };
}

/**
 * Quads the surface draws for a grid — one instance each. A grid needs at
 * least two cells on both axes to have a single quad, so a 1-D field (height
 * 1) simulates but draws nothing, which is the honest result rather than a
 * degenerate strip.
 */
export function waterQuadCount(spec: Pick<ShallowWaterSpec, "width" | "height">): number {
  return Math.max(0, (spec.width - 1) * Math.max(0, spec.height - 1));
}

/**
 * Pack the draw uniform. `out` must hold `WATER_DRAW_PARAMS_BYTES / 4` floats.
 * Mirrored by the WGSL `WaterDrawParams` struct:
 *   0..15  viewProj
 *   16..18 cameraPos        · 19 alignment padding (asserted zero)
 *   20..22 lightDirection   · 23 specularPower
 *   24..26 shallowTint      · 27 foamStrength
 *   28..30 deepTint         · 31 refractionStrength
 *   32..34 extinctionPerM   · 35 fresnelF0
 *   36..38 skyZenith        · 39 specularIntensity
 *   40..42 skyHorizon       · 43 shorelineFadeM
 *   44 gridOriginX · 45 gridOriginZ · 46 cellSize · 47 gridWidth
 *   48 gridHeight · 49 flowFoamSpeed · 50 minViewCosine · 51 padding
 *
 * Every slot but 19 and 51 is read by the shader. The two padding slots exist
 * only because vec4 alignment demands them; the tests assert they stay zero so
 * a future parameter cannot be smuggled in here and then applied nowhere.
 */
export function packWaterDrawParams(
  out: Float32Array,
  viewProj: Mat4,
  cameraPos: Vec3,
  lightDir: Vec3,
  spec: ShallowWaterSpec,
  look: WaterLook,
): Float32Array {
  const needed = WATER_DRAW_PARAMS_BYTES / 4;
  if (out.length < needed) {
    throw new Error(`packWaterDrawParams needs ${needed} floats, got ${out.length}`);
  }
  out.fill(0);
  out.set(viewProj as unknown as ArrayLike<number>, 0);
  out.set(cameraPos, 16);
  out.set(lightDir, 20);
  out[23] = look.specularPower;
  out.set(look.shallowTint, 24);
  out[27] = look.foamStrength;
  out.set(look.deepTint, 28);
  out[31] = look.refractionStrength;
  out.set(look.extinctionPerM, 32);
  out[35] = look.fresnelF0;
  out.set(look.skyZenith, 36);
  out[39] = look.specularIntensity;
  out.set(look.skyHorizon, 40);
  out[43] = look.shorelineFadeM;
  out[44] = spec.origin[0];
  out[45] = spec.origin[1];
  out[46] = spec.cellSize;
  out[47] = spec.width;
  out[48] = spec.height;
  out[49] = look.flowFoamSpeed;
  out[50] = look.minViewCosine;
  return out;
}

/** Premultiplied source-over — matches the shader's `vec4(rgb * a, a)` output. */
export function waterBlendState(): GPUBlendState {
  return {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
}

// ──────────────────────────────────────────────────────────────────────
// WGSL
// ──────────────────────────────────────────────────────────────────────

export const WATER_SURFACE_WGSL = /* wgsl */ `
${WATER_CELL_STRUCT_WGSL}
struct WaterDrawParams {
  viewProj: mat4x4<f32>,
  // xyz = camera world position, w = alignment padding
  cameraPos: vec4<f32>,
  // xyz = direction the light travels, w = specular power
  lightDirection: vec4<f32>,
  // xyz = shallow tint, w = foam strength
  shallowTint: vec4<f32>,
  // xyz = deep tint, w = refraction strength
  deepTint: vec4<f32>,
  // xyz = per-channel extinction (1/m), w = Fresnel F0
  extinction: vec4<f32>,
  // xyz = sky zenith colour, w = specular intensity
  skyZenith: vec4<f32>,
  // xyz = sky horizon colour, w = shoreline fade depth (m)
  skyHorizon: vec4<f32>,
  // x = grid origin X, y = grid origin Z, z = cell size, w = grid width
  grid: vec4<f32>,
  // x = grid height, y = flow foam speed, z = min view cosine, w = padding
  extra: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> cells: array<WaterCell>;
@group(0) @binding(1) var<uniform> params: WaterDrawParams;

@group(1) @binding(0) var sceneColor: texture_2d<f32>;
@group(1) @binding(1) var sceneSampler: sampler;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  // The clip position again, as a varying. Interpolating it and dividing by w
  // in the fragment stage recovers a perspective-correct screen UV without
  // needing the viewport size as a uniform.
  @location(0) screen: vec4<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) depth: f32,
  @location(4) speed: f32,
}

// Surface elevation at a clamped grid coordinate.
fn eta_at(i: i32, j: i32, gridWidth: i32, gridHeight: i32) -> f32 {
  let ci = clamp(i, 0, gridWidth - 1);
  let cj = clamp(j, 0, gridHeight - 1);
  let cell = cells[u32(cj * gridWidth + ci)];
  return cell.bed + cell.height;
}

// Mirrors waterSurfaceNormal() in shallow-water.ts, including the one-sided
// fallback at the grid edges. The surface is eta(x, z), whose tangents are
// (1, d/dx, 0) and (0, d/dz, 1); their cross product is (-d/dx, 1, -d/dz).
fn surface_normal(i: i32, j: i32, gridWidth: i32, gridHeight: i32, cellSize: f32) -> vec3<f32> {
  let iLo = max(i - 1, 0);
  let iHi = min(i + 1, gridWidth - 1);
  let jLo = max(j - 1, 0);
  let jHi = min(j + 1, gridHeight - 1);
  let spanX = f32(iHi - iLo) * cellSize;
  let spanZ = f32(jHi - jLo) * cellSize;
  var slopeX = 0.0;
  if (spanX > 0.0) {
    slopeX = (eta_at(iHi, j, gridWidth, gridHeight) - eta_at(iLo, j, gridWidth, gridHeight)) / spanX;
  }
  var slopeZ = 0.0;
  if (spanZ > 0.0) {
    slopeZ = (eta_at(i, jHi, gridWidth, gridHeight) - eta_at(i, jLo, gridWidth, gridHeight)) / spanZ;
  }
  return normalize(vec3<f32>(-slopeX, 1.0, -slopeZ));
}

fn fresnel_schlick(cosTheta: f32, f0: f32) -> f32 {
  let c = clamp(cosTheta, 0.0, 1.0);
  let m = 1.0 - c;
  return f0 + (1.0 - f0) * m * m * m * m * m;
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VsOut {
  let gridWidth = i32(params.grid.w);
  let gridHeight = i32(params.extra.x);
  let cellSize = params.grid.z;
  let quadsX = max(gridWidth - 1, 1);

  // Two triangles per grid quad. Winding is irrelevant: the pipeline draws
  // with culling off so the surface is visible from either side.
  var corners = array<vec2<i32>, 6>(
    vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1),
    vec2<i32>(0, 1), vec2<i32>(1, 0), vec2<i32>(1, 1),
  );
  let corner = corners[vi];
  let quadX = i32(ii) % quadsX;
  let quadZ = i32(ii) / quadsX;
  let i = clamp(quadX + corner.x, 0, gridWidth - 1);
  let j = clamp(quadZ + corner.y, 0, gridHeight - 1);
  let cell = cells[u32(j * gridWidth + i)];

  let worldPos = vec3<f32>(
    params.grid.x + f32(i) * cellSize,
    cell.bed + cell.height,
    params.grid.y + f32(j) * cellSize,
  );

  var out: VsOut;
  out.clip = params.viewProj * vec4<f32>(worldPos, 1.0);
  out.screen = out.clip;
  out.worldPos = worldPos;
  out.normal = surface_normal(i, j, gridWidth, gridHeight, cellSize);
  out.depth = cell.height;
  out.speed = length(vec2<f32>(cell.velocityX, cell.velocityZ));
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let normal = normalize(in.normal);
  let viewDir = normalize(params.cameraPos.xyz - in.worldPos);
  let cosTheta = max(dot(normal, viewDir), 0.0);
  let reflectance = fresnel_schlick(cosTheta, params.extinction.w);

  // ── Reflection: analytic sky gradient along the reflected ray, plus a
  // Blinn-Phong sun glint whose peak exceeds 1.0 so bloom catches it.
  let reflectDir = reflect(-viewDir, normal);
  let skyMix = pow(clamp(reflectDir.y, 0.0, 1.0), 0.6);
  let skyRgb = mix(params.skyHorizon.xyz, params.skyZenith.xyz, skyMix);
  let toLight = normalize(-params.lightDirection.xyz);
  let halfVector = normalize(toLight + viewDir);
  let glint = pow(max(dot(normal, halfVector), 0.0), params.lightDirection.w) * params.skyZenith.w;
  let reflected = skyRgb + vec3<f32>(glint);

  // ── Refraction: the scene colour copied after the opaque pass, offset
  // along the surface normal. Thicker water bends more, so the offset scales
  // with the simulated depth.
  let ndc = in.screen.xy / in.screen.w;
  let baseUv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let bend = normal.xz * params.deepTint.w * min(in.depth, 1.0);
  let uv = clamp(baseUv + bend, vec2<f32>(0.001, 0.001), vec2<f32>(0.999, 0.999));
  let behind = textureSample(sceneColor, sceneSampler, uv).rgb;

  // ── Absorption: Beer-Lambert over the path a view ray takes through the
  // column. Depth is the solver's own h, not a depth-buffer reconstruction.
  let pathLength = in.depth / max(abs(viewDir.y), params.extra.z);
  let transmittance = exp(-params.extinction.xyz * pathLength);
  let transmitted =
    behind * params.shallowTint.xyz * transmittance +
    params.deepTint.xyz * (vec3<f32>(1.0, 1.0, 1.0) - transmittance);

  var rgb = mix(transmitted, reflected, reflectance);

  // ── Foam from two simulated quantities: shallow water and fast flow.
  let shoreFoam = 1.0 - smoothstep(0.0, params.skyHorizon.w, in.depth);
  let flowFoam = smoothstep(0.0, params.extra.y, in.speed);
  let foam = clamp((shoreFoam + flowFoam) * params.shallowTint.w, 0.0, 1.0);
  rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), foam);

  // Dry cells vanish instead of drawing a zero-thickness sheet over the bed.
  let coverage = smoothstep(0.0, params.skyHorizon.w * 0.5, in.depth);
  // Premultiplied output — matches waterBlendState().
  return vec4<f32>(rgb * coverage, coverage);
}
`;

// ──────────────────────────────────────────────────────────────────────
// Pass
// ──────────────────────────────────────────────────────────────────────

export interface WaterSurfacePassOptions {
  /** Colour target format. Must match the main render pass (renderer: rgba16float). */
  targetFormat: GPUTextureFormat;
  /** MSAA sample count. Must match the main render pass. */
  sampleCount: 1 | 4;
  /** Depth attachment format of the main render pass. */
  depthFormat?: GPUTextureFormat;
}

export interface WaterSurfaceStats {
  /** Water surfaces attached. */
  surfaces: number;
  /** Cell-buffer bytes across every field. */
  gpuBytes: number;
  /** Grid quads drawn per frame, summed over surfaces. */
  quads: number;
  /** Simulation cells, summed over surfaces. */
  cells: number;
  /**
   * True once a real scene-colour copy has been supplied. While false the
   * refraction term samples a 1×1 black placeholder, so refraction shows as
   * black rather than as scene colour. Not a claim that refraction looks right
   * — only that the binding is live.
   */
  sceneColorBound: boolean;
}

interface SurfaceSlot {
  id: string;
  spec: ShallowWaterSpec;
  look: WaterLook;
  drawParamsBuffer: GPUBuffer;
  drawBindGroup: GPUBindGroup;
  /** The cell buffer the bind group currently points at. Compared by identity
   *  on `update` so a field reallocation can never leave the draw bound to a
   *  destroyed buffer. */
  cellBuffer: GPUBuffer;
  quadCount: number;
}

/**
 * Owns the surface render pipeline, one draw uniform per surface, and the
 * `ShallowWaterSim` that produces the geometry. The live renderer drives it.
 */
export class WaterSurfacePass {
  private readonly device: GPUDevice;
  private readonly sim: ShallowWaterSim;
  private readonly surfaceLayout: GPUBindGroupLayout;
  private readonly sceneLayout: GPUBindGroupLayout;
  private readonly pipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly placeholderTexture: GPUTexture;
  private sceneBindGroup: GPUBindGroup;
  private sceneColorBound = false;
  private readonly surfaces = new Map<string, SurfaceSlot>();
  private readonly drawScratch = new Float32Array(WATER_DRAW_PARAMS_BYTES / 4);

  constructor(device: GPUDevice, options: WaterSurfacePassOptions) {
    this.device = device;
    this.sim = new ShallowWaterSim(device);

    this.surfaceLayout = device.createBindGroupLayout({
      label: "water/surface-layout",
      entries: [
        // Vertex-stage storage reads must be read-only; WebGPU forbids
        // read_write storage outside compute.
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });
    this.sceneLayout = device.createBindGroupLayout({
      label: "water/scene-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    this.sampler = device.createSampler({
      label: "water/scene-sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    // A 1×1 stand-in so the bind group is always valid, even before the
    // renderer has a scene-colour copy to hand over. WebGPU zero-initialises
    // textures, so refraction samples black until `setSceneColor` is called.
    this.placeholderTexture = device.createTexture({
      label: "water/scene-color-placeholder",
      size: [1, 1],
      format: options.targetFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneBindGroup = this.makeSceneBindGroup(this.placeholderTexture);

    const module = device.createShaderModule({
      label: "water/surface-shader",
      code: WATER_SURFACE_WGSL,
    });
    this.pipeline = device.createRenderPipeline({
      label: "water/surface",
      layout: device.createPipelineLayout({
        label: "water/surface-pipeline-layout",
        bindGroupLayouts: [this.surfaceLayout, this.sceneLayout],
      }),
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: options.targetFormat, blend: waterBlendState() }],
      },
      // Two-sided: a camera that dips below the waterline still sees the
      // surface rather than looking through a one-way sheet.
      primitive: { topology: "triangle-list", cullMode: "none" },
      // Water DOES write depth. It is a surface, not a volumetric effect, and
      // anything encoded after it (the particle pass) must be occluded by it
      // when it is behind the water.
      depthStencil: {
        format: options.depthFormat ?? "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: { count: options.sampleCount },
    });
  }

  private makeSceneBindGroup(texture: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      label: "water/scene-bg",
      layout: this.sceneLayout,
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  /**
   * Hand over the texture holding the opaque scene colour for this frame.
   * Called by the renderer whenever its targets are (re)allocated, not per
   * frame — the bind group is rebuilt here, which allocates.
   */
  setSceneColor(texture: GPUTexture | null): void {
    this.sceneBindGroup = this.makeSceneBindGroup(texture ?? this.placeholderTexture);
    this.sceneColorBound = texture !== null;
  }

  /**
   * Create or replace a water surface: one shallow-water field plus its draw.
   * Re-attaching the same id replaces it, so this is idempotent.
   */
  attach(
    id: string,
    options: WaterSurfaceOptions = {},
    base: Readonly<ShallowWaterSpec> = POOL_PRESET,
    lookBase: Readonly<WaterLook> = OCEAN_WATER_LOOK,
  ): { spec: ShallowWaterSpec; look: WaterLook } {
    this.detach(id);
    const spec = this.sim.attach(id, options, base);
    const look = resolveWaterLook(options, lookBase);
    const cellBuffer = this.sim.cellBufferFor(id);
    if (!cellBuffer) throw new Error(`water surface ${id}: field allocation failed`);
    const drawParamsBuffer = this.device.createBuffer({
      label: `water/draw-params[${id}]`,
      size: WATER_DRAW_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.surfaces.set(id, {
      id,
      spec,
      look,
      drawParamsBuffer,
      drawBindGroup: this.device.createBindGroup({
        label: `water/draw-bg[${id}]`,
        layout: this.surfaceLayout,
        entries: [
          { binding: 0, resource: { buffer: cellBuffer } },
          { binding: 1, resource: { buffer: drawParamsBuffer } },
        ],
      }),
      cellBuffer,
      quadCount: waterQuadCount(spec),
    });
    return { spec, look };
  }

  /**
   * Patch solver and/or look knobs. A grid-geometry change restarts that
   * field, which reallocates its cell buffer, so the bind group is rebuilt.
   */
  update(id: string, patch: WaterSurfaceOptions): { spec: ShallowWaterSpec; look: WaterLook } | null {
    const slot = this.surfaces.get(id);
    if (!slot) return null;
    const look = resolveWaterLook({ ...slot.look, ...patch }, slot.look);
    const spec = this.sim.update(id, patch);
    if (!spec) return null;
    const cellBuffer = this.sim.cellBufferFor(id);
    if (!cellBuffer) return null;
    slot.spec = spec;
    slot.look = look;
    slot.quadCount = waterQuadCount(spec);
    // Identity, not a field-by-field guess at which knobs reallocate: the sim
    // decides that, and a stale bind group would point at a destroyed buffer.
    if (cellBuffer !== slot.cellBuffer) {
      slot.cellBuffer = cellBuffer;
      slot.drawBindGroup = this.device.createBindGroup({
        label: `water/draw-bg[${id}]`,
        layout: this.surfaceLayout,
        entries: [
          { binding: 0, resource: { buffer: cellBuffer } },
          { binding: 1, resource: { buffer: slot.drawParamsBuffer } },
        ],
      });
    }
    return { spec, look };
  }

  getSpec(id: string): ShallowWaterSpec | null {
    return this.surfaces.get(id)?.spec ?? null;
  }

  getLook(id: string): WaterLook | null {
    return this.surfaces.get(id)?.look ?? null;
  }

  getFieldStats(id: string): ShallowWaterFieldStats | null {
    return this.sim.getStats(id);
  }

  has(id: string): boolean {
    return this.surfaces.has(id);
  }

  get surfaceCount(): number {
    return this.surfaces.size;
  }

  ids(): string[] {
    return [...this.surfaces.keys()];
  }

  /** Queue a splash into a field. Applied on the next `simulate()`. */
  splash(id: string, worldX: number, worldZ: number, radius: number, amplitude: number): boolean {
    return this.sim.splash(id, worldX, worldZ, radius, amplitude);
  }

  /** Step every field. Encodes one compute pass into the caller's encoder. */
  simulate(encoder: GPUCommandEncoder, dt: number): void {
    this.sim.simulate(encoder, dt);
  }

  /**
   * Draw every surface into an already-open render pass. Must run in a pass
   * whose colour attachment is NOT the texture handed to `setSceneColor`.
   */
  encode(pass: GPURenderPassEncoder, viewProj: Mat4, cameraPos: Vec3, lightDir: Vec3): void {
    if (this.surfaces.size === 0) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(1, this.sceneBindGroup);
    for (const slot of this.surfaces.values()) {
      if (slot.quadCount === 0) continue;
      packWaterDrawParams(this.drawScratch, viewProj, cameraPos, lightDir, slot.spec, slot.look);
      this.device.queue.writeBuffer(slot.drawParamsBuffer, 0, this.drawScratch);
      pass.setBindGroup(0, slot.drawBindGroup);
      pass.draw(WATER_VERTS_PER_QUAD, slot.quadCount);
    }
  }

  /** Read a field's cells back. Tests and tooling only — it stalls the GPU. */
  readFieldForDebug(id: string): Promise<Float32Array | null> {
    return this.sim.readFieldForDebug(id);
  }

  getStats(): WaterSurfaceStats {
    let quads = 0;
    let cells = 0;
    for (const slot of this.surfaces.values()) {
      quads += slot.quadCount;
      cells += waterCellCount(slot.spec);
    }
    return {
      surfaces: this.surfaces.size,
      gpuBytes: this.sim.gpuBytes(),
      quads,
      cells,
      sceneColorBound: this.sceneColorBound,
    };
  }

  /** Release a surface's GPU resources. Safe on an unknown id. */
  detach(id: string): void {
    const slot = this.surfaces.get(id);
    if (!slot) return;
    slot.drawParamsBuffer.destroy();
    this.surfaces.delete(id);
    this.sim.detach(id);
  }

  destroy(): void {
    for (const id of [...this.surfaces.keys()]) this.detach(id);
    this.sim.destroy();
    this.placeholderTexture.destroy();
  }
}

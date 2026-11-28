/**
 * Post-processing pipeline — Phase 9.
 *
 * A `PostPipeline` is an ordered list of `PostPass` instances. Each pass
 * reads from an input texture and writes to an output (typically a
 * ping-pong pair). The renderer's main pass writes to `pipeline.input`;
 * the pipeline drives passes in order; the final pass writes to the
 * canvas color attachment.
 *
 * Phase 9 ships the architectural skeleton + WGSL source for the
 * three foundational passes: bloom (extract → blur → composite), ACES
 * tonemap, and FXAA. Concrete WebGPU pipeline wiring is fully implemented
 * here.
 */

import { PhysicalBloomChain } from "./bloom-chain.js";

export interface PostPassDescriptor {
  /** Stable name used for telemetry + debugging. */
  name: string;
  /** WGSL shader source. */
  shader: string;
  /** Uniform layout — caller writes these per-frame. */
  uniforms?: PostUniformLayout;
}

export interface PostUniformLayout {
  /** Total uniform-buffer size in bytes (16-byte aligned). */
  size: number;
  /** Per-uniform-name → byte offset. */
  fields: Record<string, { offset: number; type: "f32" | "vec2" | "vec4" }>;
}

export interface PostPassInstance {
  descriptor: PostPassDescriptor;
  setUniform(name: string, value: number | [number, number] | [number, number, number, number]): void;
  /** Encode this pass's render commands into the supplied encoder. */
  encode(encoder: unknown, inputView: unknown, outputView: unknown): void;
}

export interface PostPipeline {
  passes: PostPassInstance[];
  /** Write the renderer's color output to `inputView`; pipeline composites onto canvas. */
  encode(encoder: unknown, inputView: unknown, canvasView: unknown): void;
}

// ──────────────────────────────────────────────────────────────────────
// Stock pass descriptors — WGSL only.
// ──────────────────────────────────────────────────────────────────────

export const BLOOM_EXTRACT_PASS: PostPassDescriptor = {
  name: "bloom-extract",
  shader: /* wgsl */ `
// outTexel = 1 / output size. The bloom chain renders at half resolution, so
// the UV must come from the OUTPUT pixel grid, not the input's. Sampling the
// full-res input at a half-res pixel centre lands on the corner of a 2×2
// block, so the bilinear tap is a free 2×2 box downsample.
struct Uniforms { threshold: f32, knee: f32, outTexel: vec2<f32>, };
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var t_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy * u.outTexel;
  let c = textureSample(t_input, t_sampler, uv);
  let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  // Soft-knee threshold curve (Karis 2013).
  let soft = clamp(lum - u.threshold + u.knee, 0.0, 2.0 * u.knee);
  let factor = max(soft * soft / (4.0 * u.knee + 1e-5), max(lum - u.threshold, 0.0));
  return vec4<f32>(c.rgb * (factor / max(lum, 1e-5)), 1.0);
}
`,
  uniforms: {
    size: 16,
    fields: {
      threshold: { offset: 0, type: "f32" },
      knee: { offset: 4, type: "f32" },
      outTexel: { offset: 8, type: "vec2" },
    },
  },
};

export const BLOOM_BLUR_PASS: PostPassDescriptor = {
  name: "bloom-blur",
  shader: /* wgsl */ `
struct Uniforms { texelSize: vec2<f32>, direction: vec2<f32>, };
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var t_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// 9-tap separable Gaussian (Karis weights). Run twice — once horizontal,
// once vertical — for a 17-tap effective radius.
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy * u.texelSize;
  let off1 = u.direction * 1.3846153846 * u.texelSize;
  let off2 = u.direction * 3.2307692308 * u.texelSize;
  var c = textureSample(t_input, t_sampler, uv).rgb * 0.2270270270;
  c = c + textureSample(t_input, t_sampler, uv + off1).rgb * 0.3162162162;
  c = c + textureSample(t_input, t_sampler, uv - off1).rgb * 0.3162162162;
  c = c + textureSample(t_input, t_sampler, uv + off2).rgb * 0.0702702703;
  c = c + textureSample(t_input, t_sampler, uv - off2).rgb * 0.0702702703;
  return vec4<f32>(c, 1.0);
}
`,
  uniforms: {
    size: 16,
    fields: {
      texelSize: { offset: 0, type: "vec2" },
      direction: { offset: 8, type: "vec2" },
    },
  },
};

export const BLOOM_COMPOSITE_PASS: PostPassDescriptor = {
  name: "bloom-composite",
  shader: /* wgsl */ `
struct Uniforms { intensity: f32, _pad: vec4<f32>, };
@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var t_sampler: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / vec2<f32>(textureDimensions(t_scene));
  let scene = textureSample(t_scene, t_sampler, uv).rgb;
  let bloom = textureSample(t_bloom, t_sampler, uv).rgb;
  return vec4<f32>(scene + bloom * u.intensity, 1.0);
}
`,
  uniforms: {
    size: 16,
    fields: { intensity: { offset: 0, type: "f32" } },
  },
};

export const ACES_TONEMAP_PASS: PostPassDescriptor = {
  name: "aces-tonemap",
  shader: /* wgsl */ `
struct Uniforms { exposure: f32, saturation: f32, _pad: vec2<f32>, };
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var t_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// ACES filmic tonemap (Narkowicz 2015 fit). Standard for engine-level HDR→SDR.
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Scalar ACES — tonemap a single channel (the brightest), used to drive the
// hue/saturation-preserving path below.
fn aces_scalar(x: f32) -> f32 {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / vec2<f32>(textureDimensions(t_input));
  let hdr = textureSample(t_input, t_sampler, uv).rgb * u.exposure;
  // Per-channel ACES rolls bright neon toward white (each channel clamps to 1
  // independently, so a saturated cyan/magenta loses its chroma in the highs).
  let perChannel = aces_filmic(hdr);
  // Hue/saturation-preserving ACES: tonemap only the brightest channel and
  // scale the original color by that ratio, so the chroma ratio (the neon hue)
  // survives the HDR→SDR compression instead of bleaching to white.
  let peak = max(hdr.r, max(hdr.g, hdr.b));
  let tmPeak = aces_scalar(peak);
  let preserve = select(vec3<f32>(0.0), hdr * (tmPeak / max(peak, 1e-5)), peak > 1e-5);
  // Blend between the two: u.saturation=0 = filmic neutral, 1 = full neon punch.
  let sdr = clamp(mix(perChannel, preserve, u.saturation), vec3<f32>(0.0), vec3<f32>(1.0));
  // Linear -> sRGB gamma. WebGPU canvases are sRGB-aware; if the canvas is
  // configured with format = rgba8unorm-srgb, skip this. We default to
  // applying gamma 2.2 to be safe.
  let srgb = pow(sdr, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb, 1.0);
}
`,
  uniforms: {
    size: 16,
    fields: {
      exposure: { offset: 0, type: "f32" },
      saturation: { offset: 4, type: "f32" },
    },
  },
};

/**
 * Tonemappers + display grade, shared by the composite pass and the fused
 * TAA resolve (temporal.ts). Input: exposed linear HDR. Output: graded,
 * gamma-encoded display colour in [0, 1].
 */
export const TONEMAP_WGSL = /* wgsl */ `
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn aces_scalar(x: f32) -> f32 {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
// Khronos PBR Neutral (glTF reference tonemapper): exact colour below 0.76,
// a soft shoulder above it, mild desaturation of the brightest values.
fn pbr_neutral(c: vec3<f32>) -> vec3<f32> {
  let start = 0.76;
  let desat = 0.15;
  let x = min(c.r, min(c.g, c.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  var col = c - offset;
  let peak = max(col.r, max(col.g, col.b));
  if (peak < start) { return col; }
  let d = 1.0 - start;
  let newPeak = 1.0 - d * d / (peak + d - start);
  col = col * (newPeak / peak);
  let g = 1.0 - 1.0 / (desat * (peak - newPeak) + 1.0);
  return mix(col, vec3<f32>(newPeak), g);
}
// AgX (minimal fit, B. Wrensch 2023): log2 encode, sigmoid, back to linear.
fn agx_curve(x: vec3<f32>) -> vec3<f32> {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(c: vec3<f32>) -> vec3<f32> {
  let inset = mat3x3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                          0.0784335999999992, 0.878468636469772, 0.0784336,
                          0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let outset = mat3x3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                           -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                           -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var v = inset * c;
  v = clamp(log2(max(v, vec3<f32>(1e-10))), vec3<f32>(minEv), vec3<f32>(maxEv));
  v = agx_curve((v - minEv) / (maxEv - minEv));
  v = outset * v;
  // The curve lands in ~gamma-2.2 display space; return linear like the others.
  return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

// mode: 0 = ACES (hue-preserve mixed by saturation), 1 = Khronos PBR Neutral, 2 = AgX.
fn tonemap_display(hdr: vec3<f32>, mode: f32, saturation: f32, contrast: f32, vibrance: f32, lift: f32) -> vec3<f32> {
  var sdr: vec3<f32>;
  if (mode > 1.5) {
    sdr = agx(hdr);
  } else if (mode > 0.5) {
    sdr = clamp(pbr_neutral(hdr), vec3<f32>(0.0), vec3<f32>(1.0));
  } else {
    let perChannel = aces_filmic(hdr);
    let peak = max(hdr.r, max(hdr.g, hdr.b));
    let preserve = select(vec3<f32>(0.0), hdr * (aces_scalar(peak) / max(peak, 1e-5)), peak > 1e-5);
    sdr = clamp(mix(perChannel, preserve, saturation), vec3<f32>(0.0), vec3<f32>(1.0));
  }
  var outc = pow(sdr, vec3<f32>(1.0 / 2.2));
  // Display grade in gamma space: vibrance (saturation about luma), contrast
  // about mid-grey, then lift. Identity at (1, 1, 0).
  let luma = dot(outc, vec3<f32>(0.2126, 0.7152, 0.0722));
  outc = mix(vec3<f32>(luma), outc, vibrance);
  outc = (outc - 0.5) * contrast + 0.5;
  outc = outc + lift * (vec3<f32>(1.0) - outc);
  return clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0));
}
`;

/**
 * Camera effects shared by the composite and the fused TAA resolve:
 * procedural lens dirt (soft smudges that modulate bloom; off by default)
 * and film grain (display-space, luminance-weighted; off by default).
 */
export const CAMERA_FX_WGSL = /* wgsl */ `
fn fx_hash(p: vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(0.1031, 0.1030));
  let r = q + dot(q, q.yx + 33.33);
  return fract((r.x + r.y) * r.x);
}
// Lens dirt: a fixed field of soft blotches (value noise at two scales).
fn fx_lens_dirt(uv: vec2<f32>) -> f32 {
  var d = 0.0;
  for (var o = 0; o < 2; o = o + 1) {
    let s = select(7.0, 19.0, o == 1);
    let g = uv * vec2<f32>(s * 1.6, s);
    let i = floor(g);
    let f = fract(g);
    let w = f * f * (3.0 - 2.0 * f);
    let n = mix(mix(fx_hash(i), fx_hash(i + vec2<f32>(1.0, 0.0)), w.x), mix(fx_hash(i + vec2<f32>(0.0, 1.0)), fx_hash(i + vec2<f32>(1.0, 1.0)), w.x), w.y);
    d = d + smoothstep(0.62, 0.95, n) * select(1.0, 0.5, o == 1);
  }
  return d;
}
// Film grain in display space, strongest in the mid-tones.
fn fx_grain(c: vec3<f32>, frag: vec2<f32>, frame: f32, amount: f32) -> vec3<f32> {
  let n = fx_hash(frag + vec2<f32>(frame * 13.7, frame * 7.3)) - 0.5;
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return clamp(c + vec3<f32>(n * amount * 4.0 * l * (1.0 - l)), vec3<f32>(0.0), vec3<f32>(1.0));
}
`;

/**
 * Bloom composite + ACES tonemap fused into one full-screen pass. Same math as
 * BLOOM_COMPOSITE_PASS followed by ACES_TONEMAP_PASS, minus one full-resolution
 * rgba16float write and read — on a bandwidth-starved integrated GPU that
 * round trip is a measurable slice of the frame. The live runner uses this;
 * the two separate descriptors stay exported for custom chains.
 */
export const COMPOSITE_TONEMAP_PASS: PostPassDescriptor = {
  name: "composite-tonemap",
  shader: /* wgsl */ `
// contrast / vibrance / lift: parametric display grade (1, 1, 0 = no grade).
// outTexel = 1 / output size (the scene may be smaller than the output).
// bloomMode 1 = physical (energy-conserving mix of the summed mip chain,
// divided by its level count); 0 = additive threshold bloom.
struct Uniforms { intensity: f32, exposure: f32, saturation: f32, mode: f32, contrast: f32, vibrance: f32, lift: f32, bloomMode: f32, outTexel: vec2<f32>, grain: f32, frame: f32, dirt: f32, levels: f32, _p: vec2<f32>, };
@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var t_bloom: texture_2d<f32>;
@group(0) @binding(2) var t_sampler: sampler;
@group(0) @binding(3) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}
${TONEMAP_WGSL}
${CAMERA_FX_WGSL}
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy * u.outTexel;
  let scene = textureSample(t_scene, t_sampler, uv).rgb;
  // intensity is a uniform, so the bloom fetch is uniform control flow.
  // A zero intensity skips the half-res texture, which is unbound noise
  // when the bloom passes did not run.
  var bloom = vec3<f32>(0.0);
  if (u.intensity > 0.0) {
    bloom = textureSample(t_bloom, t_sampler, uv).rgb;
  }
  if (u.dirt > 0.0) { bloom = bloom * (1.0 + u.dirt * fx_lens_dirt(uv)); }
  var hdr: vec3<f32>;
  if (u.bloomMode > 0.5 && u.intensity > 0.0) {
    hdr = mix(scene, bloom / max(u.levels, 1.0), u.intensity) * u.exposure;
  } else {
    hdr = (scene + bloom * u.intensity) * u.exposure;
  }
  var outc = tonemap_display(hdr, u.mode, u.saturation, u.contrast, u.vibrance, u.lift);
  if (u.grain > 0.0) { outc = fx_grain(outc, frag.xy, u.frame, u.grain); }
  return vec4<f32>(outc, 1.0);
}
`,
  uniforms: {
    size: 64,
    fields: {
      intensity: { offset: 0, type: "f32" },
      exposure: { offset: 4, type: "f32" },
      saturation: { offset: 8, type: "f32" },
      mode: { offset: 12, type: "f32" },
      contrast: { offset: 16, type: "f32" },
      vibrance: { offset: 20, type: "f32" },
      lift: { offset: 24, type: "f32" },
      bloomMode: { offset: 28, type: "f32" },
      outTexel: { offset: 32, type: "vec2" },
      grain: { offset: 40, type: "f32" },
      frame: { offset: 44, type: "f32" },
      dirt: { offset: 48, type: "f32" },
      levels: { offset: 52, type: "f32" },
    },
  },
};

export type BloomMode = "threshold" | "physical";

export type Tonemapper = "aces" | "neutral" | "agx";
const TONEMAPPER_MODE: Record<Tonemapper, number> = { aces: 0, neutral: 1, agx: 2 };

/** Half-resolution size used by the bloom chain (never below 1 px). */
export function bloomSize(width: number, height: number): { width: number; height: number } {
  return { width: Math.max(1, Math.ceil(width / 2)), height: Math.max(1, Math.ceil(height / 2)) };
}

export const FXAA_PASS: PostPassDescriptor = {
  name: "fxaa",
  shader: /* wgsl */ `
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var t_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// Conservative FXAA 3.11 fit. Run AFTER tonemap (LDR space).
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let dim = vec2<f32>(textureDimensions(t_input));
  let texel = 1.0 / dim;
  let uv = frag.xy * texel;
  let center = textureSample(t_input, t_sampler, uv).rgb;
  let n = textureSample(t_input, t_sampler, uv + vec2<f32>(0.0, -texel.y)).rgb;
  let s = textureSample(t_input, t_sampler, uv + vec2<f32>(0.0,  texel.y)).rgb;
  let e = textureSample(t_input, t_sampler, uv + vec2<f32>( texel.x, 0.0)).rgb;
  let w = textureSample(t_input, t_sampler, uv + vec2<f32>(-texel.x, 0.0)).rgb;
  let lumW = vec3<f32>(0.2126, 0.7152, 0.0722);
  let lc = dot(center, lumW);
  let ln = dot(n, lumW);
  let ls = dot(s, lumW);
  let le = dot(e, lumW);
  let lw = dot(w, lumW);
  let lmin = min(lc, min(min(ln, ls), min(le, lw)));
  let lmax = max(lc, max(max(ln, ls), max(le, lw)));
  let range = lmax - lmin;
  if (range < 0.01) { return vec4<f32>(center, 1.0); }
  // 4-tap blend along the contrast direction.
  let blend = (n + s + e + w) * 0.25;
  return vec4<f32>(mix(center, blend, 0.5), 1.0);
}
`,
};

export const DEFAULT_POST_PIPELINE: PostPassDescriptor[] = [
  BLOOM_EXTRACT_PASS,
  BLOOM_BLUR_PASS, // horizontal
  BLOOM_BLUR_PASS, // vertical
  BLOOM_COMPOSITE_PASS,
  ACES_TONEMAP_PASS,
  FXAA_PASS,
];

// ──────────────────────────────────────────────────────────────────────
// Concrete WebGPU Post-Processing Implementations.
// ──────────────────────────────────────────────────────────────────────

export class WebGpuPostPass {
  pipeline: GPURenderPipeline;
  uniformBuffer?: GPUBuffer;
  bindGroup: GPUBindGroup | null = null;
  descriptor: PostPassDescriptor;
  private inputLayoutType: "single" | "composite" | "fxaa";

  constructor(
    device: GPUDevice,
    descriptor: PostPassDescriptor,
    inputLayoutType: "single" | "composite" | "fxaa",
    targetFormat: GPUTextureFormat,
    uniformSize?: number
  ) {
    this.descriptor = descriptor;
    this.inputLayoutType = inputLayoutType;
    const module = device.createShaderModule({
      label: `post-shader-${descriptor.name}`,
      code: descriptor.shader,
    });

    let entries: GPUBindGroupLayoutEntry[] = [];
    if (inputLayoutType === "single") {
      entries = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ];
    } else if (inputLayoutType === "composite") {
      entries = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ];
    } else if (inputLayoutType === "fxaa") {
      entries = [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ];
    }

    const bindGroupLayout = device.createBindGroupLayout({
      label: `post-bg-layout-${descriptor.name}`,
      entries,
    });

    const pipelineLayout = device.createPipelineLayout({
      label: `post-pipeline-layout-${descriptor.name}`,
      bindGroupLayouts: [bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: `post-pipeline-${descriptor.name}`,
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: targetFormat }],
      },
    });

    const realUniformSize = uniformSize ?? descriptor.uniforms?.size;
    if (realUniformSize) {
      // Pad uniform buffer to 16 bytes boundary
      const paddedSize = Math.ceil(realUniformSize / 16) * 16;
      this.uniformBuffer = device.createBuffer({
        label: `post-uniform-buf-${descriptor.name}`,
        size: paddedSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  writeUniforms(device: GPUDevice, data: Float32Array): void {
    if (this.uniformBuffer) {
      device.queue.writeBuffer(this.uniformBuffer, 0, data as unknown as BufferSource);
    }
  }

  updateBindGroup(
    device: GPUDevice,
    textures: GPUTextureView[],
    sampler: GPUSampler
  ): void {
    this.bindGroup = this.makeBindGroup(device, textures, sampler);
  }

  /** A bind group for these inputs without replacing `bindGroup`. */
  makeBindGroup(
    device: GPUDevice,
    textures: GPUTextureView[],
    sampler: GPUSampler
  ): GPUBindGroup {
    let entries: GPUBindGroupEntry[] = [];
    if (this.inputLayoutType === "single") {
      entries = [
        { binding: 0, resource: textures[0]! },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer! } },
      ];
    } else if (this.inputLayoutType === "composite") {
      entries = [
        { binding: 0, resource: textures[0]! },
        { binding: 1, resource: textures[1]! },
        { binding: 2, resource: sampler },
        { binding: 3, resource: { buffer: this.uniformBuffer! } },
      ];
    } else if (this.inputLayoutType === "fxaa") {
      entries = [
        { binding: 0, resource: textures[0]! },
        { binding: 1, resource: sampler },
      ];
    }

    return device.createBindGroup({
      label: `post-bindgroup-${this.descriptor.name}`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    });
  }
}

export class PostProcessingRunner {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private sampler: GPUSampler;

  hdrTexture: GPUTexture | null = null;
  /** Cached view of hdrTexture (stable between resizes). */
  hdrView: GPUTextureView | null = null;
  bloomTexA: GPUTexture | null = null;
  bloomTexB: GPUTexture | null = null;
  ldrTexture: GPUTexture | null = null;

  extractPass: WebGpuPostPass | null = null;
  blurHPass: WebGpuPostPass | null = null;
  blurVPass: WebGpuPostPass | null = null;
  /** Bloom composite + tonemap in one pass (see COMPOSITE_TONEMAP_PASS). */
  compositeTonemapPass: WebGpuPostPass | null = null;
  fxaaPass: WebGpuPostPass | null = null;

  width = 0;
  height = 0;
  /** Bloom chain resolution — half of width × height. */
  bloomWidth = 0;
  bloomHeight = 0;

  // Per-frame scratch, allocated once: run() used to build five Float32Arrays
  // and five texture views every frame, which is steady GC churn for nothing.
  private extractUniforms = new Float32Array(4);
  private blurHUniforms = new Float32Array(4);
  private blurVUniforms = new Float32Array(4);
  private compositeTonemapUniforms = new Float32Array(16);
  /** Energy-conserving mip-chain bloom (bloom-chain.ts); level 0 is bloomTexA. */
  private chainBloom: PhysicalBloomChain;
  private fxFrame = 0;
  /** Internal (scene) render size; equals width × height unless a render scale is set. */
  sceneWidth = 0;
  sceneHeight = 0;
  private bloomViewA: GPUTextureView | null = null;
  private bloomViewB: GPUTextureView | null = null;
  private ldrView: GPUTextureView | null = null;

  // Tunable look parameters. Defaults bias toward saturated neon glow (the
  // OVERCLOCK / stylized-arena target) rather than neutral filmic. Callers can
  // override via setLook() — e.g. a daylight scene would drop saturation +
  // raise the bloom threshold.
  bloomThreshold = 0.9;
  bloomKnee = 0.2;
  bloomIntensity = 0.6;
  exposure = 1.1;
  tonemapSaturation = 0.7;
  /**
   * "full" is extract, two blurs, tonemap, and FXAA.
   * "tonemap" is one full-screen ACES pass to the canvas. The proof room
   * uses it: the neon bloom chain was costing passes the character gate
   * does not need.
   */
  chain: "full" | "tonemap" = "full";

  setChain(chain: "full" | "tonemap"): void {
    this.chain = chain;
  }

  /** Override any subset of the look parameters. */
  /** Tonemapper + display grade. Defaults (aces, 1, 1, 0) reproduce the old look. */
  tonemapper: Tonemapper = "aces";
  contrast = 1;
  vibrance = 1;
  lift = 0;
  /** "threshold" (default, the neon look) or "physical" (bloom-chain.ts). */
  bloomMode: BloomMode = "threshold";
  /** Physical bloom: fraction of light spread by the lens (default 0.04). */
  bloomStrength = 0.04;
  /** Lens dirt on bloom (0 = off, the default). */
  lensDirt = 0;
  /** Film grain amount (0 = off, the default). */
  filmGrain = 0;
  /** Set by the renderer while natural light is on (forces physical bloom). */
  forceBloomMode: BloomMode | null = null;

  /** Bloom mode in effect this frame. */
  get effectiveBloomMode(): BloomMode {
    return this.forceBloomMode ?? this.bloomMode;
  }

  setLook(look: Partial<{
    bloomThreshold: number;
    bloomKnee: number;
    bloomIntensity: number;
    exposure: number;
    tonemapSaturation: number;
    tonemapper: Tonemapper;
    contrast: number;
    vibrance: number;
    lift: number;
    bloomMode: BloomMode;
    bloomStrength: number;
    lensDirt: number;
    filmGrain: number;
  }>): void {
    if (look.bloomMode !== undefined) this.bloomMode = look.bloomMode;
    if (look.bloomStrength !== undefined && Number.isFinite(look.bloomStrength)) this.bloomStrength = Math.min(1, Math.max(0, look.bloomStrength));
    if (look.lensDirt !== undefined && Number.isFinite(look.lensDirt)) this.lensDirt = Math.max(0, look.lensDirt);
    if (look.filmGrain !== undefined && Number.isFinite(look.filmGrain)) this.filmGrain = Math.max(0, look.filmGrain);
    if (look.tonemapper !== undefined) this.tonemapper = look.tonemapper;
    if (look.contrast !== undefined) this.contrast = look.contrast;
    if (look.vibrance !== undefined) this.vibrance = look.vibrance;
    if (look.lift !== undefined) this.lift = look.lift;
    if (look.bloomThreshold !== undefined) this.bloomThreshold = look.bloomThreshold;
    if (look.bloomKnee !== undefined) this.bloomKnee = look.bloomKnee;
    if (look.bloomIntensity !== undefined) this.bloomIntensity = look.bloomIntensity;
    if (look.exposure !== undefined) this.exposure = look.exposure;
    if (look.tonemapSaturation !== undefined) this.tonemapSaturation = look.tonemapSaturation;
  }

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    this.sampler = device.createSampler({
      label: "post-sampler",
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.initPasses();
    this.chainBloom = new PhysicalBloomChain(device);
  }

  private initPasses(): void {
    this.extractPass = new WebGpuPostPass(this.device, BLOOM_EXTRACT_PASS, "single", "rgba16float");
    this.blurHPass = new WebGpuPostPass(this.device, BLOOM_BLUR_PASS, "single", "rgba16float");
    this.blurVPass = new WebGpuPostPass(this.device, BLOOM_BLUR_PASS, "single", "rgba16float");
    this.compositeTonemapPass = new WebGpuPostPass(this.device, COMPOSITE_TONEMAP_PASS, "composite", this.format);
    this.fxaaPass = new WebGpuPostPass(this.device, FXAA_PASS, "fxaa", this.format);
  }

  /**
   * @param sceneWidth / sceneHeight the main pass's render size (render
   *   scale). The HDR target is allocated at that size; bloom and LDR targets
   *   stay at the output size.
   */
  resize(width: number, height: number, sceneWidth = width, sceneHeight = height): void {
    if (this.width === width && this.height === height && this.sceneWidth === sceneWidth && this.sceneHeight === sceneHeight) return;
    this.width = width;
    this.height = height;
    this.sceneWidth = sceneWidth;
    this.sceneHeight = sceneHeight;
    const bloom = bloomSize(width, height);
    this.bloomWidth = bloom.width;
    this.bloomHeight = bloom.height;

    this.destroyTextures();

    this.hdrTexture = this.device.createTexture({
      label: "post-hdr-color",
      size: [sceneWidth, sceneHeight],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.bloomTexA = this.device.createTexture({
      label: "post-bloom-a",
      size: [bloom.width, bloom.height],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.bloomTexB = this.device.createTexture({
      label: "post-bloom-b",
      size: [bloom.width, bloom.height],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.ldrTexture = this.device.createTexture({
      label: "post-ldr-color",
      size: [width, height],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.hdrView = this.hdrTexture.createView();
    this.bloomViewA = this.bloomTexA.createView();
    this.chainBloom.resize(width, height, this.bloomTexA, this.bloomViewA);
    this.bloomViewB = this.bloomTexB.createView();
    this.ldrView = this.ldrTexture.createView();

    // Rebuild bind groups. Chain: hdr → extract(A, half) → blurH(B) →
    // blurV(A) → composite+tonemap(hdr + A → ldr) → fxaa(ldr → canvas).
    this.extractPass!.updateBindGroup(this.device, [this.hdrTexture.createView()], this.sampler);
    this.blurHPass!.updateBindGroup(this.device, [this.bloomViewA], this.sampler);
    this.blurVPass!.updateBindGroup(this.device, [this.bloomViewB], this.sampler);
    this.compositeTonemapPass!.updateBindGroup(
      this.device,
      [this.hdrTexture.createView(), this.bloomViewA],
      this.sampler
    );
    this.fxaaPass!.updateBindGroup(this.device, [this.ldrView], this.sampler);
  }

  /** Half-res bloom result after runBloom() (null before the first resize). */
  get bloomView(): GPUTextureView | null {
    return this.bloomViewA;
  }

  /** The tonemap / grade values the fused TAA resolve needs. */
  lookUniforms(): { bloomIntensity: number; exposure: number; saturation: number; mode: number; contrast: number; vibrance: number; lift: number; bloomMode: number; bloomLevels: number; lensDirt: number; filmGrain: number; frame: number } {
    const physical = this.effectiveBloomMode === "physical";
    return {
      bloomIntensity: this.chain === "full" ? (physical ? this.bloomStrength : this.bloomIntensity) : 0,
      exposure: this.exposure,
      saturation: this.tonemapSaturation,
      mode: TONEMAPPER_MODE[this.tonemapper],
      contrast: this.contrast,
      vibrance: this.vibrance,
      lift: this.lift,
      bloomMode: physical ? 1 : 0,
      bloomLevels: this.chainBloom.levelCount,
      lensDirt: this.lensDirt,
      filmGrain: this.filmGrain,
      frame: this.fxFrame++ % 1024,
    };
  }

  /** Fill the composite uniform (bloom term, grade, camera effects). */
  private fillComposite(t: Float32Array, bloomOn: boolean): void {
    const physical = this.effectiveBloomMode === "physical";
    t[0] = bloomOn ? (physical ? this.bloomStrength : this.bloomIntensity) : 0;
    t[1] = this.exposure; t[2] = this.tonemapSaturation; t[3] = TONEMAPPER_MODE[this.tonemapper];
    t[4] = this.contrast; t[5] = this.vibrance; t[6] = this.lift; t[7] = physical ? 1 : 0;
    t[8] = 1 / this.width; t[9] = 1 / this.height; t[10] = this.filmGrain; t[11] = this.fxFrame++ % 1024;
    t[12] = this.lensDirt; t[13] = this.chainBloom.levelCount; t[14] = 0; t[15] = 0;
  }

  /**
   * Bloom only (extract → blur H → blur V into bloomView), for the TAA path,
   * where the resolve pass composites + tonemaps itself. Runs from the
   * internal-resolution HDR target. No-op on the "tonemap" chain.
   */
  runBloom(encoder: GPUCommandEncoder, beginTimestampWrites?: GPURenderPassTimestampWrites, endTimestampWrites?: GPURenderPassTimestampWrites): boolean {
    if (this.chain !== "full" || !this.bloomViewA || !this.bloomViewB) return false;
    if (this.effectiveBloomMode === "physical" && this.hdrView) {
      this.chainBloom.run(encoder, this.hdrView, this.sceneWidth, this.sceneHeight, beginTimestampWrites, endTimestampWrites);
      return true;
    }
    const texelW = 1.0 / this.bloomWidth;
    const texelH = 1.0 / this.bloomHeight;
    const ex = this.extractUniforms;
    ex[0] = this.bloomThreshold; ex[1] = this.bloomKnee; ex[2] = texelW; ex[3] = texelH;
    this.extractPass!.writeUniforms(this.device, ex);
    const bh = this.blurHUniforms;
    bh[0] = texelW; bh[1] = texelH; bh[2] = 1.0; bh[3] = 0.0;
    this.blurHPass!.writeUniforms(this.device, bh);
    const bv = this.blurVUniforms;
    bv[0] = texelW; bv[1] = texelH; bv[2] = 0.0; bv[3] = 1.0;
    this.blurVPass!.writeUniforms(this.device, bv);
    const passes: Array<[WebGpuPostPass, GPUTextureView, string]> = [
      [this.extractPass!, this.bloomViewA, "post-pass-extract"],
      [this.blurHPass!, this.bloomViewB, "post-pass-blur-h"],
      [this.blurVPass!, this.bloomViewA, "post-pass-blur-v"],
    ];
    passes.forEach(([p, target, label], i) => {
      const desc: GPURenderPassDescriptor = {
        label,
        colorAttachments: [{ view: target, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      };
      if (i === 0 && beginTimestampWrites) desc.timestampWrites = beginTimestampWrites;
      if (i === passes.length - 1 && endTimestampWrites) {
        desc.timestampWrites = { ...(desc.timestampWrites ?? {}), ...endTimestampWrites } as GPURenderPassTimestampWrites;
      }
      const pass = encoder.beginRenderPass(desc);
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, p.bindGroup!);
      pass.draw(3);
      pass.end();
    });
    return true;
  }

  /**
   * @param endTimestampWrites optional — attached to the final (FXAA) pass so
   *   a frame-spanning GPU timer can close after the whole post chain.
   * @param beginTimestampWrites optional — attached to the FIRST post pass so a
   *   per-pass timer can measure the post chain on its own. When the chain is
   *   one pass, begin and end land on that pass (same query set required).
   */
  run(
    encoder: GPUCommandEncoder,
    canvasView: GPUTextureView,
    endTimestampWrites?: GPURenderPassTimestampWrites,
    beginTimestampWrites?: GPURenderPassTimestampWrites,
  ): void {
    if (!this.hdrTexture || !this.bloomTexA || !this.bloomTexB || !this.ldrTexture) return;
    if (!this.bloomViewA || !this.bloomViewB || !this.ldrView) return;
    const compositeBG = this.compositeTonemapPass!.bindGroup!;
    const extractBG = this.extractPass!.bindGroup!;
    const outTexelX = 1 / this.width;
    const outTexelY = 1 / this.height;
    if (this.chain === "tonemap") {
      if (beginTimestampWrites) {
        const endIndex = endTimestampWrites?.endOfPassWriteIndex;
        endTimestampWrites = endIndex !== undefined
          ? { ...beginTimestampWrites, endOfPassWriteIndex: endIndex }
          : beginTimestampWrites;
      }
      const tone = this.compositeTonemapUniforms;
      this.fillComposite(tone, false);
      void outTexelX; void outTexelY;
      this.compositeTonemapPass!.writeUniforms(this.device, tone);
      const toneDesc: GPURenderPassDescriptor = {
        label: "post-pass-tonemap",
        colorAttachments: [{
          view: canvasView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      };
      if (endTimestampWrites) toneDesc.timestampWrites = endTimestampWrites;
      const tonePass = encoder.beginRenderPass(toneDesc);
      tonePass.setPipeline(this.compositeTonemapPass!.pipeline);
      tonePass.setBindGroup(0, compositeBG);
      tonePass.draw(3);
      tonePass.end();
      return;
    }

    // Blur texel size is the HALF-resolution bloom texel: a 5-tap kernel at
    // half res covers the same screen radius as a 9-tap one at full res.
    const texelW = 1.0 / this.bloomWidth;
    const texelH = 1.0 / this.bloomHeight;

    // 1. Write uniform buffers
    // Bloom tuned for neon: lower threshold so emissive HDR surfaces bloom
    // well before pure white, wider soft knee for a smoother halo, and a
    // stronger composite so the glow reads as light, not a faint rim.
    const ex = this.extractUniforms;
    ex[0] = this.bloomThreshold; ex[1] = this.bloomKnee; ex[2] = texelW; ex[3] = texelH;
    this.extractPass!.writeUniforms(this.device, ex);
    const bh = this.blurHUniforms;
    bh[0] = texelW; bh[1] = texelH; bh[2] = 1.0; bh[3] = 0.0; // dir=[1,0]
    this.blurHPass!.writeUniforms(this.device, bh);
    const bv = this.blurVUniforms;
    bv[0] = texelW; bv[1] = texelH; bv[2] = 0.0; bv[3] = 1.0; // dir=[0,1]
    this.blurVPass!.writeUniforms(this.device, bv);
    const ct = this.compositeTonemapUniforms;
    this.fillComposite(ct, true);
    this.compositeTonemapPass!.writeUniforms(this.device, ct);

    const physicalBloom = this.effectiveBloomMode === "physical" && !!this.hdrView;
    if (physicalBloom) this.chainBloom.run(encoder, this.hdrView!, this.sceneWidth, this.sceneHeight, beginTimestampWrites);

    // 2. Pass 1: Extract Bloom Highlights (full res → half res)
    if (!physicalBloom) {
      const extractDesc: GPURenderPassDescriptor = {
        label: "post-pass-extract",
        colorAttachments: [{
          view: this.bloomViewA,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      };
      if (beginTimestampWrites) extractDesc.timestampWrites = beginTimestampWrites;
      const pass = encoder.beginRenderPass(extractDesc);
      pass.setPipeline(this.extractPass!.pipeline);
      pass.setBindGroup(0, extractBG);
      pass.draw(3);
      pass.end();
    }

    // 3. Pass 2: Blur Bloom Horizontal
    if (!physicalBloom) {
      const pass = encoder.beginRenderPass({
        label: "post-pass-blur-h",
        colorAttachments: [{
          view: this.bloomViewB,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.blurHPass!.pipeline);
      pass.setBindGroup(0, this.blurHPass!.bindGroup!);
      pass.draw(3);
      pass.end();
    }

    // 4. Pass 3: Blur Bloom Vertical
    if (!physicalBloom) {
      const pass = encoder.beginRenderPass({
        label: "post-pass-blur-v",
        colorAttachments: [{
          view: this.bloomViewA,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.blurVPass!.pipeline);
      pass.setBindGroup(0, this.blurVPass!.bindGroup!);
      pass.draw(3);
      pass.end();
    }

    // 5. Pass 4: Composite bloom + ACES tonemap (one full-res pass → LDR)
    {
      const pass = encoder.beginRenderPass({
        label: "post-pass-composite-tonemap",
        colorAttachments: [{
          view: this.ldrView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.compositeTonemapPass!.pipeline);
      pass.setBindGroup(0, compositeBG);
      pass.draw(3);
      pass.end();
    }

    // 6. Pass 5: FXAA (to final presentation canvas target)
    {
      const fxaaDescriptor: GPURenderPassDescriptor = {
        label: "post-pass-fxaa",
        colorAttachments: [{
          view: canvasView,
          loadOp: "clear",
          storeOp: "store",
        }],
      };
      if (endTimestampWrites) fxaaDescriptor.timestampWrites = endTimestampWrites;
      const pass = encoder.beginRenderPass(fxaaDescriptor);
      pass.setPipeline(this.fxaaPass!.pipeline);
      pass.setBindGroup(0, this.fxaaPass!.bindGroup!);
      pass.draw(3);
      pass.end();
    }
  }

  destroyTextures(): void {
    this.hdrTexture?.destroy();
    this.bloomTexA?.destroy();
    this.bloomTexB?.destroy();
    this.ldrTexture?.destroy();
    this.hdrTexture = null;
    this.hdrView = null;
    this.bloomTexA = null;
    this.bloomTexB = null;
    this.ldrTexture = null;
    this.bloomViewA = null;
    this.bloomViewB = null;
    this.ldrView = null;
  }

  destroy(): void {
    this.destroyTextures();
    this.chainBloom.destroy();
    this.extractPass = null;
    this.blurHPass = null;
    this.blurVPass = null;
    this.compositeTonemapPass = null;
    this.fxaaPass = null;
  }
}

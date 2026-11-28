/**
 * Temporal anti-aliasing / upscaling (TAAU) and depth-only SSAO for the
 * WebGPU live renderer.
 *
 * The main pass renders at `renderScale` × the canvas size with a sub-pixel
 * projection jitter (Halton 2,3, 8 samples). The resolve pass then
 * reconstructs a full-resolution image each frame from the current low-res
 * frame plus the reprojected full-resolution history:
 *
 *  - Reprojection: where motion.ts drew a moving object (skinned, morph
 *    targets, changed model matrix) the history is fetched along that
 *    object's own motion vector; everywhere else the closest depth in a
 *    3×3 cross is unprojected with the current view-projection and projected
 *    with last frame's (exact for static geometry).
 *  - History is clipped to the current 3×3 neighbourhood's YCoCg mean ± k·σ
 *    (variance clipping) and blended with luma weighting (Karis 2014) to
 *    damp firefly flicker.
 *
 * The resolve is fused with the tonemap: one full-resolution pass writes both
 * the new HDR history (MRT 0) and the final display colour to the canvas
 * (MRT 1), with bloom (built beforehand from the internal-resolution frame),
 * exposure, tonemapper, grade and a luma sharpen applied to the display only.
 * On Intel UHD a full-resolution pass costs ~2 ms, so not having a separate
 * composite pass is what pays for TAA.
 *
 * SSAO runs at half the internal resolution from depth alone (normals are
 * reconstructed from depth), with a per-pixel rotated kernel that changes
 * every frame. Its output (ao, view depth) is upsampled inside the resolve
 * pass with depth-aware weights and accumulated into the history's alpha
 * channel, so the TAA history doubles as the AO denoiser. The display output
 * multiplies colour by that alpha. This darkens all lighting, not just
 * ambient — a deliberate compromise that keeps AO out of every material
 * shader.
 */

import { CAMERA_FX_WGSL, TONEMAP_WGSL } from "./post-processing.js";

/** Radical inverse in `base` of `index` (1-based for Halton). */
export function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

/** Number of distinct jitter positions before the sequence repeats. */
export const TAA_JITTER_SAMPLES = 8;

/**
 * Sub-pixel jitter for frame `frameIndex`, in pixels, centred on zero
 * (each component in [-0.5, 0.5)). Halton(2,3) over TAA_JITTER_SAMPLES.
 */
export function taaJitter(frameIndex: number, samples = TAA_JITTER_SAMPLES): [number, number] {
  const n = (((frameIndex % samples) + samples) % samples) + 1;
  return [halton(n, 2) - 0.5, halton(n, 3) - 0.5];
}

/**
 * Offset a column-major perspective projection so the rendered image moves
 * by (jx, jy) pixels on a width×height target (+x right, +y up in NDC).
 * For this engine's projection, ndc.x = (P0·x + P8·z) / (−z), so subtracting
 * δ from P8 adds δ to ndc.x.
 */
export function applyProjectionJitter(proj: Float32Array | number[], jx: number, jy: number, width: number, height: number): void {
  proj[8] = (proj[8] ?? 0) - (2 * jx) / width;
  proj[9] = (proj[9] ?? 0) - (2 * jy) / height;
}

/**
 * Jitter in UV units (y down) for the resolve pass: content that sits at uv
 * in an unjittered frame is found at uv + jitterUv in the jittered one.
 */
export function jitterToUv(jx: number, jy: number, width: number, height: number): [number, number] {
  return [jx / width, -jy / height];
}

/**
 * View distance from a WebGPU depth value for this engine's GL-style
 * projection: ndc.z = (A·z + B) / (−z) with A = P10, B = P14 ⇒ dist = B / (d + A).
 */
export function linearizeDepth(depth: number, p10: number, p14: number): number {
  return p14 / (depth + p10);
}

/** Internal render size for a canvas size and scale (never below 1 px). */
export function scaledSize(width: number, height: number, scale: number): { width: number; height: number } {
  const s = Math.min(1, Math.max(0.25, scale));
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)) };
}

const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}
`;

/** View distance (m) where SSAO starts fading out, and where it is gone. */
export const AO_FADE_START = 40;
export const AO_FADE_END = 70;

export const SSAO_WGSL = /* wgsl */ `
const AO_FADE_START: f32 = ${AO_FADE_START.toFixed(1)};
const AO_FADE_END: f32 = ${AO_FADE_END.toFixed(1)};
struct U {
  invProj: mat4x4<f32>,
  proj: mat4x4<f32>,
  inSize: vec2<f32>,
  outSize: vec2<f32>,
  radius: f32,
  intensity: f32,
  samples: f32,
  frame: f32,
};
@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> u: U;
${FULLSCREEN_VS}
fn loadDepth(p: vec2<i32>) -> f32 {
  let m = vec2<i32>(u.inSize) - vec2<i32>(1);
  return textureLoad(t_depth, clamp(p, vec2<i32>(0), m), 0);
}
fn viewPosAt(p: vec2<i32>, d: f32) -> vec3<f32> {
  let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / u.inSize;
  let v = u.invProj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  return v.xyz / v.w;
}
fn viewPos(p: vec2<i32>) -> vec3<f32> { return viewPosAt(p, loadDepth(p)); }
// Interleaved gradient noise (Jimenez 2014).
fn ign(p: vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let px = vec2<i32>(floor(frag.xy * (u.inSize / u.outSize)));
  let d = loadDepth(px);
  if (d >= 1.0) { return vec4<f32>(1.0, 1e4, 0.0, 1.0); }
  let P = viewPosAt(px, d);
  let dist = -P.z;
  if (dist >= AO_FADE_END) { return vec4<f32>(1.0, dist, 0.0, 1.0); }
  // Normal from depth: per axis take the smaller-difference neighbour so
  // silhouettes do not smear the normal across a depth step.
  let pr = viewPos(px + vec2<i32>(1, 0));
  let pl = viewPos(px - vec2<i32>(1, 0));
  let pu = viewPos(px - vec2<i32>(0, 1));
  let pd = viewPos(px + vec2<i32>(0, 1));
  let dx = select(P - pl, pr - P, abs(pr.z - P.z) < abs(P.z - pl.z));
  let dy = select(P - pu, pd - P, abs(pd.z - P.z) < abs(P.z - pu.z));
  var N = normalize(cross(dx, dy));
  if (dot(N, -P) < 0.0) { N = -N; }
  let angle = ign(frag.xy + vec2<f32>(u.frame * 5.588238)) * 6.2831853;
  let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(N.x) > 0.9);
  let T = normalize(cross(helper, N));
  let B = cross(N, T);
  let n = i32(u.samples);
  var occl = 0.0;
  for (var i = 0; i < n; i = i + 1) {
    let t = (f32(i) + 0.5) / f32(n);
    let a = f32(i) * 2.39996323 + angle;
    let r = sqrt(t);
    // Cosine-ish hemisphere direction, scaled so samples cluster near P.
    let local = vec3<f32>(r * cos(a), r * sin(a), sqrt(max(1.0 - t, 0.0)));
    let scale = mix(0.15, 1.0, t * t);
    let s = P + (T * local.x + B * local.y + N * local.z) * (u.radius * scale);
    let c = u.proj * vec4<f32>(s, 1.0);
    if (c.w <= 0.0) { continue; }
    let suv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
    if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) { continue; }
    let sp = vec2<i32>(suv * u.inSize);
    let sz = viewPos(sp).z;
    // View space looks down −z: a larger z is closer to the camera.
    let range = smoothstep(0.0, 1.0, u.radius / max(abs(P.z - sz), 1e-4));
    // 24-bit depth loses precision with distance, so a flat far surface
    // reconstructs as a stair of planes and occludes itself. The bias grows
    // with view distance to stay above that step.
    occl = occl + select(0.0, range, sz >= s.z + 0.02 * u.radius + 0.004 * dist);
  }
  // Fade AO out with distance: past ~40 m the sample radius is under a pixel
  // and what is left is the dark band where the far road meets buildings.
  let fade = 1.0 - smoothstep(AO_FADE_START, AO_FADE_END, dist);
  let ao = clamp(1.0 - occl / f32(max(n, 1)) * u.intensity * fade, 0.0, 1.0);
  return vec4<f32>(ao, dist, 0.0, 1.0);
}
`;

/**
 * 4×4 depth-aware box blur of the half-res AO (ao, view depth). The per-pixel
 * kernel rotation repeats every 4×4 pixels' worth of noise, so a 4×4 support
 * removes the pattern; taps across a depth edge are rejected.
 */
export const AO_BLUR_WGSL = /* wgsl */ `
@group(0) @binding(0) var t_ao: texture_2d<f32>;
${FULLSCREEN_VS}
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(frag.xy);
  let m = vec2<i32>(textureDimensions(t_ao)) - vec2<i32>(1);
  let c = textureLoad(t_ao, p, 0);
  var acc = 0.0;
  var wsum = 0.0;
  for (var y = -2; y < 2; y = y + 1) {
    for (var x = -2; x < 2; x = x + 1) {
      let s = textureLoad(t_ao, clamp(p + vec2<i32>(x, y), vec2<i32>(0), m), 0);
      let w = select(0.0, 1.0, abs(s.y - c.y) < 0.05 * c.y + 0.02);
      acc = acc + s.x * w;
      wsum = wsum + w;
    }
  }
  return vec4<f32>(select(c.x, acc / wsum, wsum > 0.0), c.y, 0.0, 1.0);
}
`;

export const TAA_WGSL = /* wgsl */ `
struct U {
  invVP: mat4x4<f32>,
  prevVP: mat4x4<f32>,
  jitterUv: vec2<f32>,
  outSize: vec2<f32>,
  inSize: vec2<f32>,
  aoSize: vec2<f32>,
  projAB: vec2<f32>,
  feedback: f32,
  flags: f32,
  // bloomIntensity, exposure, saturation, tonemapper mode
  look0: vec4<f32>,
  // contrast, vibrance, lift, sharpen
  look1: vec4<f32>,
  // bloom mode (1 = physical mix), bloom levels, lens dirt, film grain
  look2: vec4<f32>,
  // x = frame (grain seed)
  look3: vec4<f32>,
};
@group(0) @binding(0) var t_cur: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_hist: texture_2d<f32>;
@group(0) @binding(3) var t_ao: texture_2d<f32>;
@group(0) @binding(4) var s_lin: sampler;
@group(0) @binding(5) var<uniform> u: U;
@group(0) @binding(6) var t_bloom: texture_2d<f32>;
// Per-object motion (motion.ts): uv now − uv last frame; x < −100 = none.
@group(0) @binding(7) var t_vel: texture_2d<f32>;
${FULLSCREEN_VS}
${TONEMAP_WGSL}
${CAMERA_FX_WGSL}
fn toYCoCg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(c, vec3<f32>(0.25, 0.5, 0.25)), dot(c, vec3<f32>(0.5, 0.0, -0.5)), dot(c, vec3<f32>(-0.25, 0.5, -0.25)));
}
fn fromYCoCg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
fn linDepth(d: f32) -> f32 { return u.projAB.y / (d + u.projAB.x); }
// Catmull-Rom history fetch from 5 bilinear taps (Jimenez 2016, corners
// dropped). Bilinear history resampling is what softens a TAA image a little
// more every frame; this keeps the reconstruction sharp. History alpha (AO)
// takes the plain bilinear value: it is a smooth signal.
fn sampleHistoryCR(uvIn: vec2<f32>) -> vec4<f32> {
  let size = u.outSize;
  let pos = uvIn * size;
  let c = floor(pos - vec2<f32>(0.5)) + vec2<f32>(0.5);
  let f = pos - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let p0 = (c - vec2<f32>(1.0)) / size;
  let p3 = (c + vec2<f32>(2.0)) / size;
  let p12 = (c + o12) / size;
  var r = vec3<f32>(0.0);
  r = r + textureSampleLevel(t_hist, s_lin, vec2<f32>(p12.x, p0.y), 0.0).rgb * (w12.x * w0.y);
  r = r + textureSampleLevel(t_hist, s_lin, vec2<f32>(p0.x, p12.y), 0.0).rgb * (w0.x * w12.y);
  let mid = textureSampleLevel(t_hist, s_lin, p12, 0.0);
  r = r + mid.rgb * (w12.x * w12.y);
  r = r + textureSampleLevel(t_hist, s_lin, vec2<f32>(p3.x, p12.y), 0.0).rgb * (w3.x * w12.y);
  r = r + textureSampleLevel(t_hist, s_lin, vec2<f32>(p12.x, p3.y), 0.0).rgb * (w12.x * w3.y);
  let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return vec4<f32>(max(r / wsum, vec3<f32>(0.0)), mid.a);
}
fn clipAabb(q: vec3<f32>, c: vec3<f32>, e: vec3<f32>) -> vec3<f32> {
  let r = q - c;
  let ru = r / max(e, vec3<f32>(1e-4));
  let m = max(abs(ru.x), max(abs(ru.y), abs(ru.z)));
  if (m > 1.0) { return c + r / m; }
  return q;
}

struct Out {
  @location(0) history: vec4<f32>,
  @location(1) display: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> Out {
  let uv = frag.xy / u.outSize;
  let inMax = vec2<i32>(u.inSize) - vec2<i32>(1);
  let ic = clamp(vec2<i32>(uv * u.inSize), vec2<i32>(0), inMax);
  let crossOff = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));

  // Closest depth in a cross: reprojecting with it pulls history from the
  // foreground at silhouettes, which is what removes edge ghosting.
  let dCenter = textureLoad(t_depth, ic, 0);
  var dClosest = dCenter;
  var pClosest = ic;
  for (var i = 0; i < 4; i = i + 1) {
    let q = clamp(ic + crossOff[i], vec2<i32>(0), inMax);
    let dq = textureLoad(t_depth, q, 0);
    if (dq < dClosest) { dClosest = dq; pClosest = q; }
  }
  let flagBits = u32(u.flags + 0.5);

  // Neighbourhood statistics in YCoCg (5-tap cross) for variance clipping.
  let c0 = toYCoCg(textureLoad(t_cur, ic, 0).rgb);
  var m1 = c0;
  var m2 = c0 * c0;
  for (var i = 0; i < 4; i = i + 1) {
    let c = toYCoCg(textureLoad(t_cur, clamp(ic + crossOff[i], vec2<i32>(0), inMax), 0).rgb);
    m1 = m1 + c;
    m2 = m2 + c * c;
  }
  let mean = m1 / 5.0;
  let sigma = sqrt(max(m2 / 5.0 - mean * mean, vec3<f32>(0.0)));

  let cur = textureSampleLevel(t_cur, s_lin, uv + u.jitterUv, 0.0).rgb;

  // Depth-aware upsample of the half-resolution AO.
  var ao = 1.0;
  let aoOn = (flagBits & 2u) != 0u;
  if (aoOn) {
    let zc = linDepth(dCenter);
    let aoMax = vec2<i32>(u.aoSize) - vec2<i32>(1);
    let g = uv * u.aoSize - vec2<f32>(0.5);
    let base = vec2<i32>(floor(g));
    let f = fract(g);
    var acc = 0.0;
    var wsum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
      let o = vec2<i32>(j & 1, j >> 1u);
      let sm = textureLoad(t_ao, clamp(base + o, vec2<i32>(0), aoMax), 0);
      let bw = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
      let dw = 1.0 / (1e-3 + abs(sm.y - zc) / max(zc, 1e-3) * 20.0);
      let w = bw * dw;
      acc = acc + sm.x * w;
      wsum = wsum + w;
    }
    ao = select(1.0, acc / wsum, wsum > 1e-5);
  }

  let historyValid = (flagBits & 1u) != 0u;
  var outc = cur;
  var outAo = ao;
  if (historyValid) {
    // Moving objects carry their own motion; everything else reprojects
    // from depth with last frame's camera (exact for static geometry).
    var puv = vec2<f32>(-1.0);
    var pw = 1.0;
    var moving = false;
    if ((flagBits & 4u) != 0u) {
      let vel = textureLoad(t_vel, pClosest, 0).xy;
      if (vel.x > -100.0) {
        puv = uv - vel;
        moving = true;
      }
    }
    if (!moving) {
      let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, dClosest, 1.0);
      let wp = u.invVP * ndc;
      let world = vec4<f32>(wp.xyz / wp.w, 1.0);
      let pc = u.prevVP * world;
      pw = pc.w;
      puv = vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
    }
    if (pw > 0.0 && all(puv >= vec2<f32>(0.0)) && all(puv <= vec2<f32>(1.0))) {
      // Bilinear. A 5-tap Catmull-Rom fetch (sampleHistoryCR) measured no
      // visible gain on the proof-head face crops and is kept only for an
      // A/B; see results/renderer-benchmark/SCORECARD.md (2026-11-23 iq2).
      let h = textureSampleLevel(t_hist, s_lin, puv, 0.0);
      // Moving surfaces (a talking face) get a tighter clip: their history is
      // aligned by the motion vectors, so the box only needs to absorb
      // resampling error, and a tight box keeps lips and lids sharp.
      let k = select(1.25, 0.9, moving);
      let hc = fromYCoCg(clipAabb(toYCoCg(h.rgb), mean, sigma * k));
      let lc = dot(cur, vec3<f32>(0.2126, 0.7152, 0.0722));
      let lh = dot(hc, vec3<f32>(0.2126, 0.7152, 0.0722));
      let wc = u.feedback / (1.0 + lc);
      let wh = (1.0 - u.feedback) / (1.0 + lh);
      outc = (cur * wc + hc * wh) / (wc + wh);
      outAo = select(1.0, mix(h.a, ao, 0.2), aoOn);
    }
  }
  outc = max(outc, vec3<f32>(0.0));

  // Display path only (history stays unsharpened, unoccluded, linear):
  // luma sharpen against the current-frame neighbourhood mean, limited to one
  // standard deviation so edges cannot ring; then AO, bloom, tonemap, grade.
  // Overshoot limited to a quarter σ: a full σ left a dark halo around bright
  // silhouettes (the proof-room head) on the dark side of the edge.
  let dl = clamp((toYCoCg(outc).x - mean.x) * u.look1.w, -0.25 * sigma.x, 0.25 * sigma.x);
  var disp = max(outc + vec3<f32>(dl), vec3<f32>(0.0)) * outAo;
  if (u.look0.x > 0.0) {
    var b = textureSampleLevel(t_bloom, s_lin, uv, 0.0).rgb;
    if (u.look2.z > 0.0) { b = b * (1.0 + u.look2.z * fx_lens_dirt(uv)); }
    if (u.look2.x > 0.5) {
      // Physical bloom: energy-conserving mix toward the summed mip chain.
      disp = mix(disp, b / max(u.look2.y, 1.0), u.look0.x);
    } else {
      disp = disp + b * u.look0.x;
    }
  }
  var o: Out;
  o.history = vec4<f32>(outc, outAo);
  var shown = tonemap_display(disp * u.look0.y, u.look0.w, u.look0.z, u.look1.x, u.look1.y, u.look1.z);
  if (u.look2.w > 0.0) { shown = fx_grain(shown, frag.xy, u.look3.x, u.look2.w); }
  o.display = vec4<f32>(shown, 1.0);
  return o;
}
`;

export interface TemporalFrame {
  /** Current frame's UNJITTERED view-projection inverse. */
  invViewProj: Float32Array;
  /** Previous frame's UNJITTERED view-projection. */
  prevViewProj: Float32Array;
  /** Unjittered projection and its inverse (AO). */
  proj: Float32Array;
  invProj: Float32Array;
  jitterUv: [number, number];
  historyValid: boolean;
  frameIndex: number;
}

const TAA_UNIFORM_FLOATS = 60; // 240 bytes
const SSAO_UNIFORM_FLOATS = 40; // 160 bytes

/** Tonemap / grade values for the fused display output (from PostProcessingRunner). */
export interface TemporalLook {
  bloomIntensity: number;
  exposure: number;
  saturation: number;
  mode: number;
  contrast: number;
  vibrance: number;
  lift: number;
  /** 1 = physical (energy-conserving) bloom mix, 0 = additive. */
  bloomMode?: number;
  bloomLevels?: number;
  lensDirt?: number;
  filmGrain?: number;
  frame?: number;
}

/**
 * Owns the TAA history ping-pong, the half-res AO target, and their passes.
 * The renderer feeds it the internal-resolution HDR colour + depth each frame;
 * `resolve()` writes the new history and the final display colour.
 */
export class TemporalResolver {
  private device: GPUDevice;
  private sampler: GPUSampler;
  private taaPipeline: GPURenderPipeline;
  private aoPipeline: GPURenderPipeline;
  private aoBlurPipeline: GPURenderPipeline;
  private taaUniform: GPUBuffer;
  private aoUniform: GPUBuffer;
  private taaScratch = new Float32Array(TAA_UNIFORM_FLOATS);
  private aoScratch = new Float32Array(SSAO_UNIFORM_FLOATS);
  history: [GPUTexture | null, GPUTexture | null] = [null, null];
  historyViews: [GPUTextureView | null, GPUTextureView | null] = [null, null];
  private aoTexture: GPUTexture | null = null;
  private aoView: GPUTextureView | null = null;
  /** Blurred AO — what the resolve samples. */
  private aoBlurTexture: GPUTexture | null = null;
  private aoBlurView: GPUTextureView | null = null;
  private aoBlurBindGroup: GPUBindGroup | null = null;
  private dummyAo: GPUTexture;
  private dummyAoView: GPUTextureView;
  private dummyBloom: GPUTexture;
  private dummyBloomView: GPUTextureView;
  private dummyVel: GPUTexture;
  private dummyVelView: GPUTextureView;
  private taaBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [null, null];
  private aoBindGroup: GPUBindGroup | null = null;
  private boundSources: { cur: GPUTextureView; depth: GPUTextureView; bloom: GPUTextureView; vel: GPUTextureView } | null = null;
  outWidth = 0;
  outHeight = 0;
  inWidth = 0;
  inHeight = 0;
  aoWidth = 0;
  aoHeight = 0;
  /** Index of the history texture written last frame (read this frame). */
  private last = 0;

  /** Current-frame weight in the history blend (0.1 = 10% new). */
  feedback = 0.1;
  aoEnabled = false;
  aoSamples = 6;
  aoRadius = 0.6;
  aoIntensity = 1.0;
  /** Luma sharpen on the display output (0 = off). */
  sharpen = 0.25;

  constructor(device: GPUDevice, displayFormat: GPUTextureFormat) {
    this.device = device;
    this.sampler = device.createSampler({
      label: "taa/linear",
      minFilter: "linear",
      magFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    const taaModule = device.createShaderModule({ label: "taa/resolve", code: TAA_WGSL });
    this.taaPipeline = device.createRenderPipeline({
      label: "taa/resolve",
      layout: "auto",
      vertex: { module: taaModule, entryPoint: "vs_main" },
      fragment: { module: taaModule, entryPoint: "fs_main", targets: [{ format: "rgba16float" }, { format: displayFormat }] },
    });
    const aoModule = device.createShaderModule({ label: "ssao", code: SSAO_WGSL });
    this.aoPipeline = device.createRenderPipeline({
      label: "ssao",
      layout: "auto",
      vertex: { module: aoModule, entryPoint: "vs_main" },
      fragment: { module: aoModule, entryPoint: "fs_main", targets: [{ format: "rg16float" }] },
    });
    const blurModule = device.createShaderModule({ label: "ssao/blur", code: AO_BLUR_WGSL });
    this.aoBlurPipeline = device.createRenderPipeline({
      label: "ssao/blur",
      layout: "auto",
      vertex: { module: blurModule, entryPoint: "vs_main" },
      fragment: { module: blurModule, entryPoint: "fs_main", targets: [{ format: "rg16float" }] },
    });
    this.taaUniform = device.createBuffer({ label: "taa/uniform", size: TAA_UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aoUniform = device.createBuffer({ label: "ssao/uniform", size: SSAO_UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummyAo = device.createTexture({ label: "ssao/off", size: [1, 1], format: "rg16float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.dummyAoView = this.dummyAo.createView();
    this.dummyBloom = device.createTexture({ label: "taa/no-bloom", size: [1, 1], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.dummyBloomView = this.dummyBloom.createView();
    this.dummyVel = device.createTexture({ label: "taa/no-velocity", size: [1, 1], format: "rg16float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.dummyVelView = this.dummyVel.createView();
  }

  /** (Re)allocate targets. History is invalidated. */
  resize(outW: number, outH: number, inW: number, inH: number): void {
    if (this.outWidth === outW && this.outHeight === outH && this.inWidth === inW && this.inHeight === inH) return;
    this.outWidth = outW;
    this.outHeight = outH;
    this.inWidth = inW;
    this.inHeight = inH;
    this.aoWidth = Math.max(1, Math.ceil(inW / 2));
    this.aoHeight = Math.max(1, Math.ceil(inH / 2));
    this.destroyTargets();
    for (let i = 0; i < 2; i++) {
      const t = this.device.createTexture({
        label: `taa/history-${i}`,
        size: [outW, outH],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.history[i] = t;
      this.historyViews[i] = t.createView();
    }
    this.aoTexture = this.device.createTexture({
      label: "ssao/half",
      size: [this.aoWidth, this.aoHeight],
      format: "rg16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aoView = this.aoTexture.createView();
    this.aoBlurTexture = this.device.createTexture({
      label: "ssao/half-blurred",
      size: [this.aoWidth, this.aoHeight],
      format: "rg16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aoBlurView = this.aoBlurTexture.createView();
    this.aoBlurBindGroup = this.device.createBindGroup({
      label: "ssao/blur-bg",
      layout: this.aoBlurPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.aoView }],
    });
    this.boundSources = null;
  }

  private bind(cur: GPUTextureView, depth: GPUTextureView, bloom: GPUTextureView, vel: GPUTextureView): void {
    const b = this.boundSources;
    if (b && b.cur === cur && b.depth === depth && b.bloom === bloom && b.vel === vel) return;
    const aoSrc = this.aoBlurView ?? this.dummyAoView;
    for (let i = 0; i < 2; i++) {
      // Bind group i reads history[1 - i] and is used while writing history[i].
      this.taaBindGroups[i] = this.device.createBindGroup({
        label: `taa/bg-${i}`,
        layout: this.taaPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: cur },
          { binding: 1, resource: depth },
          { binding: 2, resource: this.historyViews[1 - i]! },
          { binding: 3, resource: aoSrc },
          { binding: 4, resource: this.sampler },
          { binding: 5, resource: { buffer: this.taaUniform } },
          { binding: 6, resource: bloom },
          { binding: 7, resource: vel },
        ],
      });
    }
    this.aoBindGroup = this.device.createBindGroup({
      label: "ssao/bg",
      layout: this.aoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: depth },
        { binding: 1, resource: { buffer: this.aoUniform } },
      ],
    });
    this.boundSources = { cur, depth, bloom, vel };
  }

  /**
   * Encode AO (when enabled) and the fused TAA resolve: writes the new
   * history and the tonemapped display colour into `displayView` (the canvas).
   * `bloom` is the half-res bloom result, or null when the chain has none.
   */
  resolve(
    encoder: GPUCommandEncoder,
    cur: GPUTextureView,
    depth: GPUTextureView,
    frame: TemporalFrame,
    displayView: GPUTextureView,
    bloom: GPUTextureView | null,
    look: TemporalLook,
    timing?: { ao?: GPURenderPassTimestampWrites; taa?: GPURenderPassTimestampWrites },
    velocity: GPUTextureView | null = null,
  ): boolean {
    if (!this.historyViews[0] || !this.historyViews[1] || !this.aoView) return false;
    this.bind(cur, depth, bloom ?? this.dummyBloomView, velocity ?? this.dummyVelView);
    if (this.aoEnabled) {
      const a = this.aoScratch;
      a.set(frame.invProj, 0);
      a.set(frame.proj, 16);
      a[32] = this.inWidth; a[33] = this.inHeight; a[34] = this.aoWidth; a[35] = this.aoHeight;
      a[36] = this.aoRadius; a[37] = this.aoIntensity; a[38] = this.aoSamples; a[39] = frame.frameIndex % 64;
      this.device.queue.writeBuffer(this.aoUniform, 0, a);
      const aoDesc: GPURenderPassDescriptor = {
        label: "ssao-pass",
        colorAttachments: [{ view: this.aoView, clearValue: { r: 1, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      };
      // The AO timer spans the raw pass + the blur.
      const raw = timing?.ao ? { querySet: timing.ao.querySet, beginningOfPassWriteIndex: timing.ao.beginningOfPassWriteIndex } : undefined;
      const end = timing?.ao ? { querySet: timing.ao.querySet, endOfPassWriteIndex: timing.ao.endOfPassWriteIndex } : undefined;
      if (raw) aoDesc.timestampWrites = raw as GPURenderPassTimestampWrites;
      const p = encoder.beginRenderPass(aoDesc);
      p.setPipeline(this.aoPipeline);
      p.setBindGroup(0, this.aoBindGroup!);
      p.draw(3);
      p.end();
      const blurDesc: GPURenderPassDescriptor = {
        label: "ssao-blur",
        colorAttachments: [{ view: this.aoBlurView!, clearValue: { r: 1, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      };
      if (end) blurDesc.timestampWrites = end as GPURenderPassTimestampWrites;
      const b = encoder.beginRenderPass(blurDesc);
      b.setPipeline(this.aoBlurPipeline);
      b.setBindGroup(0, this.aoBlurBindGroup!);
      b.draw(3);
      b.end();
    }
    const write = 1 - this.last;
    const t = this.taaScratch;
    t.set(frame.invViewProj, 0);
    t.set(frame.prevViewProj, 16);
    t[32] = frame.jitterUv[0]; t[33] = frame.jitterUv[1];
    t[34] = this.outWidth; t[35] = this.outHeight;
    t[36] = this.inWidth; t[37] = this.inHeight;
    t[38] = this.aoWidth; t[39] = this.aoHeight;
    t[40] = frame.proj[10]!; t[41] = frame.proj[14]!;
    t[42] = this.feedback;
    // flags: bit0 history valid, bit1 AO on, bit2 per-object velocity bound.
    t[43] = (frame.historyValid ? 1 : 0) + (this.aoEnabled ? 2 : 0) + (velocity ? 4 : 0);
    t[44] = bloom ? look.bloomIntensity : 0; t[45] = look.exposure; t[46] = look.saturation; t[47] = look.mode;
    t[48] = look.contrast; t[49] = look.vibrance; t[50] = look.lift; t[51] = this.sharpen;
    t[52] = look.bloomMode ?? 0; t[53] = look.bloomLevels ?? 1; t[54] = look.lensDirt ?? 0; t[55] = look.filmGrain ?? 0;
    t[56] = look.frame ?? 0; t[57] = 0; t[58] = 0; t[59] = 0;
    this.device.queue.writeBuffer(this.taaUniform, 0, t);
    const desc: GPURenderPassDescriptor = {
      label: "taa-resolve",
      colorAttachments: [
        { view: this.historyViews[write]!, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        { view: displayView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    };
    if (timing?.taa) desc.timestampWrites = timing.taa;
    const pass = encoder.beginRenderPass(desc);
    pass.setPipeline(this.taaPipeline);
    pass.setBindGroup(0, this.taaBindGroups[write]!);
    pass.draw(3);
    pass.end();
    this.last = write;
    return true;
  }

  private destroyTargets(): void {
    this.history[0]?.destroy();
    this.history[1]?.destroy();
    this.history = [null, null];
    this.historyViews = [null, null];
    this.aoTexture?.destroy();
    this.aoTexture = null;
    this.aoView = null;
    this.aoBlurTexture?.destroy();
    this.aoBlurTexture = null;
    this.aoBlurView = null;
    this.aoBlurBindGroup = null;
    this.taaBindGroups = [null, null];
    this.aoBindGroup = null;
  }

  destroy(): void {
    this.destroyTargets();
    this.taaUniform.destroy();
    this.aoUniform.destroy();
    this.dummyAo.destroy();
    this.dummyBloom.destroy();
    this.dummyVel.destroy();
  }
}

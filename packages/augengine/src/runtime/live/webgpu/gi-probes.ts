/**
 * Dynamic diffuse GI probes ("DDGI-lite") for the natural-light tier.
 *
 * Why probes and not SSGI (measured on Intel UHD, Loading Range 1440×900,
 * 2026-11-27): the existing half-res SSAO pass alone costs 0.88–1.26 ms
 * (6–12 samples) plus ~0.5 ms in the TAA resolve; screen-space GI needs that
 * plus colour gathering, ray marching and a denoiser, and it cannot see the
 * sunlit wall behind the camera, so it fails exactly when time of day moves
 * the light off screen. Probes cost per probe, not per pixel, and see the
 * whole scene.
 *
 * What it is (after Majercik et al., "Dynamic Diffuse Global Illumination
 * with Ray-Traced Irradiance Fields", JCGT 2019 — simplified):
 *   - A camera-centred window of probe COLUMNS × COLUMNS × 2 layers written
 *     into the renderer's existing probe ring (probe-ring.ts), so the
 *     material shaders sample them with no change (per-vertex trilinear L1).
 *   - Each update, every probe in this frame's subset traces RAYS rays in a
 *     per-frame-rotated spherical Fibonacci pattern against a simplified
 *     scene: up to MAX_GI_BOXES world AABBs of nearby meshes (with their base
 *     colour) and an infinite ground plane.
 *   - A hit's radiance = albedo × (sun illuminance × N·L × sun visibility
 *     (shadow ray) / π + irradiance from the probes themselves at the hit),
 *     so bounces accumulate over frames (infinite-bounce via history). A miss
 *     returns the sky (environment cube, sun excluded).
 *   - The rays are projected to L1 irradiance/π — k0 = mean(L),
 *     k = 2·mean(L·ω) (see gi-probes.test.ts) — and blended into history with
 *     hysteresis, so a moving sun re-converges in ~1/(alpha) updates.
 *   - Probes inside a proxy box are marked invalid; the sampler renormalises
 *     over the valid ones (cheap leak reduction).
 * Not included vs. full DDGI: per-probe depth (visibility) moments, octahedral
 * irradiance textures, probe relocation. Light can leak through thin walls
 * that do not contain a probe.
 */

import type { Vec3 } from "../../../scene/types.js";
import { PROBE_RING, PROBE_LAYERS } from "./probe-ring.js";

export const MAX_GI_BOXES = 256;
const BOX_FLOATS = 12;
const GI_UNIFORM_FLOATS = 24 + 36;

export interface GiSettings {
  /** Probe columns per axis around the camera (≤ PROBE_RING / 2). */
  columns: number;
  /** Rays per probe per update (32 or 64). */
  rays: number;
  /** Update 1 / stride of the probes each frame. */
  stride: number;
  /** Column spacing (m). */
  spacing: number;
  /** Height of layer 0 above the ground plane (m). */
  height: number;
  /** Spacing between the two layers (m). */
  layerSpacing: number;
  /** History weight of a new measurement (0.05–1). */
  alpha: number;
}

export const DEFAULT_GI: GiSettings = { columns: 12, rays: 32, stride: 2, spacing: 4, height: 1.2, layerSpacing: 4, alpha: 0.15 };

/** L1 irradiance/π coefficients [k0, kx, ky, kz] from radiance samples on uniformly distributed directions. */
export function projectL1(samples: ReadonlyArray<{ dir: Vec3; L: number }>): [number, number, number, number] {
  let k0 = 0, kx = 0, ky = 0, kz = 0;
  for (const s of samples) {
    k0 += s.L;
    kx += s.L * s.dir[0];
    ky += s.L * s.dir[1];
    kz += s.L * s.dir[2];
  }
  const n = Math.max(samples.length, 1);
  return [k0 / n, (2 * kx) / n, (2 * ky) / n, (2 * kz) / n];
}

/** Spherical Fibonacci direction i of n with a per-frame offset in [0, 1) (mirrors the WGSL). */
export function fibonacciDir(i: number, n: number, offset = 0): Vec3 {
  const y = 1 - (2 * ((i + 0.5) / n));
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const phi = 2 * Math.PI * (((i * 0.6180339887) + offset) % 1);
  return [Math.cos(phi) * r, y, Math.sin(phi) * r];
}

/** Ray vs AABB slab test: entry distance or -1. */
export function rayBox(o: Vec3, d: Vec3, mn: Vec3, mx: Vec3): number {
  let t0 = -Infinity, t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    const inv = 1 / (Math.abs(d[a]!) < 1e-12 ? 1e-12 : d[a]!);
    let ta = (mn[a]! - o[a]!) * inv, tb = (mx[a]! - o[a]!) * inv;
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta);
    t1 = Math.min(t1, tb);
  }
  return t1 >= Math.max(t0, 0) && t0 > 1e-4 ? t0 : -1;
}

const GI_WGSL = /* wgsl */ `
const RING: i32 = ${PROBE_RING};
const MAX_RAYS: u32 = 64u;
struct GiU {
  sunDir: vec4<f32>,   // xyz toward the sun, w = sun on
  sunE: vec4<f32>,     // rgb illuminance (lux), w = hysteresis alpha
  grid: vec4<f32>,     // x spacing, y layer-0 height, z layer spacing, w ground y
  win: vec4<f32>,      // x first column i, y first column j, z columns per axis, w frame
  counts: vec4<f32>,   // x boxes, y rays, z ground albedo, w update stride
  ground: vec4<f32>,   // rgb ground albedo colour
  sh: array<vec4<f32>, 9>,
};
@group(0) @binding(0) var<uniform> g: GiU;
@group(0) @binding(1) var<storage, read_write> probes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> boxes: array<vec4<f32>>;
@group(0) @binding(3) var t_env: texture_cube<f32>;
@group(0) @binding(4) var s_env: sampler;

var<workgroup> acc: array<vec4<f32>, 96>;

fn base_of(i: i32, j: i32, k: i32) -> u32 {
  let si = ((i % RING) + RING) % RING;
  let sj = ((j % RING) + RING) % RING;
  return u32((k * RING + sj) * RING + si) * 4u;
}

fn sky_irr(n: vec3<f32>) -> vec3<f32> {
  var r = g.sh[0].rgb + g.sh[1].rgb * n.y + g.sh[2].rgb * n.z + g.sh[3].rgb * n.x;
  r = r + g.sh[4].rgb * (n.x * n.y) + g.sh[5].rgb * (n.y * n.z) + g.sh[6].rgb * (3.0 * n.z * n.z - 1.0);
  r = r + g.sh[7].rgb * (n.x * n.z) + g.sh[8].rgb * (n.x * n.x - n.y * n.y);
  return max(r, vec3<f32>(0.0));
}

struct Hit { t: f32, n: vec3<f32>, albedo: vec3<f32> };

fn trace(o: vec3<f32>, d: vec3<f32>, maxT: f32) -> Hit {
  var h: Hit;
  h.t = maxT;
  h.n = vec3<f32>(0.0, 1.0, 0.0);
  h.albedo = vec3<f32>(0.0);
  if (d.y < -1e-5) {
    let tg = (g.grid.w - o.y) / d.y;
    if (tg > 1e-3 && tg < h.t) { h.t = tg; h.n = vec3<f32>(0.0, 1.0, 0.0); h.albedo = g.ground.rgb; }
  }
  let inv = 1.0 / select(d, vec3<f32>(1e-6), abs(d) < vec3<f32>(1e-6));
  let nb = u32(g.counts.x);
  for (var b = 0u; b < nb; b = b + 1u) {
    let mn = boxes[b * 3u];
    let mx = boxes[b * 3u + 1u];
    let ta = (mn.xyz - o) * inv;
    let tb = (mx.xyz - o) * inv;
    let tmin = min(ta, tb);
    let tmax = max(ta, tb);
    let t0 = max(max(tmin.x, tmin.y), tmin.z);
    let t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 >= t0 && t0 > 1e-3 && t0 < h.t) {
      h.t = t0;
      // Entry face: the axis whose slab entry is the last one.
      if (t0 == tmin.x) { h.n = vec3<f32>(-sign(d.x), 0.0, 0.0); }
      else if (t0 == tmin.y) { h.n = vec3<f32>(0.0, -sign(d.y), 0.0); }
      else { h.n = vec3<f32>(0.0, 0.0, -sign(d.z)); }
      h.albedo = vec3<f32>(mn.w, mx.w, boxes[b * 3u + 2u].x);
    }
  }
  return h;
}

fn occluded(o: vec3<f32>, d: vec3<f32>) -> bool {
  let inv = 1.0 / select(d, vec3<f32>(1e-6), abs(d) < vec3<f32>(1e-6));
  let nb = u32(g.counts.x);
  for (var b = 0u; b < nb; b = b + 1u) {
    let ta = (boxes[b * 3u].xyz - o) * inv;
    let tb = (boxes[b * 3u + 1u].xyz - o) * inv;
    let tmin = min(ta, tb);
    let tmax = max(ta, tb);
    let t0 = max(max(tmin.x, tmin.y), tmin.z);
    let t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 >= max(t0, 0.0) && t0 > 1e-3) { return true; }
  }
  return false;
}

// Irradiance/π from the probe field at P (previous values — the bounce).
fn field(P: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
  let gx = P.x / g.grid.x - 0.5;
  let gz = P.z / g.grid.x - 0.5;
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gz));
  let fx = gx - floor(gx);
  let fz = gz - floor(gz);
  let gy = clamp((P.y - g.grid.w - g.grid.y) / g.grid.z, 0.0, 1.0);
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var c = 0; c < 8; c = c + 1) {
    let di = c & 1;
    let dj = (c >> 1u) & 1;
    let k = c >> 2u;
    let b = base_of(i0 + di, j0 + dj, k);
    let m = probes[b + 3u];
    if (!(m.w > 0.5 && i32(round(m.x)) == i0 + di && i32(round(m.y)) == j0 + dj)) { continue; }
    let w = select(1.0 - fx, fx, di == 1) * select(1.0 - fz, fz, dj == 1) * select(1.0 - gy, gy, k == 1);
    let n4 = vec4<f32>(1.0, n);
    sum = sum + w * max(vec3<f32>(dot(probes[b], n4), dot(probes[b + 1u], n4), dot(probes[b + 2u], n4)), vec3<f32>(0.0));
    wsum = wsum + w;
  }
  if (wsum < 1e-3) { return vec4<f32>(sky_irr(n) * 0.6, 0.0); }
  return vec4<f32>(sum / wsum, 1.0);
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let cols = i32(g.win.z);
  let stride = u32(g.counts.w);
  let phase = u32(g.win.w) % stride;
  let pIndex = i32(wg.x * stride + phase);
  if (pIndex >= cols * cols * 2) { return; }
  let k = pIndex / (cols * cols);
  let rem = pIndex % (cols * cols);
  let gi = i32(g.win.x) + rem % cols;
  let gj = i32(g.win.y) + rem / cols;
  let P = vec3<f32>((f32(gi) + 0.5) * g.grid.x, g.grid.w + g.grid.y + f32(k) * g.grid.z, (f32(gj) + 0.5) * g.grid.x);

  // Probe inside a proxy box: invalid (the sampler skips it).
  var inside = false;
  let nb = u32(g.counts.x);
  for (var b = 0u; b < nb; b = b + 1u) {
    let mn = boxes[b * 3u].xyz;
    let mx = boxes[b * 3u + 1u].xyz;
    if (all(P > mn) && all(P < mx)) { inside = true; }
  }

  let rays = u32(g.counts.y);
  let frameOff = fract(g.win.w * 0.7548776662);
  var r = vec4<f32>(0.0);
  var gg = vec4<f32>(0.0);
  var bb = vec4<f32>(0.0);
  if (!inside) {
    for (var ri = li; ri < rays; ri = ri + 32u) {
      let y = 1.0 - 2.0 * ((f32(ri) + 0.5) / f32(rays));
      let rr = sqrt(max(0.0, 1.0 - y * y));
      let phi = 6.28318530718 * fract(f32(ri) * 0.6180339887 + frameOff);
      let d = vec3<f32>(cos(phi) * rr, y, sin(phi) * rr);
      let h = trace(P, d, 200.0);
      var L: vec3<f32>;
      if (h.t >= 200.0) {
        L = textureSampleLevel(t_env, s_env, d, 2.0).rgb;
      } else {
        let hp = P + d * h.t + h.n * 0.02;
        var direct = vec3<f32>(0.0);
        let ndl = dot(h.n, g.sunDir.xyz);
        if (g.sunDir.w > 0.5 && ndl > 0.0 && !occluded(hp, g.sunDir.xyz)) {
          direct = g.sunE.rgb * ndl / 3.14159265;
        }
        L = h.albedo * (direct + field(hp, h.n).rgb);
      }
      r = r + vec4<f32>(L.r, L.r * d.x, L.r * d.y, L.r * d.z);
      gg = gg + vec4<f32>(L.g, L.g * d.x, L.g * d.y, L.g * d.z);
      bb = bb + vec4<f32>(L.b, L.b * d.x, L.b * d.y, L.b * d.z);
    }
  }
  acc[li * 3u] = r;
  acc[li * 3u + 1u] = gg;
  acc[li * 3u + 2u] = bb;
  workgroupBarrier();
  if (li != 0u) { return; }
  let base = base_of(gi, gj, k);
  if (inside) {
    probes[base + 3u] = vec4<f32>(f32(gi), f32(gj), g.grid.w, 0.0);
    return;
  }
  var sr = vec4<f32>(0.0);
  var sg = vec4<f32>(0.0);
  var sb = vec4<f32>(0.0);
  for (var t = 0u; t < 32u; t = t + 1u) {
    sr = sr + acc[t * 3u];
    sg = sg + acc[t * 3u + 1u];
    sb = sb + acc[t * 3u + 2u];
  }
  // L1 irradiance/π: k0 = mean(L), k = 2·mean(L·ω).
  let s = vec4<f32>(1.0, 2.0, 2.0, 2.0) / f32(rays);
  let m = probes[base + 3u];
  let valid = m.w > 0.5 && i32(round(m.x)) == gi && i32(round(m.y)) == gj;
  let a = select(1.0, g.sunE.w, valid);
  probes[base] = mix(probes[base], sr * s, a);
  probes[base + 1u] = mix(probes[base + 1u], sg * s, a);
  probes[base + 2u] = mix(probes[base + 2u], sb * s, a);
  probes[base + 3u] = vec4<f32>(f32(gi), f32(gj), g.grid.w, 1.0);
}
`;

export interface GiBox { min: Vec3; max: Vec3; albedo: Vec3 }

export class GiProbes {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private uniform: GPUBuffer;
  private boxBuf: GPUBuffer;
  private boxData = new Float32Array(MAX_GI_BOXES * BOX_FLOATS);
  private boxUploaded = new Float32Array(MAX_GI_BOXES * BOX_FLOATS);
  private boxCount = 0;
  private scratch = new Float32Array(GI_UNIFORM_FLOATS);
  private bindGroup: GPUBindGroup | null = null;
  private bound: { ring: GPUBuffer; env: GPUTextureView } | null = null;
  settings: GiSettings = { ...DEFAULT_GI };
  frame = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      label: "gi/probes",
      layout: "auto",
      compute: { module: device.createShaderModule({ label: "gi/probes", code: GI_WGSL }), entryPoint: "main" },
    });
    this.uniform = device.createBuffer({ label: "gi/uniform", size: GI_UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.boxBuf = device.createBuffer({ label: "gi/boxes", size: MAX_GI_BOXES * BOX_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  configure(s: Partial<GiSettings>): void {
    const n = { ...this.settings, ...s };
    n.columns = Math.max(2, Math.min(PROBE_RING / 2, Math.round(n.columns)));
    n.rays = Math.max(8, Math.min(64, Math.round(n.rays)));
    n.stride = Math.max(1, Math.min(8, Math.round(n.stride)));
    n.alpha = Math.max(0.02, Math.min(1, n.alpha));
    this.settings = n;
  }

  /** Upload the proxy boxes (only when they changed). */
  setBoxes(list: readonly GiBox[]): void {
    const n = Math.min(list.length, MAX_GI_BOXES);
    const d = this.boxData;
    for (let i = 0; i < n; i++) {
      const b = list[i]!;
      const o = i * BOX_FLOATS;
      d[o] = b.min[0]; d[o + 1] = b.min[1]; d[o + 2] = b.min[2]; d[o + 3] = b.albedo[0];
      d[o + 4] = b.max[0]; d[o + 5] = b.max[1]; d[o + 6] = b.max[2]; d[o + 7] = b.albedo[1];
      d[o + 8] = b.albedo[2]; d[o + 9] = 0; d[o + 10] = 0; d[o + 11] = 0;
    }
    const floats = n * BOX_FLOATS;
    let same = n === this.boxCount;
    for (let i = 0; same && i < floats; i++) if (d[i] !== this.boxUploaded[i]) same = false;
    this.boxCount = n;
    if (!same && floats > 0) {
      this.boxUploaded.set(d.subarray(0, floats));
      this.device.queue.writeBuffer(this.boxBuf, 0, d as unknown as BufferSource, 0, floats);
    }
  }

  /** First probe column (i, j) of the window centred on `center`. */
  windowOrigin(center: Vec3): [number, number] {
    const s = this.settings;
    const half = Math.floor(s.columns / 2);
    return [Math.floor(center[0] / s.spacing) - half, Math.floor(center[2] / s.spacing) - half];
  }

  encode(
    encoder: GPUCommandEncoder,
    ring: GPUBuffer,
    env: GPUTextureView,
    envSampler: GPUSampler,
    p: { center: Vec3; groundY: number; groundAlbedo: Vec3; sunDir: Vec3; sunIlluminance: Vec3; sh: Float32Array },
    timing?: GPUComputePassTimestampWrites,
  ): void {
    const s = this.settings;
    if (!this.bindGroup || !this.bound || this.bound.ring !== ring || this.bound.env !== env) {
      this.bindGroup = this.device.createBindGroup({
        label: "gi/probes",
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: ring } },
          { binding: 2, resource: { buffer: this.boxBuf } },
          { binding: 3, resource: env },
          { binding: 4, resource: envSampler },
        ],
      });
      this.bound = { ring, env };
    }
    const [wi, wj] = this.windowOrigin(p.center);
    const sunOn = luminanceOf(p.sunIlluminance) > 0 ? 1 : 0;
    const u = this.scratch;
    u.set([p.sunDir[0], p.sunDir[1], p.sunDir[2], sunOn], 0);
    u.set([p.sunIlluminance[0], p.sunIlluminance[1], p.sunIlluminance[2], s.alpha], 4);
    u.set([s.spacing, s.height, s.layerSpacing, p.groundY], 8);
    u.set([wi, wj, s.columns, this.frame], 12);
    u.set([this.boxCount, s.rays, 0, s.stride], 16);
    u.set([p.groundAlbedo[0], p.groundAlbedo[1], p.groundAlbedo[2], 0], 20);
    u.set(p.sh.subarray(0, 36), 24);
    this.device.queue.writeBuffer(this.uniform, 0, u as unknown as BufferSource);
    const pass = encoder.beginComputePass(timing ? { label: "gi-probes", timestampWrites: timing } : { label: "gi-probes" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const probes = s.columns * s.columns * PROBE_LAYERS;
    pass.dispatchWorkgroups(Math.ceil(probes / s.stride));
    pass.end();
    this.frame++;
  }

  destroy(): void {
    this.uniform.destroy();
    this.boxBuf.destroy();
  }
}

function luminanceOf(c: Vec3): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

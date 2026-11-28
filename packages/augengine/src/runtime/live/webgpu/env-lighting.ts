/**
 * Environment lighting bake: the physical sky (or an HDR equirect image) →
 * a radiance cubemap → a GGX-prefiltered specular cubemap, plus the diffuse
 * irradiance as order-2 SH (sh9.ts, computed on the CPU from the same source).
 *
 * Pipeline (all compute, re-run only when the sun/sky/image changes):
 *   1. source → `radiance` cube mip 0 (sky: SKY_MODEL_WGSL per texel without
 *      the sun disc — the sun is the key light; HDR: equirect sample)
 *   2. box-downsample `radiance` to a full mip chain
 *   3. `specular` mip 0 = copy of radiance mip 0 (mirror); mip m > 0 is the
 *      split-sum prefiltered radiance for roughness m / (mips − 1): GGX
 *      importance sampling with N = V = R (Karis 2013) and filtered importance
 *      sampling — each sample reads the radiance mip whose texel solid angle
 *      matches the sample's PDF (Colbert & Křivánek, GPU Gems 3 ch. 20) — so
 *      16–64 samples are noise-free.
 * Materials read `specular` at lod = roughness × (mips − 1).
 *
 * Cube face orientation is the GL/Vulkan/D3D table (the one hardware uses to
 * sample a cube), so write and read agree regardless of world handedness.
 */

import type { Vec3 } from "../../../scene/types.js";
import { SKY_MODEL_FLOATS, SKY_MODEL_WGSL, packSkyModel, skyRadianceAt, type SkyModel } from "./sky-model.js";
import { convolveSh9Diffuse, projectEquirectSh9, projectSh9, scaleSh9, sh9Zero, type Sh9 } from "./sh9.js";

export interface EnvBakeSettings {
  /** Cube face size of mip 0 (power of two, 16–512). */
  size: number;
  /** GGX samples per prefiltered texel. */
  samples: number;
}

const CUBE_DIR_WGSL = /* wgsl */ `
// Direction through texel centre (x, y) of a cube face of the given size.
fn cube_dir(face: u32, x: u32, y: u32, size: u32) -> vec3<f32> {
  let u = (f32(x) + 0.5) / f32(size) * 2.0 - 1.0;
  let v = (f32(y) + 0.5) / f32(size) * 2.0 - 1.0;
  var d: vec3<f32>;
  switch (face) {
    case 0u: { d = vec3<f32>(1.0, -v, -u); }
    case 1u: { d = vec3<f32>(-1.0, -v, u); }
    case 2u: { d = vec3<f32>(u, 1.0, v); }
    case 3u: { d = vec3<f32>(u, -1.0, -v); }
    case 4u: { d = vec3<f32>(u, -v, 1.0); }
    default: { d = vec3<f32>(-u, -v, -1.0); }
  }
  return normalize(d);
}
`;

const SKY_TO_CUBE_WGSL = /* wgsl */ `
${SKY_MODEL_WGSL}
@group(0) @binding(0) var<uniform> sky: SkyModelU;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
${CUBE_DIR_WGSL}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst).x;
  if (id.x >= size || id.y >= size) { return; }
  let d = cube_dir(id.z, id.x, id.y, size);
  // fp16 storage: keep below its 65504 max.
  let L = min(sky_radiance(d) * sky.ground.w, vec3<f32>(60000.0));
  textureStore(dst, vec2<i32>(id.xy), i32(id.z), vec4<f32>(L, 1.0));
}
`;

const EQUIRECT_TO_CUBE_WGSL = /* wgsl */ `
struct P { rotation: f32, intensity: f32, _a: f32, _b: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
${CUBE_DIR_WGSL}
fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
  let phi = atan2(d.x, -d.z) - p.rotation;
  return vec2<f32>(fract(phi / 6.28318530718 + 0.5), acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359);
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst).x;
  if (id.x >= size || id.y >= size) { return; }
  // The equirect has no mips: a 4×4 supersample per cube texel keeps a large
  // image (and a small, very bright sun in it) from aliasing into the cube.
  var L = vec3<f32>(0.0);
  for (var j = 0u; j < 4u; j = j + 1u) {
    for (var i = 0u; i < 4u; i = i + 1u) {
      let u = (f32(id.x) + (f32(i) + 0.5) / 4.0) / f32(size) * 2.0 - 1.0;
      let v = (f32(id.y) + (f32(j) + 0.5) / 4.0) / f32(size) * 2.0 - 1.0;
      var d: vec3<f32>;
      switch (id.z) {
        case 0u: { d = vec3<f32>(1.0, -v, -u); }
        case 1u: { d = vec3<f32>(-1.0, -v, u); }
        case 2u: { d = vec3<f32>(u, 1.0, v); }
        case 3u: { d = vec3<f32>(u, -1.0, -v); }
        case 4u: { d = vec3<f32>(u, -v, 1.0); }
        default: { d = vec3<f32>(-u, -v, -1.0); }
      }
      L = L + textureSampleLevel(src, smp, equirect_uv(normalize(d)), 0.0).rgb;
    }
  }
  L = L / 16.0 * p.intensity;
  textureStore(dst, vec2<i32>(id.xy), i32(id.z), vec4<f32>(min(L, vec3<f32>(60000.0)), 1.0));
}
`;

const DOWNSAMPLE_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst).x;
  if (id.x >= size || id.y >= size) { return; }
  let b = vec2<i32>(id.xy) * 2;
  let f = i32(id.z);
  let c = textureLoad(src, b, f, 0) + textureLoad(src, b + vec2<i32>(1, 0), f, 0)
        + textureLoad(src, b + vec2<i32>(0, 1), f, 0) + textureLoad(src, b + vec2<i32>(1, 1), f, 0);
  textureStore(dst, vec2<i32>(id.xy), f, c * 0.25);
}
`;

const PREFILTER_WGSL = /* wgsl */ `
struct P { roughness: f32, samples: f32, srcSize: f32, srcMips: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var src: texture_cube<f32>;
@group(0) @binding(3) var smp: sampler;
${CUBE_DIR_WGSL}
fn radical_inverse(bitsIn: u32) -> f32 {
  var bits = bitsIn;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst).x;
  if (id.x >= size || id.y >= size) { return; }
  let N = cube_dir(id.z, id.x, id.y, size);
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.y) > 0.999);
  let T = normalize(cross(up, N));
  let B = cross(N, T);
  let a = max(p.roughness * p.roughness, 1e-4);
  let a2 = a * a;
  let n = u32(p.samples);
  // Solid angle of one source texel at mip 0.
  let omegaP = 4.0 * 3.14159265 / (6.0 * p.srcSize * p.srcSize);
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < n; i = i + 1u) {
    let xi = vec2<f32>((f32(i) + 0.5) / f32(n), radical_inverse(i));
    let phi = 6.28318530718 * xi.x;
    let cosT = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    let H = T * (sinT * cos(phi)) + B * (sinT * sin(phi)) + N * cosT;
    let L = normalize(2.0 * dot(N, H) * H - N);
    let NdotL = dot(N, L);
    if (NdotL <= 0.0) { continue; }
    // pdf(L) with N = V: D(h)·(n·h) / (4 (v·h)) = D / 4.
    let d = cosT * cosT * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159265 * d * d);
    let omegaS = 1.0 / (f32(n) * D * 0.25 + 1e-6);
    let lod = clamp(0.5 * log2(omegaS / omegaP) + 1.0, 0.0, p.srcMips - 1.0);
    acc = acc + textureSampleLevel(src, smp, L, lod).rgb * NdotL;
    wsum = wsum + NdotL;
  }
  textureStore(dst, vec2<i32>(id.xy), i32(id.z), vec4<f32>(acc / max(wsum, 1e-6), 1.0));
}
`;

/** IEEE half-float bits for a float (round-to-nearest, clamps to ±65504). */
export function toHalf(v: number): number {
  if (Number.isNaN(v)) return 0x7e00;
  const f = Math.max(-65504, Math.min(65504, v));
  const sign = f < 0 ? 0x8000 : 0;
  const a = Math.abs(f);
  if (a < 6.103515625e-5) return sign | Math.round(a / 5.960464477539063e-8); // subnormal
  let e = Math.floor(Math.log2(a));
  let m = a / Math.pow(2, e) - 1;
  let mi = Math.round(m * 1024);
  if (mi === 1024) { mi = 0; e += 1; }
  return sign | ((e + 15) << 10) | mi;
}

/** RGB(A) float image → RGBA half-float texels. */
export function toHalfRgba(data: Float32Array, width: number, height: number, channels: number): Uint16Array {
  const out = new Uint16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    out[i * 4] = toHalf(data[i * channels]!);
    out[i * 4 + 1] = toHalf(data[i * channels + 1]!);
    out[i * 4 + 2] = toHalf(data[i * channels + 2]!);
    out[i * 4 + 3] = 0x3c00;
  }
  return out;
}

export interface EquirectImage {
  width: number;
  height: number;
  /** Linear radiance, `channels` floats per pixel, row 0 = zenith. */
  data: Float32Array;
  channels: number;
}

export class EnvironmentLighting {
  readonly device: GPUDevice;
  settings: EnvBakeSettings;
  private radiance!: GPUTexture;
  private specular!: GPUTexture;
  /** Prefiltered specular cube (what materials sample). Recreated only on resize. */
  specularView!: GPUTextureView;
  /** Radiance cube with its mip chain (for the background / debug). */
  radianceView!: GPUTextureView;
  specularMips = 1;
  private radianceMips = 1;
  readonly sampler: GPUSampler;
  private skyPipeline: GPUComputePipeline;
  private equirectPipeline: GPUComputePipeline;
  private downsamplePipeline: GPUComputePipeline;
  private prefilterPipeline: GPUComputePipeline;
  private skyUniform: GPUBuffer;
  private skyScratch = new Float32Array(SKY_MODEL_FLOATS);
  private equirectUniform: GPUBuffer;
  private prefilterUniforms: GPUBuffer[] = [];
  /** Equirect source on the GPU (rgba16float) — also sampled by the background pass. */
  equirectTexture: GPUTexture | null = null;
  equirectView: GPUTextureView | null = null;
  private dummyEquirect: GPUTexture;
  readonly dummyEquirectView: GPUTextureView;
  /** Diffuse irradiance / π, SH9 (sh9.ts). */
  sh: Sh9 = sh9Zero();
  /** Bumps whenever the bound views change (resize / new equirect). */
  generation = 0;
  /** Bumps on every bake. */
  bakes = 0;

  constructor(device: GPUDevice, settings: EnvBakeSettings) {
    this.device = device;
    this.settings = { ...settings };
    this.sampler = device.createSampler({
      label: "env/linear",
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "clamp-to-edge",
    });
    const mk = (label: string, code: string) => device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    this.skyPipeline = mk("env/sky-to-cube", SKY_TO_CUBE_WGSL);
    this.equirectPipeline = mk("env/equirect-to-cube", EQUIRECT_TO_CUBE_WGSL);
    this.downsamplePipeline = mk("env/downsample", DOWNSAMPLE_WGSL);
    this.prefilterPipeline = mk("env/prefilter", PREFILTER_WGSL);
    this.skyUniform = device.createBuffer({ label: "env/sky", size: SKY_MODEL_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.equirectUniform = device.createBuffer({ label: "env/equirect", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummyEquirect = device.createTexture({ label: "env/no-equirect", size: [1, 1], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.dummyEquirectView = this.dummyEquirect.createView();
    this.allocate();
  }

  /** Change the cube size / sample count (re-allocates; caller re-bakes). */
  configure(settings: Partial<EnvBakeSettings>): void {
    const next = { ...this.settings, ...settings };
    next.size = Math.min(512, Math.max(16, 2 ** Math.round(Math.log2(next.size))));
    next.samples = Math.min(256, Math.max(4, Math.round(next.samples)));
    const resize = next.size !== this.settings.size;
    this.settings = next;
    if (resize) {
      this.radiance.destroy();
      this.specular.destroy();
      for (const b of this.prefilterUniforms) b.destroy();
      this.prefilterUniforms = [];
      this.allocate();
    }
  }

  private allocate(): void {
    const size = this.settings.size;
    // Radiance: full chain down to 4×4 (the prefilter reads coarse mips for wide lobes).
    this.radianceMips = Math.max(1, Math.log2(size) - 1);
    // Specular: roughness 0 … 1 over mips down to 8×8.
    this.specularMips = Math.max(2, Math.log2(size) - 2);
    this.radiance = this.device.createTexture({
      label: `env/radiance-${size}`,
      size: [size, size, 6],
      format: "rgba16float",
      mipLevelCount: this.radianceMips,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.specular = this.device.createTexture({
      label: `env/specular-${size}`,
      size: [size, size, 6],
      format: "rgba16float",
      mipLevelCount: this.specularMips,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.radianceView = this.radiance.createView({ dimension: "cube" });
    this.specularView = this.specular.createView({ dimension: "cube" });
    for (let m = 0; m < this.specularMips; m++) {
      this.prefilterUniforms.push(this.device.createBuffer({ label: `env/prefilter-${m}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    this.generation++;
  }

  private storageView(tex: GPUTexture, mip: number): GPUTextureView {
    return tex.createView({ dimension: "2d-array", baseMipLevel: mip, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 6 });
  }

  /** Upload an HDR equirect for bakeEquirect() and the background pass. */
  setEquirect(img: EquirectImage | null): void {
    this.equirectTexture?.destroy();
    this.equirectTexture = null;
    this.equirectView = null;
    if (img) {
      const tex = this.device.createTexture({
        label: `env/equirect-${img.width}x${img.height}`,
        size: [img.width, img.height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const half = toHalfRgba(img.data, img.width, img.height, img.channels);
      this.device.queue.writeTexture({ texture: tex }, half as unknown as BufferSource, { bytesPerRow: img.width * 8 }, [img.width, img.height]);
      this.equirectTexture = tex;
      this.equirectView = tex.createView();
    }
    this.generation++;
  }

  /**
   * Bake from the analytic sky. Diffuse SH comes from the same model on the
   * CPU (sun disc excluded), scaled by `intensity`.
   */
  bakeSky(encoder: GPUCommandEncoder, model: SkyModel, intensity = 1, shSamples = 1024, timing?: GPUComputePassTimestampWrites): void {
    packSkyModel(this.skyScratch, 0, model, [1, 1, 1], 0, intensity);
    this.device.queue.writeBuffer(this.skyUniform, 0, this.skyScratch);
    const pass = encoder.beginComputePass(timing ? { label: "env-bake", timestampWrites: timing } : { label: "env-bake" });
    const size = this.settings.size;
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.skyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.skyUniform } },
        { binding: 1, resource: this.storageView(this.radiance, 0) },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), 6);
    this.encodeChain(pass);
    pass.end();
    this.copyMirrorAndPrefilter(encoder);
    this.sh = scaleSh9(convolveSh9Diffuse(projectSh9((d) => skyRadianceAt(model, d), shSamples)), intensity);
    this.bakes++;
  }

  /** Bake from the equirect set with setEquirect(). SH from the CPU image. */
  bakeEquirect(encoder: GPUCommandEncoder, img: EquirectImage, intensity = 1, rotation = 0, timing?: GPUComputePassTimestampWrites): void {
    if (!this.equirectView) this.setEquirect(img);
    this.device.queue.writeBuffer(this.equirectUniform, 0, new Float32Array([rotation, intensity, 0, 0]));
    const pass = encoder.beginComputePass(timing ? { label: "env-bake", timestampWrites: timing } : { label: "env-bake" });
    const size = this.settings.size;
    pass.setPipeline(this.equirectPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.equirectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.equirectUniform } },
        { binding: 1, resource: this.storageView(this.radiance, 0) },
        { binding: 2, resource: this.equirectView! },
        { binding: 3, resource: this.sampler },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), 6);
    this.encodeChain(pass);
    pass.end();
    this.copyMirrorAndPrefilter(encoder);
    const step = Math.max(1, Math.floor(img.width / 512));
    this.sh = scaleSh9(convolveSh9Diffuse(projectEquirectSh9(img.data, img.width, img.height, img.channels, step, rotation)), intensity);
    this.bakes++;
  }

  private encodeChain(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.downsamplePipeline);
    for (let m = 1; m < this.radianceMips; m++) {
      const size = Math.max(1, this.settings.size >> m);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.radiance.createView({ dimension: "2d-array", baseMipLevel: m - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.storageView(this.radiance, m) },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), 6);
    }
  }

  private copyMirrorAndPrefilter(encoder: GPUCommandEncoder): void {
    const size = this.settings.size;
    encoder.copyTextureToTexture(
      { texture: this.radiance, mipLevel: 0 },
      { texture: this.specular, mipLevel: 0 },
      [size, size, 6],
    );
    const pass = encoder.beginComputePass({ label: "env-prefilter" });
    pass.setPipeline(this.prefilterPipeline);
    for (let m = 1; m < this.specularMips; m++) {
      const roughness = m / (this.specularMips - 1);
      this.device.queue.writeBuffer(this.prefilterUniforms[m]!, 0, new Float32Array([roughness, this.settings.samples, size, this.radianceMips]));
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: this.prefilterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.prefilterUniforms[m]! } },
          { binding: 1, resource: this.storageView(this.specular, m) },
          { binding: 2, resource: this.radianceView },
          { binding: 3, resource: this.sampler },
        ],
      }));
      const s = Math.max(1, size >> m);
      pass.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 6);
    }
    pass.end();
  }

  /** Approximate radiance (cd/m²) toward the horizon, averaged over azimuth — the aerial-perspective / fog colour. */
  static horizonRadiance(model: SkyModel, elevationDeg = 4): Vec3 {
    const acc: [number, number, number] = [0, 0, 0];
    const el = (elevationDeg * Math.PI) / 180;
    const n = 16;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * 2 * Math.PI;
      const L = skyRadianceAt(model, [Math.cos(el) * Math.cos(a), Math.sin(el), Math.cos(el) * Math.sin(a)]);
      acc[0] += L[0] / n; acc[1] += L[1] / n; acc[2] += L[2] / n;
    }
    return acc;
  }

  destroy(): void {
    this.radiance.destroy();
    this.specular.destroy();
    this.skyUniform.destroy();
    this.equirectUniform.destroy();
    for (const b of this.prefilterUniforms) b.destroy();
    this.equirectTexture?.destroy();
    this.dummyEquirect.destroy();
  }
}

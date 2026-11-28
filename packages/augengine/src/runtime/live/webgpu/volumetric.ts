/**
 * Volumetric sun shafts ("god rays") for the natural-light tier.
 *
 * Not froxels: a per-pixel ray march from the camera to the depth buffer
 * (capped at `maxDistance`) through the key light's shadow map, which is
 * what a froxel grid would integrate, without the 3D texture. Each of
 * `steps` samples (jittered by interleaved gradient noise that changes per
 * frame, so TAA integrates them) tests sun visibility in the near cascade;
 * the visible length × transmittance gives single-scattered sun radiance
 *   L = E_sun · p_HG(θ) · σ_s · Σ T(s) · vis(s) · Δs
 * marched at HALF resolution into its own target, then added to the HDR
 * target (bilinear upsample, additive blend) before tonemapping, pre-exposed
 * like everything else. Full resolution cost 7.3 ms for 12 steps on Intel
 * UHD (2048² cascades); half resolution is a quarter of the rays. Needs the
 * single-sample depth buffer (TAA / 1× MSAA); skipped otherwise. The aerial-perspective fog in the material
 * shaders provides the unshadowed haze; this pass adds only the shadowed
 * shafts' sunlit share.
 */

import { m4Invert } from "../../../scene/math.js";
import type { Mat4, Vec3 } from "../../../scene/types.js";

const UNIFORM_FLOATS = 16 + 16 + 4 + 4 + 4 + 4 + 4; // invVP, shadowVP, cam, sunDir, sunE, params, params2

const VOLUMETRIC_WGSL = /* wgsl */ `
struct VolU {
  invViewProj: mat4x4<f32>,
  shadowViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  sunDir: vec4<f32>,     // xyz toward the sun
  sunE: vec4<f32>,       // rgb sun illuminance × pre-exposure × white balance
  params: vec4<f32>,     // x density σ (1/m), y HG g, z steps, w frame
  params2: vec4<f32>,    // x max distance (m), y shadow depth bias, z cascades (1/2), w unused
};
@group(0) @binding(0) var<uniform> u: VolU;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_shadow: texture_depth_2d;
@group(0) @binding(3) var s_shadow: sampler_comparison;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

fn sun_visible(p: vec3<f32>) -> f32 {
  let lp = u.shadowViewProj * vec4<f32>(p, 1.0);
  let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
  if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || lp.z >= 1.0) { return 1.0; }
  return textureSampleCompareLevel(t_shadow, s_shadow, vec2<f32>(uv.x / u.params2.z, uv.y), lp.z - u.params2.y);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  // Half-resolution target: each texel covers 2×2 depth texels.
  let dims = vec2<f32>(textureDimensions(t_depth));
  let full = min(frag.xy * 2.0, dims - vec2<f32>(1.0));
  let d = textureLoad(t_depth, vec2<i32>(full), 0);
  let uv = (frag.xy * 2.0) / dims;
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let farH = u.invViewProj * vec4<f32>(ndc, min(d, 0.999999), 1.0);
  let endP = farH.xyz / farH.w;
  let toEnd = endP - u.cameraPos.xyz;
  let dist = min(length(toEnd), u.params2.x);
  let dir = toEnd / max(length(toEnd), 1e-4);
  let steps = u.params.z;
  let ds = dist / steps;
  let jitter = fract(52.9829189 * fract(dot(frag.xy + vec2<f32>(u.params.w * 5.588238), vec2<f32>(0.06711056, 0.00583715))));
  let sigma = u.params.x;
  var acc = 0.0;
  var T = 1.0;
  let stepT = exp(-sigma * ds);
  for (var i = 0.0; i < steps; i = i + 1.0) {
    let p = u.cameraPos.xyz + dir * ((i + jitter) * ds);
    acc = acc + T * sun_visible(p) * ds;
    T = T * stepT;
  }
  let g = u.params.y;
  let cosT = dot(dir, u.sunDir.xyz);
  let phase = (1.0 - g * g) / (12.5663706 * pow(max(1.0 + g * g - 2.0 * g * cosT, 1e-4), 1.5));
  let L = u.sunE.rgb * (phase * sigma * acc);
  return vec4<f32>(min(L, vec3<f32>(60000.0)), 0.0);
}
`;

export interface VolumetricParams {
  viewProj: Mat4;
  cameraPos: Vec3;
  shadowViewProj: Float32Array;
  cascades: number;
  sunDir: Vec3;
  /** Sun illuminance already multiplied by pre-exposure × white balance. */
  sunE: Vec3;
  density: number;
  anisotropy: number;
  steps: number;
  maxDistance: number;
  frame: number;
}

const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var t_vol: texture_2d<f32>;
@group(0) @binding(1) var s_vol: sampler;
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let full = vec2<f32>(textureDimensions(t_vol)) * 2.0;
  return vec4<f32>(textureSampleLevel(t_vol, s_vol, frag.xy / full, 0.0).rgb, 0.0);
}
`;

export class VolumetricPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private uniform: GPUBuffer;
  private scratch = new Float32Array(UNIFORM_FLOATS);
  private invVP = new Float32Array(16) as unknown as Mat4;
  private bindGroup: GPUBindGroup | null = null;
  private bound: { depth: GPUTextureView; shadow: GPUTextureView } | null = null;
  private sampler: GPUSampler;
  private composite: GPURenderPipeline;
  private linear: GPUSampler;
  private half: GPUTexture | null = null;
  private halfView: GPUTextureView | null = null;
  private compositeBg: GPUBindGroup | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ label: "volumetric", code: VOLUMETRIC_WGSL });
    this.pipeline = device.createRenderPipeline({
      label: "volumetric",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: "rgba16float" }] },
    });
    const cm = device.createShaderModule({ label: "volumetric/composite", code: COMPOSITE_WGSL });
    this.composite = device.createRenderPipeline({
      label: "volumetric/composite",
      layout: "auto",
      vertex: { module: cm, entryPoint: "vs_main" },
      fragment: {
        module: cm,
        entryPoint: "fs_main",
        targets: [{
          format: "rgba16float",
          blend: {
            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
          },
        }],
      },
    });
    this.linear = device.createSampler({ label: "volumetric/linear", magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.uniform = device.createBuffer({ label: "volumetric/u", size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ label: "volumetric/shadow", compare: "less", magFilter: "linear", minFilter: "linear" });
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, depth: GPUTextureView, shadow: GPUTextureView, size: { width: number; height: number }, p: VolumetricParams, timing?: GPURenderPassTimestampWrites): void {
    if (!m4Invert(this.invVP, p.viewProj)) return;
    const hw = Math.max(1, Math.ceil(size.width / 2)), hh = Math.max(1, Math.ceil(size.height / 2));
    if (!this.half || this.half.width !== hw || this.half.height !== hh) {
      this.half?.destroy();
      this.half = this.device.createTexture({ label: "volumetric/half", size: [hw, hh], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.halfView = this.half.createView();
      this.compositeBg = this.device.createBindGroup({
        label: "volumetric/composite",
        layout: this.composite.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.halfView }, { binding: 1, resource: this.linear }],
      });
    }
    if (!this.bindGroup || !this.bound || this.bound.depth !== depth || this.bound.shadow !== shadow) {
      this.bindGroup = this.device.createBindGroup({
        label: "volumetric",
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: depth },
          { binding: 2, resource: shadow },
          { binding: 3, resource: this.sampler },
        ],
      });
      this.bound = { depth, shadow };
    }
    const s = this.scratch;
    s.set(this.invVP as unknown as ArrayLike<number>, 0);
    s.set(p.shadowViewProj.subarray(0, 16), 16);
    s.set([p.cameraPos[0], p.cameraPos[1], p.cameraPos[2], 1], 32);
    s.set([p.sunDir[0], p.sunDir[1], p.sunDir[2], 0], 36);
    s.set([p.sunE[0], p.sunE[1], p.sunE[2], 0], 40);
    s.set([p.density, p.anisotropy, Math.max(4, Math.round(p.steps)), p.frame % 64], 44);
    s.set([p.maxDistance, 0.0008, p.cascades, 0], 48);
    this.device.queue.writeBuffer(this.uniform, 0, s as unknown as BufferSource);
    const pass = encoder.beginRenderPass({
      label: "volumetric",
      colorAttachments: [{ view: this.halfView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
      ...(timing ? { timestampWrites: { querySet: timing.querySet, ...(timing.beginningOfPassWriteIndex !== undefined ? { beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex } : {}) } } : {}),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    const comp = encoder.beginRenderPass({
      label: "volumetric-composite",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
      ...(timing ? { timestampWrites: { querySet: timing.querySet, ...(timing.endOfPassWriteIndex !== undefined ? { endOfPassWriteIndex: timing.endOfPassWriteIndex } : {}) } } : {}),
    });
    comp.setPipeline(this.composite);
    comp.setBindGroup(0, this.compositeBg!);
    comp.draw(3);
    comp.end();
  }

  destroy(): void {
    this.uniform.destroy();
    this.half?.destroy();
  }
}

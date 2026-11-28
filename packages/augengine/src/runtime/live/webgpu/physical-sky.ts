/**
 * Physical sky background: the visible half of the natural-light tier.
 *
 * Same far-plane fullscreen triangle as the gradient SkyPass (sky.ts):
 * `depthCompare: "equal"` against the clear depth, so it only shades pixels
 * no geometry covered. Per pixel it evaluates either
 *   - mode 0: the analytic sky (SKY_MODEL_WGSL, Preetham + CIE overcast) plus
 *     the limb-darkened sun disc, or
 *   - mode 1: the HDR equirect image at full resolution,
 * then applies pre-exposure × white balance (the same factor the material
 * shaders apply), so sky and geometry share one exposure.
 *
 * Aerial perspective toward the horizon is not added here: distant geometry
 * fades toward the sky colour in the material fog, and the sky itself IS the
 * fully in-scattered limit.
 */

import { m4Invert } from "../../../scene/math.js";
import type { Mat4 } from "../../../scene/types.js";
import { SKY_MODEL_FLOATS, SKY_MODEL_WGSL } from "./sky-model.js";

const PHYSICAL_SKY_WGSL = /* wgsl */ `
${SKY_MODEL_WGSL}
struct BgU {
  invViewProj: mat4x4<f32>,
  // x = equirect rotation (radians), y = sun disc on (1/0)
  params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> bg: BgU;
@group(0) @binding(1) var<uniform> sky: SkyModelU;
@group(0) @binding(2) var t_equirect: texture_2d<f32>;
@group(0) @binding(3) var s_equirect: sampler;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var out: VsOut;
  out.clip = vec4<f32>(pos[i], 1.0, 1.0);
  out.ndc = pos[i];
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let near = bg.invViewProj * vec4<f32>(in.ndc, 0.0, 1.0);
  let far = bg.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - near.xyz / near.w);
  var col: vec3<f32>;
  if (sky.expo.w > 0.5) {
    let phi = atan2(dir.x, -dir.z) - bg.params.x;
    let uv = vec2<f32>(fract(phi / 6.28318530718 + 0.5), acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265359);
    col = textureSampleLevel(t_equirect, s_equirect, uv, 0.0).rgb * sky.ground.w;
  } else {
    col = sky_radiance(dir) * sky.ground.w;
    if (bg.params.y > 0.5) { col = col + sun_disc(dir); }
  }
  // Pre-exposed; clamp below the fp16 maximum so bloom never sees inf.
  return vec4<f32>(min(col * sky.expo.rgb, vec3<f32>(60000.0)), 1.0);
}
`;

export class PhysicalSkyPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bgUniform: GPUBuffer;
  private skyUniform: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private boundEquirect: GPUTextureView | null = null;
  private scratch = new Float32Array(20);
  private invVP: Mat4 = new Float32Array(16) as unknown as Mat4;
  private sampler: GPUSampler;

  constructor(device: GPUDevice, targetFormat: GPUTextureFormat, sampleCount: 1 | 4) {
    this.device = device;
    const module = device.createShaderModule({ label: "physical-sky", code: PHYSICAL_SKY_WGSL });
    this.pipeline = device.createRenderPipeline({
      label: "physical-sky",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: targetFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { depthWriteEnabled: false, depthCompare: "equal", format: "depth24plus" },
      multisample: { count: sampleCount },
    });
    this.bgUniform = device.createBuffer({ label: "physical-sky/bg", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.skyUniform = device.createBuffer({ label: "physical-sky/model", size: SKY_MODEL_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ label: "physical-sky/equirect", magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" });
  }

  /** Upload the packed sky model (sky-model.ts packSkyModel, with exposure × WB in expo.rgb). */
  setSkyModel(packed: Float32Array): void {
    this.device.queue.writeBuffer(this.skyUniform, 0, packed as unknown as BufferSource);
  }

  encode(pass: GPURenderPassEncoder, viewProj: Mat4, equirect: GPUTextureView, rotation: number, sunDisc: boolean): void {
    const inv = m4Invert(this.invVP, viewProj);
    if (!inv) return;
    if (!this.bindGroup || this.boundEquirect !== equirect) {
      this.bindGroup = this.device.createBindGroup({
        label: "physical-sky/bg",
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.bgUniform } },
          { binding: 1, resource: { buffer: this.skyUniform } },
          { binding: 2, resource: equirect },
          { binding: 3, resource: this.sampler },
        ],
      });
      this.boundEquirect = equirect;
    }
    const s = this.scratch;
    s.set(inv as unknown as ArrayLike<number>, 0);
    s[16] = rotation; s[17] = sunDisc ? 1 : 0; s[18] = 0; s[19] = 0;
    this.device.queue.writeBuffer(this.bgUniform, 0, s as unknown as BufferSource);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
  }

  destroy(): void {
    this.bgUniform.destroy();
    this.skyUniform.destroy();
  }
}

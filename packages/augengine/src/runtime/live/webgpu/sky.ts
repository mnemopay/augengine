/**
 * Gradient-sky background pass.
 *
 * The live renderer had no skybox — the upper half of every scene rendered as
 * the flat `clearColor`, which reads as a dead void above the horizon. This
 * pass fills that gap: a fullscreen triangle drawn AFTER opaque geometry,
 * reconstructing the per-pixel world-space view ray from the inverse
 * view-projection and shading it as a vertical horizon→zenith gradient.
 *
 * The triangle sits on the far plane (clip z = 1) and does not write depth.
 * `depthCompare: "equal"` keeps it on pixels the scene left at the clear
 * depth, so covered pixels are rejected instead of shaded and then overdrawn.
 * No giant emissive sphere, no extra scene entity; it's a pure screen-space pass.
 *
 * Gradient is configurable (zenith / horizon color + a horizon-glow tint and
 * its falloff). Defaults are baked for OVERCLOCK's neon-night mood: a deep
 * indigo zenith fading through a dark teal horizon band with a faint magenta
 * glow hugging the skyline.
 */

import { m4Invert } from "../../../scene/math.js";
import type { Mat4, Vec3 } from "../../../scene/types.js";

export interface SkyGradient {
  /** Linear RGB at straight-up (ray.y → +1). */
  zenith: Vec3;
  /** Linear RGB at the horizon line (ray.y → 0). */
  horizon: Vec3;
  /** Linear RGB glow tint blended into the band just above the horizon. */
  horizonGlow: Vec3;
  /**
   * Exponent shaping how fast the gradient climbs from horizon to zenith.
   * Higher = the horizon color clings lower and the zenith dominates more of
   * the dome. ~1.2–2.0 reads well for a moody night sky.
   */
  exponent: number;
  /**
   * Tightness of the horizon glow band. Higher = a thinner, sharper glow line
   * hugging the skyline; lower = a broad wash up the lower sky.
   */
  glowFalloff: number;
}

/** OVERCLOCK neon-night default — deep indigo zenith, dark teal horizon, faint
 *  magenta skyline glow. Tuned to sit behind the cyan grid + magenta accents
 *  without competing with the bloomed geometry. */
export const OVERCLOCK_SKY: SkyGradient = {
  zenith: [0.012, 0.018, 0.06],
  horizon: [0.03, 0.09, 0.13],
  horizonGlow: [0.18, 0.04, 0.16],
  exponent: 1.35,
  glowFalloff: 7.0,
};

// std140-ish layout, 16-byte aligned. invViewProj(64) + cameraPos(16) +
// zenith(16) + horizon(16) + horizonGlow(16) + params(16) = 144 bytes.
const SKY_UNIFORM_SIZE = 64 + 16 + 16 + 16 + 16 + 16; // 144

const SKY_WGSL = /* wgsl */ `
struct SkyUniforms {
  invViewProj: mat4x4<f32>,
  cameraPos:   vec4<f32>,
  zenith:      vec4<f32>,
  horizon:     vec4<f32>,
  horizonGlow: vec4<f32>,
  // params.x = gradient exponent, params.y = glow falloff.
  params:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: SkyUniforms;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
  // Oversized fullscreen triangle.
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var out: VsOut;
  let p = pos[i];
  out.clip = vec4<f32>(p, 1.0, 1.0);
  out.ndc = p;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  // Unproject the near + far NDC points and take the world-space ray between
  // them — independent of the (negative) clip-z convention.
  let near = u.invViewProj * vec4<f32>(in.ndc, 0.0, 1.0);
  let far  = u.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
  let nearW = near.xyz / near.w;
  let farW  = far.xyz / far.w;
  let dir = normalize(farW - nearW);

  // Vertical factor: 0 at/below horizon, 1 straight up.
  let up = clamp(dir.y, 0.0, 1.0);
  let t = pow(up, max(u.params.x, 0.05));
  var col = mix(u.horizon.rgb, u.zenith.rgb, t);

  // Horizon glow — strongest right at the skyline, falls off going up. Also
  // fade it out below the horizon so the floor void stays dark.
  let glow = exp(-up * max(u.params.y, 0.01)) * smoothstep(-0.15, 0.05, dir.y);
  col = col + u.horizonGlow.rgb * glow;

  return vec4<f32>(col, 1.0);
}
`;

/**
 * Owns the sky render pipeline + its per-frame uniform buffer. Compiled once
 * at renderer init against the same MSAA sample count + HDR target format the
 * main pass uses, so it can be encoded into the live render pass directly.
 */
export class SkyPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private uniformBuf: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private scratch = new Float32Array(SKY_UNIFORM_SIZE / 4);
  private invVP: Mat4 = new Float32Array(16) as unknown as Mat4;

  gradient: SkyGradient;
  enabled = true;

  constructor(
    device: GPUDevice,
    targetFormat: GPUTextureFormat,
    sampleCount: 1 | 4,
    gradient: SkyGradient = OVERCLOCK_SKY,
  ) {
    this.device = device;
    this.gradient = gradient;

    const module = device.createShaderModule({ label: "sky-shader", code: SKY_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      label: "sky-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "sky-pipeline-layout",
      bindGroupLayouts: [bindGroupLayout],
    });
    this.pipeline = device.createRenderPipeline({
      label: "sky-pipeline",
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: targetFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // Far-plane fill. Opaque geometry has already written closer depth, so
      // equal-to-clear keeps the sky on the uncovered pixels only.
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "equal",
        format: "depth24plus",
      },
      multisample: { count: sampleCount },
    });

    this.uniformBuf = device.createBuffer({
      label: "sky-uniform",
      size: SKY_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = device.createBindGroup({
      label: "sky-bindgroup",
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });
  }

  /** Replace any subset of the gradient parameters. */
  setGradient(g: Partial<SkyGradient>): void {
    this.gradient = { ...this.gradient, ...g };
  }

  /**
   * Encode the sky into an already-open render pass. Call it AFTER opaque
   * scene draws: the far-plane depth test fills only pixels still at the
   * clear depth. `viewProj` is the camera's combined matrix; `cameraPos` its eye.
   */
  encode(pass: GPURenderPassEncoder, viewProj: Mat4, cameraPos: Vec3): void {
    if (!this.enabled) return;
    const inv = m4Invert(this.invVP, viewProj);
    if (!inv) return; // degenerate camera — skip the sky this frame.
    const s = this.scratch;
    s.set(inv as unknown as ArrayLike<number>, 0);
    s.set([cameraPos[0], cameraPos[1], cameraPos[2], 1], 16);
    const g = this.gradient;
    s.set([g.zenith[0], g.zenith[1], g.zenith[2], 0], 20);
    s.set([g.horizon[0], g.horizon[1], g.horizon[2], 0], 24);
    s.set([g.horizonGlow[0], g.horizonGlow[1], g.horizonGlow[2], 0], 28);
    s.set([g.exponent, g.glowFalloff, 0, 0], 32);
    this.device.queue.writeBuffer(this.uniformBuf, 0, s as unknown as BufferSource);

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
  }

  destroy(): void {
    this.uniformBuf.destroy();
  }
}

/**
 * Physically weighted bloom: the energy-conserving mip-chain bloom of
 * Jimenez, "Next Generation Post Processing in Call of Duty: Advanced
 * Warfare" (SIGGRAPH 2014).
 *
 * No threshold. The HDR frame is downsampled into a chain of half-size
 * levels with a 13-tap filter (the first level weights its five 2×2 boxes by
 * the Karis average 1/(1 + luma) so single bright pixels cannot flicker),
 * then upsampled back with a 3×3 tent, each level adding the blurred level
 * below it. The result — the sum of all levels — is composited as
 *   hdr = mix(scene, bloom / levels, strength)
 * so bloom is a FRACTION of all light (≈ 4% by default) spread with a long,
 * physically plausible tail: a lens / eye point-spread function, not a glow
 * effect on bright pixels only. The legacy threshold bloom stays for the
 * neon scenes.
 *
 * Level 0 is the post runner's existing half-resolution bloom target, so
 * the composite and TAA resolve read the chain through the same binding.
 */

const FULLSCREEN = /* wgsl */ `
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}
`;

export const BLOOM_DOWN_WGSL = /* wgsl */ `
struct U { srcTexel: vec2<f32>, outTexel: vec2<f32>, karis: f32, _p: f32, _q: vec2<f32> };
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_lin: sampler;
@group(0) @binding(2) var<uniform> u: U;
${FULLSCREEN}
fn tap(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(t_src, s_lin, uv, 0.0).rgb; }
fn kw(c: vec3<f32>) -> f32 { return 1.0 / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))); }
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy * u.outTexel;
  let t = u.srcTexel;
  let a = tap(uv + t * vec2<f32>(-2.0, -2.0)); let b = tap(uv + t * vec2<f32>(0.0, -2.0)); let c = tap(uv + t * vec2<f32>(2.0, -2.0));
  let d = tap(uv + t * vec2<f32>(-1.0, -1.0)); let e = tap(uv + t * vec2<f32>(1.0, -1.0));
  let f = tap(uv + t * vec2<f32>(-2.0, 0.0)); let g = tap(uv); let h = tap(uv + t * vec2<f32>(2.0, 0.0));
  let i = tap(uv + t * vec2<f32>(-1.0, 1.0)); let j = tap(uv + t * vec2<f32>(1.0, 1.0));
  let k = tap(uv + t * vec2<f32>(-2.0, 2.0)); let l = tap(uv + t * vec2<f32>(0.0, 2.0)); let m = tap(uv + t * vec2<f32>(2.0, 2.0));
  // Five overlapping 2×2 boxes: centre (0.5) + four corners (0.125 each).
  let b0 = (d + e + i + j) * 0.25;
  let b1 = (a + b + f + g) * 0.25;
  let b2 = (b + c + g + h) * 0.25;
  let b3 = (f + g + k + l) * 0.25;
  let b4 = (g + h + l + m) * 0.25;
  if (u.karis > 0.5) {
    let w0 = kw(b0) * 0.5; let w1 = kw(b1) * 0.125; let w2 = kw(b2) * 0.125; let w3 = kw(b3) * 0.125; let w4 = kw(b4) * 0.125;
    return vec4<f32>((b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4) / (w0 + w1 + w2 + w3 + w4), 1.0);
  }
  return vec4<f32>(b0 * 0.5 + (b1 + b2 + b3 + b4) * 0.125, 1.0);
}
`;

export const BLOOM_UP_WGSL = /* wgsl */ `
struct U { srcTexel: vec2<f32>, outTexel: vec2<f32>, radius: f32, _p: f32, _q: vec2<f32> };
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_lin: sampler;
@group(0) @binding(2) var<uniform> u: U;
${FULLSCREEN}
fn tap(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(t_src, s_lin, uv, 0.0).rgb; }
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy * u.outTexel;
  let t = u.srcTexel * u.radius;
  // 3×3 tent: 1 2 1 / 2 4 2 / 1 2 1, /16. Added onto this level (blend one + one).
  var c = tap(uv) * 4.0;
  c = c + (tap(uv + vec2<f32>(t.x, 0.0)) + tap(uv - vec2<f32>(t.x, 0.0)) + tap(uv + vec2<f32>(0.0, t.y)) + tap(uv - vec2<f32>(0.0, t.y))) * 2.0;
  c = c + tap(uv + t) + tap(uv - t) + tap(uv + vec2<f32>(t.x, -t.y)) + tap(uv + vec2<f32>(-t.x, t.y));
  return vec4<f32>(c / 16.0, 1.0);
}
`;

/** Sizes of the chain: level 0 = half the output, each next level half again, ≥ 8 px, at most `maxLevels`. */
export function bloomChainSizes(width: number, height: number, maxLevels = 6): Array<{ width: number; height: number }> {
  const out: Array<{ width: number; height: number }> = [];
  let w = Math.max(1, Math.ceil(width / 2)), h = Math.max(1, Math.ceil(height / 2));
  while (out.length < maxLevels) {
    out.push({ width: w, height: h });
    if (Math.min(w, h) <= 16) break;
    w = Math.max(1, Math.ceil(w / 2));
    h = Math.max(1, Math.ceil(h / 2));
  }
  return out;
}

export class PhysicalBloomChain {
  private device: GPUDevice;
  private down: GPURenderPipeline;
  private up: GPURenderPipeline;
  private sampler: GPUSampler;
  private levels: GPUTexture[] = [];
  private views: GPUTextureView[] = [];
  /** Separate down / up uniforms: queue writes all land before the frame runs, so one buffer cannot serve both. */
  private downU: GPUBuffer[] = [];
  private upU: GPUBuffer[] = [];
  private downBgs: GPUBindGroup[] = [];
  private upBgs: GPUBindGroup[] = [];
  private boundSource: GPUTextureView | null = null;
  private sourceSize = { width: 0, height: 0 };
  sizes: Array<{ width: number; height: number }> = [];

  constructor(device: GPUDevice) {
    this.device = device;
    const mk = (label: string, code: string, blend: boolean) => {
      const module = device.createShaderModule({ label, code });
      return device.createRenderPipeline({
        label,
        layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: {
          module,
          entryPoint: "fs_main",
          targets: [blend
            ? { format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "zero", operation: "add" } } }
            : { format: "rgba16float" }],
        },
      });
    };
    this.down = mk("bloom/down", BLOOM_DOWN_WGSL, false);
    this.up = mk("bloom/up", BLOOM_UP_WGSL, true);
    this.sampler = device.createSampler({ label: "bloom/linear", magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  }

  /** Number of levels (the composite divides the summed chain by this). */
  get levelCount(): number {
    return this.sizes.length;
  }

  /**
   * (Re)build for an output size. `level0` is the caller's half-resolution
   * bloom target (must match sizes[0]).
   */
  resize(width: number, height: number, level0: GPUTexture, level0View: GPUTextureView): void {
    this.destroyLevels();
    this.sizes = bloomChainSizes(width, height);
    this.levels = [level0];
    this.views = [level0View];
    for (let i = 1; i < this.sizes.length; i++) {
      const s = this.sizes[i]!;
      const t = this.device.createTexture({ label: `bloom/level-${i}`, size: [s.width, s.height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.levels.push(t);
      this.views.push(t.createView());
    }
    const mkU = (label: string) => this.device.createBuffer({ label, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.downU = this.sizes.map((_, i) => mkU(`bloom/down-u-${i}`));
    this.upU = this.sizes.slice(0, -1).map((_, i) => mkU(`bloom/up-u-${i}`));
    for (let i = 1; i < this.sizes.length; i++) {
      const s = this.sizes[i]!, src = this.sizes[i - 1]!;
      this.device.queue.writeBuffer(this.downU[i]!, 0, new Float32Array([1 / src.width, 1 / src.height, 1 / s.width, 1 / s.height, 0, 0, 0, 0]));
    }
    for (let i = 0; i < this.sizes.length - 1; i++) {
      const s = this.sizes[i]!, src = this.sizes[i + 1]!;
      this.device.queue.writeBuffer(this.upU[i]!, 0, new Float32Array([1 / src.width, 1 / src.height, 1 / s.width, 1 / s.height, 1, 0, 0, 0]));
    }
    this.boundSource = null;
    this.sourceSize = { width: 0, height: 0 };
  }

  private rebuildBindGroups(source: GPUTextureView): void {
    this.downBgs = this.sizes.map((_, i) => this.device.createBindGroup({
      label: `bloom/down-${i}`,
      layout: this.down.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: i === 0 ? source : this.views[i - 1]! },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.downU[i]! } },
      ],
    }));
    this.upBgs = this.upU.map((buf, i) => this.device.createBindGroup({
      label: `bloom/up-${i}`,
      layout: this.up.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.views[i + 1]! },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: buf } },
      ],
    }));
    this.boundSource = source;
  }

  /** Encode down + up passes from the HDR `source` (size sw × sh). */
  run(encoder: GPUCommandEncoder, source: GPUTextureView, sw: number, sh: number, begin?: GPURenderPassTimestampWrites, end?: GPURenderPassTimestampWrites): void {
    const n = this.sizes.length;
    if (n === 0) return;
    if (this.boundSource !== source) this.rebuildBindGroups(source);
    if (this.sourceSize.width !== sw || this.sourceSize.height !== sh) {
      const s0 = this.sizes[0]!;
      this.device.queue.writeBuffer(this.downU[0]!, 0, new Float32Array([1 / sw, 1 / sh, 1 / s0.width, 1 / s0.height, 1, 0, 0, 0]));
      this.sourceSize = { width: sw, height: sh };
    }
    for (let i = 0; i < n; i++) {
      const pass = encoder.beginRenderPass({
        label: `bloom-down-${i}`,
        colorAttachments: [{ view: this.views[i]!, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
        ...(i === 0 && begin ? { timestampWrites: begin } : {}),
      });
      pass.setPipeline(this.down);
      pass.setBindGroup(0, this.downBgs[i]!);
      pass.draw(3);
      pass.end();
    }
    for (let i = n - 2; i >= 0; i--) {
      const pass = encoder.beginRenderPass({
        label: `bloom-up-${i}`,
        colorAttachments: [{ view: this.views[i]!, loadOp: "load", storeOp: "store" }],
        ...(i === 0 && end ? { timestampWrites: end } : {}),
      });
      pass.setPipeline(this.up);
      pass.setBindGroup(0, this.upBgs[i]!);
      pass.draw(3);
      pass.end();
    }
  }

  private destroyLevels(): void {
    for (const t of this.levels.slice(1)) t.destroy();
    for (const b of this.downU) b.destroy();
    for (const b of this.upU) b.destroy();
    this.levels = [];
    this.views = [];
    this.downU = [];
    this.upU = [];
  }

  destroy(): void {
    this.destroyLevels();
  }
}

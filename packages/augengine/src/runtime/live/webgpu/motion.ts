/**
 * Per-object motion vectors for TAA (backend A).
 *
 * The TAA resolve reprojects history from depth with last frame's camera,
 * which is exact for anything that did not move. For things that did — skinned
 * characters, morph-target faces, entities whose model matrix changed — this
 * pass draws just those meshes a second time into a small velocity target:
 * each fragment stores (uv now − uv last frame), computed from the current
 * and previous model matrix, the current and previous skinned positions, and
 * the current and previous blendshape weights. Pixels the pass does not touch
 * keep the clear sentinel and the resolve falls back to depth reprojection.
 *
 * The rasterised position uses the frame's JITTERED view-projection with the
 * same expression as the main pass (both are @invariant), so `less-equal`
 * against the main pass's depth keeps exactly the visible surface. Velocity
 * itself is computed from the UNJITTERED matrices, so it carries no jitter.
 *
 * Static scenes pay nothing: with no moving draw the pass is skipped and the
 * resolve is told velocity is off.
 */

/** Clear value meaning "no object motion here — reproject from depth". */
export const VELOCITY_NONE = -1000;

/** Velocity in UV units (y down) between two clip positions. */
export function clipToUvVelocity(curClip: ArrayLike<number>, prevClip: ArrayLike<number>): [number, number] {
  const cw = curClip[3]!, pw = prevClip[3]!;
  const cu = curClip[0]! / cw * 0.5 + 0.5, cv = 0.5 - curClip[1]! / cw * 0.5;
  const pu = prevClip[0]! / pw * 0.5 + 0.5, pv = 0.5 - prevClip[1]! / pw * 0.5;
  return [cu - pu, cv - pv];
}

const MOTION_FRAME_WGSL = /* wgsl */ `
struct MotionFrame {
  jitteredVP: mat4x4<f32>,
  curVP: mat4x4<f32>,
  prevVP: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u_mf: MotionFrame;
struct MotionDraw { model: mat4x4<f32>, prevModel: mat4x4<f32>, };
@group(1) @binding(0) var<uniform> u_md: MotionDraw;
struct VOut {
  @builtin(position) @invariant clipPos: vec4<f32>,
  @location(0) cur: vec4<f32>,
  @location(1) prev: vec4<f32>,
};
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let cu = vec2<f32>(in.cur.x / in.cur.w * 0.5 + 0.5, 0.5 - in.cur.y / in.cur.w * 0.5);
  let pu = vec2<f32>(in.prev.x / in.prev.w * 0.5 + 0.5, 0.5 - in.prev.y / in.prev.w * 0.5);
  return vec4<f32>(cu - pu, 0.0, 1.0);
}
`;

/** Rigid or skinned: slot 0 = current positions, slot 1 = last frame's. */
export const MOTION_RIGID_WGSL = MOTION_FRAME_WGSL + /* wgsl */ `
@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) prevPosition: vec3<f32>) -> VOut {
  var out: VOut;
  let world = u_md.model * vec4<f32>(position, 1.0);
  out.clipPos = u_mf.jitteredVP * world;
  out.cur = u_mf.curVP * world;
  out.prev = u_mf.prevVP * (u_md.prevModel * vec4<f32>(prevPosition, 1.0));
  return out;
}
`;

/** Morph targets: the same deltas weighted by this frame's and last frame's weights. */
export const MOTION_MORPH_WGSL = MOTION_FRAME_WGSL + /* wgsl */ `
const BS_COUNT: u32 = 52u;
struct Weights { values: array<vec4<f32>, 13> };
@group(2) @binding(0) var<uniform> u_w: Weights;
@group(2) @binding(1) var<uniform> u_wPrev: Weights;
@group(2) @binding(2) var<storage, read> u_deltas: array<vec3<f32>>;
fn lane(v: vec4<f32>, k: u32) -> f32 {
  let l = k & 3u;
  if (l == 0u) { return v.x; }
  if (l == 1u) { return v.y; }
  if (l == 2u) { return v.z; }
  return v.w;
}
@vertex
fn vs_main(@location(0) position: vec3<f32>, @builtin(vertex_index) vi: u32) -> VOut {
  var out: VOut;
  var off = vec3<f32>(0.0);
  var offPrev = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < BS_COUNT; k = k + 1u) {
    let w = lane(u_w.values[k >> 2u], k);
    let wp = lane(u_wPrev.values[k >> 2u], k);
    if (w == 0.0 && wp == 0.0) { continue; }
    let d = u_deltas[vi * BS_COUNT + k];
    off = off + w * d;
    offPrev = offPrev + wp * d;
  }
  // Same expression as the main pass's morph vertex shader (invariant clipPos).
  let morphedPos = position + off;
  let world = u_md.model * vec4<f32>(morphedPos, 1.0);
  out.clipPos = u_mf.jitteredVP * world;
  out.cur = u_mf.curVP * world;
  out.prev = u_mf.prevVP * (u_md.prevModel * vec4<f32>(position + offPrev, 1.0));
  return out;
}
`;

/** One moving draw for the motion pass. */
export interface MotionDraw {
  entityId: string;
  vbuf: GPUBuffer;
  /** Last frame's positions (skinned meshes); null = same buffer as `vbuf`. */
  prevVbuf: GPUBuffer | null;
  ibuf: GPUBuffer;
  indexCount: number;
  vertexStride: number;
  model: ArrayLike<number>;
  /** Model matrix last frame (same as `model` for a first appearance). */
  prevModel: ArrayLike<number>;
  /** Morph targets: current weights buffer, last frame's weights, deltas. */
  morph: { weights: GPUBuffer; prevWeights: Float32Array; deltas: GPUBuffer } | null;
}

interface EntityMotionGpu {
  buf: GPUBuffer;
  bindGroup: GPUBindGroup;
  scratch: Float32Array;
  morphBindGroup: GPUBindGroup | null;
  morphWeights: GPUBuffer | null;
  prevWeightsBuf: GPUBuffer | null;
  deltas: GPUBuffer | null;
  seenFrame: number;
}

export class MotionPass {
  private device: GPUDevice;
  private frameBuf: GPUBuffer;
  private frameBindGroup: GPUBindGroup;
  private frameLayout: GPUBindGroupLayout;
  private drawLayout: GPUBindGroupLayout;
  private morphLayout: GPUBindGroupLayout;
  private rigidModule: GPUShaderModule;
  private morphModule: GPUShaderModule;
  private pipelines = new Map<string, GPURenderPipeline>();
  private entities = new Map<string, EntityMotionGpu>();
  private frameScratch = new Float32Array(48);
  private frameIndex = 0;
  texture: GPUTexture | null = null;
  view: GPUTextureView | null = null;
  width = 0;
  height = 0;
  /** Draws encoded last frame (diagnostics). */
  lastDrawCount = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.frameLayout = device.createBindGroupLayout({
      label: "motion/frame",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });
    this.drawLayout = device.createBindGroupLayout({
      label: "motion/draw",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });
    this.morphLayout = device.createBindGroupLayout({
      label: "motion/morph",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
    this.frameBuf = device.createBuffer({ label: "motion/frame-uniform", size: 48 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.frameBindGroup = device.createBindGroup({ layout: this.frameLayout, entries: [{ binding: 0, resource: { buffer: this.frameBuf } }] });
    this.rigidModule = device.createShaderModule({ label: "motion/rigid", code: MOTION_RIGID_WGSL });
    this.morphModule = device.createShaderModule({ label: "motion/morph", code: MOTION_MORPH_WGSL });
  }

  resize(width: number, height: number): void {
    if (this.texture && this.width === width && this.height === height) return;
    this.texture?.destroy();
    this.width = width;
    this.height = height;
    this.texture = this.device.createTexture({
      label: "motion/velocity",
      size: [width, height],
      format: "rg16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView();
  }

  private pipelineFor(stride: number, morph: boolean): GPURenderPipeline {
    const key = `${stride}|${morph ? "m" : "r"}`;
    let p = this.pipelines.get(key);
    if (p) return p;
    const buffers: GPUVertexBufferLayout[] = morph
      ? [{ arrayStride: stride * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }]
      : [
          { arrayStride: stride * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
          { arrayStride: stride * 4, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
        ];
    p = this.device.createRenderPipeline({
      label: `motion/${key}`,
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: morph ? [this.frameLayout, this.drawLayout, this.morphLayout] : [this.frameLayout, this.drawLayout],
      }),
      vertex: { module: morph ? this.morphModule : this.rigidModule, entryPoint: "vs_main", buffers },
      fragment: { module: morph ? this.morphModule : this.rigidModule, entryPoint: "fs_main", targets: [{ format: "rg16float" }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" },
    });
    this.pipelines.set(key, p);
    return p;
  }

  private entityFor(d: MotionDraw): EntityMotionGpu {
    let e = this.entities.get(d.entityId);
    if (!e) {
      const buf = this.device.createBuffer({ label: `motion/${d.entityId}`, size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      e = {
        buf,
        bindGroup: this.device.createBindGroup({ layout: this.drawLayout, entries: [{ binding: 0, resource: { buffer: buf } }] }),
        scratch: new Float32Array(32),
        morphBindGroup: null,
        morphWeights: null,
        prevWeightsBuf: null,
        deltas: null,
        seenFrame: -1,
      };
      this.entities.set(d.entityId, e);
    }
    return e;
  }

  /**
   * Encode the velocity pass. Returns false (and encodes nothing) when there
   * is no moving draw, so a static scene skips the clear as well.
   */
  encode(
    encoder: GPUCommandEncoder,
    depthView: GPUTextureView,
    draws: readonly MotionDraw[],
    jitteredVP: ArrayLike<number>,
    curVP: ArrayLike<number>,
    prevVP: ArrayLike<number>,
    timing?: GPURenderPassTimestampWrites,
  ): boolean {
    this.frameIndex++;
    this.lastDrawCount = draws.length;
    if (draws.length === 0 || !this.view) {
      this.forgetStale();
      return false;
    }
    const f = this.frameScratch;
    f.set(jitteredVP as ArrayLike<number>, 0);
    f.set(curVP as ArrayLike<number>, 16);
    f.set(prevVP as ArrayLike<number>, 32);
    this.device.queue.writeBuffer(this.frameBuf, 0, f);
    const desc: GPURenderPassDescriptor = {
      label: "motion-pass",
      colorAttachments: [{ view: this.view, clearValue: { r: VELOCITY_NONE, g: VELOCITY_NONE, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store", depthReadOnly: false },
    };
    if (timing) desc.timestampWrites = timing;
    const pass = encoder.beginRenderPass(desc);
    pass.setBindGroup(0, this.frameBindGroup);
    let bound = "";
    for (const d of draws) {
      const e = this.entityFor(d);
      e.seenFrame = this.frameIndex;
      const s = e.scratch;
      s.set(d.model, 0);
      s.set(d.prevModel, 16);
      this.device.queue.writeBuffer(e.buf, 0, s as unknown as BufferSource);
      const morph = !!d.morph;
      const key = `${d.vertexStride}|${morph ? "m" : "r"}`;
      if (key !== bound) {
        pass.setPipeline(this.pipelineFor(d.vertexStride, morph));
        bound = key;
      }
      pass.setBindGroup(1, e.bindGroup);
      if (d.morph) {
        if (!e.prevWeightsBuf) {
          e.prevWeightsBuf = this.device.createBuffer({ label: `motion/prev-weights/${d.entityId}`, size: 208, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        }
        this.device.queue.writeBuffer(e.prevWeightsBuf, 0, d.morph.prevWeights as unknown as BufferSource);
        if (!e.morphBindGroup || e.morphWeights !== d.morph.weights || e.deltas !== d.morph.deltas) {
          e.morphBindGroup = this.device.createBindGroup({
            layout: this.morphLayout,
            entries: [
              { binding: 0, resource: { buffer: d.morph.weights } },
              { binding: 1, resource: { buffer: e.prevWeightsBuf } },
              { binding: 2, resource: { buffer: d.morph.deltas } },
            ],
          });
          e.morphWeights = d.morph.weights;
          e.deltas = d.morph.deltas;
        }
        pass.setBindGroup(2, e.morphBindGroup);
        pass.setVertexBuffer(0, d.vbuf);
      } else {
        pass.setVertexBuffer(0, d.vbuf);
        pass.setVertexBuffer(1, d.prevVbuf ?? d.vbuf);
      }
      pass.setIndexBuffer(d.ibuf, "uint32");
      pass.drawIndexed(d.indexCount);
    }
    pass.end();
    this.forgetStale();
    return true;
  }

  /** Drop per-entity buffers for entities not drawn for a few frames. */
  private forgetStale(): void {
    if (this.frameIndex % 120 !== 0) return;
    for (const [id, e] of this.entities) {
      if (this.frameIndex - e.seenFrame > 120) {
        e.buf.destroy();
        e.prevWeightsBuf?.destroy();
        this.entities.delete(id);
      }
    }
  }

  destroy(): void {
    this.texture?.destroy();
    this.texture = null;
    this.view = null;
    this.frameBuf.destroy();
    for (const e of this.entities.values()) {
      e.buf.destroy();
      e.prevWeightsBuf?.destroy();
    }
    this.entities.clear();
    this.pipelines.clear();
  }
}

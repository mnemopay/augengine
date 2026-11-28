/**
 * GPU luminance histogram for auto-exposure (math in exposure.ts).
 *
 * After the main pass, one compute dispatch bins log2 luminance of every
 * second pixel (both axes) of the pre-exposed HDR target into 64 bins, each
 * pixel voting 1–4 times by a centre-weighted (slightly low) metering mask —
 * workgroup-local atomics, then one global atomicAdd per bin per workgroup.
 * The 256-byte result is copied into one of three mappable readback buffers;
 * the CPU reads it 1–3 frames later (never stalls the frame), converts it to
 * a target EV100 with the pre-exposure that frame was rendered with, and the
 * renderer adapts toward it.
 */

import { DEFAULT_HISTOGRAM, type HistogramLayout } from "./exposure.js";

const HISTOGRAM_WGSL = (h: HistogramLayout) => /* wgsl */ `
const BINS: u32 = ${h.bins}u;
const MIN_LOG2: f32 = ${h.minLog2.toFixed(4)};
const MAX_LOG2: f32 = ${h.maxLog2.toFixed(4)};
@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>, ${h.bins}>;
var<workgroup> local: array<atomic<u32>, ${h.bins}>;

fn bin_of(lum: f32) -> u32 {
  if (lum <= 1e-9) { return 0u; }
  let t = (log2(lum) - MIN_LOG2) / (MAX_LOG2 - MIN_LOG2);
  return clamp(u32(floor(t * f32(BINS - 1u))) + 1u, 1u, BINS - 1u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (li < BINS) { atomicStore(&local[li], 0u); }
  workgroupBarrier();
  let dim = textureDimensions(t_hdr);
  let p = gid.xy * 2u;
  if (p.x < dim.x && p.y < dim.y) {
    let c = textureLoad(t_hdr, vec2<i32>(p), 0).rgb;
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Centre-weighted metering (1–4 votes per pixel), biased below centre
    // where the subject usually is: a bright sky across the top half must
    // not underexpose the ground, as a camera's centre-weighted mode avoids.
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(dim);
    let d = uv - vec2<f32>(0.5, 0.6);
    let w = 1u + u32(round(3.0 * exp(-dot(d, d) / (2.0 * 0.22 * 0.22))));
    atomicAdd(&local[bin_of(lum)], w);
  }
  workgroupBarrier();
  if (li < BINS) {
    let v = atomicLoad(&local[li]);
    if (v > 0u) { atomicAdd(&counts[li], v); }
  }
}
`;

interface Slot { buf: GPUBuffer; busy: boolean; preExposure: number }

export class AutoExposureMeter {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private counts: GPUBuffer;
  private slots: Slot[] = [];
  private bindGroup: GPUBindGroup | null = null;
  private boundView: GPUTextureView | null = null;
  readonly layout: HistogramLayout;
  /** Most recent resolved histogram + the pre-exposure it was measured under. */
  latest: { counts: Uint32Array; preExposure: number; frame: number } | null = null;
  private frame = 0;

  constructor(device: GPUDevice, layout: HistogramLayout = DEFAULT_HISTOGRAM) {
    this.device = device;
    this.layout = layout;
    this.pipeline = device.createComputePipeline({
      label: "exposure/histogram",
      layout: "auto",
      compute: { module: device.createShaderModule({ label: "exposure/histogram", code: HISTOGRAM_WGSL(layout) }), entryPoint: "main" },
    });
    const bytes = layout.bins * 4;
    this.counts = device.createBuffer({ label: "exposure/counts", size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    for (let i = 0; i < 3; i++) {
      this.slots.push({ buf: device.createBuffer({ label: `exposure/readback-${i}`, size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false, preExposure: 1 });
    }
  }

  /**
   * Encode the histogram of `hdr` (rendered with `preExposure`). Returns the
   * slot index that will receive it, or -1 when every readback is still in
   * flight (the frame is simply not measured).
   */
  encode(encoder: GPUCommandEncoder, hdr: GPUTextureView, width: number, height: number, preExposure: number, timing?: GPUComputePassTimestampWrites): number {
    const slotIndex = this.slots.findIndex((s) => !s.busy);
    if (slotIndex < 0) return -1;
    if (!this.bindGroup || this.boundView !== hdr) {
      this.bindGroup = this.device.createBindGroup({
        label: "exposure/histogram",
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: hdr },
          { binding: 1, resource: { buffer: this.counts } },
        ],
      });
      this.boundView = hdr;
    }
    encoder.clearBuffer(this.counts);
    const pass = encoder.beginComputePass(timing ? { label: "exposure-histogram", timestampWrites: timing } : { label: "exposure-histogram" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 32), Math.ceil(height / 32));
    pass.end();
    const slot = this.slots[slotIndex]!;
    encoder.copyBufferToBuffer(this.counts, 0, slot.buf, 0, this.layout.bins * 4);
    slot.busy = true;
    slot.preExposure = preExposure;
    return slotIndex;
  }

  /** Call after queue.submit() with the index encode() returned. */
  collect(slotIndex: number): void {
    if (slotIndex < 0) return;
    const slot = this.slots[slotIndex]!;
    const frame = ++this.frame;
    slot.buf.mapAsync(GPUMapMode.READ).then(() => {
      const counts = new Uint32Array(slot.buf.getMappedRange().slice(0));
      slot.buf.unmap();
      slot.busy = false;
      if (!this.latest || frame > this.latest.frame) this.latest = { counts, preExposure: slot.preExposure, frame };
    }).catch(() => {
      slot.busy = false;
    });
  }

  destroy(): void {
    this.counts.destroy();
    for (const s of this.slots) s.buf.destroy();
  }
}

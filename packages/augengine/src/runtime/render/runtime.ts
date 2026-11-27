/**
 * RenderRuntime — frame-deterministic render path. Drives the WebGPU
 * renderer at exactly `1/fps` per tick, captures each frame into a
 * VideoFrame, hands to the WebCodecs encoder, then muxes the result
 * into an MP4 (phase 1.1).
 *
 * Same Scene + Camera + AI scheduler + Audio graph as live runtime —
 * deterministic because:
 *   - Time advances by frame index, not wall clock
 *   - All RNG threads through a clip-level seed
 *   - AI artifacts are pulled from the manifest cache (decision #4)
 *
 * Phase 1: works against an OffscreenCanvas, returns the raw chunks +
 * stub-muxed bytes. Phase 1.1 wires real MP4 muxing.
 */

import { WebGpuLiveRenderer, type RenderFrameOptions } from "../live/webgpu/renderer.js";
import type { Scene } from "../../scene/scene.js";
import {
  WebCodecsEncoder,
  muxToMp4,
  type EncodedFrame,
} from "./encoder.js";
import type { EncoderOptions } from "../types.js";

export interface RenderJobOptions {
  scene: Scene;
  encoder: EncoderOptions;
  fps: number;
  frames: number;
  /** Deterministic seed threaded into RNG used by sim ticks. */
  seed: number;
  /** Hook called per frame for animation / sim ticks. */
  onSimTick?: (frameIndex: number, t: number) => void;
  /** Optional render-frame options (lighting, clear). */
  renderOpts?: RenderFrameOptions;
  /** Progress callback (0..1). */
  onProgress?: (p: number) => void;
}

export interface RenderJobResult {
  /** Raw encoded chunks if you want to hand-mux. */
  frames: EncodedFrame[];
  /** Stub-muxed payload (NOT valid MP4 in phase 1; see encoder.muxToMp4). */
  mp4: Uint8Array;
  /** Wall-clock duration of the render in ms. */
  elapsedMs: number;
  framesEncoded: number;
}

export class RenderRuntime {
  private canvas: OffscreenCanvas;
  private renderer: WebGpuLiveRenderer;

  constructor(canvas: OffscreenCanvas) {
    this.canvas = canvas;
    this.renderer = new WebGpuLiveRenderer(canvas);
  }

  async init(): Promise<void> {
    await this.renderer.init();
  }

  async run(opts: RenderJobOptions): Promise<RenderJobResult> {
    const start = performance.now();
    const enc = new WebCodecsEncoder(opts.encoder);
    await enc.configure();
    const dt = 1 / opts.fps;
    for (let i = 0; i < opts.frames; i++) {
      const t = i * dt;
      opts.onSimTick?.(i, t);
      this.renderer.render(opts.scene, opts.renderOpts ?? {});
      // Snapshot canvas → VideoFrame.
      const tsUs = Math.round((i / opts.fps) * 1_000_000);
      enc.encodeFrame(this.canvas, tsUs);
      opts.onProgress?.((i + 1) / opts.frames);
    }
    const frames = await enc.close();
    const mp4 = muxToMp4(frames, opts.encoder);
    return {
      frames,
      mp4,
      framesEncoded: enc.framesEncoded(),
      elapsedMs: performance.now() - start,
    };
  }

  destroy(): void {
    this.renderer.destroy();
  }
}

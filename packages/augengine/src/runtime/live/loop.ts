/**
 * LiveRuntime — wall-clock RAF loop driving the WebGPU renderer against
 * a Scene. Fixed-step simulation tick at 60 Hz; render at display refresh.
 *
 * Emits frame metrics on each tick for telemetry / overlay.
 */

import { WebGpuLiveRenderer, type RenderFrameOptions } from "./webgpu/renderer.js";
import type { Scene } from "../../scene/scene.js";

export interface LiveRuntimeOptions {
  /** Fixed simulation tick rate (Hz). Default 60. */
  simHz?: number;
  /** Render-frame options per tick (lighting, clear). */
  renderOpts?: RenderFrameOptions;
  /** Hook called every fixed sim tick. dt is in seconds. */
  onSimTick?: (dt: number, totalElapsed: number) => void;
  /** Hook called every render frame. metric.fps is rolling. */
  onRenderTick?: (metric: FrameMetric) => void;
}

export interface FrameMetric {
  fps: number;
  frameTimeMs: number;
  frameIndex: number;
}

export class LiveRuntime {
  readonly renderer: WebGpuLiveRenderer;
  readonly scene: Scene;
  private opts: LiveRuntimeOptions;

  private rafHandle: number | null = null;
  private lastNow = 0;
  private simAccum = 0;
  private totalElapsed = 0;
  private frameIndex = 0;
  private fpsAccum = 0;
  private fpsCount = 0;
  private fpsLastEmit = 0;
  private rollingFps = 0;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, scene: Scene, opts: LiveRuntimeOptions = {}) {
    this.renderer = new WebGpuLiveRenderer(canvas);
    this.scene = scene;
    this.opts = opts;
  }

  async init(): Promise<void> {
    await this.renderer.init();
  }

  start(): void {
    if (this.rafHandle !== null) return;
    this.lastNow = performance.now();
    this.fpsLastEmit = this.lastNow;
    const tick = (now: number) => {
      const dt = (now - this.lastNow) / 1000;
      this.lastNow = now;
      this.totalElapsed += dt;

      // Fixed-step sim ticks.
      const simDt = 1 / (this.opts.simHz ?? 60);
      this.simAccum += dt;
      let safety = 6;
      while (this.simAccum >= simDt && safety-- > 0) {
        this.opts.onSimTick?.(simDt, this.totalElapsed);
        this.simAccum -= simDt;
      }

      // Render the latest scene state.
      const t0 = performance.now();
      this.renderer.render(this.scene, this.opts.renderOpts ?? {});
      const t1 = performance.now();

      this.frameIndex += 1;
      this.fpsAccum += 1;
      this.fpsCount += 1;
      if (now - this.fpsLastEmit >= 500) {
        this.rollingFps = (this.fpsAccum / (now - this.fpsLastEmit)) * 1000;
        this.fpsAccum = 0;
        this.fpsLastEmit = now;
      }
      this.opts.onRenderTick?.({
        fps: this.rollingFps,
        frameTimeMs: t1 - t0,
        frameIndex: this.frameIndex,
      });

      this.rafHandle = requestAnimationFrame(tick);
    };
    this.rafHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
  }

  destroy(): void {
    this.stop();
    this.renderer.destroy();
  }
}

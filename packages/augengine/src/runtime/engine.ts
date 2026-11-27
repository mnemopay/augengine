import type { Scene } from "../scene/scene.js";
import type { RenderFrameOptions } from "./live/webgpu/renderer.js";
import type { LiveRenderer } from "./live/renderer-api.js";
import { CoupledSimulation } from "./simulation.js";
import { AIScheduler } from "../ai/scheduler.js";

export interface EngineOptions {
  scene: Scene;
  /** Any backend behind the renderer adapter (WebGPU or three.js). */
  renderer: LiveRenderer;
  targetFps?: number;
  /** Per-frame render options forwarded to the renderer. */
  renderOpts?: RenderFrameOptions;
  /**
   * Step rigid bodies, contacts, and chemistry each frame. Default true.
   * The character proof room turns this off: that scene is a lit character
   * on a dais, and the extraction-game physics tick was running anyway.
   */
  simulate?: boolean;
}

/**
 * The central Engine runner.
 *
 * Manages the client-side game loop, ticking Physics and Rendering in
 * deterministic order. AI dispatch is request-driven via AIScheduler —
 * subsystems call `scheduler.dispatch()` on-demand and race each request
 * against its own deadline. The scheduler is exposed here so engine
 * consumers can register providers and share one cache.
 */
export class EngineRunner {
  private scene: Scene;
  private renderer: LiveRenderer;
  readonly simulation: CoupledSimulation;
  readonly ai: AIScheduler;
  
  /** Asynchronous time manager for non-blocking systems (AI, Economy) */
  private asyncSystems: Set<{ tick: (scene: Scene, dt: number) => Promise<void> }> = new Set();

  private targetFps: number;
  private renderOpts: RenderFrameOptions;
  private simulate: boolean;
  private isRunning = false;
  private lastTime = 0;
  private animationFrameId: number | null = null;
  private startGeneration = 0;

  constructor(opts: EngineOptions) {
    this.scene = opts.scene;
    this.renderer = opts.renderer;
    this.targetFps = opts.targetFps ?? 60;
    this.renderOpts = opts.renderOpts ?? { ambient: [0.2, 0.2, 0.25] };
    this.simulate = opts.simulate !== false;
    this.simulation = new CoupledSimulation();
    this.ai = new AIScheduler();
    void this.targetFps;
  }

  registerAsyncSystem(system: { tick: (scene: Scene, dt: number) => Promise<void> }): void {
    this.asyncSystems.add(system);
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;
    const generation = ++this.startGeneration;
    try {
      if (this.simulate) await this.simulation.init();
    }
    catch (error) { if (generation === this.startGeneration) this.isRunning = false; throw error; }
    if (!this.isRunning || generation !== this.startGeneration) return;
    this.lastTime = performance.now();
    this.animationFrameId = requestAnimationFrame(this.tick);
    console.log("[EngineRunner] Started.");
  }

  stop(): void {
    this.isRunning = false;
    this.startGeneration++;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    console.log("[EngineRunner] Stopped.");
  }

  private tick = (currentTime: number): void => {
    if (!this.isRunning) return;

    // RAF timestamps mark the start of the frame and may precede a
    // performance.now() sampled while that frame was being scheduled.
    const dt = Math.max(0, (currentTime - this.lastTime) / 1000);
    this.lastTime = currentTime;

    if (this.simulate) this.simulation.advance(this.scene, dt);
    
    // Asynchronous Time Manager: Tick heavy systems without blocking render frame
    for (const sys of this.asyncSystems) {
      sys.tick(this.scene, dt).catch(e => console.error("[EngineRunner] Async System Error:", e));
    }

    this.renderer.render(this.scene, this.renderOpts);

    this.animationFrameId = requestAnimationFrame(this.tick);
  };

  destroy(): void { this.stop(); this.simulation.dispose(); }
}

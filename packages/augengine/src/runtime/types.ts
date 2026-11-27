/**
 * AugEngine runtime types — the dual-renderer abstraction.
 *
 * Live runtime drives the loop with wall clock; render runtime drives it with
 * frame index. Both implement the same `Renderer` trait so character /
 * audio / AI subsystems are agnostic.
 */

import type { SceneSpec, EntityId } from "../scene/types.js";

export interface Renderer {
  /** Initialize GPU device + canvas + pipelines. */
  init(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<void>;

  /** Replace the current scene. Disposes previous GPU resources. */
  loadScene(spec: SceneSpec): Promise<void>;

  /** Set active camera by entity ID. */
  setCamera(entity: EntityId): void;

  /** Advance simulation by `dt` seconds and submit a frame. Returns time the
   *  frame represents (in seconds since loadScene). For live mode this matches
   *  wall clock; for render mode this is `frameIndex / fps`. */
  step(dt: number): Promise<number>;

  /** Tear down GPU resources. */
  destroy(): Promise<void>;
}

/** Live runtime drives `step(dt)` from `requestAnimationFrame`. */
export interface LiveRenderer extends Renderer {
  /** True after `init()` succeeds. */
  readonly ready: boolean;
  /** Backend identification — useful for fallback decisions. */
  readonly backend: "webgpu" | "webgl2";
}

/** Render runtime drives `step(1/fps)` deterministically and emits MP4. */
export interface RenderRenderer extends Renderer {
  /** Configure the encoder before first `step()`. */
  configureEncoder(opts: EncoderOptions): Promise<void>;

  /** Run the full render loop: load scene → step `frames` times → finalize MP4 → return URL. */
  renderToMp4(opts: { fps: number; frames: number; seed: number }): Promise<{ url: string; bytes: number }>;
}

export interface EncoderOptions {
  /** "h264" | "hevc" | "av1" (av1 currently software-only on most hardware in 2026). */
  codec: "h264" | "hevc" | "av1";
  /** Width × height in physical pixels. */
  width: number;
  /** Height in physical pixels. */
  height: number;
  /** Frames per second. */
  fps: number;
  /** Encoded bitrate in bits/second. */
  bitrate: number;
  /** "constant" | "variable". */
  bitrateMode: "constant" | "variable";
  /** Frames between keyframes. */
  keyframeInterval: number;
  /** Optional audio track. */
  audio?: {
    codec: "aac" | "opus";
    sampleRate: number;
    channels: number;
    bitrate: number;
  };
}

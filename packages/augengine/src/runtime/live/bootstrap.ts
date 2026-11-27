/**
 * Single playable-renderer constructor.
 *
 * Studio, the Character Proof Room, smoke-game, OVERCLOCK, and the
 * Ledgerfall slice all boot through `createLiveRenderer`, which is the
 * WebGPU backend. The three.js backend (`ThreeLiveRenderer`, imported from
 * `@mnemopay/augengine/runtime/live/three`) sits behind the same
 * `LiveRenderer` adapter as an A/B candidate; it is opt-in per app (the
 * smoke-game's `?renderer=three`) and is deliberately not reachable from
 * this module, so apps without three installed never bundle it. See
 * results/renderer-benchmark/SCORECARD.md for the A/B decision.
 */

import { WebGpuLiveRenderer, type LiveRendererOptions } from "./webgpu/renderer.js";

/** The only renderer kind a playable AugEngine shell may construct. */
export const PLAYABLE_RENDERER = "webgpu" as const;

export type PlayableRendererKind = typeof PLAYABLE_RENDERER;

export function createLiveRenderer(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  opts: LiveRendererOptions = {},
): WebGpuLiveRenderer {
  return new WebGpuLiveRenderer(canvas, opts);
}

/**
 * Guard for shells that still accept a renderer-kind string. Throws if
 * anything other than WebGPU is requested so a Three.js fallback cannot
 * silently re-enter the playable graph.
 */
export function assertPlayableRenderer(kind: string = PLAYABLE_RENDERER): asserts kind is PlayableRendererKind {
  if (kind !== PLAYABLE_RENDERER) {
    throw new Error(
      `AugEngine playable path is WebGpuLiveRenderer (got "${kind}"). Three.js is retired.`,
    );
  }
}

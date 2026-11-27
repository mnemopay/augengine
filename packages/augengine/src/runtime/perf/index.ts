/**
 * Performance tooling — Track 6 (AUGENGINE-9-TRACK-PLAN.md).
 *
 * Public surface for:
 *   - WebGPU timestamp queries (TimestampSession + helpers)
 *   - CPU + GPU frame-budget HUD overlay (mountFrameBudgetHud)
 *   - Golden-path deterministic replay (ReplayRecorder + ReplayPlayer)
 *
 * Import from "@mnemopay/augengine/runtime/perf".
 */

export {
  TimestampSession,
  PassTimestampSession,
  PASS_LABELS,
  decodePassTimings,
  type PassLabel,
  type PassTimingSample,
  RollingGpuMs,
  decodeTimestampDeltaMs,
  probeTimestampQuerySupport,
  requiredFeaturesForTimestamps,
  type GpuMsSource,
  type TimestampQuerySupport,
} from "./timestamp-queries.js";

export {
  mountFrameBudgetHud,
  formatHud,
  type HudHandle,
} from "./hud.js";

export {
  ReplayRecorder,
  ReplayPlayer,
  serializeReplay,
  parseReplay,
  computeReplayResult,
  mulberry32,
  type ReplayEvent,
  type ReplayInputSnapshot,
  type ReplayMeta,
  type ReplayStream,
  type ReplayResult,
  type ReplayDeltaFrame,
  type RecorderOptions,
  type PlayerOptions,
} from "./replay.js";

/**
 * Publish a single (gpu_ms, gpu_ms_p95, gpu_ms_source) sample onto
 * `window.__augengineFrameBudget.latest` without disturbing existing
 * fields (Tracks 1, 3, 4 share the object). Idempotent — safe to call
 * every frame. No-op when the budget hook hasn't been mounted yet.
 *
 * The HUD + Track 1 proof-room sidecar both read these fields without
 * any further wiring.
 */
export {
  isHardwareAdapter,
  formatPerformanceRow,
  type ProofCaptureSample,
} from "./proof-capture.js";

export function publishGpuMs(sample: {
  gpu_ms: number;
  gpu_ms_p95: number;
  gpu_ms_source: "timestamp-query" | "cpu-fallback";
}): void {
  if (typeof window === "undefined") return;
  const g = window as unknown as {
    __augengineFrameBudget?: {
      latest?: {
        gpu_ms?: number;
        gpu_ms_p95?: number;
        gpu_ms_source?: "timestamp-query" | "cpu-fallback";
      } | null;
      /** Last GPU sample, kept so a newly emitted window can carry it. */
      lastGpu?: typeof sample;
    };
  };
  const hook = g.__augengineFrameBudget;
  if (!hook) return;
  hook.lastGpu = sample;
  if (!hook.latest) return;
  hook.latest.gpu_ms = sample.gpu_ms;
  hook.latest.gpu_ms_p95 = sample.gpu_ms_p95;
  hook.latest.gpu_ms_source = sample.gpu_ms_source;
}

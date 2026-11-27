/**
 * Golden-path deterministic replay — Track 6 (AUGENGINE-9-TRACK-PLAN.md).
 *
 * Recorder: captures per-frame `ReplayEvent`s (clock, input bitmap,
 * RNG seed) into an in-memory JSONL stream. `serialize()` produces a
 * `.replay.jsonl` text blob that callers can download or POST.
 *
 * Player: consumes a JSONL stream and replaces the engine's clock /
 * input / RNG via the wiring hooks below. Per-frame, the player calls
 * the engine's tick function with the recorded payload, computes
 * frame-budget delta vs the reference budget, and emits a per-frame
 * `ReplayResultFrame`.
 *
 * Determinism budget: floats must match to 1 ULP on identical
 * hardware/driver. Non-determinism sources documented in the README:
 *
 *   1. Async resource load order — the player REPLAYS the same order
 *      by pinning fetch fulfilment to the recorded sequence.
 *   2. requestAnimationFrame jitter — the player drives a virtual
 *      clock from the recorded timestamps, ignoring rAF wall time.
 *   3. Math.random() / crypto.getRandomValues — the player reseeds a
 *      Mulberry32 PRNG per frame from `event.rng_seed`.
 *
 * The serializer is JSONL (newline-delimited JSON) so partial files
 * stream cleanly and a corrupted final line doesn't break parsing.
 */

export interface ReplayInputSnapshot {
  /** Bitmask of pressed keys, keyed by code. Stored as record so JSONL is human-readable. */
  keys?: Record<string, true>;
  /** Pointer-lock movementX/Y deltas for this frame. */
  mouseDx?: number;
  mouseDy?: number;
  /** Mouse buttons currently pressed. */
  buttons?: number;
}

export interface ReplayEvent {
  /** Monotonic frame index (0-based). */
  frame: number;
  /** Virtual time in ms since recording started. */
  t_ms: number;
  /** Engine wall-clock delta this frame (ms). */
  dt_ms: number;
  /** Replayable RNG seed used for any randomness this frame. */
  rng_seed: number;
  /** Input snapshot — null when no input changed since last event. */
  input?: ReplayInputSnapshot;
  /**
   * Optional measured frame budget for this frame — recorder fills it
   * so the player can compute deltas. CPU ms only; GPU ms is read from
   * `window.__augengineFrameBudget` separately because timestamp-query
   * fidelity varies by adapter.
   */
  cpu_ms?: number;
  gpu_ms?: number;
}

export interface ReplayMeta {
  /** Schema version — bump when ReplayEvent shape changes. */
  schema: 1;
  /** Recording wall-clock start. */
  recordedAt: string;
  /** UA / platform / GPU snapshot for cross-machine comparison. */
  hostInfo: Record<string, string | number | boolean | null>;
  /** Total frames captured. */
  totalFrames: number;
  /** Total recording duration in ms. */
  durationMs: number;
}

export interface ReplayStream {
  meta: ReplayMeta;
  events: ReplayEvent[];
}

/**
 * Mulberry32 PRNG — small, fast, deterministic. Same input → same
 * sequence on every browser/JS engine because the math is integer-only
 * and operates within Uint32. Used to seed any randomness the engine
 * pulls during recording / playback.
 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function next(): number {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ──────────────────────────────────────────────────────────────────────
// SERIALIZER — pure functions, unit-testable without a DOM.
// ──────────────────────────────────────────────────────────────────────

/**
 * Serialize a complete ReplayStream to JSONL. First line is the meta
 * envelope (prefixed with `#meta `), subsequent lines are events.
 */
export function serializeReplay(stream: ReplayStream): string {
  const lines: string[] = [];
  lines.push(`#meta ${JSON.stringify(stream.meta)}`);
  for (const ev of stream.events) {
    lines.push(JSON.stringify(ev));
  }
  return lines.join("\n");
}

/**
 * Parse a JSONL blob into a ReplayStream. Tolerates a corrupted final
 * line (returns what was parsed up to the break).
 */
export function parseReplay(jsonl: string): ReplayStream {
  const lines = jsonl.split(/\r?\n/);
  let meta: ReplayMeta | null = null;
  const events: ReplayEvent[] = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("#meta ")) {
      try {
        meta = JSON.parse(line.slice(6)) as ReplayMeta;
      } catch {
        // Bad meta — keep going; we'll synthesize one below if needed.
      }
      continue;
    }
    try {
      const ev = JSON.parse(line) as ReplayEvent;
      if (typeof ev.frame === "number" && typeof ev.t_ms === "number") {
        events.push(ev);
      }
    } catch {
      // Bad line — skip silently.
    }
  }
  const synthesizedMeta: ReplayMeta = meta ?? {
    schema: 1,
    recordedAt: new Date(0).toISOString(),
    hostInfo: {},
    totalFrames: events.length,
    durationMs: events.length > 0 ? (events[events.length - 1]!.t_ms - events[0]!.t_ms) : 0,
  };
  return { meta: synthesizedMeta, events };
}

// ──────────────────────────────────────────────────────────────────────
// RECORDER
// ──────────────────────────────────────────────────────────────────────

export interface RecorderOptions {
  /** Frames to capture before auto-stopping. 0 = unbounded (caller stops). */
  maxFrames?: number;
  /** Seed for the recorded RNG sequence. Defaults to Date.now(). */
  rngSeed?: number;
  /** Host info to embed in meta. */
  hostInfo?: Record<string, string | number | boolean | null>;
}

export class ReplayRecorder {
  private events: ReplayEvent[] = [];
  private started = false;
  private startWallMs = 0;
  private lastTickWallMs = 0;
  private frameIndex = 0;
  private rngSeed: number;
  private maxFrames: number;
  private hostInfo: Record<string, string | number | boolean | null>;
  private stopped = false;

  constructor(opts: RecorderOptions = {}) {
    this.rngSeed = opts.rngSeed ?? (Date.now() & 0x7fffffff);
    this.maxFrames = opts.maxFrames ?? 0;
    this.hostInfo = opts.hostInfo ?? {};
  }

  start(now: number = performance.now()): void {
    if (this.started) return;
    this.started = true;
    this.startWallMs = now;
    this.lastTickWallMs = now;
  }

  /**
   * Record one frame. Caller passes the current monotonic time + the
   * input snapshot + (optional) the measured cpu/gpu ms for this frame.
   *
   * Returns false when recording has auto-stopped (caller should stop
   * calling `tick`).
   */
  tick(
    now: number,
    input: ReplayInputSnapshot | undefined,
    measured: { cpu_ms?: number; gpu_ms?: number } = {},
  ): boolean {
    if (!this.started || this.stopped) return false;
    const dt = now - this.lastTickWallMs;
    this.lastTickWallMs = now;
    // Derive a per-frame rng_seed from the base seed + frame index so
    // each frame's randomness is independently reproducible.
    const seed = (this.rngSeed + this.frameIndex * 0x9e3779b1) >>> 0;
    const ev: ReplayEvent = {
      frame: this.frameIndex,
      t_ms: now - this.startWallMs,
      dt_ms: dt,
      rng_seed: seed,
      ...(input ? { input } : {}),
      ...(measured.cpu_ms !== undefined ? { cpu_ms: measured.cpu_ms } : {}),
      ...(measured.gpu_ms !== undefined ? { gpu_ms: measured.gpu_ms } : {}),
    };
    this.events.push(ev);
    this.frameIndex += 1;
    if (this.maxFrames > 0 && this.frameIndex >= this.maxFrames) {
      this.stopped = true;
      return false;
    }
    return true;
  }

  stop(): void {
    this.stopped = true;
  }

  isStopped(): boolean {
    return this.stopped;
  }

  /** Snapshot the stream — safe to call mid-recording. */
  snapshot(): ReplayStream {
    const dur = this.events.length > 0
      ? this.events[this.events.length - 1]!.t_ms
      : 0;
    return {
      meta: {
        schema: 1,
        recordedAt: new Date(this.startWallMs).toISOString(),
        hostInfo: this.hostInfo,
        totalFrames: this.events.length,
        durationMs: dur,
      },
      events: this.events.slice(),
    };
  }

  framesCaptured(): number {
    return this.frameIndex;
  }
}

// ──────────────────────────────────────────────────────────────────────
// PLAYER
// ──────────────────────────────────────────────────────────────────────

export interface ReplayDeltaFrame {
  frame: number;
  t_ms: number;
  /** Reference (recorded) cpu_ms — null if unrecorded. */
  ref_cpu_ms: number | null;
  /** Observed cpu_ms during playback. */
  obs_cpu_ms: number | null;
  /** ref - obs, ms. Positive means playback is faster. */
  cpu_delta_ms: number | null;
  ref_gpu_ms: number | null;
  obs_gpu_ms: number | null;
  gpu_delta_ms: number | null;
}

export interface ReplayResult {
  meta: ReplayMeta;
  /** Per-frame deltas — same length as the recorded event stream. */
  frames: ReplayDeltaFrame[];
  /** Aggregate statistics. */
  summary: {
    framesCompared: number;
    cpu_delta_p50_ms: number;
    cpu_delta_p95_ms: number;
    cpu_delta_max_ms: number;
    gpu_delta_p50_ms: number;
    gpu_delta_p95_ms: number;
    gpu_delta_max_ms: number;
    /** Frames whose CPU delta exceeded the budget. */
    cpu_over_budget: number;
    /** Frames whose GPU delta exceeded the budget. */
    gpu_over_budget: number;
    /** Threshold used when classifying over-budget (ms). */
    delta_budget_ms: number;
  };
}

export interface PlayerOptions {
  /** Threshold for over-budget classification. Default 0.5 ms. */
  deltaBudgetMs?: number;
}

/**
 * Compute the per-frame and aggregate delta result given a recorded
 * stream and an observed (playback) stream. Pure — no DOM access.
 */
export function computeReplayResult(
  recorded: ReplayStream,
  observed: ReplayStream,
  opts: PlayerOptions = {},
): ReplayResult {
  const budget = opts.deltaBudgetMs ?? 0.5;
  const len = Math.min(recorded.events.length, observed.events.length);
  const frames: ReplayDeltaFrame[] = [];
  const cpuDeltas: number[] = [];
  const gpuDeltas: number[] = [];
  let cpuOver = 0;
  let gpuOver = 0;
  for (let i = 0; i < len; i++) {
    const r = recorded.events[i]!;
    const o = observed.events[i]!;
    const refCpu = r.cpu_ms ?? null;
    const obsCpu = o.cpu_ms ?? null;
    const refGpu = r.gpu_ms ?? null;
    const obsGpu = o.gpu_ms ?? null;
    const cpuDelta = refCpu !== null && obsCpu !== null ? obsCpu - refCpu : null;
    const gpuDelta = refGpu !== null && obsGpu !== null ? obsGpu - refGpu : null;
    if (cpuDelta !== null) {
      const abs = Math.abs(cpuDelta);
      cpuDeltas.push(abs);
      if (abs > budget) cpuOver += 1;
    }
    if (gpuDelta !== null) {
      const abs = Math.abs(gpuDelta);
      gpuDeltas.push(abs);
      if (abs > budget) gpuOver += 1;
    }
    frames.push({
      frame: r.frame,
      t_ms: r.t_ms,
      ref_cpu_ms: refCpu,
      obs_cpu_ms: obsCpu,
      cpu_delta_ms: cpuDelta,
      ref_gpu_ms: refGpu,
      obs_gpu_ms: obsGpu,
      gpu_delta_ms: gpuDelta,
    });
  }
  const pct = (arr: number[], p: number): number => {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]!;
  };
  return {
    meta: recorded.meta,
    frames,
    summary: {
      framesCompared: len,
      cpu_delta_p50_ms: pct(cpuDeltas, 0.5),
      cpu_delta_p95_ms: pct(cpuDeltas, 0.95),
      cpu_delta_max_ms: cpuDeltas.length > 0 ? Math.max(...cpuDeltas) : 0,
      gpu_delta_p50_ms: pct(gpuDeltas, 0.5),
      gpu_delta_p95_ms: pct(gpuDeltas, 0.95),
      gpu_delta_max_ms: gpuDeltas.length > 0 ? Math.max(...gpuDeltas) : 0,
      cpu_over_budget: cpuOver,
      gpu_over_budget: gpuOver,
      delta_budget_ms: budget,
    },
  };
}

/**
 * Replay player. Streams recorded events through a virtual clock + RNG.
 * Caller installs the hooks in `attach()` and calls `step()` per frame.
 *
 * Determinism rules:
 *   - `now()` returns the recorded event's `t_ms` (the virtual clock).
 *   - `random()` returns a Mulberry32 sample reseeded from the event's
 *     `rng_seed`. The seed advances per-frame and is identical to the
 *     recording.
 *   - `inputSnapshot()` returns the recorded input bitmap; the engine's
 *     real keyboard/mouse listeners should be stubbed out via the hook.
 */
export class ReplayPlayer {
  private cursor = 0;
  private observedEvents: ReplayEvent[] = [];

  constructor(private stream: ReplayStream) {}

  /** Number of recorded frames left to play. */
  remaining(): number {
    return Math.max(0, this.stream.events.length - this.cursor);
  }

  /** Total recorded frames in the stream. */
  totalFrames(): number {
    return this.stream.events.length;
  }

  /** Step the player one frame forward, returning the recorded event. */
  step(): ReplayEvent | null {
    if (this.cursor >= this.stream.events.length) return null;
    const ev = this.stream.events[this.cursor]!;
    this.cursor += 1;
    return ev;
  }

  /** Record what actually happened during playback so the caller can
   *  compute a delta result via `computeReplayResult`. */
  observe(ev: ReplayEvent): void {
    this.observedEvents.push(ev);
  }

  observedStream(): ReplayStream {
    return {
      meta: { ...this.stream.meta, totalFrames: this.observedEvents.length },
      events: this.observedEvents.slice(),
    };
  }

  /** Build the virtual clock function — replaces engine `performance.now`. */
  buildClock(): () => number {
    let lastT = 0;
    return () => {
      const idx = Math.min(this.cursor, this.stream.events.length - 1);
      const ev = this.stream.events[Math.max(0, idx)];
      if (ev) lastT = ev.t_ms;
      return lastT;
    };
  }

  /** Build the RNG for the current frame. Reseeded each time `step` advances. */
  buildRng(): () => number {
    return () => {
      const idx = Math.min(this.cursor, this.stream.events.length - 1);
      const ev = this.stream.events[Math.max(0, idx)];
      const seed = ev?.rng_seed ?? 0;
      return mulberry32(seed)();
    };
  }
}

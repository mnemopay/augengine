/**
 * WebGPU timestamp-query lifecycle helper — Track 6 (AUGENGINE-9-TRACK-PLAN.md).
 *
 * Wraps the 2-tick GPU timestamp pattern:
 *   1. begin pass:   timestampWrites { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
 *   2. end pass
 *   3. encoder.resolveQuerySet(querySet, 0, 2, resolveBuf, 0)
 *   4. encoder.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16)
 *   5. after submit:  await readBuf.mapAsync(GPUMapMode.READ)
 *   6. decode BigInt64Array → ms
 *
 * If the device doesn't expose `timestamp-query` we fall back to a
 * CPU-side wall-clock wrapped around `device.queue.submit()` — coarser
 * (driver buffers commands) but always available. `sourceForFrame()`
 * reports which path the last sample came from.
 *
 * The math (BigInt64Array → ms) is exported as a pure function so the
 * vitest unit test can exercise it without a GPU.
 */

/// <reference types="@webgpu/types" />

export type GpuMsSource = "timestamp-query" | "cpu-fallback";

export interface TimestampQuerySupport {
  /** Whether the adapter advertised `timestamp-query`. */
  supported: boolean;
  /** Why we fell back — useful in HUD/diagnostics. Empty when supported. */
  reason: string;
}

/**
 * Decode the two-entry timestamp buffer (nanosecond u64s) into a delta
 * in milliseconds. Exported pure so the unit test can hit it without a
 * GPU device.
 *
 * Note: timestamps from WebGPU are in nanoseconds (Chrome 122+ when
 * `--enable-webgpu-developer-features` is set; otherwise quantized to
 * 100µs but still nanosecond units).
 */
export function decodeTimestampDeltaMs(bytes: ArrayBuffer): number {
  const view = new BigInt64Array(bytes);
  if (view.length < 2) return 0;
  const t0 = view[0]!;
  const t1 = view[1]!;
  // Guard against a clock that went backwards (driver quirks, rare).
  const delta = t1 > t0 ? t1 - t0 : 0n;
  return Number(delta) / 1e6;
}

/**
 * Probe whether the supplied adapter advertises `timestamp-query`.
 * Returns a structured result so callers can surface the failure reason
 * (rather than just a boolean) in HUD / diagnostics.
 */
export function probeTimestampQuerySupport(adapter: GPUAdapter | null): TimestampQuerySupport {
  if (!adapter) return { supported: false, reason: "no adapter" };
  const has = (adapter.features as ReadonlySet<string>).has?.("timestamp-query")
    ?? Array.from(adapter.features as unknown as Iterable<string>).includes("timestamp-query");
  return has
    ? { supported: true, reason: "" }
    : { supported: false, reason: "adapter does not advertise timestamp-query" };
}

/**
 * Compute the required features array to pass to `adapter.requestDevice`.
 * Caller is responsible for passing through other features it needs.
 */
export function requiredFeaturesForTimestamps(adapter: GPUAdapter | null): GPUFeatureName[] {
  const probe = probeTimestampQuerySupport(adapter);
  return probe.supported ? (["timestamp-query"] as GPUFeatureName[]) : [];
}

/**
 * Per-frame timestamp-query session. Allocates a 2-slot query set and
 * two persistent buffers (resolve + readback). Reused across frames by
 * design — the GPU buffers are large enough for a single render pass.
 *
 * Multi-pass usage is supported by allocating multiple `TimestampSession`
 * instances; that's intentionally out of scope for Track 6's single
 * main-pass wiring.
 */
export class TimestampSession {
  readonly querySet: GPUQuerySet;
  private resolveBuf: GPUBuffer;
  private readBuf: GPUBuffer;
  /** True while a previous frame's readBuf is still mapped or pending. */
  private inflight = false;
  /** True when this frame's encoder copied fresh queries into readBuf. */
  private copied = false;
  /** Last successfully-decoded GPU pass duration in ms. */
  private lastGpuMs: number = 0;
  /** Where the last sample came from. */
  private lastSource: GpuMsSource = "cpu-fallback";

  constructor(private device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    this.resolveBuf = device.createBuffer({
      size: 16, // 2 × u64
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Build the `timestampWrites` payload for `encoder.beginRenderPass()`.
   * Caller spreads it into the pass descriptor when the device supports
   * timestamps.
   */
  passTimestampWrites(): GPURenderPassTimestampWrites {
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }

  /**
   * Span several passes: put `frameBeginWrites()` on the FIRST pass of the
   * frame and `frameEndWrites()` on the LAST. The delta then covers every
   * pass in between (shadow, opaque, water, post chain), which is the number
   * that decides whether a frame makes vsync — a main-pass-only timer hides
   * the post chain entirely.
   */
  frameBeginWrites(): GPURenderPassTimestampWrites {
    return { querySet: this.querySet, beginningOfPassWriteIndex: 0 };
  }

  frameEndWrites(): GPURenderPassTimestampWrites {
    return { querySet: this.querySet, endOfPassWriteIndex: 1 };
  }

  /**
   * After the pass has ended, resolve queries into the readback buffer.
   * Caller already holds the encoder. Safe to call every frame: while the
   * previous readback is still mapped or pending, the copy is skipped
   * (submitting a copy into a mapped buffer is a validation error).
   */
  resolveAfterPass(encoder: GPUCommandEncoder): void {
    if (this.inflight) return;
    encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, 16);
    this.copied = true;
  }

  /**
   * After `device.queue.submit([encoder.finish()])`, map the readback
   * buffer and decode the delta. Returns null when the previous read is
   * still pending — caller falls back to the prior sample (don't await
   * inside the hot loop).
   */
  async readFrameMs(): Promise<number | null> {
    if (this.inflight || !this.copied) return null;
    this.copied = false;
    this.inflight = true;
    try {
      await this.readBuf.mapAsync(GPUMapMode.READ);
      const copy = this.readBuf.getMappedRange().slice(0);
      this.readBuf.unmap();
      const ms = decodeTimestampDeltaMs(copy);
      this.lastGpuMs = ms;
      this.lastSource = "timestamp-query";
      return ms;
    } catch {
      // Driver dropped the readback — record nothing, keep going.
      return null;
    } finally {
      this.inflight = false;
    }
  }

  /** CPU-fallback path: caller wraps `device.queue.submit()` in a wall-
   *  clock measurement and feeds the result back via this method. */
  recordCpuFallback(ms: number): void {
    this.lastGpuMs = ms;
    this.lastSource = "cpu-fallback";
  }

  lastMs(): number {
    return this.lastGpuMs;
  }

  source(): GpuMsSource {
    return this.lastSource;
  }

  destroy(): void {
    this.querySet.destroy();
    this.resolveBuf.destroy();
    this.readBuf.destroy();
  }
}

/**
 * Labels the per-pass timer knows about. A frame only reports the passes it
 * actually encoded; "frame" is derived (first begin → last end).
 */
export type PassLabel = "skin" | "shadow" | "main" | "water" | "post" | "ao" | "taa" | "motion";
export const PASS_LABELS: readonly PassLabel[] = ["skin", "shadow", "main", "water", "post", "ao", "taa", "motion"];

/** One resolved frame of per-pass GPU durations, in milliseconds. */
export interface PassTimingSample {
  /** First pass begin → last pass end, including idle gaps between passes. */
  frame_ms: number;
  /** Sum of the individual pass durations (excludes the gaps). */
  sum_ms: number;
  passes: Partial<Record<PassLabel, number>>;
}

/**
 * Decode a resolved query buffer (2 u64 ticks per label, in PASS_LABELS
 * order) into per-pass milliseconds. Only labels in `used` are reported;
 * unwritten queries resolve to 0 and would otherwise read as garbage spans.
 * Exported pure so it can be unit-tested without a GPU.
 */
export function decodePassTimings(bytes: ArrayBuffer, used: ReadonlySet<PassLabel>): PassTimingSample {
  const view = new BigInt64Array(bytes);
  const passes: Partial<Record<PassLabel, number>> = {};
  let first: bigint | null = null;
  let last: bigint | null = null;
  let sum = 0;
  PASS_LABELS.forEach((label, i) => {
    if (!used.has(label)) return;
    const t0 = view[i * 2];
    const t1 = view[i * 2 + 1];
    if (t0 === undefined || t1 === undefined || t0 === 0n || t1 === 0n || t1 < t0) return;
    const ms = Number(t1 - t0) / 1e6;
    passes[label] = ms;
    sum += ms;
    if (first === null || t0 < first) first = t0;
    if (last === null || t1 > last) last = t1;
  });
  const frame = first !== null && last !== null ? Number((last as bigint) - (first as bigint)) / 1e6 : 0;
  return { frame_ms: frame, sum_ms: sum, passes };
}

/**
 * Per-pass GPU timer. One query set holds a begin/end pair for every label in
 * PASS_LABELS; each instrumented pass gets its own pair, so a frame reports
 * shadow / main / water / post separately instead of one frame-spanning
 * number. A span covering several passes (the post chain) puts `spanBegin`
 * on its first pass and `spanEnd` on its last.
 *
 * Same readback discipline as TimestampSession: the copy is skipped while the
 * previous readback is mapped, and the listener receives the most recently
 * resolved frame (one or two frames behind the encoder).
 */
export class PassTimestampSession {
  readonly querySet: GPUQuerySet;
  private resolveBuf: GPUBuffer;
  private readBuf: GPUBuffer;
  private inflight = false;
  private copied = false;
  /** Labels written by the frame currently being encoded. */
  private usedThisFrame = new Set<PassLabel>();
  /** Labels belonging to the bytes sitting in readBuf. */
  private usedInCopy = new Set<PassLabel>();
  private readonly bytes = PASS_LABELS.length * 2 * 8;

  constructor(private device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: PASS_LABELS.length * 2 });
    this.resolveBuf = device.createBuffer({
      size: this.bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: this.bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /** Call once at the start of encoding a frame. */
  beginFrame(): void {
    this.usedThisFrame.clear();
  }

  private index(label: PassLabel): number {
    return PASS_LABELS.indexOf(label) * 2;
  }

  /** begin + end writes for a single pass (render or compute). */
  writes(label: PassLabel): GPURenderPassTimestampWrites & GPUComputePassTimestampWrites {
    this.usedThisFrame.add(label);
    const i = this.index(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 };
  }

  /** Begin-only write: first pass of a multi-pass span. */
  spanBegin(label: PassLabel): GPURenderPassTimestampWrites {
    this.usedThisFrame.add(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: this.index(label) };
  }

  /** End-only write: last pass of a multi-pass span. */
  spanEnd(label: PassLabel): GPURenderPassTimestampWrites {
    this.usedThisFrame.add(label);
    return { querySet: this.querySet, endOfPassWriteIndex: this.index(label) + 1 };
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (this.inflight || this.usedThisFrame.size === 0) return;
    encoder.resolveQuerySet(this.querySet, 0, PASS_LABELS.length * 2, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, this.bytes);
    this.usedInCopy = new Set(this.usedThisFrame);
    this.copied = true;
  }

  async read(): Promise<PassTimingSample | null> {
    if (this.inflight || !this.copied) return null;
    this.copied = false;
    this.inflight = true;
    try {
      await this.readBuf.mapAsync(GPUMapMode.READ);
      const copy = this.readBuf.getMappedRange().slice(0);
      this.readBuf.unmap();
      return decodePassTimings(copy, this.usedInCopy);
    } catch {
      return null;
    } finally {
      this.inflight = false;
    }
  }

  destroy(): void {
    this.querySet.destroy();
    this.resolveBuf.destroy();
    this.readBuf.destroy();
  }
}

/**
 * Rolling p95 over the last N (default 60) gpu_ms samples. Shared shape
 * with the existing frame-budget rolling window so the HUD + proof
 * sidecar can read it the same way.
 */
export class RollingGpuMs {
  private buf: number[] = [];
  constructor(private cap: number = 60) {}

  push(ms: number): void {
    this.buf.push(ms);
    if (this.buf.length > this.cap) this.buf.shift();
  }

  avg(): number {
    if (this.buf.length === 0) return 0;
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length;
  }

  p95(): number {
    if (this.buf.length === 0) return 0;
    const sorted = [...this.buf].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))]!;
  }

  clear(): void {
    this.buf.length = 0;
  }
}

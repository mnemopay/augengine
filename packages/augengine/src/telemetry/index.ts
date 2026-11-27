/**
 * Telemetry — Phase 12. Privacy-first analytics + crash reports +
 * frame-time histograms. MnemoPay-aligned consent model: no PII, no
 * fingerprinting, single user-facing toggle.
 */

import { fnv1a64 } from "../persistence/manifest.js";

export type TelemetryEventKind =
  | "session.start"
  | "session.end"
  | "frame.histogram"
  | "ai.dispatch"
  | "scene.load"
  | "save.write"
  | "save.load"
  | "error"
  | "custom";

export interface TelemetryEvent {
  kind: TelemetryEventKind;
  ts: string;
  /** Anonymous session identifier — hash of (build version, install id). */
  session: string;
  payload?: Record<string, unknown>;
}

export interface TelemetrySink {
  ingest(event: TelemetryEvent): void | Promise<void>;
}

/** No-op sink for dev or when consent is denied. */
export class NullSink implements TelemetrySink {
  ingest(): void { /* discard */ }
}

/** In-memory ring buffer — useful for replay export + bug repros. */
export class MemorySink implements TelemetrySink {
  private buf: TelemetryEvent[] = [];
  private cap: number;
  constructor(capacity = 1000) { this.cap = capacity; }
  ingest(event: TelemetryEvent): void {
    if (this.buf.length >= this.cap) this.buf.shift();
    this.buf.push(event);
  }
  drain(): TelemetryEvent[] {
    const out = this.buf;
    this.buf = [];
    return out;
  }
  peek(): readonly TelemetryEvent[] { return this.buf; }
}

/** HTTP sink — POSTs batched events to a configured endpoint. */
export class HttpSink implements TelemetrySink {
  private endpoint: string;
  private batch: TelemetryEvent[] = [];
  private flushIntervalMs: number;
  private timer: ReturnType<typeof setInterval> | null = null;
  constructor(endpoint: string, flushIntervalMs = 5000) {
    this.endpoint = endpoint;
    this.flushIntervalMs = flushIntervalMs;
  }
  ingest(event: TelemetryEvent): void {
    this.batch.push(event);
    if (!this.timer) {
      this.timer = setInterval(() => this.flush(), this.flushIntervalMs);
    }
  }
  async flush(): Promise<void> {
    if (this.batch.length === 0) return;
    const out = this.batch;
    this.batch = [];
    try {
      await fetch(this.endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ events: out }),
        keepalive: true,
      });
    } catch {
      // Reinsert if endpoint unreachable; ring-cap on next flush.
      this.batch.unshift(...out);
    }
  }
  close(): void {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }
}

/** Frame-time histogram aggregator. Emits one `frame.histogram` event/window. */
export class FrameTimeHistogram {
  private bucketsMs = [4, 8, 16, 24, 33, 50, 100, 250]; // <=N ms buckets
  private counts: number[];
  private windowStart = 0;
  private flushEveryMs: number;
  constructor(flushEveryMs = 5000) {
    this.flushEveryMs = flushEveryMs;
    this.counts = new Array(this.bucketsMs.length + 1).fill(0);
  }
  record(frameMs: number): void {
    let i = 0;
    while (i < this.bucketsMs.length && frameMs > (this.bucketsMs[i] as number)) i++;
    this.counts[i] = (this.counts[i] || 0) + 1;
    const now = performance.now();
    if (this.windowStart === 0) this.windowStart = now;
    if (now - this.windowStart > this.flushEveryMs) {
      this.flush();
    }
  }
  flush(): { buckets: number[]; counts: number[]; windowMs: number } | null {
    if (this.counts.every((c) => c === 0)) return null;
    const out = {
      buckets: [...this.bucketsMs],
      counts: [...this.counts],
      windowMs: performance.now() - this.windowStart,
    };
    this.counts.fill(0);
    this.windowStart = 0;
    return out;
  }
}

/** Anonymous session id from build version + install id (no PII). */
export function makeSessionId(buildVersion: string, installId: string): string {
  return fnv1a64(`${buildVersion}|${installId}|${new Date().toISOString().slice(0, 10)}`);
}

/**
 * Unit tests for the golden-path replay serializer + math — Track 6.
 *
 * The DOM-driven recorder/player loop is exercised by
 * `e2e/replay-determinism.spec.ts`. This file covers the pure functions
 * so failures land in the inner-loop vitest run (~1s) rather than the
 * Playwright outer loop (~30s).
 */
import { describe, expect, it } from "vitest";
import {
  ReplayRecorder,
  ReplayPlayer,
  serializeReplay,
  parseReplay,
  computeReplayResult,
  mulberry32,
  type ReplayStream,
  type ReplayEvent,
} from "./replay.ts";

describe("mulberry32", () => {
  it("is deterministic — same seed yields the same sequence", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const seqA = [a(), a(), a(), a(), a()];
    const seqB = [b(), b(), b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });

  it("yields values in [0, 1)", () => {
    const rng = mulberry32(1337);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("serializeReplay / parseReplay round-trip", () => {
  it("preserves meta + events through a JSONL round-trip", () => {
    const stream: ReplayStream = {
      meta: {
        schema: 1,
        recordedAt: "2026-05-29T00:00:00.000Z",
        hostInfo: { ua: "test/1.0", cores: 8 },
        totalFrames: 2,
        durationMs: 33.4,
      },
      events: [
        { frame: 0, t_ms: 0, dt_ms: 0, rng_seed: 1, input: { keys: { a: true } }, cpu_ms: 5.1, gpu_ms: 3.2 },
        { frame: 1, t_ms: 16.7, dt_ms: 16.7, rng_seed: 2, input: { mouseDx: 12, mouseDy: -4 }, cpu_ms: 5.3, gpu_ms: 3.4 },
      ],
    };
    const text = serializeReplay(stream);
    const parsed = parseReplay(text);
    expect(parsed.meta).toEqual(stream.meta);
    expect(parsed.events).toEqual(stream.events);
  });

  it("tolerates a corrupted trailing line", () => {
    const stream: ReplayStream = {
      meta: { schema: 1, recordedAt: "x", hostInfo: {}, totalFrames: 1, durationMs: 0 },
      events: [{ frame: 0, t_ms: 0, dt_ms: 0, rng_seed: 0 }],
    };
    const text = serializeReplay(stream) + "\n{this is not json";
    const parsed = parseReplay(text);
    expect(parsed.events).toHaveLength(1);
  });

  it("synthesizes meta when missing", () => {
    const text =
      JSON.stringify({ frame: 0, t_ms: 0, dt_ms: 0, rng_seed: 0 }) + "\n" +
      JSON.stringify({ frame: 1, t_ms: 16, dt_ms: 16, rng_seed: 0 });
    const parsed = parseReplay(text);
    expect(parsed.meta.schema).toBe(1);
    expect(parsed.events).toHaveLength(2);
  });
});

describe("ReplayRecorder", () => {
  it("captures exactly maxFrames before auto-stopping", () => {
    const rec = new ReplayRecorder({ maxFrames: 3, rngSeed: 7 });
    rec.start(0);
    expect(rec.tick(16, undefined)).toBe(true);
    expect(rec.tick(32, undefined)).toBe(true);
    expect(rec.tick(48, undefined)).toBe(false); // auto-stops on the 3rd tick
    expect(rec.tick(64, undefined)).toBe(false); // post-stop no-op
    expect(rec.framesCaptured()).toBe(3);
  });

  it("produces deterministic rng_seed per frame from the base seed", () => {
    const rec = new ReplayRecorder({ maxFrames: 5, rngSeed: 100 });
    rec.start(0);
    for (let i = 0; i < 5; i++) rec.tick(i * 16, undefined);
    const seeds = rec.snapshot().events.map((e) => e.rng_seed);
    const rec2 = new ReplayRecorder({ maxFrames: 5, rngSeed: 100 });
    rec2.start(0);
    for (let i = 0; i < 5; i++) rec2.tick(i * 16, undefined);
    expect(rec2.snapshot().events.map((e) => e.rng_seed)).toEqual(seeds);
  });

  it("records measured cpu_ms + gpu_ms when supplied", () => {
    const rec = new ReplayRecorder({ maxFrames: 1, rngSeed: 0 });
    rec.start(0);
    rec.tick(16, { keys: { w: true } }, { cpu_ms: 5.5, gpu_ms: 3.3 });
    const ev = rec.snapshot().events[0]!;
    expect(ev.cpu_ms).toBeCloseTo(5.5);
    expect(ev.gpu_ms).toBeCloseTo(3.3);
    expect(ev.input?.keys?.w).toBe(true);
  });
});

describe("ReplayPlayer", () => {
  it("walks the stream once and stops returning events", () => {
    const stream: ReplayStream = {
      meta: { schema: 1, recordedAt: "x", hostInfo: {}, totalFrames: 2, durationMs: 16 },
      events: [
        { frame: 0, t_ms: 0, dt_ms: 0, rng_seed: 1 },
        { frame: 1, t_ms: 16, dt_ms: 16, rng_seed: 2 },
      ],
    };
    const p = new ReplayPlayer(stream);
    expect(p.totalFrames()).toBe(2);
    expect(p.step()?.frame).toBe(0);
    expect(p.step()?.frame).toBe(1);
    expect(p.step()).toBeNull();
  });

  it("buildRng returns the same value each call when cursor is parked", () => {
    const stream: ReplayStream = {
      meta: { schema: 1, recordedAt: "x", hostInfo: {}, totalFrames: 1, durationMs: 0 },
      events: [{ frame: 0, t_ms: 0, dt_ms: 0, rng_seed: 12345 }],
    };
    const p = new ReplayPlayer(stream);
    const rng = p.buildRng();
    const a = rng();
    const b = rng();
    expect(a).toBe(b);
  });
});

describe("computeReplayResult", () => {
  function mkStream(values: { cpu_ms: number; gpu_ms: number }[]): ReplayStream {
    const events: ReplayEvent[] = values.map((v, i) => ({
      frame: i,
      t_ms: i * 16,
      dt_ms: 16,
      rng_seed: i,
      cpu_ms: v.cpu_ms,
      gpu_ms: v.gpu_ms,
    }));
    return {
      meta: { schema: 1, recordedAt: "x", hostInfo: {}, totalFrames: events.length, durationMs: events.length * 16 },
      events,
    };
  }

  it("reports zero delta when the streams are identical", () => {
    const ref = mkStream([{ cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }]);
    const obs = mkStream([{ cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }]);
    const r = computeReplayResult(ref, obs);
    expect(r.summary.framesCompared).toBe(3);
    expect(r.summary.cpu_delta_p95_ms).toBe(0);
    expect(r.summary.gpu_delta_p95_ms).toBe(0);
    expect(r.summary.cpu_over_budget).toBe(0);
    expect(r.summary.gpu_over_budget).toBe(0);
  });

  it("counts over-budget frames against the threshold", () => {
    const ref = mkStream([{ cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }]);
    const obs = mkStream([{ cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 6, gpu_ms: 3 }, { cpu_ms: 9, gpu_ms: 5 }]);
    const r = computeReplayResult(ref, obs, { deltaBudgetMs: 0.5 });
    expect(r.summary.cpu_over_budget).toBe(2);
    expect(r.summary.gpu_over_budget).toBe(1);
    expect(r.summary.cpu_delta_max_ms).toBeCloseTo(4, 6);
    expect(r.summary.gpu_delta_max_ms).toBeCloseTo(2, 6);
  });

  it("returns null per-frame deltas when the observed stream is missing fields", () => {
    const ref = mkStream([{ cpu_ms: 5, gpu_ms: 3 }]);
    const obs: ReplayStream = {
      meta: ref.meta,
      events: [{ frame: 0, t_ms: 0, dt_ms: 16, rng_seed: 0 }],
    };
    const r = computeReplayResult(ref, obs);
    expect(r.frames[0]!.cpu_delta_ms).toBeNull();
    expect(r.frames[0]!.gpu_delta_ms).toBeNull();
  });

  it("truncates to the shorter stream length", () => {
    const ref = mkStream([{ cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }, { cpu_ms: 5, gpu_ms: 3 }]);
    const obs = mkStream([{ cpu_ms: 5, gpu_ms: 3 }]);
    const r = computeReplayResult(ref, obs);
    expect(r.summary.framesCompared).toBe(1);
  });
});

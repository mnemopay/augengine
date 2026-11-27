/**
 * Unit tests for WebGPU timestamp-query helpers — Track 6.
 *
 * The session lifecycle (createQuerySet → resolveQuerySet → mapAsync)
 * requires a real GPU device and is exercised by `e2e/proof-room-capture.spec.ts`
 * and `e2e/replay-determinism.spec.ts`. This unit file covers the pure
 * math and pure feature-probe paths so regressions land before they
 * reach the Playwright run (~30s feedback vs ~10s here).
 */
import { beforeAll, describe, expect, it } from "vitest";
import {
  decodePassTimings,
  decodeTimestampDeltaMs,
  PASS_LABELS,
  probeTimestampQuerySupport,
  requiredFeaturesForTimestamps,
  RollingGpuMs,
  TimestampSession,
} from "./timestamp-queries.ts";

describe("decodeTimestampDeltaMs", () => {
  it("converts a 1ms delta in nanoseconds to 1.0 ms", () => {
    // 1ms = 1_000_000 ns. Pack as two u64.
    const buf = new ArrayBuffer(16);
    const view = new BigInt64Array(buf);
    view[0] = 0n;
    view[1] = 1_000_000n;
    expect(decodeTimestampDeltaMs(buf)).toBeCloseTo(1.0, 6);
  });

  it("handles sub-millisecond precision down to 1 microsecond", () => {
    const buf = new ArrayBuffer(16);
    const view = new BigInt64Array(buf);
    view[0] = 1_000n;
    view[1] = 2_000n; // 1µs apart
    expect(decodeTimestampDeltaMs(buf)).toBeCloseTo(0.001, 6);
  });

  it("returns 0 when the clock goes backwards (driver quirk)", () => {
    const buf = new ArrayBuffer(16);
    const view = new BigInt64Array(buf);
    view[0] = 5_000_000n;
    view[1] = 4_000_000n;
    expect(decodeTimestampDeltaMs(buf)).toBe(0);
  });

  it("returns 0 for a too-small buffer", () => {
    expect(decodeTimestampDeltaMs(new ArrayBuffer(8))).toBe(0);
  });

  it("handles very large nanosecond timestamps without precision loss above µs", () => {
    const buf = new ArrayBuffer(16);
    const view = new BigInt64Array(buf);
    // 1 hour in ns = 3_600_000_000_000
    view[0] = 1_000_000_000_000n;
    view[1] = 1_000_016_667_000n; // +16.667 ms
    expect(decodeTimestampDeltaMs(buf)).toBeCloseTo(16.667, 3);
  });
});

describe("probeTimestampQuerySupport", () => {
  it("returns supported=true when feature is present (Set)", () => {
    const fakeAdapter = { features: new Set(["timestamp-query", "depth-clip-control"]) } as unknown as GPUAdapter;
    const r = probeTimestampQuerySupport(fakeAdapter);
    expect(r.supported).toBe(true);
    expect(r.reason).toBe("");
  });

  it("returns supported=false with reason when feature absent", () => {
    const fakeAdapter = { features: new Set(["depth-clip-control"]) } as unknown as GPUAdapter;
    const r = probeTimestampQuerySupport(fakeAdapter);
    expect(r.supported).toBe(false);
    expect(r.reason).toMatch(/timestamp-query/);
  });

  it("returns supported=false on null adapter", () => {
    const r = probeTimestampQuerySupport(null);
    expect(r.supported).toBe(false);
    expect(r.reason).toBe("no adapter");
  });

  it("requiredFeaturesForTimestamps returns [] when unsupported", () => {
    const fakeAdapter = { features: new Set([]) } as unknown as GPUAdapter;
    expect(requiredFeaturesForTimestamps(fakeAdapter)).toEqual([]);
  });

  it("requiredFeaturesForTimestamps returns ['timestamp-query'] when supported", () => {
    const fakeAdapter = { features: new Set(["timestamp-query"]) } as unknown as GPUAdapter;
    expect(requiredFeaturesForTimestamps(fakeAdapter)).toEqual(["timestamp-query"]);
  });
});

describe("RollingGpuMs", () => {
  it("computes a stable p95 over a uniform window", () => {
    const r = new RollingGpuMs(60);
    for (let i = 0; i < 60; i++) r.push(8);
    expect(r.avg()).toBeCloseTo(8, 6);
    expect(r.p95()).toBeCloseTo(8, 6);
  });

  it("separates avg from p95 with a tail spike", () => {
    const r = new RollingGpuMs(60);
    for (let i = 0; i < 57; i++) r.push(4);
    r.push(20);
    r.push(40);
    r.push(60);
    // p95 of a 60-sample sorted window: index = floor(60*0.95) = 57 → 20 (the smallest of the spikes).
    expect(r.p95()).toBeGreaterThanOrEqual(20);
    expect(r.avg()).toBeLessThan(r.p95());
  });

  it("drops the oldest sample when capacity is exceeded", () => {
    const r = new RollingGpuMs(3);
    r.push(100);
    r.push(1);
    r.push(1);
    r.push(1); // pushes 100 out
    expect(r.avg()).toBeCloseTo(1, 6);
  });
});

describe("decodePassTimings", () => {
  function pack(pairs: Array<[bigint, bigint]>): ArrayBuffer {
    const buf = new ArrayBuffer(PASS_LABELS.length * 16);
    const v = new BigInt64Array(buf);
    pairs.forEach(([a, b], i) => { v[i * 2] = a; v[i * 2 + 1] = b; });
    return buf;
  }

  it("reports each used pass and the frame span across them", () => {
    // skin 1.0, shadow 0.5, main 4.0, water unused, post 2.0 (ns ticks)
    const buf = pack([
      [1_000_000n, 2_000_000n],
      [2_100_000n, 2_600_000n],
      [2_700_000n, 6_700_000n],
      [0n, 0n],
      [7_000_000n, 9_000_000n],
    ]);
    const s = decodePassTimings(buf, new Set(["skin", "shadow", "main", "post"]));
    expect(s.passes.skin).toBeCloseTo(1.0, 6);
    expect(s.passes.shadow).toBeCloseTo(0.5, 6);
    expect(s.passes.main).toBeCloseTo(4.0, 6);
    expect(s.passes.post).toBeCloseTo(2.0, 6);
    expect(s.passes.water).toBeUndefined();
    expect(s.sum_ms).toBeCloseTo(7.5, 6);
    expect(s.frame_ms).toBeCloseTo(8.0, 6); // 1.0 → 9.0 ms, gaps included
  });

  it("ignores labels the frame did not write, even if the slot has stale ticks", () => {
    const buf = pack([[0n, 0n], [5n, 99_000_000n], [1_000_000n, 3_000_000n], [0n, 0n], [0n, 0n]]);
    const s = decodePassTimings(buf, new Set(["main"]));
    expect(Object.keys(s.passes)).toEqual(["main"]);
    expect(s.frame_ms).toBeCloseTo(2.0, 6);
  });

  it("drops a pass whose clock went backwards", () => {
    const buf = pack([[0n, 0n], [0n, 0n], [5_000_000n, 4_000_000n], [0n, 0n], [0n, 0n]]);
    const s = decodePassTimings(buf, new Set(["main"]));
    expect(s.passes.main).toBeUndefined();
    expect(s.frame_ms).toBe(0);
  });
});

describe("TimestampSession readback lifecycle", () => {
  // Node does not define the WebGPU globals; stand in with the spec's bit values.
  beforeAll(() => {
    const g = globalThis as Record<string, unknown>;
    g.GPUBufferUsage ??= {
      MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
      VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
    };
    g.GPUMapMode ??= { READ: 1, WRITE: 2 };
  });

  // Minimal fake device: records copies into the readback buffer and lets the
  // test resolve mapAsync by hand, so we can assert nothing is copied into the
  // readback buffer while it is mapped (WebGPU: "used in submit while mapped").
  function fakeDevice() {
    const ns = new BigInt64Array([1_000_000n, 3_500_000n]);
    let resolveMap: (() => void) | null = null;
    const readBuf = {
      mapAsync: () => new Promise<void>((r) => { resolveMap = r; }),
      getMappedRange: () => ns.buffer.slice(0),
      unmap: () => {},
      destroy: () => {},
    };
    const other = { destroy: () => {} };
    let made = 0;
    const device = {
      createQuerySet: () => other,
      createBuffer: () => (made++ === 0 ? other : readBuf),
    } as unknown as GPUDevice;
    const copies: unknown[] = [];
    const encoder = {
      resolveQuerySet: () => {},
      copyBufferToBuffer: (_s: unknown, _so: number, dst: unknown) => copies.push(dst),
    } as unknown as GPUCommandEncoder;
    return { device, encoder, copies, readBuf, finishMap: () => resolveMap?.() };
  }

  it("skips the copy while the previous readback is still pending", async () => {
    const f = fakeDevice();
    const s = new TimestampSession(f.device);
    s.resolveAfterPass(f.encoder);
    const pending = s.readFrameMs();
    expect(f.copies.length).toBe(1);
    s.resolveAfterPass(f.encoder); // next frame while mapAsync is pending
    expect(f.copies.length).toBe(1);
    f.finishMap();
    expect(await pending).toBeCloseTo(2.5, 6);
    s.resolveAfterPass(f.encoder); // buffer free again
    expect(f.copies.length).toBe(2);
  });

  it("does not map when no fresh copy was recorded", async () => {
    const f = fakeDevice();
    const s = new TimestampSession(f.device);
    expect(await s.readFrameMs()).toBeNull();
  });
});

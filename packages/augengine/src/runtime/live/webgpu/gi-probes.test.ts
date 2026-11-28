import { describe, expect, it } from "vitest";
import { DEFAULT_GI, fibonacciDir, projectL1, rayBox } from "./gi-probes.js";
import { LIGHTING_PRESETS } from "./lighting-presets.js";
import { PROBE_RING } from "./probe-ring.js";

const dirs = (n: number, off = 0) => Array.from({ length: n }, (_, i) => fibonacciDir(i, n, off));

describe("GI probe L1 projection", () => {
  it("uniform radiance L → irradiance/π = L at every normal", () => {
    const k = projectL1(dirs(512).map((dir) => ({ dir, L: 3 })));
    expect(k[0]).toBeCloseTo(3, 9);
    for (const c of k.slice(1)) expect(Math.abs(c)).toBeLessThan(0.02);
  });

  it("sky above (L=1 for ω·y > 0): ≈1 facing up, ≈0 facing down, ½ sideways (L1 is exact for these)", () => {
    const k = projectL1(dirs(4096, 0.37).map((dir) => ({ dir, L: dir[1] > 0 ? 1 : 0 })));
    const ev = (n: [number, number, number]) => k[0] + k[1] * n[0] + k[2] * n[1] + k[3] * n[2];
    expect(ev([0, 1, 0])).toBeCloseTo(1, 2);
    expect(ev([0, -1, 0])).toBeCloseTo(0, 2);
    expect(ev([1, 0, 0])).toBeCloseTo(0.5, 2);
  });

  it("Fibonacci directions are unit length and cover the sphere evenly", () => {
    const d = dirs(256, 0.5);
    let mx = 0, my = 0, mz = 0;
    for (const v of d) {
      expect(Math.hypot(...v)).toBeCloseTo(1, 9);
      mx += v[0]; my += v[1]; mz += v[2];
    }
    expect(Math.abs(my / 256)).toBeLessThan(1e-9);
    expect(Math.hypot(mx, mz) / 256).toBeLessThan(0.02);
  });
});

describe("GI ray vs proxy box", () => {
  it("slab test hits the near face, misses beside it, ignores boxes behind or around the origin", () => {
    expect(rayBox([0, 1, 0], [1, 0, 0], [2, 0, -1], [3, 2, 1])).toBeCloseTo(2, 9);
    expect(rayBox([0, 1, 0], [1, 0, 0], [2, 3, -1], [3, 4, 1])).toBe(-1);
    expect(rayBox([0, 1, 0], [-1, 0, 0], [2, 0, -1], [3, 2, 1])).toBe(-1);
    // Origin inside: no entry ahead (the probe-inside case is handled separately).
    expect(rayBox([2.5, 1, 0], [1, 0, 0], [2, 0, -1], [3, 2, 1])).toBe(-1);
  });
});

describe("GI presets", () => {
  it("windows fit the probe ring without aliasing; low has no GI", () => {
    expect(LIGHTING_PRESETS.low.gi).toBeNull();
    for (const p of ["medium", "high", "ultra"] as const) {
      const g = { ...DEFAULT_GI, ...LIGHTING_PRESETS[p].gi! };
      expect(g.columns).toBeLessThanOrEqual(PROBE_RING / 2);
      expect(g.rays % 32).toBe(0);
    }
  });
});

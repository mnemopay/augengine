import { describe, expect, it } from "vitest";
import {
  TAA_JITTER_SAMPLES,
  applyProjectionJitter,
  halton,
  jitterToUv,
  linearizeDepth,
  scaledSize,
  taaJitter,
} from "./temporal.js";
import { m4Create, m4Perspective } from "../../../scene/math.js";

describe("halton / taaJitter", () => {
  it("matches the radical inverse in base 2 and 3", () => {
    expect([1, 2, 3, 4].map((i) => halton(i, 2))).toEqual([0.5, 0.25, 0.75, 0.125]);
    expect(halton(1, 3)).toBeCloseTo(1 / 3, 12);
    expect(halton(2, 3)).toBeCloseTo(2 / 3, 12);
    expect(halton(3, 3)).toBeCloseTo(1 / 9, 12);
  });

  it("stays inside half a pixel, averages near zero, and repeats every 8 frames", () => {
    const pts = Array.from({ length: TAA_JITTER_SAMPLES }, (_, i) => taaJitter(i));
    for (const [x, y] of pts) {
      expect(Math.abs(x)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(y)).toBeLessThanOrEqual(0.5);
    }
    const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
    const my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
    expect(Math.abs(mx)).toBeLessThan(0.1);
    expect(Math.abs(my)).toBeLessThan(0.1);
    expect(new Set(pts.map((p) => p.join(","))).size).toBe(TAA_JITTER_SAMPLES);
    expect(taaJitter(TAA_JITTER_SAMPLES + 3)).toEqual(taaJitter(3));
    expect(taaJitter(-1)).toEqual(taaJitter(TAA_JITTER_SAMPLES - 1));
  });
});

/** Project a view-space point with a column-major mat4; returns NDC xy. */
function ndc(p: Float32Array, v: [number, number, number]): [number, number] {
  const x = p[0]! * v[0] + p[4]! * v[1] + p[8]! * v[2] + p[12]!;
  const y = p[1]! * v[0] + p[5]! * v[1] + p[9]! * v[2] + p[13]!;
  const w = p[3]! * v[0] + p[7]! * v[1] + p[11]! * v[2] + p[15]!;
  return [x / w, y / w];
}

describe("applyProjectionJitter", () => {
  it("shifts every projected point by exactly (jx, jy) pixels", () => {
    const w = 1008, h = 630;
    const base = m4Create() as unknown as Float32Array;
    m4Perspective(base as never, Math.PI / 3, w / h, 0.1, 100);
    const jit = Float32Array.from(base);
    applyProjectionJitter(jit, 0.25, -0.375, w, h);
    for (const v of [[0, 0, -1], [3, -2, -10], [-7, 4, -60]] as [number, number, number][]) {
      const a = ndc(base, v);
      const b = ndc(jit, v);
      // NDC spans 2 units across the target: one pixel = 2 / size.
      expect(((b[0] - a[0]) * w) / 2).toBeCloseTo(0.25, 5);
      expect(((b[1] - a[1]) * h) / 2).toBeCloseTo(-0.375, 5);
    }
  });

  it("jitterToUv flips y (uv grows downward) and scales by the target size", () => {
    expect(jitterToUv(0.5, 0.25, 100, 50)).toEqual([0.005, -0.005]);
  });
});

describe("linearizeDepth", () => {
  it("inverts the engine's projection depth for points along -z", () => {
    const p = m4Create() as unknown as Float32Array;
    m4Perspective(p as never, Math.PI / 3, 1.6, 0.1, 100);
    for (const dist of [0.5, 2, 17, 80]) {
      const z = -dist;
      const depth = (p[10]! * z + p[14]!) / -z;
      expect(linearizeDepth(depth, p[10]!, p[14]!)).toBeCloseTo(dist, 4);
    }
  });
});

describe("scaledSize", () => {
  it("rounds, clamps the scale to [0.25, 1], and never returns 0", () => {
    expect(scaledSize(1440, 900, 0.7)).toEqual({ width: 1008, height: 630 });
    expect(scaledSize(1440, 900, 2)).toEqual({ width: 1440, height: 900 });
    expect(scaledSize(1440, 900, 0.01)).toEqual({ width: 360, height: 225 });
    expect(scaledSize(1, 1, 0.5)).toEqual({ width: 1, height: 1 });
  });
});

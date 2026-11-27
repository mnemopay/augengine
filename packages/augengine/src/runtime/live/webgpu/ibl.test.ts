import { describe, expect, it } from "vitest";
import { evalZonalAmbient, packSkyAmbient, skyRadiance, zonalAmbientSH } from "./ibl.js";
import { OVERCLOCK_SKY } from "./sky.js";
import type { Vec3 } from "../../../scene/types.js";

/** Brute-force cosine-weighted irradiance / π for normal (0, ny, ·) of a y-only radiance. */
function bruteAmbient(radiance: (y: number) => number, ny: number): number {
  const nx = Math.sqrt(Math.max(0, 1 - ny * ny));
  let sum = 0;
  const nT = 400, nP = 400;
  for (let i = 0; i < nT; i++) {
    const y = -1 + ((i + 0.5) * 2) / nT;
    const r = Math.sqrt(1 - y * y);
    for (let j = 0; j < nP; j++) {
      const phi = ((j + 0.5) * 2 * Math.PI) / nP;
      const cos = Math.max(0, nx * r * Math.cos(phi) + ny * y);
      sum += radiance(y) * cos * (2 / nT) * ((2 * Math.PI) / nP);
    }
  }
  return sum / Math.PI;
}

describe("zonalAmbientSH", () => {
  it("a constant sky gives a constant ambient equal to that radiance", () => {
    const k = zonalAmbientSH(() => [0.5, 0.25, 1]);
    for (const ny of [-1, -0.3, 0, 0.6, 1]) {
      const a = evalZonalAmbient(k, ny);
      expect(a[0]).toBeCloseTo(0.5, 3);
      expect(a[1]).toBeCloseTo(0.25, 3);
      expect(a[2]).toBeCloseTo(1, 3);
    }
  });

  it("matches brute-force cosine convolution for a linear and a step sky", () => {
    for (const f of [(y: number) => 0.5 + 0.5 * y, (y: number) => (y > 0 ? 1 : 0.2)]) {
      const k = zonalAmbientSH((y) => [f(y), f(y), f(y)] as Vec3);
      for (const ny of [-1, -0.5, 0, 0.5, 1]) {
        // L2 truncation of a step is within a few percent of the true irradiance.
        expect(evalZonalAmbient(k, ny)[0]).toBeCloseTo(bruteAmbient(f, ny), 1);
      }
    }
  });
});

describe("packSkyAmbient", () => {
  it("keeps the sphere-average equal to the flat ambient and is brighter facing up", () => {
    const out = new Float32Array(12);
    const ambient: Vec3 = [0.12, 0.14, 0.22];
    packSkyAmbient(out, 0, OVERCLOCK_SKY, ambient);
    expect(out[3]).toBe(1);
    const k: [Vec3, Vec3, Vec3] = [
      [out[0]!, out[1]!, out[2]!], [out[4]!, out[5]!, out[6]!], [out[8]!, out[9]!, out[10]!],
    ];
    // Average over the sphere of a(ny): integrate ny uniformly in [-1, 1].
    for (let c = 0; c < 3; c++) {
      let avg = 0;
      for (let i = 0; i < 1000; i++) avg += evalZonalAmbient(k, -1 + (i + 0.5) / 500)[c]! / 1000;
      expect(avg).toBeCloseTo(ambient[c]!, 4);
    }
    const up = evalZonalAmbient(k, 1), down = evalZonalAmbient(k, -1);
    expect(up[0] + up[1] + up[2]).toBeGreaterThan(down[0] + down[1] + down[2]);
  });

  it("uses a neutral dome when there is no sky gradient", () => {
    const out = new Float32Array(12);
    packSkyAmbient(out, 0, null, [0.2, 0.2, 0.2]);
    expect(out[0]).toBeCloseTo(0.2, 5);
    expect(out[4]!).toBeGreaterThan(0); // brighter upward
  });

  it("skyRadiance below the horizon is the horizon colour times the bounce", () => {
    expect(skyRadiance(OVERCLOCK_SKY, -0.5, 0.5)).toEqual(OVERCLOCK_SKY.horizon.map((c) => c * 0.5));
  });
});

describe("sky specular (split-sum)", () => {
  it("environment BRDF approx matches the known DFG ends", async () => {
    const { envBrdfApprox } = await import("./ibl.js");
    // Smooth, head-on: almost all F0 (A≈1, B≈0).
    const [a0, b0] = envBrdfApprox(0, 1);
    expect(a0).toBeGreaterThan(0.9);
    expect(Math.abs(b0)).toBeLessThan(0.05);
    // Rough: A and B both small; grazing adds B (Fresnel).
    const [a1] = envBrdfApprox(1, 1);
    expect(a1).toBeLessThan(0.6);
    const [, bg] = envBrdfApprox(0.2, 0.05);
    expect(bg).toBeGreaterThan(b0);
  });

  it("damps higher SH bands more as roughness grows (blurrier reflection)", async () => {
    const { specularBandWeight } = await import("./ibl.js");
    for (const l of [1, 2]) {
      expect(specularBandWeight(l, 0.1)).toBeGreaterThan(specularBandWeight(l, 0.5));
      expect(specularBandWeight(l, 0.5)).toBeGreaterThanOrEqual(specularBandWeight(l, 1));
    }
    expect(specularBandWeight(0, 0.9)).toBe(1);
    expect(specularBandWeight(2, 0.05)).toBeGreaterThan(0.99);
  });

  it("raw-radiance SH reconstructs a smooth gradient and packs with the enable flag", async () => {
    const { zonalRadianceSH, packSkySpecular } = await import("./ibl.js");
    const lin = (y: number): [number, number, number] => [0.5 + 0.3 * y, 0.5, 0.5];
    const [c0, c1, c2] = zonalRadianceSH(lin);
    for (const y of [-0.8, 0, 0.7]) {
      const r = c0[0] + c1[0] * y + c2[0] * (3 * y * y - 1);
      expect(r).toBeCloseTo(lin(y)[0], 2);
    }
    const out = new Float32Array(12);
    packSkySpecular(out, 0, null, [0.2, 0.2, 0.2]);
    expect(out[3]).toBe(1);
    expect(out[0]).toBeGreaterThan(0);
  });
});

import { describe, expect, it } from "vitest";
import { envBrdfApprox } from "./ibl.js";
import { DFG_LUT_SIZE, DFG_LUT_WGSL, dfgLutCoords, envIblResponse, integrateDfg } from "./ibl-brdf.js";

describe("split-sum DFG LUT", () => {
  it("smooth surfaces reflect everything at normal incidence (A + B → 1)", () => {
    const [a, b] = integrateDfg(0.02, 0.99);
    expect(a + b).toBeGreaterThan(0.98);
    expect(a + b).toBeLessThanOrEqual(1.001);
  });

  it("single-scatter GGX energy (F0 = 1) is ≤ 1 and falls with roughness", () => {
    const e = [0.1, 0.5, 1].map((r) => integrateDfg(r, 0.7).reduce((s, x) => s + x, 0));
    for (const v of e) expect(v).toBeLessThanOrEqual(1.001);
    expect(e[0]!).toBeGreaterThan(e[2]!);
  });

  it("converges: the LUT's sample count is within 0.01 of a 4× denser integral", () => {
    for (const [r, nv] of [[0.3, 0.2], [0.6, 0.5], [0.9, 0.9]] as const) {
      const lut = integrateDfg(r, nv);
      const ref = integrateDfg(r, nv, 1024, 4);
      expect(Math.abs(lut[0] - ref[0])).toBeLessThan(0.01);
      expect(Math.abs(lut[1] - ref[1])).toBeLessThan(0.01);
    }
  });

  it("the WGSL generator uses the same sample counts and texel-centre mapping", () => {
    expect(DFG_LUT_WGSL).toContain("256u");
    expect(DFG_LUT_WGSL).toContain("4u");
    expect(dfgLutCoords(0, DFG_LUT_SIZE - 1)).toEqual([0.5 / DFG_LUT_SIZE, (DFG_LUT_SIZE - 0.5) / DFG_LUT_SIZE]);
  });

  it("records how far Karis's analytic fit (legacy path) is from the integral", () => {
    let worst = 0;
    for (const r of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      for (const nv of [0.1, 0.3, 0.6, 0.95]) {
        const [a, b] = envBrdfApprox(r, nv);
        const [A, B] = integrateDfg(r, nv, 128, 4);
        worst = Math.max(worst, Math.abs(a - A), Math.abs(b - B));
      }
    }
    // eslint-disable-next-line no-console
    console.info(`[furnace] Karis fit max |fit − integral|: ${worst.toFixed(3)} (why the natural tier uses a LUT)`);
    expect(worst).toBeGreaterThan(0.1);
  });
});

describe("white furnace (uniform environment of radiance 1)", () => {
  it("multiple-scattering IBL never creates energy; a white metal reflects ~1 at every roughness", () => {
    for (const r of [0.05, 0.25, 0.5, 0.75, 1]) {
      for (const nv of [0.2, 0.5, 1]) {
        const metal = envIblResponse(1, 1, r, nv);
        expect(metal).toBeLessThanOrEqual(1.005);
        expect(metal).toBeGreaterThan(0.99);
        const diel = envIblResponse(1, 0, r, nv);
        expect(diel).toBeLessThanOrEqual(1.005);
        expect(diel).toBeGreaterThan(0.95);
      }
    }
  });

  it("without the compensation a rough white metal loses energy (what the multiscatter term fixes)", () => {
    const ss = envIblResponse(1, 1, 1, 0.5, false);
    const ms = envIblResponse(1, 1, 1, 0.5, true);
    // eslint-disable-next-line no-console
    console.info(`[furnace] rough white metal: single-scatter ${ss.toFixed(3)}, with multiscatter ${ms.toFixed(3)}`);
    expect(ss).toBeLessThan(0.8);
    expect(ms).toBeGreaterThan(0.99);
  });

  it("an 18% grey dielectric reflects ~18% plus its specular rim, never more than 1", () => {
    const v = envIblResponse(0.18, 0, 0.6, 0.8);
    expect(v).toBeGreaterThan(0.18);
    expect(v).toBeLessThan(0.3);
  });
});

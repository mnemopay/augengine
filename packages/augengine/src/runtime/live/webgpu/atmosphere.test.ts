import { describe, expect, it } from "vitest";
import { aerialExtinction, henyeyGreenstein } from "./sky-model.js";
import { FRAME_LAYOUT, FRAME_UNIFORM_SIZE } from "./lights.js";
import { buildWgsl } from "./pipeline-factory.js";
import { LIGHTING_PRESETS } from "./lighting-presets.js";

describe("aerial perspective", () => {
  it("turbidity 2.5: σ(550 nm) ≈ 1.4e-4 /m → Koschmieder visibility ≈ 28 km, bluer light extinguished more", () => {
    const s = aerialExtinction(2.5);
    expect(s[1]).toBeGreaterThan(1.2e-4);
    expect(s[1]).toBeLessThan(1.6e-4);
    expect(3.912 / s[1] / 1000).toBeGreaterThan(24);
    expect(3.912 / s[1] / 1000).toBeLessThan(33);
    expect(s[2]).toBeGreaterThan(s[1]);
    expect(s[1]).toBeGreaterThan(s[0]);
  });

  it("hazier air extinguishes more", () => {
    expect(aerialExtinction(8)[1]).toBeGreaterThan(aerialExtinction(2)[1] * 2);
  });
});

describe("Henyey–Greenstein phase", () => {
  it("integrates to 1 over the sphere and is isotropic at g = 0", () => {
    for (const g of [0, 0.5, 0.8, -0.3]) {
      let sum = 0;
      const n = 4000;
      for (let i = 0; i < n; i++) {
        const mu = -1 + (2 * (i + 0.5)) / n;
        sum += henyeyGreenstein(mu, g) * 2 * Math.PI * (2 / n);
      }
      expect(sum).toBeCloseTo(1, 3);
    }
    expect(henyeyGreenstein(0.3, 0)).toBeCloseTo(1 / (4 * Math.PI), 9);
    // Forward scattering: brighter toward the light.
    expect(henyeyGreenstein(1, 0.7)).toBeGreaterThan(henyeyGreenstein(-1, 0.7) * 50);
  });
});

describe("atmosphere wiring", () => {
  it("frame uniform carries the atmosphere block (1152 bytes, 16-byte aligned)", () => {
    expect(FRAME_LAYOUT.atmos % 4).toBe(0);
    expect(FRAME_LAYOUT.atmosSigma).toBe(FRAME_LAYOUT.atmos + 4);
    expect(FRAME_UNIFORM_SIZE).toBe(1152);
  });

  it("every lit material applies fog through augengine_fog_apply", () => {
    for (const kind of ["pbr", "cel", "vrm-mtoon"] as const) {
      const src = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1 });
      expect(src).toContain("augengine_fog_apply(lit, in.worldPos)");
      expect(src).not.toContain("mix(lit, u_frame.fog.rgb, augengine_fog_factor(in.worldPos))");
    }
  });

  it("volumetric shafts only on high/ultra presets", () => {
    expect(LIGHTING_PRESETS.low.volumetricSteps).toBe(0);
    expect(LIGHTING_PRESETS.medium.volumetricSteps).toBe(0);
    expect(LIGHTING_PRESETS.high.volumetricSteps).toBeGreaterThan(0);
    expect(LIGHTING_PRESETS.ultra.volumetricSteps).toBeGreaterThan(LIGHTING_PRESETS.high.volumetricSteps);
  });
});

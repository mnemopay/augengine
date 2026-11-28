import { describe, expect, it } from "vitest";
import { aerialExtinction, henyeyGreenstein } from "./sky-model.js";
import { FRAME_LAYOUT, FRAME_UNIFORM_SIZE } from "./lights.js";
import { buildWgsl } from "./pipeline-factory.js";
import { legacyFogVeil, resolveNaturalLight } from "./renderer.js";
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

describe("legacy scene fog under natural light", () => {
  // CPU mirror of augengine_fog_apply's legacy optical depth with the veil.
  const smoothstep = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const legacyOpacity = (density: number, dist: number, veil: number) => 1 - Math.exp(-density * dist * (veil > 0 ? smoothstep(0, veil, dist) : 1));

  it("veil is the default and keeps the Loading Range's streaming edge hidden as before", () => {
    expect(resolveNaturalLight({}).legacyFog).toBe("veil");
    const veil = legacyFogVeil("veil", 0.012, null);
    expect(veil).toBeCloseTo(125, 9);
    // At and past the veil distance the legacy fog is exactly what it was.
    for (const d of [125, 200, 400]) expect(legacyOpacity(0.012, d, veil)).toBeCloseTo(legacyOpacity(0.012, d, 0), 12);
    // At the 112 m load radius it still hides tiles almost as well (73% vs 74%).
    expect(legacyOpacity(0.012, 112, veil)).toBeGreaterThan(0.72);
    // In the middle distance the scene keeps its contrast: 30 m goes from 30% fog to 5%.
    expect(legacyOpacity(0.012, 30, 0)).toBeCloseTo(0.30, 2);
    expect(legacyOpacity(0.012, 30, veil)).toBeLessThan(0.06);
  });

  it("keep / off / no fog / explicit distance", () => {
    expect(legacyFogVeil("keep", 0.012, null)).toBe(0);
    expect(legacyFogVeil("off", 0.012, null)).toBe(0);
    expect(legacyFogVeil("veil", 0, null)).toBe(0);
    expect(legacyFogVeil("veil", 0.012, 80)).toBe(80);
    expect(resolveNaturalLight({ legacyFog: "keep" }).legacyFog).toBe("keep");
    expect(resolveNaturalLight({ legacyFogVeilDistance: -5 }).legacyFogVeilDistance).toBeNull();
  });

  it("the shader scales the legacy optical depth by the veil only when atmos.w > 0", () => {
    const src = buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1, natural: true });
    expect(src).toContain("if (u_frame.atmos.w > 0.0) { optical = optical * smoothstep(0.0, u_frame.atmos.w, dist); }");
  });
});

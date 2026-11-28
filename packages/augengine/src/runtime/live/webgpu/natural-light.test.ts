import { describe, expect, it } from "vitest";
import {
  airMass,
  buildSkyModel,
  horizontalSkyIlluminance,
  iesClearSky,
  luminance,
  perez,
  preethamCoefficients,
  preethamZenith,
  PREETHAM_CALIBRATION,
  skyRadianceAt,
  sunFromAngles,
  sunIlluminanceClear,
  sunPosition,
  xyYToLinearSrgb,
} from "./sky-model.js";
import {
  convolveSh9Diffuse,
  equirectDirection,
  evalSh9,
  projectEquirectSh9,
  projectSh9,
} from "./sh9.js";
import {
  DEFAULT_AUTO_EXPOSURE,
  DEFAULT_HISTOGRAM,
  adaptEv100,
  binCenterLog2,
  ev100FromAverageLuminance,
  ev100FromCamera,
  exposureFromEv100,
  histogramAverageLog2,
  histogramBin,
  kelvinToLinearSrgb,
  planckianXy,
  targetEv100FromHistogram,
  whiteBalanceGains,
} from "./exposure.js";

const DEG = Math.PI / 180;

describe("sun position", () => {
  it("equinox solar noon at 40°N: elevation 50°, due south", () => {
    // Day 80 ≈ 21 March; Cooper's declination is within ~0.5° of zero there.
    const s = sunPosition({ latitudeDeg: 40, dayOfYear: 80, solarHour: 12 });
    expect(s.elevation / DEG).toBeCloseTo(50, 0);
    expect(s.azimuth / DEG).toBeCloseTo(180, 3);
    // South = +z in the engine's world (−z is north).
    expect(s.dir[2]).toBeGreaterThan(0.6);
    expect(Math.abs(s.dir[0])).toBeLessThan(1e-9);
  });

  it("summer solstice noon at the Tropic of Cancer: sun overhead", () => {
    const s = sunPosition({ latitudeDeg: 23.44, dayOfYear: 172, solarHour: 12 });
    expect(s.elevation / DEG).toBeGreaterThan(89.5);
  });

  it("equator, equinox, 06:00 solar: sun on the eastern horizon", () => {
    const s = sunPosition({ latitudeDeg: 0, dayOfYear: 80, solarHour: 6 });
    expect(Math.abs(s.elevation / DEG)).toBeLessThan(1);
    expect(s.azimuth / DEG).toBeCloseTo(90, 0);
    expect(s.dir[0]).toBeGreaterThan(0.99); // +x = east
  });

  it("afternoon sun is in the west, and angles round-trip", () => {
    const s = sunPosition({ latitudeDeg: 51.5, dayOfYear: 200, solarHour: 16 });
    expect(s.dir[0]).toBeLessThan(0);
    const back = sunFromAngles(s.elevation / DEG, s.azimuth / DEG);
    for (let i = 0; i < 3; i++) expect(back.dir[i]).toBeCloseTo(s.dir[i]!, 9);
  });
});

describe("Preetham sky model samples", () => {
  it("coefficient table matches the paper at T = 2", () => {
    const c = preethamCoefficients(2);
    expect(c.Y.A).toBeCloseTo(-1.1056, 4);
    expect(c.Y.B).toBeCloseTo(-0.2833, 4);
    expect(c.Y.C).toBeCloseTo(5.2797, 4);
    expect(c.Y.D).toBeCloseTo(-2.3359, 4);
    expect(c.Y.E).toBeCloseTo(0.2363, 4);
  });

  it("zenith sample equals the paper's zenith luminance and chromaticity", () => {
    const sun = sunFromAngles(45, 180);
    const m = buildSkyModel({ sunDir: sun.dir, turbidity: 3 });
    const z = preethamZenith(3, 45 * DEG);
    // At the zenith F(θ=0, γ=θs) / F(0, θs) = 1 → exactly the zenith values.
    const rgb = skyRadianceAt(m, [0, 1, 0]);
    const expected = xyYToLinearSrgb(z.x, z.y, z.Y * PREETHAM_CALIBRATION);
    for (let i = 0; i < 3; i++) expect(rgb[i]! - m.nightFloor * [0.6, 0.8, 1.3][i]!).toBeCloseTo(expected[i]!, 6);
    // A clear daytime zenith is bluish and a few kcd/m².
    expect(z.x).toBeGreaterThan(0.24);
    expect(z.x).toBeLessThan(0.3);
    expect(z.Y).toBeGreaterThan(2000);
    expect(z.Y).toBeLessThan(15000);
    expect(rgb[2]).toBeGreaterThan(rgb[0]);
  });

  it("circumsolar region is brighter than the anti-solar sky; horizon brighter than mid-sky away from sun", () => {
    const sun = sunFromAngles(30, 180);
    const m = buildSkyModel({ sunDir: sun.dir, turbidity: 2.5 });
    const near = luminance(skyRadianceAt(m, sunFromAngles(33, 180).dir));
    const away = luminance(skyRadianceAt(m, sunFromAngles(33, 0).dir));
    expect(near).toBeGreaterThan(away * 3);
    // Symmetric about the solar meridian.
    const l = luminance(skyRadianceAt(m, sunFromAngles(20, 150).dir));
    const r = luminance(skyRadianceAt(m, sunFromAngles(20, 210).dir));
    expect(l).toBeCloseTo(r, 6);
    expect(perez(1, 0.3, m.coeffs.Y)).toBeGreaterThan(0);
  });

  it("raw Preetham is ~1.75–2× IES; the single 45° calibration keeps 15°–75° within ±15% of IES", () => {
    const ratios: number[] = [];
    for (const el of [15, 30, 45, 60, 75]) {
      const ref = iesClearSky(el * DEG).skyHorizontal;
      const raw = buildSkyModel({ sunDir: sunFromAngles(el, 180).dir, turbidity: 2.5, calibration: 1 });
      const cal = buildSkyModel({ sunDir: sunFromAngles(el, 180).dir, turbidity: 2.5 });
      const rawRatio = luminance(horizontalSkyIlluminance(raw, 32, 64)) / ref;
      expect(rawRatio).toBeGreaterThan(1.5);
      expect(rawRatio).toBeLessThan(2.2);
      ratios.push(luminance(horizontalSkyIlluminance(cal, 32, 64)) / ref);
    }
    // Recorded for docs/LIGHTING.md — calibrated Preetham vs IES.
    // eslint-disable-next-line no-console
    console.info("[sky] calibrated Preetham/IES horizontal sky illuminance (15..75°):", ratios.map((r) => r.toFixed(2)).join(" "));
    for (const r of ratios) {
      expect(r).toBeGreaterThan(0.85);
      expect(r).toBeLessThan(1.15);
    }
    expect(PREETHAM_CALIBRATION).toBeGreaterThan(0.45);
    expect(PREETHAM_CALIBRATION).toBeLessThan(0.6);
  });

  it("direct sun illuminance tracks the IES direct-normal equation within ±15% (T = 2.5, sun ≥ 15°)", () => {
    for (const el of [15, 30, 60, 90]) {
      const E = luminance(sunIlluminanceClear(el * DEG, 2.5));
      const ref = iesClearSky(el * DEG).sunNormal;
      expect(E / ref).toBeGreaterThan(0.85);
      expect(E / ref).toBeLessThan(1.15);
    }
    // Lower sun → redder light.
    const low = sunIlluminanceClear(5 * DEG, 2.5);
    const high = sunIlluminanceClear(60 * DEG, 2.5);
    expect(low[0] / low[2]).toBeGreaterThan(high[0] / high[2]);
    expect(airMass(0)).toBeCloseTo(1, 3);
  });

  it("overcast: neutral CIE distribution, sun hidden, zenith 3× horizon", () => {
    const m = buildSkyModel({ sunDir: sunFromAngles(40, 180).dir, cloudCover: 1 });
    expect(luminance(m.sunIlluminance)).toBe(0);
    const z = luminance(skyRadianceAt(m, [0, 1, 0]));
    const h = luminance(skyRadianceAt(m, [1, 0.0001, 0]));
    expect(z / h).toBeCloseTo(3, 1);
    // CIE overcast horizontal illuminance = 7π/9 · Lz.
    const E = luminance(horizontalSkyIlluminance(m, 32, 64));
    expect(E / ((7 * Math.PI) / 9 * m.overcastZenith)).toBeCloseTo(1, 1);
  });

  it("night: sun below −8° leaves only the dim floor", () => {
    const m = buildSkyModel({ sunDir: sunFromAngles(-12, 180).dir });
    expect(luminance(m.sunIlluminance)).toBe(0);
    expect(luminance(skyRadianceAt(m, [0, 1, 0]))).toBeLessThan(0.1);
  });
});

describe("SH9 projection", () => {
  it("a uniform environment of radiance L evaluates to L at every normal", () => {
    const sh = convolveSh9Diffuse(projectSh9(() => [2, 1, 0.5], 4096));
    for (const n of [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0.577, -0.577, 0.577]] as const) {
      const e = evalSh9(sh, n);
      expect(e[0]).toBeCloseTo(2, 3);
      expect(e[1]).toBeCloseTo(1, 3);
      expect(e[2]).toBeCloseTo(0.5, 3);
    }
  });

  it("upper-hemisphere sky of radiance 1: E/π = 1 facing up, ½ sideways, ~0 facing down", () => {
    const sh = convolveSh9Diffuse(projectSh9((d) => (d[1] > 0 ? [1, 1, 1] : [0, 0, 0]), 8192));
    // Exact: up 1, horizon 0.5, down 0. Order-2 SH is within a few percent.
    expect(evalSh9(sh, [0, 1, 0])[0]).toBeCloseTo(1, 1);
    expect(evalSh9(sh, [1, 0, 0])[0]).toBeCloseTo(0.5, 2);
    expect(Math.abs(evalSh9(sh, [0, -1, 0])[0])).toBeLessThan(0.08);
  });

  it("equirect projection agrees with direct sphere projection", () => {
    const f = (d: readonly number[]): [number, number, number] => [Math.max(d[1]!, 0) * 3 + 0.2, Math.max(d[0]!, 0), 0.1 + Math.max(-d[2]!, 0)];
    const W = 256, H = 128;
    const img = new Float32Array(W * H * 3);
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const d = equirectDirection((c + 0.5) / W, (r + 0.5) / H);
        img.set(f(d), (r * W + c) * 3);
      }
    }
    const a = convolveSh9Diffuse(projectEquirectSh9(img, W, H));
    const b = convolveSh9Diffuse(projectSh9(f, 8192));
    for (let i = 0; i < 27; i++) expect(a[i]).toBeCloseTo(b[i]!, 2);
  });

  it("equirect mapping: centre column looks north (−z), three-quarters east (+x), top row up", () => {
    const n = equirectDirection(0.5, 0.5);
    expect(n[2]).toBeCloseTo(-1, 9);
    const e = equirectDirection(0.75, 0.5);
    expect(e[0]).toBeCloseTo(1, 9);
    expect(equirectDirection(0.3, 0)[1]).toBeCloseTo(1, 9);
  });

});

describe("exposure math", () => {
  it("Sunny-16: f/16, 1/100 s, ISO 100 ≈ EV100 14.6", () => {
    expect(ev100FromCamera(16, 1 / 100, 100)).toBeCloseTo(Math.log2(25600), 6);
    expect(ev100FromCamera(16, 1 / 100, 100)).toBeCloseTo(14.64, 2);
  });

  it("average luminance ↔ EV100 with K = 12.5", () => {
    // L = 12.5/100 · 2^EV → EV 0 at 0.125 cd/m², EV 10 at 128.
    expect(ev100FromAverageLuminance(0.125)).toBeCloseTo(0, 9);
    expect(ev100FromAverageLuminance(128)).toBeCloseTo(10, 9);
  });

  it("saturation exposure maps the metered average to 1/(1.2·8) ≈ 0.104", () => {
    const L = 4000;
    const H = exposureFromEv100(ev100FromAverageLuminance(L));
    expect(L * H).toBeCloseTo(12.5 / 100 / 1.2, 9);
  });

  it("histogram bins + percentile average recover a uniform scene's luminance", () => {
    const counts = new Array(DEFAULT_HISTOGRAM.bins).fill(0);
    const lum = 0.25;
    counts[histogramBin(lum)] = 1000;
    const avg = histogramAverageLog2(counts, 0.5, 0.95)!;
    // Within half a bin (18 stops / 63 bins).
    expect(Math.abs(avg - Math.log2(lum))).toBeLessThan(0.5 * (18 / 63) + 1e-9);
    expect(histogramBin(0)).toBe(0);
    expect(histogramBin(1e9)).toBe(DEFAULT_HISTOGRAM.bins - 1);
    expect(binCenterLog2(1)).toBeGreaterThan(DEFAULT_HISTOGRAM.minLog2);
  });

  it("percentile window ignores a tiny very bright region (the sun)", () => {
    const counts = new Array(64).fill(0);
    counts[histogramBin(0.1)] = 990;
    counts[histogramBin(50)] = 10;
    const avg = histogramAverageLog2(counts, 0.5, 0.95)!;
    expect(Math.abs(avg - Math.log2(0.1))).toBeLessThan(0.2);
  });

  it("target EV divides the pre-exposure back out", () => {
    const counts = new Array(64).fill(0);
    const scene = 5000; // cd/m²
    const H = 1e-4;
    counts[histogramBin(scene * H)] = 100;
    const ev = targetEv100FromHistogram(counts, H, { ...DEFAULT_AUTO_EXPOSURE, compensation: 0 })!;
    expect(Math.abs(ev - ev100FromAverageLuminance(scene))).toBeLessThan(0.2);
    // Compensation shifts the target one-for-one.
    expect(targetEv100FromHistogram(counts, H, { ...DEFAULT_AUTO_EXPOSURE, compensation: -1 })!).toBeCloseTo(ev - 1, 9);
  });

  it("adaptation approaches the target without overshoot and respects the rate", () => {
    let ev = 5;
    for (let i = 0; i < 600; i++) ev = adaptEv100(ev, 15, 1 / 60);
    expect(ev).toBeGreaterThan(14.9);
    expect(ev).toBeLessThanOrEqual(15);
    // One frame never moves more than speed·dt.
    expect(adaptEv100(5, 15, 1 / 60) - 5).toBeLessThanOrEqual(3 / 60 + 1e-12);
    expect(adaptEv100(Number.NaN, 7, 0.016)).toBe(7);
  });
});

describe("white balance", () => {
  it("Planckian locus samples (Kim et al.): 2856 K ≈ illuminant A, 6504 K ≈ D65", () => {
    const [xa, ya] = planckianXy(2856);
    // The spline approximates the locus to ~1e-3.
    expect(Math.abs(xa - 0.4476)).toBeLessThan(1.5e-3);
    expect(Math.abs(ya - 0.4074)).toBeLessThan(1.5e-3);
    const [x65, y65] = planckianXy(6504);
    expect(x65).toBeCloseTo(0.3135, 2);
    expect(y65).toBeCloseTo(0.3236, 2);
  });

  it("neutral at 6504 K, warm light gets blue-boosting gains, luminance preserved", () => {
    const g0 = whiteBalanceGains(6504);
    for (const v of g0) expect(v).toBeCloseTo(1, 9);
    const g = whiteBalanceGains(3200);
    expect(g[2]).toBeGreaterThan(g[0]);
    expect(luminance(g)).toBeCloseTo(1, 9);
    // Applying the gains to that light's own colour makes it (nearly) neutral relative to 6504 K.
    const c = kelvinToLinearSrgb(3200);
    const ref = kelvinToLinearSrgb(6504);
    const balanced = [c[0] * g[0], c[1] * g[1], c[2] * g[2]];
    const ratio = balanced.map((v, i) => v / ref[i]!);
    expect(ratio[0]! / ratio[2]!).toBeCloseTo(1, 6);
  });
});

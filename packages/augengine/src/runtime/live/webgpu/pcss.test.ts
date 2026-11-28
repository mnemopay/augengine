import { describe, expect, it } from "vitest";
import { FRAME_LAYOUT, FRAME_UNIFORM_FLOATS, cascadeDepthRange, vogelDisc, collectSceneLights, fitDirectionalShadow, packFrameLights, pcssPenumbraWidth, shadowSettingsFor, transformPoint } from "./lights.js";
import { Scene } from "../../../scene/scene.js";
import { buildWgsl } from "./pipeline-factory.js";

const SUN_TAN = Math.tan((0.533 / 2) * Math.PI / 180);

describe("contact-hardening (PCSS) shadows", () => {
  it("penumbra grows linearly with blocker height: sun ≈ 4.65 mm per metre", () => {
    expect(pcssPenumbraWidth(0, SUN_TAN)).toBe(0);
    expect(pcssPenumbraWidth(1, SUN_TAN)).toBeCloseTo(0.00930, 4);
    expect(pcssPenumbraWidth(10, SUN_TAN) / pcssPenumbraWidth(5, SUN_TAN)).toBeCloseTo(2, 9);
    expect(pcssPenumbraWidth(-3, SUN_TAN)).toBe(0);
  });

  it("cascade depth range matches fitDirectionalShadow (1 depth unit = 3.99 r metres)", () => {
    const r = 8;
    const m = fitDirectionalShadow([0, -1, 0], [0, 0, 0], r, 1024);
    // Two points 1 m apart along the light travel direction differ in depth by 1 / range.
    const a = transformPoint(m, [0, 0, 0]);
    const b = transformPoint(m, [0, -1, 0]);
    expect((b[2] - a[2]) * cascadeDepthRange(2 * r)).toBeCloseTo(1, 3);
  });

  it("soft qualities enable PCSS taps; low/high keep fixed PCF", () => {
    expect(shadowSettingsFor("soft")!.pcss).toBe(8);
    expect(shadowSettingsFor("soft-high", 2)).toEqual({ mapSize: 2048, pcfTaps: 1, cascades: 2, pcss: 16 });
    expect(shadowSettingsFor("low")!.pcss).toBe(0);
    expect(shadowSettingsFor("high")!.pcss).toBe(0);
    expect(shadowSettingsFor("off")).toBeNull();
  });

  it("packs the PCSS block (and zeroes it when off)", () => {
    const lights = collectSceneLights(new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [] }), [0, -1, 0], [0, 0, 0]);
    const out = new Float32Array(FRAME_UNIFORM_FLOATS).fill(7);
    const vp = fitDirectionalShadow([0, -1, 0], [0, 0, 0], 8, 1024);
    packFrameLights(out, lights, { enabled: true, viewProj: vp, texelSize: 1 / 1024, depthBias: 0.0006, normalOffset: 0.02, pcfTaps: 1, soft: { tanLight: SUN_TAN, worldSize: 16, worldSizeFar: 64, taps: 12, frame: 70, searchHeight: 16 } });
    expect(Array.from(out.slice(FRAME_LAYOUT.shadowSoft, FRAME_LAYOUT.shadowSoft + 4))).toEqual([Math.fround(SUN_TAN), 16, 64, 12]);
    expect(out[FRAME_LAYOUT.shadowSoft2]).toBe(6); // 70 mod 64
    packFrameLights(out, lights, { enabled: true, viewProj: vp, texelSize: 1 / 1024, depthBias: 0.0006, normalOffset: 0.02, pcfTaps: 1 });
    expect(Array.from(out.slice(FRAME_LAYOUT.shadowSoft, FRAME_LAYOUT.shadowSoft2 + 4))).toEqual(new Array(8).fill(0));
  });

  it("Vogel disc: inside the unit disc, outermost first, the first four taps cover all quadrants", () => {
    const d = vogelDisc(8, 0.3);
    for (const [x, y] of d) expect(Math.hypot(x, y)).toBeLessThanOrEqual(1);
    for (let i = 1; i < d.length; i++) expect(Math.hypot(...d[i]!)).toBeLessThan(Math.hypot(...d[i - 1]!));
    const quadrants = new Set(d.slice(0, 4).map(([x, y]) => `${x > 0}${y > 0}`));
    expect(quadrants.size).toBeGreaterThanOrEqual(3);
    // Shader's incremental rotation constant = the golden angle.
    expect(Math.cos(2.39996323)).toBeCloseTo(-0.7373688, 6);
    expect(Math.sin(2.39996323)).toBeCloseTo(0.6754903, 6);
  });

  it("every lit material passes the fragment position to the shadow lookup", () => {
    for (const kind of ["pbr", "cel", "vrm-mtoon"] as const) {
      const src = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1 });
      expect(src).toContain("augengine_shadow(in.worldPos, N, in.clipPos.xy)");
      expect(src).toContain("fn augengine_pcss(");
    }
  });
});

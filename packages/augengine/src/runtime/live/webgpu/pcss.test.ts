import { describe, expect, it } from "vitest";
import { FRAME_LAYOUT, FRAME_UNIFORM_FLOATS, PCSS_ALONG_FILTER_TEXELS, PCSS_MIN_FILTER_TEXELS, RPDB_MAX_GRADIENT, cascadeDepthRange, vogelDisc, collectSceneLights, fitDirectionalShadow, packFrameLights, pcssFilterAxes, pcssPenumbraWidth, receiverPlaneDepthGradient, shadowSettingsFor, transformPoint } from "./lights.js";
import { Scene } from "../../../scene/scene.js";
import { buildWgsl, pipelineKeyString, specializeWgsl, wgslF32 } from "./pipeline-factory.js";
import type { Vec3 } from "../../../scene/types.js";

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
      const src = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1, pcss: true });
      expect(src).toContain("augengine_shadow(in.worldPos, N, in.clipPos.xy)");
      expect(src).toContain("fn augengine_pcss(");
      // Fixed-PCF pipelines leave PCSS out entirely, but still shadow.
      const legacy = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1 });
      expect(legacy).toContain("augengine_shadow(in.worldPos, N, in.clipPos.xy)");
      expect(legacy).not.toContain("augengine_pcss");
      expect(legacy).not.toContain("dpdx");
    }
  });

  it("receiver plane: the gradient predicts the shadow depth of other points on the same plane", () => {
    // The parlour wall case: a vertical wall facing +z under a high sun (N·L ≈ 0.28).
    const sunDir: Vec3 = [-0.1198, 0.9525, 0.28];
    const m = fitDirectionalShadow([-sunDir[0], -sunDir[1], -sunDir[2]], [0, 1.5, 3], 15, 1024);
    const light = (p: Vec3) => transformPoint(m, p);
    const uv = (l: [number, number, number]) => [l[0] * 0.5 + 0.5, 0.5 - l[1] * 0.5] as const;
    const p0: Vec3 = [0.3, 2.1, 3.125];
    // Two arbitrary (non-parallel) steps within the wall plane z = 3.125 stand in for one screen pixel in x and y.
    const sx: Vec3 = [0.011, 0.002, 0], sy: Vec3 = [-0.003, 0.017, 0];
    const l0 = light(p0);
    const lx = light([p0[0] + sx[0], p0[1] + sx[1], p0[2]]);
    const ly = light([p0[0] + sy[0], p0[1] + sy[1], p0[2]]);
    const g = receiverPlaneDepthGradient([lx[0] - l0[0], lx[1] - l0[1], lx[2] - l0[2]], [ly[0] - l0[0], ly[1] - l0[1], ly[2] - l0[2]]);
    for (const q of [[1.7, 0.4], [-2.2, 2.9], [0.05, 2.0]] as const) {
      const lq = light([q[0], q[1], 3.125]);
      const du = uv(lq)[0] - uv(l0)[0], dv = uv(lq)[1] - uv(l0)[1];
      expect(l0[2] + g[0] * du + g[1] * dv).toBeCloseTo(lq[2], 6);
    }
    // |gradient| ≈ 0.5·tan θ for a receiver θ from facing the light (θ = acos 0.28 ≈ 73.7°).
    expect(Math.hypot(g[0], g[1])).toBeCloseTo(0.5 * Math.tan(Math.acos(0.28)), 2);
  });

  it("receiver plane: facing the light gives zero gradient; edge-on is capped; degenerate is zero", () => {
    // Screen steps that only move across the map (no depth change).
    const facing = receiverPlaneDepthGradient([0.01, 0, 0], [0, 0.01, 0]);
    expect(Math.hypot(...facing)).toBe(0);
    const steep = receiverPlaneDepthGradient([0.001, 0, 0.5], [0, 0.001, 0]);
    expect(Math.hypot(...steep)).toBeCloseTo(RPDB_MAX_GRADIENT, 9);
    expect(Math.abs(steep[1])).toBe(0);
    expect(Math.hypot(...receiverPlaneDepthGradient([0.01, 0.02, 0.1], [0.02, 0.04, 0.3]))).toBe(0);
  });

  it("PCSS kernel: isotropic when facing the light, widened only along the iso-depth line when steep", () => {
    const texel = 1 / 1024;
    const flat = pcssFilterAxes([0, 0], 0, texel);
    expect(flat.radiusN).toBeCloseTo(PCSS_MIN_FILTER_TEXELS * texel, 12);
    expect(flat.radiusT).toBeCloseTo(PCSS_MIN_FILTER_TEXELS * texel, 12);
    const steep = pcssFilterAxes([0, -2.5], 0, texel);
    expect(steep.n[0]).toBeCloseTo(0, 12);
    expect(steep.n[1]).toBeCloseTo(-1, 12);
    expect(steep.t[0] * steep.n[0] + steep.t[1] * steep.n[1]).toBeCloseTo(0, 12);
    expect(steep.radiusN).toBeCloseTo(PCSS_MIN_FILTER_TEXELS * texel, 12);
    expect(steep.radiusT).toBeCloseTo(PCSS_ALONG_FILTER_TEXELS * texel, 12);
    // A real penumbra wider than both floors stays round.
    const wide = pcssFilterAxes([0, -2.5], 10 * texel, texel);
    expect(wide.radiusN).toBeCloseTo(10 * texel, 12);
    expect(wide.radiusT).toBeCloseTo(10 * texel, 12);
  });

  it("WGSL: the receiver plane is taken before any per-pixel branch, and its constants are f32 literals", () => {
    const src = buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1, pcss: true });
    const fn = src.slice(src.indexOf("fn augengine_shadow(worldPos"));
    const rpAt = fn.indexOf("augengine_receiver_plane(worldPos)");
    expect(rpAt).toBeGreaterThan(0);
    // Derivatives need uniform control flow: only the uniform shadow-on and PCSS-on tests may precede them.
    expect(fn.indexOf("dot(N, L) <= 0.0")).toBeGreaterThan(rpAt);
    expect(src).toContain(`const AUGENGINE_RPDB_MAX: f32 = ${wgslF32(RPDB_MAX_GRADIENT)};`);
    expect(src).toContain(`const AUGENGINE_PCSS_ALONG_TEXELS: f32 = ${wgslF32(PCSS_ALONG_FILTER_TEXELS)};`);
    expect(src).not.toMatch(/\blet ref\b/); // 'ref' is reserved in WGSL
    expect(wgslF32(3)).toBe("3.0");
    expect(wgslF32(1.5)).toBe("1.5");
    expect(wgslF32(1e-7)).toBe("1e-7");
  });
});

describe("natural-light / PCSS shader variants", () => {
  const kinds = ["pbr", "cel", "vrm-mtoon", "unlit"] as const;

  it("the legacy variant compiles none of the natural-light or PCSS code", () => {
    for (const kind of kinds) {
      const src = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1 });
      for (const name of ["augengine_env_ibl", "augengine_env_irradiance", "augengine_atmosphere_apply", "augengine_pcss", "augengine_receiver_plane", "envParams.x", "atmos.x", "// @"]) {
        expect(src, `${kind}: ${name}`).not.toContain(name);
      }
      // The legacy look's own paths stay.
      expect(src).toContain("return mix(lit, u_frame.fog.rgb, f);");
      expect(src).toContain("fn augengine_ambient_sky(");
    }
    expect(buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1 })).toContain("lit = lit + augengine_sky_specular(N, V, F0, roughness);");
  });

  it("the natural variant compiles the IBL, SH9 and atmosphere code (and keeps the legacy branches as a fallback)", () => {
    for (const kind of kinds) {
      const src = buildWgsl({ materialKind: kind, vertexStride: 8, sampleCount: 1, natural: true });
      expect(src).toContain("fn augengine_env_irradiance(");
      expect(src).toContain("fn augengine_atmosphere_apply(");
      expect(src).toContain("if (u_frame.atmos.x > 0.5) { return augengine_atmosphere_apply(lit, worldPos, f); }");
      expect(src).not.toContain("// @");
    }
    const pbr = buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1, natural: true });
    expect(pbr).toMatch(/if \(u_frame\.envParams\.x > 0\.5\) \{\s+lit = lit \+ augengine_env_ibl[\s\S]*\} else \{[\s\S]*augengine_sky_specular/);
    // Instanced variants carry the flag through.
    expect(buildWgsl({ materialKind: "pbr", vertexStride: 8, sampleCount: 1, natural: true, pcss: true, instanced: true })).toContain("fn augengine_pcss(");
  });

  it("variants have distinct cache keys", () => {
    const base = { materialKind: "pbr" as const, vertexStride: 8 as const, sampleCount: 1 as const };
    const keys = [base, { ...base, natural: true }, { ...base, pcss: true }, { ...base, natural: true, pcss: true }].map(pipelineKeyString);
    expect(new Set(keys).size).toBe(4);
    expect(pipelineKeyString(base)).toBe("pbr|8|1|static|rigid|nocolor|flat|nomr|nonormal");
  });

  it("specializeWgsl removes only the disabled blocks", () => {
    const src = "a\n  // @natural-begin\nN\n  // @natural-end\nb\n// @pcss-begin\nP\n// @pcss-end\nc\n";
    expect(specializeWgsl(src, { natural: false, pcss: false })).toBe("a\nb\nc\n");
    expect(specializeWgsl(src, { natural: true, pcss: false })).toBe("a\nN\nb\nc\n");
    expect(specializeWgsl(src, { natural: false, pcss: true })).toBe("a\nb\nP\nc\n");
  });
});

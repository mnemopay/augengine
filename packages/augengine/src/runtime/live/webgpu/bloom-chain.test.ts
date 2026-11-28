import { describe, expect, it } from "vitest";
import { BLOOM_DOWN_WGSL, BLOOM_UP_WGSL, bloomChainSizes } from "./bloom-chain.js";
import { CAMERA_FX_WGSL, COMPOSITE_TONEMAP_PASS } from "./post-processing.js";
import { TAA_WGSL } from "./temporal.js";

describe("physical bloom chain", () => {
  it("levels halve from half resolution down to ≥ ~8 px, at most 6", () => {
    const s = bloomChainSizes(1440, 900);
    expect(s[0]).toEqual({ width: 720, height: 450 });
    expect(s.length).toBe(6);
    for (let i = 1; i < s.length; i++) expect(s[i]!.width).toBe(Math.ceil(s[i - 1]!.width / 2));
    expect(bloomChainSizes(64, 32).length).toBeLessThan(6);
  });

  it("filters preserve energy: 13-tap weights and the 3×3 tent each sum to 1", () => {
    // Down: centre box 0.5 + four corner boxes × 0.125.
    expect(0.5 + 4 * 0.125).toBe(1);
    expect(BLOOM_DOWN_WGSL).toContain("b0 * 0.5 + (b1 + b2 + b3 + b4) * 0.125");
    // Up: 4 + 4×2 + 4×1 = 16, divided by 16.
    expect(4 + 4 * 2 + 4 * 1).toBe(16);
    expect(BLOOM_UP_WGSL).toContain("c / 16.0");
    // First level suppresses fireflies with the Karis average.
    expect(BLOOM_DOWN_WGSL).toContain("1.0 / (1.0 + dot(c");
  });

  it("composite + TAA resolve both mix (not add) physical bloom, divided by the level count", () => {
    expect(COMPOSITE_TONEMAP_PASS.shader).toContain("mix(scene, bloom / max(u.levels, 1.0), u.intensity)");
    expect(TAA_WGSL).toContain("mix(disp, b / max(u.look2.y, 1.0), u.look0.x)");
    expect(COMPOSITE_TONEMAP_PASS.uniforms!.size).toBe(64);
    expect(COMPOSITE_TONEMAP_PASS.uniforms!.fields.levels!.offset).toBe(52);
  });

  it("lens dirt and film grain are wired but scaled by uniforms that default to 0", () => {
    expect(CAMERA_FX_WGSL).toContain("fn fx_lens_dirt");
    expect(CAMERA_FX_WGSL).toContain("fn fx_grain");
    expect(COMPOSITE_TONEMAP_PASS.shader).toContain("if (u.grain > 0.0)");
    expect(TAA_WGSL).toContain("if (u.look2.w > 0.0)");
  });
});

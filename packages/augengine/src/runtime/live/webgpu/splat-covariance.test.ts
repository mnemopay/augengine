import { describe, it, expect } from "vitest";
import {
  quatScaleToCovariance3D,
  evaluateShBand1,
  sortSplatsBackToFront,
  SPLAT_COVARIANCE_WGSL,
} from "./splat-covariance.js";

describe("quatScaleToCovariance3D", () => {
  it("identity quaternion + uniform unit scale → identity covariance", () => {
    const cov = quatScaleToCovariance3D([0, 0, 0, 1], [1, 1, 1]);
    expect(cov[0]).toBeCloseTo(1, 6); // Σ00
    expect(cov[1]).toBeCloseTo(0, 6); // Σ01
    expect(cov[2]).toBeCloseTo(0, 6); // Σ02
    expect(cov[3]).toBeCloseTo(1, 6); // Σ11
    expect(cov[4]).toBeCloseTo(0, 6); // Σ12
    expect(cov[5]).toBeCloseTo(1, 6); // Σ22
  });

  it("identity quaternion + axis-aligned anisotropic scale → diagonal Σ with sigma²", () => {
    const cov = quatScaleToCovariance3D([0, 0, 0, 1], [2, 3, 4]);
    expect(cov[0]).toBeCloseTo(4, 5);
    expect(cov[3]).toBeCloseTo(9, 5);
    expect(cov[5]).toBeCloseTo(16, 5);
    // Off-diagonals are zero for axis-aligned scales under identity rotation.
    expect(cov[1]).toBeCloseTo(0, 5);
    expect(cov[2]).toBeCloseTo(0, 5);
    expect(cov[4]).toBeCloseTo(0, 5);
  });

  it("90° rotation about Z + axis-aligned scale → swapped diagonals", () => {
    // q = (0, 0, sin(45°), cos(45°)) rotates the X axis onto Y.
    const s = Math.SQRT1_2;
    const cov = quatScaleToCovariance3D([0, 0, s, s], [2, 1, 1]);
    // After rotation, the "x sigma=2" axis aligns with Y, so Σ11 = 4.
    expect(cov[3]).toBeCloseTo(4, 4);
    expect(cov[0]).toBeCloseTo(1, 4);
    expect(cov[5]).toBeCloseTo(1, 4);
  });

  it("output is symmetric (the 6-tuple compresses a 3×3 symmetric)", () => {
    const cov = quatScaleToCovariance3D([0.1, 0.2, 0.3, 0.93], [1.5, 0.5, 2.0]);
    // Reconstruct full Σ and check transpose equality.
    const M = [
      [cov[0], cov[1], cov[2]],
      [cov[1], cov[3], cov[4]],
      [cov[2], cov[4], cov[5]],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(M[i]![j]).toBeCloseTo(M[j]![i]!, 5);
      }
    }
  });

  it("normalises a non-unit quaternion defensively", () => {
    // 4× scaled identity quat — should produce identity Σ after normalisation.
    const cov = quatScaleToCovariance3D([0, 0, 0, 4], [1, 1, 1]);
    expect(cov[0]).toBeCloseTo(1, 5);
    expect(cov[3]).toBeCloseTo(1, 5);
    expect(cov[5]).toBeCloseTo(1, 5);
  });
});

describe("evaluateShBand1", () => {
  it("zero coefficients → zero contribution regardless of direction", () => {
    const coeffs = new Float32Array(9);
    const out = evaluateShBand1(coeffs, 0.5, 0.5, 0.7);
    expect(out).toEqual([0, 0, 0]);
  });

  it("each level activates the matching basis function", () => {
    const k = 0.48860251190291992;
    // Level -1: -k · dy · coeff
    {
      const c = new Float32Array(9);
      c[0] = 1; c[1] = 0; c[2] = 0;
      const [r] = evaluateShBand1(c, 0, 1, 0);
      expect(r).toBeCloseTo(-k, 6);
    }
    // Level 0: +k · dz · coeff
    {
      const c = new Float32Array(9);
      c[3] = 1;
      const [r] = evaluateShBand1(c, 0, 0, 1);
      expect(r).toBeCloseTo(k, 6);
    }
    // Level +1: -k · dx · coeff
    {
      const c = new Float32Array(9);
      c[6] = 1;
      const [r] = evaluateShBand1(c, 1, 0, 0);
      expect(r).toBeCloseTo(-k, 6);
    }
  });

  it("returns separate contributions per RGB channel", () => {
    const c = new Float32Array(9);
    c[3] = 0.1; c[4] = 0.2; c[5] = 0.3; // level 0, all three channels
    const [r, g, b] = evaluateShBand1(c, 0, 0, 1);
    const k = 0.48860251190291992;
    expect(r).toBeCloseTo(k * 0.1, 6);
    expect(g).toBeCloseTo(k * 0.2, 6);
    expect(b).toBeCloseTo(k * 0.3, 6);
  });
});

describe("sortSplatsBackToFront", () => {
  function identityView(): Float32Array {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  }

  it("sorts splats descending by view-space Z (further first)", () => {
    // Identity view → view-space z == world z. Positions at z = 0, 5, -3, 10.
    const positions = new Float32Array([
      0, 0, 0,
      0, 0, 5,
      0, 0, -3,
      0, 0, 10,
    ]);
    const into = new Uint32Array(4);
    sortSplatsBackToFront(positions, identityView(), into);
    // Z values: 0, 5, -3, 10 → descending order is 10, 5, 0, -3 → indices 3, 1, 0, 2.
    expect([...into]).toEqual([3, 1, 0, 2]);
  });

  it("preserves stable behaviour on equal-depth splats", () => {
    const positions = new Float32Array([1, 0, 7, 0, 0, 7, -2, 0, 7]);
    const into = new Uint32Array(3);
    sortSplatsBackToFront(positions, identityView(), into);
    // All depths equal → any permutation is correct; but the function
    // should at least not duplicate or drop indices.
    const seen = new Set([...into]);
    expect(seen.size).toBe(3);
  });

  it("throws when `into` is undersized", () => {
    const positions = new Float32Array(9); // 3 splats
    expect(() =>
      sortSplatsBackToFront(positions, identityView(), new Uint32Array(2)),
    ).toThrow(/into.length/);
  });

  it("throws on a malformed view matrix", () => {
    const positions = new Float32Array([0, 0, 0]);
    expect(() =>
      sortSplatsBackToFront(positions, new Float32Array(15), new Uint32Array(1)),
    ).toThrow(/16 floats/);
  });
});

describe("SPLAT_COVARIANCE_WGSL", () => {
  it("exports the 2D covariance projection function", () => {
    expect(SPLAT_COVARIANCE_WGSL).toContain("fn augengine_splat_covariance_2d");
    expect(SPLAT_COVARIANCE_WGSL).toContain("t * s3d * transpose(t)");
  });

  it("exports the SH band-1 evaluator", () => {
    expect(SPLAT_COVARIANCE_WGSL).toContain("fn augengine_splat_eval_sh_band1");
    // Matches the K1 = 0.5 · sqrt(3/π) constant
    expect(SPLAT_COVARIANCE_WGSL).toContain("0.48860251190291992");
  });
});

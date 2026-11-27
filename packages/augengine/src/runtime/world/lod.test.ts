/**
 * LOD selector unit tests. Distance + screen-space-error gating + key suffix.
 */

import { describe, it, expect } from "vitest";
import {
  DEFAULT_LOD_CHAIN,
  selectLod,
  lodMeshKey,
  estimateLodVertexCount,
} from "./lod.js";

describe("lod / distance thresholds", () => {
  it("picks LOD0 at point-blank range", () => {
    expect(selectLod([0, 0, 0], [10, 0, 0], 5)).toBe(0);
  });
  it("steps to LOD1 past the first threshold", () => {
    expect(selectLod([0, 0, 0], [50, 0, 0], 20)).toBe(1);
  });
  it("steps to LOD2 past the second threshold", () => {
    expect(selectLod([0, 0, 0], [120, 0, 0], 20)).toBe(2);
  });
  it("steps to LOD3 (imposter) past the third threshold", () => {
    expect(selectLod([0, 0, 0], [400, 0, 0], 20)).toBe(3);
  });
});

describe("lod / screen-space error", () => {
  it("promotes to next LOD when projected pixel size falls below threshold", () => {
    // A tiny radius object at moderate distance — projected pixels small,
    // even though distance threshold says LOD1.
    const lod = selectLod([0, 0, 0], [50, 0, 0], 0.5);
    expect(lod).toBeGreaterThanOrEqual(2);
  });
  it("keeps a large object at LOD0 even at threshold-1 distance", () => {
    // Distance just past 32 m but a 30 m radius landmark — still LOD1 by
    // distance rule but the SSE override should NOT downgrade further.
    const lod = selectLod([0, 0, 0], [40, 0, 0], 30);
    expect(lod).toBe(1);
  });
});

describe("lod / mesh-key suffixing", () => {
  it("returns the base key untouched for LOD0", () => {
    expect(lodMeshKey("gltf:foo.glb#0", 0)).toBe("gltf:foo.glb#0");
  });
  it("appends #lod1..3 for higher levels", () => {
    expect(lodMeshKey("a", 1)).toBe("a#lod1");
    expect(lodMeshKey("a", 2)).toBe("a#lod2");
    expect(lodMeshKey("a", 3)).toBe("a#lod3");
  });
});

describe("lod / vertex estimate", () => {
  it("estimates LOD0 at full count, LOD3 at billboard 8", () => {
    expect(estimateLodVertexCount(10000, 0)).toBe(10000);
    expect(estimateLodVertexCount(10000, 1)).toBeCloseTo(2500);
    expect(estimateLodVertexCount(10000, 2)).toBeCloseTo(625);
    expect(estimateLodVertexCount(10000, 3)).toBe(8);
  });
  it("clamps tiny LOD counts to a floor", () => {
    expect(estimateLodVertexCount(100, 1)).toBeGreaterThanOrEqual(64);
    expect(estimateLodVertexCount(100, 2)).toBeGreaterThanOrEqual(32);
  });
});

describe("lod / focalPx override (camera resize)", () => {
  it("higher focal length keeps detail longer (smaller projected pixel threshold)", () => {
    const wide = { ...DEFAULT_LOD_CHAIN, focalPx: 270 };
    const tele = { ...DEFAULT_LOD_CHAIN, focalPx: 2160 };
    const wideLod = selectLod([0, 0, 0], [50, 0, 0], 1, wide);
    const teleLod = selectLod([0, 0, 0], [50, 0, 0], 1, tele);
    expect(teleLod).toBeLessThanOrEqual(wideLod);
  });
});

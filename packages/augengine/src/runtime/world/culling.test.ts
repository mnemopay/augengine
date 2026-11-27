/**
 * Frustum culling tests. Uses a known camera + AABB set so failures pinpoint
 * the plane extraction or the AABB test in isolation.
 */

import { describe, it, expect } from "vitest";
import {
  extractFrustum,
  aabbInFrustum,
  sphereInFrustum,
} from "./culling.js";
import { m4Create, m4LookAt, m4Mul, m4Perspective } from "../../scene/math.js";
import type { Mat4 } from "../../scene/types.js";

function buildViewProj(eye: [number, number, number], target: [number, number, number]): Mat4 {
  const view = m4Create();
  const proj = m4Create();
  const vp = m4Create();
  m4LookAt(view, eye, target, [0, 1, 0]);
  m4Perspective(proj, Math.PI / 3, 16 / 9, 0.1, 1000);
  m4Mul(vp, proj, view);
  return vp;
}

describe("culling / frustum extraction", () => {
  it("produces 6 normalised planes", () => {
    const vp = buildViewProj([0, 5, 10], [0, 0, 0]);
    const f = extractFrustum(vp);
    expect(f.planes).toHaveLength(6);
    for (const p of f.planes) {
      const len = Math.hypot(p[0], p[1], p[2]);
      expect(len).toBeCloseTo(1, 5);
    }
  });
});

describe("culling / AABB tests", () => {
  it("keeps a box directly in front of the camera", () => {
    const vp = buildViewProj([0, 5, 20], [0, 0, 0]);
    const f = extractFrustum(vp);
    const visible = aabbInFrustum(f, {
      centerWorld: [0, 0, 0],
      halfExtents: [1, 1, 1],
    });
    expect(visible).toBe(true);
  });
  it("culls a box behind the camera", () => {
    const vp = buildViewProj([0, 5, 20], [0, 0, 0]);
    const f = extractFrustum(vp);
    const culled = aabbInFrustum(f, {
      centerWorld: [0, 0, 200], // way behind eye (z > eye.z, but cam looks toward origin)
      halfExtents: [1, 1, 1],
    });
    expect(culled).toBe(false);
  });
  it("culls a box far to the side", () => {
    const vp = buildViewProj([0, 5, 20], [0, 0, 0]);
    const f = extractFrustum(vp);
    const culled = aabbInFrustum(f, {
      centerWorld: [500, 0, 0],
      halfExtents: [1, 1, 1],
    });
    expect(culled).toBe(false);
  });
  it("culls a box past the far plane", () => {
    const vp = buildViewProj([0, 5, 20], [0, 0, 0]);
    const f = extractFrustum(vp);
    const culled = aabbInFrustum(f, {
      centerWorld: [0, 0, -2000],
      halfExtents: [1, 1, 1],
    });
    expect(culled).toBe(false);
  });
});

describe("culling / sphere variant", () => {
  it("matches AABB behaviour for axis-aligned cases", () => {
    const vp = buildViewProj([0, 5, 20], [0, 0, 0]);
    const f = extractFrustum(vp);
    expect(sphereInFrustum(f, [0, 0, 0], 1)).toBe(true);
    expect(sphereInFrustum(f, [500, 0, 0], 1)).toBe(false);
  });
});

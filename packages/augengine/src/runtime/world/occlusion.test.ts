import { describe, expect, it } from "vitest";
import { m4Create, m4LookAt, m4Mul, m4Perspective } from "../../scene/math.js";
import type { Vec3 } from "../../scene/types.js";
import { SoftwareOcclusion, type OcclusionBox } from "./occlusion.js";

function camera(eye: Vec3, target: Vec3) {
  const view = m4Create(), proj = m4Create(), vp = m4Create();
  m4LookAt(view, eye, target, [0, 1, 0]);
  m4Perspective(proj, (80 * Math.PI) / 180, 16 / 10, 0.1, 400);
  m4Mul(vp, proj, view);
  return vp;
}

/** Segment eye→p hits the box (slab test), excluding the endpoint itself. */
function segmentHits(eye: Vec3, p: Vec3, b: OcclusionBox): boolean {
  let t0 = 0, t1 = 1 - 1e-4;
  for (let i = 0; i < 3; i++) {
    const o = eye[i]!, d = p[i]! - eye[i]!;
    const mn = b.center[i]! - b.halfExtents[i]!, mx = b.center[i]! + b.halfExtents[i]!;
    if (Math.abs(d) < 1e-12) {
      if (o < mn || o > mx) return false;
      continue;
    }
    let a = (mn - o) / d, c = (mx - o) / d;
    if (a > c) { const t = a; a = c; c = t; }
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, c);
    if (t0 > t1) return false;
  }
  return true;
}

/** Ground truth: is any point on the box's surface visible past the occluders? */
function anyPointVisible(eye: Vec3, box: OcclusionBox, occluders: OcclusionBox[]): boolean {
  const g = 6;
  for (let axis = 0; axis < 3; axis++) {
    for (const side of [-1, 1]) {
      for (let a = 0; a <= g; a++) {
        for (let b = 0; b <= g; b++) {
          const p = [0, 0, 0] as [number, number, number];
          const u = (axis + 1) % 3, w = (axis + 2) % 3;
          p[axis] = box.center[axis]! + side * box.halfExtents[axis]!;
          p[u] = box.center[u]! + (-1 + (2 * a) / g) * box.halfExtents[u]!;
          p[w] = box.center[w]! + (-1 + (2 * b) / g) * box.halfExtents[w]!;
          if (!occluders.some((o) => segmentHits(eye, p, o))) return true;
        }
      }
    }
  }
  return false;
}

describe("software occlusion", () => {
  const eye: Vec3 = [0, 1.7, 0];
  const vp = camera(eye, [1, 1.7, 0]);

  it("hides what is behind a wall and keeps what is beside or in front of it", () => {
    const occ = new SoftwareOcclusion();
    occ.begin(vp);
    const wall: OcclusionBox = { center: [12, 5, 0], halfExtents: [0.5, 5, 8] };
    occ.addBox(wall);
    expect(occ.isOccluded({ center: [30, 2, 0], halfExtents: [1, 2, 1] })).toBe(true);
    expect(occ.isOccluded({ center: [30, 2, 25], halfExtents: [1, 2, 1] })).toBe(false); // off to the side
    expect(occ.isOccluded({ center: [6, 1, 0], halfExtents: [1, 1, 1] })).toBe(false); // in front of the wall
    expect(occ.isOccluded({ center: [30, 14, 0], halfExtents: [1, 1, 1] })).toBe(false); // above the wall
  });

  it("never culls or occludes with geometry touching the near plane", () => {
    const occ = new SoftwareOcclusion();
    occ.begin(vp);
    occ.addBox({ center: [0.05, 1.7, 0], halfExtents: [1, 1, 1] }); // engulfs the camera
    expect(occ.stats.occluderTriangles).toBe(0);
    occ.addBox({ center: [12, 5, 0], halfExtents: [0.5, 5, 8] });
    expect(occ.isOccluded({ center: [0.5, 1.7, 0], halfExtents: [1, 1, 1] })).toBe(false);
  });

  it("is conservative: nothing it hides has a single visible point (randomised)", () => {
    let seed = 1234;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    let claimed = 0, checked = 0;
    for (let scene = 0; scene < 60; scene++) {
      const eyeR: Vec3 = [0, 1 + rand() * 4, 0];
      const yaw = rand() * Math.PI * 2;
      const vpr = camera(eyeR, [Math.cos(yaw), eyeR[1] - 0.1 + rand() * 0.2, Math.sin(yaw)]);
      const occluders: OcclusionBox[] = [];
      for (let k = 0; k < 14; k++) {
        const d = 6 + rand() * 50, a = yaw + (rand() - 0.5) * 1.6;
        occluders.push({ center: [Math.cos(a) * d, 3 + rand() * 5, Math.sin(a) * d], halfExtents: [1 + rand() * 5, 2 + rand() * 6, 1 + rand() * 5] });
      }
      const occ = new SoftwareOcclusion();
      occ.begin(vpr);
      for (const o of occluders) occ.addBox(o);
      for (let k = 0; k < 40; k++) {
        const d = 10 + rand() * 90, a = yaw + (rand() - 0.5) * 1.6;
        const box: OcclusionBox = { center: [Math.cos(a) * d, rand() * 6, Math.sin(a) * d], halfExtents: [0.3 + rand() * 2, 0.3 + rand() * 3, 0.3 + rand() * 2] };
        if (occluders.some((o) => Math.abs(o.center[0] - box.center[0]) < o.halfExtents[0] + box.halfExtents[0] && Math.abs(o.center[1] - box.center[1]) < o.halfExtents[1] + box.halfExtents[1] && Math.abs(o.center[2] - box.center[2]) < o.halfExtents[2] + box.halfExtents[2])) continue;
        checked += 1;
        if (occ.isOccluded(box)) {
          claimed += 1;
          expect(anyPointVisible(eyeR, box, occluders), `scene ${scene} box ${k}`).toBe(false);
        }
      }
    }
    // It must actually do work, not pass by culling nothing.
    expect(claimed).toBeGreaterThan(checked * 0.1);
  });

  it("raster matches the four-corner inner-coverage rule (randomised, never covers more)", () => {
    // Reference: the original per-pixel test of all four corners against all three edges.
    const W = 160, H = 96;
    const ref = new Float32Array(W * H);
    const project = (x: number, y: number, z: number): [number, number, number] => {
      const cx = vp[0]! * x + vp[4]! * y + vp[8]! * z + vp[12]!;
      const cy = vp[1]! * x + vp[5]! * y + vp[9]! * z + vp[13]!;
      const cw = vp[3]! * x + vp[7]! * y + vp[11]! * z + vp[15]!;
      return [(cx / cw * 0.5 + 0.5) * W, (0.5 - cy / cw * 0.5) * H, cw];
    };
    const refTri = (p: [number, number, number][]) => {
      const [[ax, ay, aw], [bx, by, bw], [cx, cy, cw]] = p as [[number, number, number], [number, number, number], [number, number, number]];
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (Math.abs(area) < 1e-9) return;
      const far = Math.max(aw, bw, cw), sign = area > 0 ? 1 : -1;
      const edge = (x0: number, y0: number, x1: number, y1: number, px: number, py: number) => sign * ((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0));
      for (let y = Math.max(0, Math.floor(Math.min(ay, by, cy))); y <= Math.min(H - 1, Math.ceil(Math.max(ay, by, cy))); y++) {
        for (let x = Math.max(0, Math.floor(Math.min(ax, bx, cx))); x <= Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx))); x++) {
          let inside = true;
          for (let k = 0; k < 4 && inside; k++) {
            const qx = x + (k & 1), qy = y + (k >> 1);
            if (edge(ax, ay, bx, by, qx, qy) < 0 || edge(bx, by, cx, cy, qx, qy) < 0 || edge(cx, cy, ax, ay, qx, qy) < 0) inside = false;
          }
          if (inside && far < ref[y * W + x]!) ref[y * W + x] = far;
        }
      }
    };
    let seed = 99;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    let covered = 0, mismatched = 0;
    for (let trial = 0; trial < 40; trial++) {
      const occ = new SoftwareOcclusion();
      occ.begin(vp);
      ref.fill(Infinity);
      const pos: number[] = [], idx: number[] = [];
      for (let t = 0; t < 12; t++) {
        for (let v = 0; v < 3; v++) {
          pos.push(4 + rand() * 60, rand() * 12 - 2, (rand() - 0.5) * 70);
          idx.push(idx.length);
        }
        const p = [0, 1, 2].map((v) => project(pos[(t * 3 + v) * 3]!, pos[(t * 3 + v) * 3 + 1]!, pos[(t * 3 + v) * 3 + 2]!));
        if (p.every((q) => q[2] >= 0.15)) refTri(p);
      }
      occ.addTriangles(pos, idx);
      const got = occ.depthView;
      for (let i = 0; i < W * H; i++) {
        if (ref[i] !== Infinity) covered += 1;
        // Never covered where the reference is not, and never nearer than it.
        if (ref[i] === Infinity) expect(got[i]).toBe(Infinity);
        else {
          expect(got[i]!).toBeGreaterThanOrEqual(ref[i]!);
          if (got[i] !== ref[i]) mismatched += 1;
        }
      }
    }
    expect(covered).toBeGreaterThan(1000);
    // Only borderline pixels may differ (the epsilon rejects them).
    expect(mismatched).toBeLessThan(covered * 0.002);
  });

  it("uses terrain triangles as occluders (a hill hides what is behind it)", () => {
    const occ = new SoftwareOcclusion();
    occ.begin(vp);
    // A ridge across the view: two triangles rising to 12 m at x = 20.
    const pos = [15, 0, -40, 15, 0, 40, 20, 12, -40, 20, 12, 40];
    occ.addTriangles(pos, [0, 2, 1, 1, 2, 3]);
    expect(occ.isOccluded({ center: [45, 2, 0], halfExtents: [2, 2, 2] })).toBe(true);
    expect(occ.isOccluded({ center: [45, 40, 0], halfExtents: [2, 2, 2] })).toBe(false);
  });
});

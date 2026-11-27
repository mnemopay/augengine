/**
 * world/occlusion — CPU software occlusion culling, conservative by construction.
 *
 * Each frame the big opaque things nearest the camera (building boxes, and the
 * hills' coarse terrain triangles) are rasterised into a small depth buffer.
 * An object is culled only when every pixel its screen rectangle could touch
 * is already covered by an occluder that is nearer than the object's nearest
 * point. Three rules keep it conservative, so nothing visible is ever hidden:
 *
 *  - Inner coverage: an occluder triangle marks a pixel only when the whole
 *    pixel square lies inside it (all four corners), never on partial overlap.
 *  - Far depth: a triangle writes its farthest vertex depth, not its real
 *    per-pixel depth, so occluders are treated as farther than they are.
 *  - Outer bounds: an occludee is tested on its projected box expanded by one
 *    pixel, against its nearest corner depth (exact for an AABB).
 *
 * Anything that crosses the near plane is never an occluder and is never
 * culled. Depth is clip-space w, i.e. view-space distance for a perspective
 * projection.
 */
import type { Mat4, Vec3 } from "../../scene/types.js";

export interface OcclusionBox {
  center: Vec3;
  halfExtents: Vec3;
}

export interface OcclusionStats {
  occluderTriangles: number;
  tested: number;
  occluded: number;
}

export interface SoftwareOcclusionOptions {
  width?: number;
  height?: number;
  /** Clip w below which geometry is treated as touching the near plane. */
  near?: number;
}

const BOX_TRIS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], // -x, +x
  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], // -y, +y
  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], // -z, +z
];

export class SoftwareOcclusion {
  readonly width: number;
  readonly height: number;
  private readonly near: number;
  private readonly depth: Float32Array;
  private vp: Mat4 | null = null;
  stats: OcclusionStats = { occluderTriangles: 0, tested: 0, occluded: 0 };
  // Scratch for projected corners: x, y (pixels), w.
  private readonly px = new Float64Array(8);
  private readonly py = new Float64Array(8);
  private readonly pw = new Float64Array(8);

  constructor(opts: SoftwareOcclusionOptions = {}) {
    this.width = opts.width ?? 160;
    this.height = opts.height ?? 96;
    this.near = opts.near ?? 0.15;
    this.depth = new Float32Array(this.width * this.height);
  }

  /** The depth buffer (Infinity = uncovered), row-major, y down. Read-only; for tests and debug views. */
  get depthView(): Readonly<Float32Array> {
    return this.depth;
  }

  /** Start a frame: clear the buffer and set the camera. */
  begin(viewProj: Mat4): void {
    this.vp = viewProj;
    this.depth.fill(Infinity);
    this.stats = { occluderTriangles: 0, tested: 0, occluded: 0 };
  }

  /** Rasterise an axis-aligned box as an occluder (only fully solid geometry!). */
  addBox(box: OcclusionBox): void {
    if (!this.projectBox(box)) return;
    for (const [a, b, c] of BOX_TRIS) this.rasterTri(a, b, c);
  }

  /**
   * Rasterise world-space triangles (xyz triples, indexed) as occluders —
   * e.g. a coarse terrain LOD. Triangles touching the near plane are skipped.
   */
  addTriangles(positions: ArrayLike<number>, indices: ArrayLike<number>, offset: Vec3 = [0, 0, 0], stride = 3): void {
    for (let t = 0; t + 2 < indices.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const vi = indices[t + k]! * stride;
        this.projectPoint(k, positions[vi]! + offset[0], positions[vi + 1]! + offset[1], positions[vi + 2]! + offset[2]);
      }
      if (this.pw[0]! < this.near || this.pw[1]! < this.near || this.pw[2]! < this.near) continue;
      this.rasterTri(0, 1, 2);
    }
  }

  /** True only when the box is certainly hidden behind occluders drawn so far. */
  isOccluded(box: OcclusionBox): boolean {
    this.stats.tested += 1;
    if (!this.projectBox(box)) return false;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, minW = Infinity;
    for (let k = 0; k < 8; k++) {
      const x = this.px[k]!, y = this.py[k]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (this.pw[k]! < minW) minW = this.pw[k]!;
    }
    const x0 = Math.max(0, Math.floor(minX) - 1), x1 = Math.min(this.width - 1, Math.ceil(maxX) + 1);
    const y0 = Math.max(0, Math.floor(minY) - 1), y1 = Math.min(this.height - 1, Math.ceil(maxY) + 1);
    // Entirely off-screen: that is the frustum test's job, not ours.
    if (x0 > x1 || y0 > y1) return false;
    for (let y = y0; y <= y1; y++) {
      const row = y * this.width;
      for (let x = x0; x <= x1; x++) {
        if (!(this.depth[row + x]! < minW)) return false;
      }
    }
    this.stats.occluded += 1;
    return true;
  }

  // ── internals ─────────────────────────────────────────────────────────

  /** Project the box's 8 corners; false when any corner is near/behind the camera. */
  private projectBox(box: OcclusionBox): boolean {
    const [cx, cy, cz] = box.center;
    const [hx, hy, hz] = box.halfExtents;
    for (let k = 0; k < 8; k++) {
      this.projectPoint(k, cx + (k & 4 ? hx : -hx), cy + (k & 2 ? hy : -hy), cz + (k & 1 ? hz : -hz));
      if (this.pw[k]! < this.near) return false;
    }
    return true;
  }

  private projectPoint(slot: number, x: number, y: number, z: number): void {
    const m = this.vp!;
    const cx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
    const cy = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
    const cw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
    this.pw[slot] = cw;
    const inv = cw !== 0 ? 1 / cw : 0;
    // NDC → pixels, y down.
    this.px[slot] = (cx * inv * 0.5 + 0.5) * this.width;
    this.py[slot] = (0.5 - cy * inv * 0.5) * this.height;
  }

  /**
   * Inner-coverage raster of one projected triangle at its farthest depth.
   *
   * A pixel [x, x+1] × [y, y+1] is covered when all four corners are inside
   * all three edges. Each edge function E(px, py) = A·px + B·py + C is linear,
   * so its minimum over the pixel's corners is E(x, y) + min(0, A) + min(0, B):
   * one value per edge per pixel, stepped by A along a row and B down rows.
   * (Same rule as testing the four corners, ~4× fewer operations; the small
   * epsilon only ever rejects borderline pixels, so it stays conservative.)
   */
  private rasterTri(a: number, b: number, c: number): void {
    const ax = this.px[a]!, ay = this.py[a]!, bx = this.px[b]!, by = this.py[b]!, cx = this.px[c]!, cy = this.py[c]!;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-9) return;
    this.stats.occluderTriangles += 1;
    const far = Math.max(this.pw[a]!, this.pw[b]!, this.pw[c]!);
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(this.width - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), y1 = Math.min(this.height - 1, Math.ceil(Math.max(ay, by, cy)));
    if (x0 > x1 || y0 > y1) return;
    const sign = area > 0 ? 1 : -1;
    // Edge (P → Q): sign·((Qx−Px)(py−Py) − (Qy−Py)(px−Px)) = A·px + B·py + C.
    const A0 = -sign * (by - ay), B0 = sign * (bx - ax), C0 = sign * ((by - ay) * ax - (bx - ax) * ay);
    const A1 = -sign * (cy - by), B1 = sign * (cx - bx), C1 = sign * ((cy - by) * bx - (cx - bx) * by);
    const A2 = -sign * (ay - cy), B2 = sign * (ax - cx), C2 = sign * ((ay - cy) * cx - (ax - cx) * cy);
    // Worst corner of a pixel, relative to its top-left corner.
    const k0 = Math.min(0, A0) + Math.min(0, B0);
    const k1 = Math.min(0, A1) + Math.min(0, B1);
    const k2 = Math.min(0, A2) + Math.min(0, B2);
    const eps = 1e-7 * Math.abs(area);
    const w = this.width, depth = this.depth;
    for (let y = y0; y <= y1; y++) {
      let e0 = A0 * x0 + B0 * y + C0 + k0;
      let e1 = A1 * x0 + B1 * y + C1 + k1;
      let e2 = A2 * x0 + B2 * y + C2 + k2;
      let i = y * w + x0;
      for (let x = x0; x <= x1; x++, i++, e0 += A0, e1 += A1, e2 += A2) {
        if (e0 < eps || e1 < eps || e2 < eps) continue;
        if (far < depth[i]!) depth[i] = far;
      }
    }
  }
}

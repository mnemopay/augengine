/**
 * builder — accumulate faceted geometry with normals, world-metre UVs and
 * per-part vertex colours. Every helper emits flat faces (own vertices per
 * face) so hard architectural edges and low-poly foliage shade correctly.
 */
import type { KitMesh, Rgba, Vec3 } from "./mesh.js";

export class MeshBuilder {
  private p: number[] = [];
  private n: number[] = [];
  private uv: number[] = [];
  private c: number[] = [];
  private idx: number[] = [];

  /** Planar quad a-b-c-d (counter-clockwise seen from the front). */
  quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: Rgba, uvScale = 1): this {
    const normal = faceNormal(a, b, c);
    const base = this.p.length / 3;
    // UVs in metres along the quad's own edges, so textures tile at a constant density.
    const w = dist(a, b) * uvScale;
    const h = dist(a, d) * uvScale;
    const uvs: [number, number][] = [[0, 0], [w, 0], [w, h], [0, h]];
    [a, b, c, d].forEach((v, i) => {
      this.p.push(v[0], v[1], v[2]);
      this.n.push(normal[0], normal[1], normal[2]);
      this.uv.push(uvs[i]![0], uvs[i]![1]);
      this.c.push(color[0], color[1], color[2], color[3]);
    });
    this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    return this;
  }

  /** Single triangle with its face normal. */
  tri(a: Vec3, b: Vec3, c: Vec3, color: Rgba): this {
    const normal = faceNormal(a, b, c);
    const base = this.p.length / 3;
    [a, b, c].forEach((v, i) => {
      this.p.push(v[0], v[1], v[2]);
      this.n.push(normal[0], normal[1], normal[2]);
      this.uv.push(i === 1 ? 1 : 0, i === 2 ? 1 : 0);
      this.c.push(color[0], color[1], color[2], color[3]);
    });
    this.idx.push(base, base + 1, base + 2);
    return this;
  }

  /** Axis-aligned box from min to max. `faces` can skip e.g. the hidden bottom. */
  box(min: Vec3, max: Vec3, color: Rgba, faces: Partial<Record<BoxFace, boolean>> = {}): this {
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    const on = (f: BoxFace) => faces[f] !== false;
    if (on("px")) this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], color);
    if (on("nx")) this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], color);
    if (on("py")) this.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], color);
    if (on("ny")) this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], color);
    if (on("pz")) this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], color);
    if (on("nz")) this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], color);
    return this;
  }

  /** Box centred on c with full size s. */
  boxAt(c: Vec3, s: Vec3, color: Rgba, faces: Partial<Record<BoxFace, boolean>> = {}): this {
    return this.box(
      [c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2],
      [c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2],
      color,
      faces,
    );
  }

  /**
   * Faceted frustum/cylinder along +Y from base centre `c`: bottom radius r0,
   * top radius r1, height h, `sides` facets. Caps are optional.
   */
  cylinder(c: Vec3, r0: number, r1: number, h: number, sides: number, color: Rgba, caps: { top?: boolean; bottom?: boolean } = {}): this {
    const ring = (r: number, y: number) =>
      Array.from({ length: sides }, (_, i): Vec3 => {
        const a = (i / sides) * Math.PI * 2;
        return [c[0] + Math.cos(a) * r, c[1] + y, c[2] + Math.sin(a) * r];
      });
    const lo = ring(r0, 0);
    const hi = ring(r1, h);
    for (let i = 0; i < sides; i++) {
      const j = (i + 1) % sides;
      if (r1 > 1e-6) this.quad(lo[j]!, lo[i]!, hi[i]!, hi[j]!, color);
      else this.tri(lo[j]!, lo[i]!, hi[i]!, color);
    }
    const top: Vec3 = [c[0], c[1] + h, c[2]];
    const bottom: Vec3 = [c[0], c[1], c[2]];
    if (caps.top !== false && r1 > 1e-6) for (let i = 0; i < sides; i++) this.tri(top, hi[(i + 1) % sides]!, hi[i]!, color);
    if (caps.bottom === true) for (let i = 0; i < sides; i++) this.tri(bottom, lo[i]!, lo[(i + 1) % sides]!, color);
    return this;
  }

  /** Low-poly icosphere (subdiv 0 or 1), optionally squashed, flat-shaded. */
  icosphere(c: Vec3, r: Vec3, subdiv: 0 | 1, color: Rgba, jitter?: () => number): this {
    const t = (1 + Math.sqrt(5)) / 2;
    let verts: Vec3[] = [
      [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ].map((v) => normalize(v as Vec3));
    let faces: [number, number, number][] = [
      [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    if (subdiv === 1) {
      const mid = new Map<string, number>();
      const midpoint = (a: number, b: number) => {
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        const hit = mid.get(key);
        if (hit !== undefined) return hit;
        const va = verts[a]!, vb = verts[b]!;
        verts.push(normalize([(va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2]));
        mid.set(key, verts.length - 1);
        return verts.length - 1;
      };
      const next: [number, number, number][] = [];
      for (const [a, b, cc] of faces) {
        const ab = midpoint(a, b), bc = midpoint(b, cc), ca = midpoint(cc, a);
        next.push([a, ab, ca], [b, bc, ab], [cc, ca, bc], [ab, bc, ca]);
      }
      faces = next;
    }
    // Shared per-vertex jitter keeps the surface closed while breaking the perfect sphere.
    verts = verts.map((v) => {
      const k = jitter ? 1 + (jitter() - 0.5) * 0.25 : 1;
      return [c[0] + v[0] * r[0] * k, c[1] + v[1] * r[1] * k, c[2] + v[2] * r[2] * k];
    });
    for (const [a, b, cc] of faces) this.tri(verts[a]!, verts[b]!, verts[cc]!, color);
    return this;
  }

  /** Append another builder's geometry. */
  merge(other: MeshBuilder): this {
    const base = this.p.length / 3;
    this.p.push(...other.p);
    this.n.push(...other.n);
    this.uv.push(...other.uv);
    this.c.push(...other.c);
    this.idx.push(...other.idx.map((i) => i + base));
    return this;
  }

  build(): KitMesh {
    return {
      positions: Float32Array.from(this.p),
      normals: Float32Array.from(this.n),
      uvs: Float32Array.from(this.uv),
      colors: Float32Array.from(this.c),
      indices: Uint32Array.from(this.idx),
    };
  }
}

export type BoxFace = "px" | "nx" | "py" | "ny" | "pz" | "nz";

function faceNormal(a: Vec3, b: Vec3, c: Vec3): Vec3 {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  return normalize([uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]);
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function dist(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** mulberry32 — deterministic PRNG so the same seed always builds the same asset. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

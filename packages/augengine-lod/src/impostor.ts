/**
 * impostor — LOD3 cards that show the asset's real colours.
 *
 * The LOD0 mesh is rasterized on the CPU (orthographic, z-buffered, 2x
 * supersampled) once per impostor face: five views for a "box" impostor
 * (four sides + top), four for "cross" (both sides of the two crossed
 * cards). Each pixel is the surface's **albedo** — baseColor texture
 * (decoded, sRGB → linear) × baseColorFactor × vertex colour — with no
 * lighting baked in, because the engine lights the impostor with its own
 * face normals.
 *
 * Output, by `texture`:
 *   "atlas"   the views packed into one small RGBA atlas (default 64 px
 *             per view), mapped onto the impostor's faces. Box views are
 *             opaque (holes in the silhouette are filled from their
 *             neighbours); cross views carry alpha (glTF alphaMode MASK).
 *   "vertex"  no texture: each face gets the mean albedo of its view as a
 *             vertex colour. Keeps the impostor untextured, so the engine
 *             can still instance it (the city kit uses this).
 *   "none"    one area-weighted average colour for the whole card (0.1).
 *
 * Textures the CPU cannot decode (JPEG, KTX2/Basis, WebP, or PNG without a
 * `decodeImage`) contribute their material's factor colour only; the bake
 * reports this in `warnings`.
 */
import { averageColor, groupsOf, type Aabb, type KitMesh, type Rgba, type Vec3 } from "./mesh.js";
import { textureImage, type LodImage, type LodMaterialSet } from "./material.js";
import type { RgbaImage } from "./png.js";

export type ImpostorKind = "box" | "cross";
export type ImpostorTexture = "atlas" | "vertex" | "none";

export interface ImpostorOptions {
  kind: ImpostorKind;
  texture: ImpostorTexture;
  /** Source materials; omit for vertex-coloured meshes (the kit). */
  materials?: LodMaterialSet;
  /** Decode an image to RGBA8 (the Node entry decodes PNG); return null if unsupported. */
  decodeImage?: (img: LodImage) => RgbaImage | null;
  /** Atlas pixels per view (square). Default 64. */
  tile?: number;
}

export interface ImpostorOut {
  mesh: KitMesh;
  /** glTF material for the impostor (atlas texture is attached by `withImpostorMaterial`). */
  material: any;
  atlas: RgbaImage | null;
  /** Atlas grid: views are `cols` x `rows` cells of `tile` px, in `faces` order. */
  layout: { tile: number; cols: number; rows: number; faces: string[] } | null;
  texture: ImpostorTexture;
  warnings: string[];
}

interface View {
  name: string;
  /** View direction (camera looks along it). */
  d: Vec3;
  right: Vec3;
  up: Vec3;
  /** Impostor face: 4 corners CCW seen from outside, and its normal. */
  quad: [Vec3, Vec3, Vec3, Vec3];
  normal: Vec3;
}

const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function views(kind: ImpostorKind, b: Aabb): View[] {
  const [x0, y0, z0] = b.min, [x1, y1, z1] = b.max;
  const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
  const mk = (name: string, d: Vec3, up: Vec3, quad: [Vec3, Vec3, Vec3, Vec3]): View => {
    const right = cross3(d, up);
    return { name, d, right, up, quad, normal: [-d[0], -d[1], -d[2]] };
  };
  const Y: Vec3 = [0, 1, 0];
  if (kind === "box") {
    return [
      mk("+x", [-1, 0, 0], Y, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
      mk("-x", [1, 0, 0], Y, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
      mk("+z", [0, 0, -1], Y, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
      mk("-z", [0, 0, 1], Y, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
      mk("+y", [0, -1, 0], [0, 0, -1], [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
    ];
  }
  return [
    mk("+z", [0, 0, -1], Y, [[x0, y0, cz], [x1, y0, cz], [x1, y1, cz], [x0, y1, cz]]),
    mk("-z", [0, 0, 1], Y, [[x1, y0, cz], [x0, y0, cz], [x0, y1, cz], [x1, y1, cz]]),
    mk("+x", [-1, 0, 0], Y, [[cx, y0, z1], [cx, y0, z0], [cx, y1, z0], [cx, y1, z1]]),
    mk("-x", [1, 0, 0], Y, [[cx, y0, z0], [cx, y0, z1], [cx, y1, z1], [cx, y1, z0]]),
  ];
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const linearToSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

interface MatSampler {
  factor: Rgba;
  tex: { img: RgbaImage; lin: Float32Array; wrapS: number; wrapT: number } | null;
  cutoff: number | null;
}

function buildSamplers(opts: ImpostorOptions, warnings: string[]): Map<number, MatSampler> {
  const out = new Map<number, MatSampler>();
  const set = opts.materials;
  if (!set) return out;
  const decoded = new Map<number, { img: RgbaImage; lin: Float32Array } | null>();
  set.materials.forEach((m, i) => {
    const pbr = m?.pbrMetallicRoughness ?? {};
    const f = pbr.baseColorFactor ?? [1, 1, 1, 1];
    let tex: MatSampler["tex"] = null;
    const ti = pbr.baseColorTexture;
    if (ti) {
      const imgIdx = textureImage(set, ti.index);
      if (imgIdx !== undefined) {
        if (!decoded.has(imgIdx)) {
          const src = set.images[imgIdx];
          const img = src && src.bytes.length > 0 && opts.decodeImage ? opts.decodeImage(src) : null;
          if (!img) {
            const why = !opts.decodeImage ? "no image decoder was given" : `${src?.mimeType ?? "image"} is not decodable on the CPU`;
            warnings.push(`impostor: baseColor image ${imgIdx} (${src?.mimeType ?? "?"}) not sampled (${why}); its material's factor colour is used instead`);
            decoded.set(imgIdx, null);
          } else {
            const lin = new Float32Array(img.width * img.height * 4);
            for (let p = 0; p < img.width * img.height; p++) {
              for (let k = 0; k < 3; k++) lin[p * 4 + k] = srgbToLinear(img.rgba[p * 4 + k]! / 255);
              lin[p * 4 + 3] = img.rgba[p * 4 + 3]! / 255;
            }
            decoded.set(imgIdx, { img, lin });
          }
        }
        const d = decoded.get(imgIdx);
        if (d) {
          const t = set.textures[ti.index];
          const s = t?.sampler !== undefined ? set.samplers[t.sampler] : undefined;
          tex = { ...d, wrapS: s?.wrapS ?? 10497, wrapT: s?.wrapT ?? 10497 };
        }
      }
    }
    out.set(i, { factor: [f[0], f[1], f[2], f[3]], tex, cutoff: m?.alphaMode === "MASK" ? (m.alphaCutoff ?? 0.5) : null });
  });
  return out;
}

function wrap(x: number, n: number, mode: number): number {
  if (mode === 33071) return Math.min(n - 1, Math.max(0, x)); // CLAMP_TO_EDGE
  if (mode === 33648) { // MIRRORED_REPEAT
    const p = ((x % (2 * n)) + 2 * n) % (2 * n);
    return p < n ? p : 2 * n - 1 - p;
  }
  return ((x % n) + n) % n; // REPEAT
}

function sampleBilinear(t: NonNullable<MatSampler["tex"]>, u: number, v: number, out: number[]): void {
  const W = t.img.width, H = t.img.height;
  const x = u * W - 0.5, y = v * H - 0.5;
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
  for (let k = 0; k < 4; k++) out[k] = 0;
  for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
    const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
    const p = (wrap(yi + dy, H, t.wrapT) * W + wrap(xi + dx, W, t.wrapS)) * 4;
    for (let k = 0; k < 4; k++) out[k]! += t.lin[p + k]! * w;
  }
}

/** Rasterize LOD0's albedo seen along `view` into a `size` x `size` linear RGBA buffer (alpha = coverage). */
function renderView(m: KitMesh, view: View, b: Aabb, size: number, samplers: Map<number, MatSampler>): Float32Array {
  const SS = 2, W = size * SS;
  // Extents of the bounds box along right/up (the face the view projects onto).
  const corners: Vec3[] = [];
  for (const x of [b.min[0], b.max[0]]) for (const y of [b.min[1], b.max[1]]) for (const z of [b.min[2], b.max[2]]) corners.push([x, y, z]);
  const rs = corners.map((c) => dot(c, view.right)), us = corners.map((c) => dot(c, view.up));
  const r0 = Math.min(...rs), r1 = Math.max(...rs), u0 = Math.min(...us), u1 = Math.max(...us);
  const sx = W / Math.max(1e-9, r1 - r0), sy = W / Math.max(1e-9, u1 - u0);
  const acc = new Float32Array(W * W * 4);
  const depth = new Float32Array(W * W).fill(Infinity);
  const groups = groupsOf(m);
  const px = new Float32Array(3), py = new Float32Array(3), pz = new Float32Array(3);
  const tmp = [0, 0, 0, 0];
  for (const g of groups) {
    const s = samplers.get(g.material) ?? { factor: [1, 1, 1, 1] as Rgba, tex: null, cutoff: null };
    for (let t = g.start / 3; t < (g.start + g.count) / 3; t++) {
      const id = [m.indices[t * 3]!, m.indices[t * 3 + 1]!, m.indices[t * 3 + 2]!];
      for (let k = 0; k < 3; k++) {
        const p = [m.positions[id[k]! * 3]!, m.positions[id[k]! * 3 + 1]!, m.positions[id[k]! * 3 + 2]!];
        px[k] = (dot(p, view.right) - r0) * sx;
        py[k] = (u1 - dot(p, view.up)) * sy;
        pz[k] = dot(p, view.d);
      }
      const area = (px[1]! - px[0]!) * (py[2]! - py[0]!) - (py[1]! - py[0]!) * (px[2]! - px[0]!);
      if (Math.abs(area) < 1e-12) continue;
      const minX = Math.max(0, Math.floor(Math.min(px[0]!, px[1]!, px[2]!)));
      const maxX = Math.min(W - 1, Math.ceil(Math.max(px[0]!, px[1]!, px[2]!)));
      const minY = Math.max(0, Math.floor(Math.min(py[0]!, py[1]!, py[2]!)));
      const maxY = Math.min(W - 1, Math.ceil(Math.max(py[0]!, py[1]!, py[2]!)));
      for (let yy = minY; yy <= maxY; yy++) for (let xx = minX; xx <= maxX; xx++) {
        const x = xx + 0.5, y = yy + 0.5;
        const w0 = ((px[1]! - x) * (py[2]! - y) - (py[1]! - y) * (px[2]! - x)) / area;
        const w1 = ((px[2]! - x) * (py[0]! - y) - (py[2]! - y) * (px[0]! - x)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
        const z = w0 * pz[0]! + w1 * pz[1]! + w2 * pz[2]!;
        const o = yy * W + xx;
        if (z >= depth[o]!) continue;
        let r = s.factor[0], gg = s.factor[1], bb = s.factor[2], a = s.factor[3];
        const ca = id[0]! * 4, cb = id[1]! * 4, cc = id[2]! * 4;
        r *= w0 * m.colors[ca]! + w1 * m.colors[cb]! + w2 * m.colors[cc]!;
        gg *= w0 * m.colors[ca + 1]! + w1 * m.colors[cb + 1]! + w2 * m.colors[cc + 1]!;
        bb *= w0 * m.colors[ca + 2]! + w1 * m.colors[cb + 2]! + w2 * m.colors[cc + 2]!;
        a *= w0 * m.colors[ca + 3]! + w1 * m.colors[cb + 3]! + w2 * m.colors[cc + 3]!;
        if (s.tex) {
          const u = w0 * m.uvs[id[0]! * 2]! + w1 * m.uvs[id[1]! * 2]! + w2 * m.uvs[id[2]! * 2]!;
          const v = w0 * m.uvs[id[0]! * 2 + 1]! + w1 * m.uvs[id[1]! * 2 + 1]! + w2 * m.uvs[id[2]! * 2 + 1]!;
          sampleBilinear(s.tex, u, v, tmp);
          r *= tmp[0]!; gg *= tmp[1]!; bb *= tmp[2]!; a *= tmp[3]!;
        }
        if (s.cutoff !== null && a < s.cutoff) continue;
        depth[o] = z;
        acc[o * 4] = r; acc[o * 4 + 1] = gg; acc[o * 4 + 2] = bb; acc[o * 4 + 3] = 1;
      }
    }
  }
  // Box-filter the supersampled buffer: colour averaged over covered samples, alpha = coverage.
  const out = new Float32Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let r = 0, g = 0, bl = 0, cov = 0;
    for (let dy = 0; dy < SS; dy++) for (let dx = 0; dx < SS; dx++) {
      const o = ((y * SS + dy) * W + x * SS + dx) * 4;
      if (acc[o + 3]! > 0) { r += acc[o]!; g += acc[o + 1]!; bl += acc[o + 2]!; cov++; }
    }
    const o = (y * size + x) * 4;
    if (cov > 0) { out[o] = r / cov; out[o + 1] = g / cov; out[o + 2] = bl / cov; }
    out[o + 3] = cov / (SS * SS);
  }
  return out;
}

/** Fill uncovered pixels from covered neighbours (so filtering never pulls in black), then any rest with the mean. */
function dilate(img: Float32Array, size: number): void {
  const covered = new Uint8Array(size * size);
  let mr = 0, mg = 0, mb = 0, n = 0;
  for (let p = 0; p < size * size; p++) if (img[p * 4 + 3]! > 0) {
    covered[p] = 1; mr += img[p * 4]!; mg += img[p * 4 + 1]!; mb += img[p * 4 + 2]!; n++;
  }
  for (let pass = 0; pass < size; pass++) {
    const next: [number, number, number, number][] = [];
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (covered[p]) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= size || ny >= size || !covered[ny * size + nx]) continue;
        const q = (ny * size + nx) * 4;
        r += img[q]!; g += img[q + 1]!; b += img[q + 2]!; c++;
      }
      if (c) next.push([p, r / c, g / c, b / c]);
    }
    if (next.length === 0) break;
    for (const [p, r, g, b] of next) { img[p * 4] = r; img[p * 4 + 1] = g; img[p * 4 + 2] = b; covered[p] = 1; }
  }
  for (let p = 0; p < size * size; p++) if (!covered[p] && n > 0) { img[p * 4] = mr / n; img[p * 4 + 1] = mg / n; img[p * 4 + 2] = mb / n; }
}

/** Build the LOD3 impostor for `lod0` (see the module comment). */
export function buildImpostor(lod0: KitMesh, b: Aabb, opts: ImpostorOptions): ImpostorOut {
  const warnings: string[] = [];
  const vs = views(opts.kind, b);
  const tile = Math.max(8, Math.round(opts.tile ?? 64));
  const metallic = avgFactor(opts.materials, (m) => m?.pbrMetallicRoughness?.metallicFactor ?? 1, 0);
  const roughness = avgFactor(opts.materials, (m) => m?.pbrMetallicRoughness?.roughnessFactor ?? 1, 1);
  const baseMaterial = {
    name: "lod3-impostor",
    pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: metallic, roughnessFactor: roughness },
    ...(opts.kind === "cross" && opts.texture === "atlas" ? { alphaMode: "MASK", alphaCutoff: 0.5 } : {}),
  };

  let rendered: Float32Array[] = [];
  if (opts.texture !== "none") {
    const samplers = buildSamplers(opts, warnings);
    rendered = vs.map((v) => renderView(lod0, v, b, tile, samplers));
  }
  const avg: Rgba = opts.texture === "none" ? averageColor(lod0) : [1, 1, 1, 1];
  const faceColor = (i: number): Rgba => {
    if (opts.texture !== "vertex") return opts.texture === "none" ? avg : [1, 1, 1, 1];
    const img = rendered[i]!;
    let r = 0, g = 0, bl = 0, w = 0;
    for (let p = 0; p < tile * tile; p++) {
      const a = img[p * 4 + 3]!;
      r += img[p * 4]! * a; g += img[p * 4 + 1]! * a; bl += img[p * 4 + 2]! * a; w += a;
    }
    return w > 0 ? [r / w, g / w, bl / w, 1] : [0.5, 0.5, 0.5, 1];
  };

  const cols = opts.kind === "box" ? 3 : 2, rows = 2;
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [];
  vs.forEach((v, i) => {
    const color = faceColor(i);
    const cellX = i % cols, cellY = Math.floor(i / cols);
    // UV of each corner = where the view projected that point into its cell.
    const corners: Vec3[] = [];
    for (const x of [b.min[0], b.max[0]]) for (const y of [b.min[1], b.max[1]]) for (const z of [b.min[2], b.max[2]]) corners.push([x, y, z]);
    const rs = corners.map((c) => dot(c, v.right)), us = corners.map((c) => dot(c, v.up));
    const r0 = Math.min(...rs), r1 = Math.max(...rs), u0 = Math.min(...us), u1 = Math.max(...us);
    const base = P.length / 3;
    for (const q of v.quad) {
      P.push(...q);
      N.push(...v.normal);
      const fu = (dot(q, v.right) - r0) / Math.max(1e-9, r1 - r0);
      const fv = (u1 - dot(q, v.up)) / Math.max(1e-9, u1 - u0);
      // Half-texel inset keeps bilinear filtering inside the cell.
      const inset = 0.5 / tile;
      UV.push((cellX + inset + fu * (1 - 2 * inset)) / cols, (cellY + inset + fv * (1 - 2 * inset)) / rows);
      C.push(...color);
    }
    I.push(base, base + 1, base + 2, base, base + 2, base + 3);
  });
  const mesh: KitMesh = {
    positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV),
    colors: Float32Array.from(C), indices: Uint32Array.from(I),
  };

  let atlas: RgbaImage | null = null;
  if (opts.texture === "atlas") {
    const AW = cols * tile, AH = rows * tile;
    const rgba = new Uint8Array(AW * AH * 4);
    rendered.forEach((img, i) => {
      if (opts.kind === "box") dilate(img, tile);
      else {
        // Cross cards keep alpha, but fill colour under transparent texels too (no dark fringes when filtered).
        const alpha = Float32Array.from({ length: tile * tile }, (_, p) => img[p * 4 + 3]!);
        dilate(img, tile);
        for (let p = 0; p < tile * tile; p++) img[p * 4 + 3] = alpha[p]!;
      }
      const ox = (i % cols) * tile, oy = Math.floor(i / cols) * tile;
      for (let y = 0; y < tile; y++) for (let x = 0; x < tile; x++) {
        const s = (y * tile + x) * 4, d = ((oy + y) * AW + ox + x) * 4;
        for (let k = 0; k < 3; k++) rgba[d + k] = Math.round(Math.min(1, Math.max(0, linearToSrgb(img[s + k]!))) * 255);
        rgba[d + 3] = opts.kind === "box" ? 255 : Math.round(Math.min(1, img[s + 3]!) * 255);
      }
    });
    atlas = { width: AW, height: AH, rgba };
  }
  return {
    mesh,
    material: baseMaterial,
    atlas,
    layout: opts.texture === "atlas" ? { tile, cols, rows, faces: vs.map((v) => v.name) } : null,
    texture: opts.texture,
    warnings,
  };
}

function avgFactor(set: LodMaterialSet | undefined, f: (m: any) => number, fallback: number): number {
  if (!set || set.materials.length === 0) return fallback;
  return set.materials.reduce((s, m) => s + f(m), 0) / set.materials.length;
}

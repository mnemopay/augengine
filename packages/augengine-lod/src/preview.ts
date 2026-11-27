/**
 * preview — a tiny software rasterizer that renders a mesh to PNG, so baked
 * assets can be eyeballed (and diffed) without a browser or GPU. Lambert +
 * ambient shading of vertex colours, z-buffered, 3/4 view from the front
 * right. Not the engine's renderer; it exists to check shape and LOD loss.
 */
import { deflateSync } from "node:zlib";
import { bounds, groupsOf, type KitMesh } from "./mesh.js";
import { textureImage, type LodImage, type LodMaterialSet } from "./material.js";
import { encodePngWith, type RgbaImage } from "./png.js";

export interface PreviewOptions {
  size?: number;
  /** Camera yaw / pitch in degrees. */
  yaw?: number;
  pitch?: number;
  background?: [number, number, number];
  /** Source materials: base colour factors and (decodable) baseColor textures are applied. */
  materials?: LodMaterialSet;
  decodeImage?: (img: LodImage) => RgbaImage | null;
}

/** Render to RGBA pixels. */
export function renderPreview(m: KitMesh, opts: PreviewOptions = {}): { width: number; height: number; rgba: Uint8Array } {
  const W = opts.size ?? 256, H = W;
  const yaw = ((opts.yaw ?? 35) * Math.PI) / 180;
  const pitch = ((opts.pitch ?? 20) * Math.PI) / 180;
  const bg = opts.background ?? [0.83, 0.86, 0.9];
  const rgba = new Uint8Array(W * H * 4);
  const depth = new Float32Array(W * H).fill(Infinity);
  for (let i = 0; i < W * H; i++) {
    // Vertical sky gradient so silhouettes read.
    const t = Math.floor(i / W) / H;
    rgba[i * 4] = Math.round(255 * (bg[0] - t * 0.12));
    rgba[i * 4 + 1] = Math.round(255 * (bg[1] - t * 0.12));
    rgba[i * 4 + 2] = Math.round(255 * (bg[2] - t * 0.1));
    rgba[i * 4 + 3] = 255;
  }
  const b = bounds(m);
  const c = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
  const r = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) / 2 || 1;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // View transform: rotate world around Y by yaw, then around X by pitch (orthographic).
  const project = (x: number, y: number, z: number): [number, number, number] => {
    const dx = x - c[0]!, dy = y - c[1]!, dz = z - c[2]!;
    const x1 = cy * dx - sy * dz;
    const z1 = sy * dx + cy * dz;
    const y2 = cp * dy - sp * z1;
    const z2 = sp * dy + cp * z1;
    const s = (W * 0.45) / r;
    return [W / 2 + x1 * s, H / 2 - y2 * s, -z2];
  };
  const light = normalize([0.45, 0.8, 0.4]);
  const triMat = new Uint32Array(m.indices.length / 3);
  for (const g of groupsOf(m)) triMat.fill(g.material, g.start / 3, (g.start + g.count) / 3);
  const decoded = new Map<number, RgbaImage | null>();
  const matInfo = (mi: number): { f: number[]; tex: RgbaImage | null } => {
    const mat = opts.materials?.materials[mi];
    const pbr = mat?.pbrMetallicRoughness ?? {};
    const f: number[] = pbr.baseColorFactor ?? [1, 1, 1, 1];
    let tex: RgbaImage | null = null;
    if (opts.materials && pbr.baseColorTexture && opts.decodeImage) {
      const img = textureImage(opts.materials, pbr.baseColorTexture.index);
      if (img !== undefined) {
        if (!decoded.has(img)) decoded.set(img, opts.materials.images[img] ? opts.decodeImage(opts.materials.images[img]!) : null);
        tex = decoded.get(img) ?? null;
      }
    }
    return { f, tex };
  };
  const infos = new Map<number, { f: number[]; tex: RgbaImage | null }>();
  for (let t = 0; t < m.indices.length / 3; t++) {
    const ids = [m.indices[t * 3]!, m.indices[t * 3 + 1]!, m.indices[t * 3 + 2]!];
    const pts = ids.map((v) => project(m.positions[v * 3]!, m.positions[v * 3 + 1]!, m.positions[v * 3 + 2]!));
    // Average of the corner normals: the face normal for flat meshes, a smoothed one for smooth meshes.
    const n = normalize([0, 1, 2].map((k) => m.normals[ids[0]! * 3 + k]! + m.normals[ids[1]! * 3 + k]! + m.normals[ids[2]! * 3 + k]!));
    // Back-face cull like the engine's main pipelines (cullMode "back").
    if (sp * n[1]! + cp * (sy * n[0]! + cy * n[2]!) <= 0) continue;
    const lambert = Math.max(0, n[0]! * light[0]! + n[1]! * light[1]! + n[2]! * light[2]!);
    const shade = 0.38 + 0.62 * lambert;
    const mi = triMat[t]!;
    if (!infos.has(mi)) infos.set(mi, matInfo(mi));
    const info = infos.get(mi)!;
    const col = [0, 1, 2].map((k) => ((m.colors[ids[0]! * 4 + k]! + m.colors[ids[1]! * 4 + k]! + m.colors[ids[2]! * 4 + k]!) / 3) * info.f[k]!);
    const [a, bb, cc] = pts as [[number, number, number], [number, number, number], [number, number, number]];
    const area = (bb[0] - a[0]) * (cc[1] - a[1]) - (bb[1] - a[1]) * (cc[0] - a[0]);
    if (Math.abs(area) < 1e-9) continue;
    const minX = Math.max(0, Math.floor(Math.min(a[0], bb[0], cc[0])));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(a[0], bb[0], cc[0])));
    const minY = Math.max(0, Math.floor(Math.min(a[1], bb[1], cc[1])));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(a[1], bb[1], cc[1])));
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const x = px + 0.5, y = py + 0.5;
        const w0 = ((bb[0] - x) * (cc[1] - y) - (bb[1] - y) * (cc[0] - x)) / area;
        const w1 = ((cc[0] - x) * (a[1] - y) - (cc[1] - y) * (a[0] - x)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
        const z = w0 * a[2] + w1 * bb[2] + w2 * cc[2];
        const o = py * W + px;
        if (z >= depth[o]!) continue;

        let tr = 1, tg = 1, tb = 1;
        if (info.tex) {
          const u = w0 * m.uvs[ids[0]! * 2]! + w1 * m.uvs[ids[1]! * 2]! + w2 * m.uvs[ids[2]! * 2]!;
          const v = w0 * m.uvs[ids[0]! * 2 + 1]! + w1 * m.uvs[ids[1]! * 2 + 1]! + w2 * m.uvs[ids[2]! * 2 + 1]!;
          const T = info.tex;
          const tx = ((Math.floor(u * T.width) % T.width) + T.width) % T.width, ty = ((Math.floor(v * T.height) % T.height) + T.height) % T.height;
          const q = (ty * T.width + tx) * 4;
          const mat = opts.materials?.materials[mi];
          if (mat?.alphaMode === "MASK" && info.f[3]! * T.rgba[q + 3]! / 255 < (mat.alphaCutoff ?? 0.5)) continue;
          tr = T.rgba[q]! / 255; tg = T.rgba[q + 1]! / 255; tb = T.rgba[q + 2]! / 255;
        }
        depth[o] = z;
        rgba[o * 4] = clamp255(col[0]! * tr * shade);
        rgba[o * 4 + 1] = clamp255(col[1]! * tg * shade);
        rgba[o * 4 + 2] = clamp255(col[2]! * tb * shade);
      }
    }
  }
  return { width: W, height: H, rgba };
}

/** Lay several renders side by side. */
export function contactSheet(images: { width: number; height: number; rgba: Uint8Array }[]): { width: number; height: number; rgba: Uint8Array } {
  const H = Math.max(...images.map((i) => i.height));
  const W = images.reduce((s, i) => s + i.width, 0);
  const rgba = new Uint8Array(W * H * 4);
  let ox = 0;
  for (const img of images) {
    for (let y = 0; y < img.height; y++) rgba.set(img.rgba.subarray(y * img.width * 4, (y + 1) * img.width * 4), (y * W + ox) * 4);
    ox += img.width;
  }
  return { width: W, height: H, rgba };
}

/** Encode RGBA pixels as a PNG (node:zlib deflate). */
export function encodePng(img: { width: number; height: number; rgba: Uint8Array }): Uint8Array {
  return encodePngWith(img, (d) => new Uint8Array(deflateSync(d)));
}

function clamp255(x: number): number {
  return Math.max(0, Math.min(255, Math.round(x * 255)));
}

function normalize(v: number[]): number[] {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
  return [v[0]! / l, v[1]! / l, v[2]! / l];
}

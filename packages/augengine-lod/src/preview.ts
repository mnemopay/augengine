/**
 * preview — a tiny software rasterizer that renders a mesh to PNG, so baked
 * assets can be eyeballed (and diffed) without a browser or GPU. Lambert +
 * ambient shading of vertex colours, z-buffered, 3/4 view from the front
 * right. Not the engine's renderer; it exists to check shape and LOD loss.
 */
import { deflateSync } from "node:zlib";
import { bounds, type KitMesh } from "./mesh.js";

export interface PreviewOptions {
  size?: number;
  /** Camera yaw / pitch in degrees. */
  yaw?: number;
  pitch?: number;
  background?: [number, number, number];
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
  for (let t = 0; t < m.indices.length / 3; t++) {
    const ids = [m.indices[t * 3]!, m.indices[t * 3 + 1]!, m.indices[t * 3 + 2]!];
    const pts = ids.map((v) => project(m.positions[v * 3]!, m.positions[v * 3 + 1]!, m.positions[v * 3 + 2]!));
    const n = [m.normals[ids[0]! * 3]!, m.normals[ids[0]! * 3 + 1]!, m.normals[ids[0]! * 3 + 2]!];
    // Back-face cull like the engine's main pipelines (cullMode "back").
    if (sp * n[1]! + cp * (sy * n[0]! + cy * n[2]!) <= 0) continue;
    const lambert = Math.max(0, n[0]! * light[0]! + n[1]! * light[1]! + n[2]! * light[2]!);
    const shade = 0.38 + 0.62 * lambert;
    const col = [0, 1, 2].map((k) => (m.colors[ids[0]! * 4 + k]! + m.colors[ids[1]! * 4 + k]! + m.colors[ids[2]! * 4 + k]!) / 3);
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
        depth[o] = z;
        rgba[o * 4] = clamp255(col[0]! * shade);
        rgba[o * 4 + 1] = clamp255(col[1]! * shade);
        rgba[o * 4 + 2] = clamp255(col[2]! * shade);
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

/** Encode RGBA pixels as a PNG (no dependencies). */
export function encodePng(img: { width: number; height: number; rgba: Uint8Array }): Uint8Array {
  const { width: W, height: H, rgba } = img;
  const raw = new Uint8Array(H * (W * 4 + 1));
  for (let y = 0; y < H; y++) {
    raw[y * (W * 4 + 1)] = 0;
    raw.set(rgba.subarray(y * W * 4, (y + 1) * W * 4), y * (W * 4 + 1) + 1);
  }
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, data.length);
    out.set(new TextEncoder().encode(type), 4);
    out.set(data, 8);
    dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, W);
  dv.setUint32(4, H);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", new Uint8Array(deflateSync(raw))),
    chunk("IEND", new Uint8Array(0)),
  ];
  const total = parts.reduce((s, p) => s + p.length, 0);
  const png = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { png.set(p, o); o += p.length; }
  return png;
}

let CRC_TABLE: Uint32Array | null = null;
function crc32(data: Uint8Array): number {
  if (!CRC_TABLE) {
    CRC_TABLE = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) crc = CRC_TABLE[(crc ^ data[i]!) & 0xff]! ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function clamp255(x: number): number {
  return Math.max(0, Math.min(255, Math.round(x * 255)));
}

function normalize(v: number[]): number[] {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
  return [v[0]! / l, v[1]! / l, v[2]! / l];
}

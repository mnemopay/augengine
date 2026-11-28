/**
 * Radiance .hdr (RGBE) decoder + HDR-environment helpers for the natural
 * light tier. Written for this engine (no vendored code).
 *
 * Format (Greg Ward, "Real Pixels", Graphics Gems II, 1991; the Radiance
 * file format docs):
 *   - ASCII header starting "#?RADIANCE" or "#?RGBE", lines of KEY=VALUE,
 *     FORMAT=32-bit_rle_rgbe (xyze is rejected), an empty line, then a
 *     resolution line, normally "-Y <height> +X <width>" (top row first).
 *   - Pixels are 4 bytes (R, G, B mantissas + shared exponent E):
 *     value = mantissa · 2^(E − 136), and E = 0 means black (rgbe.c).
 *   - Scanlines are either flat, or "new" adaptive RLE: 2, 2, hi(width),
 *     lo(width), then each of the 4 channels separately as runs
 *     (count > 128 → repeat next byte count−128 times; else count literals).
 *     Old-style RLE (1,1,1,n repeat markers) is also accepted.
 * OpenEXR is NOT decoded here; convert .exr to .hdr offline.
 */

import type { Vec3 } from "../../../scene/types.js";
import { equirectDirection } from "./sh9.js";

export interface HdrImage {
  width: number;
  height: number;
  /** Linear RGB, 3 floats per pixel, row 0 = top (zenith for an equirect). */
  data: Float32Array;
  channels: 3;
  /** Header EXPOSURE= product (already divided out of `data`), 1 when absent. */
  exposure: number;
}

/** RGBE → linear RGB (Ward's rgbe.c convention). */
export function rgbeToFloat(r: number, g: number, b: number, e: number, out: number[] | Float32Array = [0, 0, 0], offset = 0): void {
  if (e === 0) {
    out[offset] = 0; out[offset + 1] = 0; out[offset + 2] = 0;
    return;
  }
  const f = Math.pow(2, e - 136);
  out[offset] = r * f;
  out[offset + 1] = g * f;
  out[offset + 2] = b * f;
}

/** Linear RGB → RGBE bytes (inverse of rgbeToFloat), for tests and tooling. */
export function floatToRgbe(r: number, g: number, b: number): [number, number, number, number] {
  const v = Math.max(r, g, b);
  if (v < 1e-32) return [0, 0, 0, 0];
  const e = Math.floor(Math.log2(v)) + 1;
  const scale = Math.pow(2, -e) * 256;
  return [Math.min(255, Math.floor(r * scale)), Math.min(255, Math.floor(g * scale)), Math.min(255, Math.floor(b * scale)), e + 128];
}

export function parseRadianceHdr(buffer: ArrayBuffer | Uint8Array): HdrImage {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let pos = 0;
  const readLine = (): string => {
    let s = "";
    while (pos < bytes.length) {
      const c = bytes[pos++]!;
      if (c === 0x0a) break;
      s += String.fromCharCode(c);
    }
    return s.replace(/\r$/, "");
  };
  const magic = readLine();
  if (!magic.startsWith("#?RADIANCE") && !magic.startsWith("#?RGBE")) throw new Error("hdr: not a Radiance file");
  let format = "";
  let exposure = 1;
  for (;;) {
    if (pos >= bytes.length) throw new Error("hdr: truncated header");
    const line = readLine();
    if (line === "") break;
    if (line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim().toUpperCase();
    const value = line.slice(eq + 1).trim();
    if (key === "FORMAT") format = value;
    else if (key === "EXPOSURE") {
      const v = Number(value);
      if (Number.isFinite(v) && v > 0) exposure *= v;
    }
  }
  if (format && format !== "32-bit_rle_rgbe") throw new Error(`hdr: unsupported FORMAT ${format}`);
  const res = readLine().trim().split(/\s+/);
  if (res.length !== 4) throw new Error("hdr: bad resolution line");
  const [ya, ys, xa, xs] = res as [string, string, string, string];
  if (ya[1] !== "Y" || xa[1] !== "X") throw new Error(`hdr: unsupported orientation ${res.join(" ")}`);
  const height = Number(ys), width = Number(xs);
  if (!(width > 0 && height > 0 && width * height < 1 << 28)) throw new Error("hdr: bad dimensions");
  const flipY = ya[0] === "+"; // +Y = bottom row first
  const flipX = xa[0] === "-";
  const data = new Float32Array(width * height * 3);
  const scan = new Uint8Array(width * 4);
  const tmp = [0, 0, 0];
  for (let y = 0; y < height; y++) {
    pos = readScanline(bytes, pos, width, scan);
    const row = flipY ? height - 1 - y : y;
    for (let x = 0; x < width; x++) {
      rgbeToFloat(scan[x * 4]!, scan[x * 4 + 1]!, scan[x * 4 + 2]!, scan[x * 4 + 3]!, tmp, 0);
      const col = flipX ? width - 1 - x : x;
      const o = (row * width + col) * 3;
      data[o] = tmp[0]! / exposure;
      data[o + 1] = tmp[1]! / exposure;
      data[o + 2] = tmp[2]! / exposure;
    }
  }
  return { width, height, data, channels: 3, exposure };
}

/** Decode one scanline into `out` (width × RGBE). Returns the new read position. */
function readScanline(bytes: Uint8Array, start: number, width: number, out: Uint8Array): number {
  let pos = start;
  const need = (n: number) => { if (pos + n > bytes.length) throw new Error("hdr: truncated pixel data"); };
  need(4);
  const isNewRle = width >= 8 && width < 32768 && bytes[pos] === 2 && bytes[pos + 1] === 2 && (bytes[pos + 2]! & 0x80) === 0;
  if (isNewRle) {
    const w = (bytes[pos + 2]! << 8) | bytes[pos + 3]!;
    if (w !== width) throw new Error("hdr: scanline width mismatch");
    pos += 4;
    for (let ch = 0; ch < 4; ch++) {
      let x = 0;
      while (x < width) {
        need(1);
        let count = bytes[pos++]!;
        if (count > 128) {
          count -= 128;
          need(1);
          const v = bytes[pos++]!;
          if (x + count > width) throw new Error("hdr: bad run length");
          for (let i = 0; i < count; i++) out[(x++) * 4 + ch] = v;
        } else {
          if (count === 0 || x + count > width) throw new Error("hdr: bad literal length");
          need(count);
          for (let i = 0; i < count; i++) out[(x++) * 4 + ch] = bytes[pos++]!;
        }
      }
    }
    return pos;
  }
  // Flat pixels, possibly with old-style run markers (1, 1, 1, n).
  let x = 0;
  let shift = 0;
  while (x < width) {
    need(4);
    const r = bytes[pos]!, g = bytes[pos + 1]!, b = bytes[pos + 2]!, e = bytes[pos + 3]!;
    pos += 4;
    if (r === 1 && g === 1 && b === 1) {
      if (x === 0) throw new Error("hdr: run marker at scanline start");
      const count = e << shift;
      for (let i = 0; i < count && x < width; i++, x++) out.copyWithin(x * 4, (x - 1) * 4, x * 4);
      shift += 8;
    } else {
      out[x * 4] = r; out[x * 4 + 1] = g; out[x * 4 + 2] = b; out[x * 4 + 3] = e;
      x++;
      shift = 0;
    }
  }
  return pos;
}

export async function loadRadianceHdr(url: string): Promise<HdrImage> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`hdr: fetch ${res.status} ${url}`);
  return parseRadianceHdr(await res.arrayBuffer());
}

// ─── Sun extraction ────────────────────────────────────────────────────

export interface ExtractedSun {
  /** Unit vector toward the sun (world, same equirect mapping as sh9.ts). */
  dir: Vec3;
  /** Illuminance (per channel, image units × sr) on a surface facing it. */
  illuminance: Vec3;
  /** Number of pixels that were classed as sun. */
  pixels: number;
}

/**
 * Find the sun in an outdoor equirect and remove it, so it can be shaded as
 * the shadow-casting key light instead of being baked (blurred, fp16-clamped,
 * double counted) into the environment.
 *
 * Pixels brighter than `threshold` × the image's 99th-percentile luminance
 * (upper hemisphere) near the brightest pixel are the sun: their radiance ×
 * solid angle summed is its illuminance, the weighted centroid its direction;
 * they are replaced by the threshold radiance. Returns null when no compact
 * bright region exists (overcast / interior images).
 */
export function extractSun(img: { width: number; height: number; data: Float32Array; channels: number }, threshold = 20, maxAngleDeg = 6, rotation = 0): ExtractedSun | null {
  const { width: W, height: H, data, channels: C } = img;
  const lum = (o: number) => 0.2126 * data[o]! + 0.7152 * data[o + 1]! + 0.0722 * data[o + 2]!;
  // 99th percentile of the upper hemisphere (sampled).
  const samples: number[] = [];
  let maxL = 0, maxIdx = -1;
  for (let y = 0; y < H / 2; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * C;
      const l = lum(o);
      if (((x + y * 7) & 15) === 0) samples.push(l);
      if (l > maxL) { maxL = l; maxIdx = y * W + x; }
    }
  }
  if (maxIdx < 0 || samples.length === 0) return null;
  samples.sort((a, b) => a - b);
  const p99 = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.99))]!;
  const cut = Math.max(p99 * threshold, 1e-9);
  if (maxL < cut) return null;
  const sx = maxIdx % W, sy = Math.floor(maxIdx / W);
  const peakDir = equirectDirection((sx + 0.5) / W, (sy + 0.5) / H, rotation);
  const cosMax = Math.cos((maxAngleDeg * Math.PI) / 180);
  const E: [number, number, number] = [0, 0, 0];
  const centroid: [number, number, number] = [0, 0, 0];
  let pixels = 0;
  const dTheta = Math.PI / H, dPhi = (2 * Math.PI) / W;
  for (let y = 0; y < H; y++) {
    const st = Math.sin(((y + 0.5) / H) * Math.PI);
    const omega = st * dTheta * dPhi;
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * C;
      const l = lum(o);
      if (l < cut) continue;
      const d = equirectDirection((x + 0.5) / W, (y + 0.5) / H, rotation);
      if (d[0] * peakDir[0] + d[1] * peakDir[1] + d[2] * peakDir[2] < cosMax) continue;
      // Excess above the cut is the sun; the cut level stays in the sky.
      const k = (l - cut) / l;
      const er = data[o]! * k * omega, eg = data[o + 1]! * k * omega, eb = data[o + 2]! * k * omega;
      E[0] += er; E[1] += eg; E[2] += eb;
      const w = 0.2126 * er + 0.7152 * eg + 0.0722 * eb;
      centroid[0] += d[0] * w; centroid[1] += d[1] * w; centroid[2] += d[2] * w;
      data[o] = data[o]! * (1 - k);
      data[o + 1] = data[o + 1]! * (1 - k);
      data[o + 2] = data[o + 2]! * (1 - k);
      pixels++;
    }
  }
  const len = Math.hypot(centroid[0], centroid[1], centroid[2]);
  if (pixels === 0 || len < 1e-12) return null;
  return { dir: [centroid[0] / len, centroid[1] / len, centroid[2] / len], illuminance: E, pixels };
}

/**
 * png — dependency-free PNG decode/encode for the impostor atlas.
 *
 * Browser-safe: the zlib step is injected. The Node entry passes
 * `node:zlib` (`inflateSync` / `deflateSync`); in a browser, `decodePng`
 * needs an inflate you supply, and `encodePngStored` writes uncompressed
 * ("stored") deflate blocks so no compressor is needed at all.
 *
 * Decode supports what glTF exporters write: 8- and 16-bit greyscale, grey +
 * alpha, RGB, RGBA and 8-bit palette images, non-interlaced. Anything else
 * returns null and the caller falls back (see `impostorTexture` in bake.ts).
 */

export interface RgbaImage {
  width: number;
  height: number;
  /** 8-bit RGBA, row-major, top row first. */
  rgba: Uint8Array;
}

export type Inflate = (data: Uint8Array) => Uint8Array;
export type Deflate = (data: Uint8Array) => Uint8Array;

const SIG = [137, 80, 78, 71, 13, 10, 26, 10];

export function isPng(bytes: Uint8Array): boolean {
  return bytes.length > 8 && SIG.every((b, i) => bytes[i] === b);
}

/** Decode a PNG to RGBA8, or null when the variant is unsupported. */
export function decodePng(bytes: Uint8Array, inflate: Inflate): RgbaImage | null {
  if (!isPng(bytes)) return null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 8;
  let width = 0, height = 0, depth = 0, colorType = 0, interlace = 0;
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  while (off + 8 <= bytes.length) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(bytes[off + 4]!, bytes[off + 5]!, bytes[off + 6]!, bytes[off + 7]!);
    const body = bytes.subarray(off + 8, off + 8 + len);
    if (type === "IHDR") {
      width = dv.getUint32(off + 8);
      height = dv.getUint32(off + 12);
      depth = body[8]!;
      colorType = body[9]!;
      interlace = body[12]!;
    } else if (type === "PLTE") palette = body;
    else if (type === "tRNS") trns = body;
    else if (type === "IDAT") idat.push(body);
    else if (type === "IEND") break;
    off += 12 + len;
  }
  if (!width || !height || interlace !== 0) return null;
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
  if (!channels) return null;
  if (!(depth === 8 || (depth === 16 && colorType !== 3))) return null;
  const total = idat.reduce((s, c) => s + c.length, 0);
  const z = new Uint8Array(total);
  let o = 0;
  for (const c of idat) { z.set(c, o); o += c.length; }
  const raw = inflate(z);
  const bpp = channels * (depth / 8);
  const stride = width * bpp;
  if (raw.length < height * (stride + 1)) return null;
  const px = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)]!;
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = px.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? px.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? row[x - bpp]! : 0;
      const b = prev ? prev[x]! : 0;
      const c = prev && x >= bpp ? prev[x - bpp]! : 0;
      let v = src[x]!;
      switch (filter) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) >> 1; break;
        case 4: {
          const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
        default: return null;
      }
      row[x] = v & 0xff;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  const sample = (i: number) => (depth === 16 ? px[i * 2]! : px[i]!); // 16-bit: keep the high byte
  for (let i = 0; i < width * height; i++) {
    const s = i * channels;
    let r: number, g: number, b: number, al = 255;
    if (colorType === 3) {
      const k = px[i]!;
      r = palette?.[k * 3] ?? 0; g = palette?.[k * 3 + 1] ?? 0; b = palette?.[k * 3 + 2] ?? 0;
      al = trns && k < trns.length ? trns[k]! : 255;
    } else if (colorType === 0 || colorType === 4) {
      r = g = b = sample(s);
      if (colorType === 4) al = sample(s + 1);
    } else {
      r = sample(s); g = sample(s + 1); b = sample(s + 2);
      if (colorType === 6) al = sample(s + 3);
    }
    rgba.set([r, g, b, al], i * 4);
  }
  return { width, height, rgba };
}

/** Encode RGBA8 as PNG with the given deflate (zlib stream). */
export function encodePngWith(img: RgbaImage, deflate: Deflate): Uint8Array {
  const { width: W, height: H, rgba } = img;
  const raw = new Uint8Array(H * (W * 4 + 1));
  for (let y = 0; y < H; y++) raw.set(rgba.subarray(y * W * 4, (y + 1) * W * 4), y * (W * 4 + 1) + 1);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, W);
  dv.setUint32(4, H);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const parts = [new Uint8Array(SIG), chunk("IHDR", ihdr), chunk("IDAT", deflate(raw)), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Browser-safe PNG encode: zlib stream of uncompressed blocks (bigger file, same pixels). */
export function encodePngStored(img: RgbaImage): Uint8Array {
  return encodePngWith(img, zlibStored);
}

function zlibStored(data: Uint8Array): Uint8Array {
  const blocks = Math.max(1, Math.ceil(data.length / 65535));
  const out = new Uint8Array(2 + data.length + blocks * 5 + 4);
  out[0] = 0x78; out[1] = 0x01;
  let o = 2;
  for (let b = 0; b < blocks; b++) {
    const start = b * 65535, len = Math.min(65535, data.length - start);
    out[o] = b === blocks - 1 ? 1 : 0;
    out[o + 1] = len & 0xff; out[o + 2] = len >> 8;
    out[o + 3] = ~len & 0xff; out[o + 4] = (~len >> 8) & 0xff;
    out.set(data.subarray(start, start + len), o + 5);
    o += 5 + len;
  }
  let a = 1, s2 = 0;
  for (let i = 0; i < data.length; i++) { a = (a + data[i]!) % 65521; s2 = (s2 + a) % 65521; }
  new DataView(out.buffer).setUint32(o, ((s2 << 16) | a) >>> 0);
  return out;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

let CRC_TABLE: Uint32Array | null = null;
export function crc32(data: Uint8Array): number {
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

/** Width/height from a PNG or JPEG header without decoding (for texture-memory accounting). */
export function imageSize(bytes: Uint8Array, mimeType: string): { width: number; height: number } | null {
  if (isPng(bytes)) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return { width: dv.getUint32(16), height: dv.getUint32(20) };
  }
  if (mimeType === "image/jpeg" || (bytes[0] === 0xff && bytes[1] === 0xd8)) {
    let o = 2;
    while (o + 9 < bytes.length) {
      if (bytes[o] !== 0xff) { o++; continue; }
      const m = bytes[o + 1]!;
      const len = (bytes[o + 2]! << 8) | bytes[o + 3]!;
      // SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC)
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
        return { height: (bytes[o + 5]! << 8) | bytes[o + 6]!, width: (bytes[o + 7]! << 8) | bytes[o + 8]! };
      }
      o += 2 + len;
    }
    return null;
  }
  if (mimeType === "image/ktx2" && bytes.length >= 28) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return { width: dv.getUint32(20, true), height: dv.getUint32(24, true) };
  }
  return null;
}

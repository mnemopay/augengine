/**
 * decode — Node image decoding for the impostor atlas and the preview sheet.
 *
 *   PNG   the package's own decoder (png.ts) with node:zlib inflate
 *   JPEG  jpeg-js (pure JavaScript, no dependencies; baseline + progressive)
 *   WebP  @jsquash/webp (libwebp compiled to WebAssembly; async, so WebP
 *         images are decoded up front by `createImageDecoder`)
 *
 * No native modules: `npx @mnemopay/augengine-lod` installs and runs without
 * a compiler or prebuilt binaries (sharp would pull ~30 MB of libvips per
 * platform). Formats are recognised by their magic bytes, not the declared
 * mimeType. KTX2/Basis and anything unrecognised return null, and the
 * impostor falls back to the material's factor colour with a warning.
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { inflateSync } from "node:zlib";
import jpeg from "jpeg-js";
import type { LodImage, LodMaterialSet } from "./material.js";
import { decodePng, isJpeg, isPng, isWebp, type RgbaImage } from "./png.js";

export type ImageDecoder = (img: LodImage) => RgbaImage | null;

/** Decode a JPEG to RGBA8, or null when jpeg-js rejects it. */
export function decodeJpeg(bytes: Uint8Array): RgbaImage | null {
  if (!isJpeg(bytes)) return null;
  try {
    const out = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true, maxResolutionInMP: 128, maxMemoryUsageInMB: 1024 });
    return { width: out.width, height: out.height, rgba: new Uint8Array(out.data.buffer, out.data.byteOffset, out.data.byteLength) };
  } catch {
    return null;
  }
}

/** Synchronous decoder: PNG and JPEG (WebP needs `createImageDecoder`). */
export function decodeImageNode(img: LodImage): RgbaImage | null {
  if (isPng(img.bytes)) return decodePng(img.bytes, (d) => new Uint8Array(inflateSync(d)));
  if (isJpeg(img.bytes)) return decodeJpeg(img.bytes);
  return null;
}

let webpDecode: Promise<(bytes: Uint8Array) => Promise<RgbaImage>> | null = null;

function loadWebp(): Promise<(bytes: Uint8Array) => Promise<RgbaImage>> {
  webpDecode ??= (async () => {
    const mod = await import("@jsquash/webp/decode.js");
    const require = createRequire(import.meta.url);
    const wasm = await WebAssembly.compile(readFileSync(require.resolve("@jsquash/webp/codec/dec/webp_dec.wasm")));
    await (mod.init as unknown as (m: WebAssembly.Module) => Promise<void>)(wasm);
    return async (bytes: Uint8Array) => {
      const copy = bytes.slice(); // the codec takes an ArrayBuffer of its own
      const d = await mod.default(copy.buffer);
      return { width: d.width, height: d.height, rgba: new Uint8Array(d.data.buffer, d.data.byteOffset, d.data.byteLength) };
    };
  })();
  return webpDecode;
}

/** Decode a WebP to RGBA8 (async: the WebAssembly codec loads on first use), or null on failure. */
export async function decodeWebp(bytes: Uint8Array): Promise<RgbaImage | null> {
  if (!isWebp(bytes)) return null;
  try {
    return await (await loadWebp())(bytes);
  } catch {
    return null;
  }
}

/**
 * A synchronous decoder for every image in `set` that PNG, JPEG or WebP
 * decoding can read. WebP images are decoded now (async); PNG and JPEG on
 * first use. Results are cached per image.
 */
export async function createImageDecoder(set?: LodMaterialSet): Promise<ImageDecoder> {
  const cache = new Map<LodImage, RgbaImage | null>();
  for (const img of set?.images ?? []) if (isWebp(img.bytes)) cache.set(img, await decodeWebp(img.bytes));
  return (img) => {
    if (!cache.has(img)) cache.set(img, decodeImageNode(img));
    return cache.get(img)!;
  };
}

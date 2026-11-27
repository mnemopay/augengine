/**
 * KHR_texture_basisu / KTX2 runtime decoder hook.
 *
 * gltfpack with `-tc` converts every embedded texture to KTX2 with
 * Basis Universal supercompression. Browsers do NOT natively decode
 * KTX2 / BasisU — a transcoder must run on the client to turn the
 * compressed payload into a GPU-uploadable byte blob.
 *
 * Khronos publishes a reference WASM transcoder (~500 KB initial, can
 * be lazy-fetched after first frame). AugEngine's bundle stays lean by
 * NOT including the transcoder WASM directly — instead the application
 * (e.g. `examples/smoke-game/`) imports it on first KTX2 hit, then
 * registers it via `setKtx2Transcoder()` here.
 *
 * Target format selection — pick the best available GPU format for the
 * current WebGPU adapter:
 *   - Desktop / discrete GPUs: BC7 (4 bpp, highest quality)
 *   - Mobile / Apple Silicon: ASTC 4×4 (8 bpp, high quality)
 *   - Fallback: RGBA8 uncompressed (32 bpp — last resort)
 *
 * Spec:
 *   https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_texture_basisu
 *   https://github.khronos.org/KTX-Specification/
 */

import type { GltfImage } from "../../scene/loaders/gltf-types.js";
import type { GltfParsed } from "../../scene/loaders/gltf.js";

/** A decoded KTX2 image, ready for `texture.copyExternalImageToTexture` or `device.queue.writeTexture`. */
export interface DecodedKtx2 {
  width: number;
  height: number;
  /** WebGPU GPUTextureFormat string — e.g. "bc7-rgba-unorm", "astc-4x4-unorm", "rgba8unorm". */
  format: string;
  /** Tightly packed mip-level 0 bytes for the chosen format. */
  data: Uint8Array;
  /** Optional mip chain (level 1+). */
  mipChain?: Uint8Array[];
}

/** Transcode signature — the host provides a real WASM-backed implementation. */
export type Ktx2Transcoder = (
  source: Uint8Array,
  preferredFormat: TargetFormat,
) => Promise<DecodedKtx2>;

export type TargetFormat = "bc7" | "astc" | "rgba8";

let registeredTranscoder: Ktx2Transcoder | null = null;
let preferredFormat: TargetFormat = "bc7";

/**
 * Register the host-provided transcoder. Called by the application
 * shell after it has loaded the Basis WASM. Subsequent KTX2 image
 * loads use this transcoder.
 */
export function setKtx2Transcoder(transcoder: Ktx2Transcoder | null): void {
  registeredTranscoder = transcoder;
}

/**
 * Probe the active GPUAdapter and set the preferred target format.
 * Call once after `requestAdapter()`.
 */
export function configureKtx2TargetFormat(adapter: {
  features: ReadonlySet<string> | { has: (s: string) => boolean };
}): TargetFormat {
  const has = (f: string): boolean => {
    const feats = adapter.features;
    if (feats instanceof Set) return feats.has(f);
    return typeof (feats as { has?: unknown }).has === "function"
      ? (feats as { has: (s: string) => boolean }).has(f)
      : false;
  };
  if (has("texture-compression-bc")) preferredFormat = "bc7";
  else if (has("texture-compression-astc")) preferredFormat = "astc";
  else preferredFormat = "rgba8";
  return preferredFormat;
}

/** Cheap pre-flight — does this asset need a KTX2 transcoder? */
export function hasKtx2Textures(parsed: GltfParsed): boolean {
  const ext = parsed.root.extensionsUsed ?? [];
  if (ext.includes("KHR_texture_basisu")) return true;
  for (const image of parsed.root.images ?? []) {
    if (image.mimeType === "image/ktx2") return true;
    if (image.uri && image.uri.toLowerCase().endsWith(".ktx2")) return true;
  }
  return false;
}

/**
 * Synchronous stub transcoder — returns a 1×1 magenta debug texture in
 * RGBA8. Useful for unit tests + as a fallback when the real
 * transcoder hasn't loaded yet. Production must call
 * `setKtx2Transcoder()` with a real Basis-backed implementation.
 */
export const stubKtx2Transcoder: Ktx2Transcoder = async () => {
  return {
    width: 1,
    height: 1,
    format: "rgba8unorm",
    data: new Uint8Array([255, 0, 255, 255]),
  };
};

/**
 * Read the raw KTX2 byte payload for an image entry. Resolves either
 * the `bufferView` reference (typical for embedded GLB) or `uri` (for
 * standalone glTF). Throws on unsupported indirection.
 */
export function extractKtx2Bytes(
  parsed: GltfParsed,
  image: GltfImage,
): Uint8Array | null {
  if (image.bufferView !== undefined) {
    const bv = parsed.root.bufferViews?.[image.bufferView];
    if (!bv || !parsed.bin) return null;
    const start = bv.byteOffset ?? 0;
    return parsed.bin.subarray(start, start + bv.byteLength);
  }
  if (image.uri && image.uri.startsWith("data:")) {
    // data URI inline — caller can fetch separately. We don't decode
    // base64 here because GLB always uses bufferView, not data URIs.
    return null;
  }
  return null;
}

/**
 * Decode every KTX2 image in the asset. Returns a parallel array
 * indexed by `gltfRoot.images[i]`. Entries are `null` for non-KTX2
 * images, or `DecodedKtx2` for transcoded ones.
 *
 * Throws if any image is KTX2 but no transcoder is registered.
 */
export async function decodeGltfKtx2Images(
  parsed: GltfParsed,
): Promise<(DecodedKtx2 | null)[]> {
  const images = parsed.root.images ?? [];
  const out: (DecodedKtx2 | null)[] = new Array(images.length).fill(null);
  let hasKtx2 = false;
  for (const image of images) {
    if (image.mimeType === "image/ktx2") hasKtx2 = true;
    if (image.uri?.toLowerCase().endsWith(".ktx2")) hasKtx2 = true;
  }
  if (!hasKtx2) return out;
  const transcoder = registeredTranscoder ?? stubKtx2Transcoder;
  for (let i = 0; i < images.length; i++) {
    const image = images[i]!;
    const isKtx2 =
      image.mimeType === "image/ktx2" ||
      image.uri?.toLowerCase().endsWith(".ktx2");
    if (!isKtx2) continue;
    const bytes = extractKtx2Bytes(parsed, image);
    if (!bytes) continue;
    out[i] = await transcoder(bytes, preferredFormat);
  }
  return out;
}

/**
 * Estimate the GPU memory footprint of a decoded image set. Used by
 * the runtime to push the `texture_gpu_bytes` figure into the proof
 * sidecar (see `e2e/proof-room-capture.spec.ts`).
 */
export function estimateKtx2GpuBytes(decoded: readonly (DecodedKtx2 | null)[]): number {
  let total = 0;
  for (const img of decoded) {
    if (!img) continue;
    total += img.data.byteLength;
    if (img.mipChain) for (const mip of img.mipChain) total += mip.byteLength;
  }
  return total;
}

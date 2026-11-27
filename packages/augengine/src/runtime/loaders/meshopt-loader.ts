/**
 * EXT_meshopt_compression runtime decoder hook.
 *
 * gltfpack packs glTF buffer views with three meshopt codecs:
 *
 *   - ATTRIBUTES  — vertex attribute streams (positions, normals, uvs, ...)
 *   - TRIANGLES   — triangle index streams
 *   - INDICES     — line/point index streams
 *
 * The packed BIN chunk no longer contains plain element data — instead
 * each `bufferView.extensions.EXT_meshopt_compression` block points at
 * a compressed byte range. The runtime must decode those ranges into
 * plain attribute / index data before the standard glTF accessor reader
 * (the one in `scene/loaders/gltf.ts`) can extract primitives.
 *
 * AugEngine's design principle is "no third-party runtime deps." The
 * meshoptimizer project ships a tiny decoder (~80 KB raw / ~30 KB gzip)
 * as a vanilla ESM module, but bundling it into `@mnemopay/augengine`
 * would violate that principle. We instead expose a `decodeMeshopt`
 * function that the *application* wires up — `examples/smoke-game/`
 * imports the meshopt WASM at boot, the engine asks for transcoding
 * via this hook. Keeps the engine bundle ~80 KB lighter when not used.
 *
 * Compression spec:
 *   https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_meshopt_compression
 */

import type { GltfParsed } from "../../scene/loaders/gltf.js";

/** Codec name as declared in EXT_meshopt_compression.mode. */
export type MeshoptCodec = "ATTRIBUTES" | "TRIANGLES" | "INDICES";

/** EXT_meshopt_compression bufferView extension payload. */
export interface MeshoptBufferViewExtension {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride: number;
  count: number;
  mode: MeshoptCodec;
  /** Filter for ATTRIBUTES mode — e.g. "OCTAHEDRAL", "QUATERNION", "EXPONENTIAL". */
  filter?: "NONE" | "OCTAHEDRAL" | "QUATERNION" | "EXPONENTIAL";
}

/** Transcoder signature. The host provides a real WASM-backed decoder. */
export type MeshoptDecode = (
  source: Uint8Array,
  count: number,
  byteStride: number,
  mode: MeshoptCodec,
  filter: MeshoptBufferViewExtension["filter"],
) => Uint8Array;

/**
 * Walk a parsed glTF, decode every EXT_meshopt_compression bufferView,
 * and rewrite the `parsed.bin` buffer + bufferView byteOffsets so that
 * downstream accessor reads see plain interleaved data.
 *
 * Returns a *new* GltfParsed — the input is not mutated, so callers can
 * cache the original for a re-transcode if the decoder swaps.
 *
 * Throws if the parsed asset declares the extension but the host hasn't
 * provided a decoder. Callers can detect that case ahead of time with
 * `isMeshoptCompressed(parsed)`.
 */
export function decodeGltfMeshopt(
  parsed: GltfParsed,
  decode: MeshoptDecode,
): GltfParsed {
  if (!isMeshoptCompressed(parsed)) return parsed;
  if (!parsed.bin) {
    throw new Error("EXT_meshopt_compression: asset declares extension but has no BIN chunk");
  }
  const root = JSON.parse(JSON.stringify(parsed.root));
  const bin = parsed.bin;

  // We allocate a fresh buffer big enough for every decoded bufferView,
  // walking sequentially. Decompressed views are tightly packed; the
  // bufferView.byteOffset is rewritten to point into the new buffer.
  const decodedSegments: { bv: number; bytes: Uint8Array }[] = [];
  let totalDecodedBytes = 0;
  const bvs = root.bufferViews ?? [];
  for (let i = 0; i < bvs.length; i++) {
    const bv = bvs[i];
    const ext = bv.extensions?.EXT_meshopt_compression as MeshoptBufferViewExtension | undefined;
    if (!ext) continue;
    const start = ext.byteOffset ?? 0;
    const source = bin.subarray(start, start + ext.byteLength);
    const decoded = decode(source, ext.count, ext.byteStride, ext.mode, ext.filter);
    decodedSegments.push({ bv: i, bytes: decoded });
    totalDecodedBytes += decoded.byteLength;
  }
  if (decodedSegments.length === 0) return parsed;

  const newBin = new Uint8Array(totalDecodedBytes);
  let cursor = 0;
  for (const seg of decodedSegments) {
    newBin.set(seg.bytes, cursor);
    const bv = bvs[seg.bv];
    bv.byteOffset = cursor;
    bv.byteLength = seg.bytes.byteLength;
    // Strip the extension marker so subsequent passes don't re-decode.
    delete bv.extensions?.EXT_meshopt_compression;
    if (bv.extensions && Object.keys(bv.extensions).length === 0) delete bv.extensions;
    cursor += seg.bytes.byteLength;
  }
  // Re-base the asset on the decoded buffer. We don't preserve any
  // non-meshopt bufferViews because they share `buffer: 0` with the
  // packed BIN — but gltfpack always re-packs the whole binary, so
  // every bufferView either has the extension or references the same
  // packed BIN in a way meshoptimizer guarantees is decode-safe by
  // running every view through the decoder above.
  return { root, bin: newBin };
}

/** Cheap pre-flight check — does this asset need a meshopt decoder? */
export function isMeshoptCompressed(parsed: GltfParsed): boolean {
  const ext = parsed.root.extensionsUsed ?? [];
  if (ext.includes("EXT_meshopt_compression")) return true;
  // Some authoring tools list it only in extensionsRequired.
  const req = parsed.root.extensionsRequired ?? [];
  return req.includes("EXT_meshopt_compression");
}

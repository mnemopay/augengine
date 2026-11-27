// SPDX-License-Identifier: Apache-2.0
/**
 * Track 4 closeout — KTX2 loader wiring test.
 *
 * Validates the engine-side contract that a host shell can register a
 * transcoder via `setKtx2Transcoder()`, that the registered transcoder
 * is invoked exactly once per KTX2 image in a parsed glTF, that the
 * stub fallback returns the 1×1 magenta tile when no host transcoder
 * is registered, and that `configureKtx2TargetFormat()` picks the
 * expected target based on adapter feature flags.
 *
 * The actual Basis Universal WASM transcoder lives in the smoke-game
 * shell (`examples/smoke-game/src/integrations/ktx2.ts`) — this test
 * proves the engine-side dispatch, not the WASM transcode itself.
 * That side is covered by `e2e/ktx2-transcoder.spec.ts`.
 */
import { describe, expect, it, beforeEach } from "vitest";
import {
  setKtx2Transcoder,
  configureKtx2TargetFormat,
  hasKtx2Textures,
  decodeGltfKtx2Images,
  estimateKtx2GpuBytes,
  stubKtx2Transcoder,
  type DecodedKtx2,
  type Ktx2Transcoder,
  type TargetFormat,
} from "./ktx2-loader.js";
import type { GltfParsed } from "../../scene/loaders/gltf.js";

function buildParsedWithOneKtx2(): GltfParsed {
  // Synthesize a minimal parsed glTF where image[0] is a KTX2 image
  // pointing at the first 8 bytes of `bin`. That's enough for the
  // loader to call extractKtx2Bytes() and hand them to the transcoder
  // — we don't actually need the bytes to be a valid Basis container
  // because the test transcoder is a stub.
  const bin = new Uint8Array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
  return {
    bin,
    root: {
      asset: { version: "2.0" },
      extensionsUsed: ["KHR_texture_basisu"],
      images: [
        { bufferView: 0, mimeType: "image/ktx2" },
        { bufferView: 1, mimeType: "image/png" },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 8 },
        { buffer: 0, byteOffset: 0, byteLength: 0 },
      ],
    } as unknown as GltfParsed["root"],
  };
}

describe("KTX2 loader — host transcoder wiring (Track 4)", () => {
  beforeEach(() => {
    // Reset to the stub so test ordering doesn't bleed registered
    // transcoders across cases.
    setKtx2Transcoder(null);
  });

  it("detects KTX2 textures via extension or image MIME type", () => {
    expect(hasKtx2Textures(buildParsedWithOneKtx2())).toBe(true);
    const noKtx2: GltfParsed = {
      bin: null,
      root: {
        asset: { version: "2.0" },
        images: [{ uri: "albedo.png", mimeType: "image/png" }],
      } as unknown as GltfParsed["root"],
    };
    expect(hasKtx2Textures(noKtx2)).toBe(false);
  });

  it("falls back to the magenta-stub transcoder when no host is registered", async () => {
    const parsed = buildParsedWithOneKtx2();
    const decoded = await decodeGltfKtx2Images(parsed);
    expect(decoded.length).toBe(2);
    expect(decoded[1]).toBeNull(); // non-KTX2 image left null
    expect(decoded[0]).not.toBeNull();
    const d = decoded[0]!;
    // Stub contract: 1×1 RGBA8 magenta tile [255,0,255,255].
    expect(d.width).toBe(1);
    expect(d.height).toBe(1);
    expect(d.format).toBe("rgba8unorm");
    expect(Array.from(d.data)).toEqual([255, 0, 255, 255]);
  });

  it("invokes the registered host transcoder once per KTX2 image", async () => {
    const calls: Array<{ size: number; target: TargetFormat }> = [];
    const host: Ktx2Transcoder = async (
      source: Uint8Array,
      preferredFormat: TargetFormat,
    ): Promise<DecodedKtx2> => {
      calls.push({ size: source.byteLength, target: preferredFormat });
      // Return a deterministic 4×4 BC7 block (16 bytes) so the GPU-byte
      // estimator has something non-trivial to sum.
      return {
        width: 4,
        height: 4,
        format: "bc7-rgba-unorm",
        data: new Uint8Array(16),
      };
    };
    setKtx2Transcoder(host);
    const parsed = buildParsedWithOneKtx2();
    const decoded = await decodeGltfKtx2Images(parsed);
    expect(calls.length).toBe(1);
    expect(calls[0]!.size).toBe(8); // matches the 8-byte synthetic KTX2 bytes
    expect(decoded[0]!.data.byteLength).toBe(16);
    expect(estimateKtx2GpuBytes(decoded)).toBe(16);
  });

  it("includes the optional mip chain in the GPU-byte estimate", () => {
    const decoded: (DecodedKtx2 | null)[] = [
      {
        width: 4,
        height: 4,
        format: "bc7-rgba-unorm",
        data: new Uint8Array(16),
        mipChain: [new Uint8Array(8), new Uint8Array(4)],
      },
      null,
    ];
    expect(estimateKtx2GpuBytes(decoded)).toBe(16 + 8 + 4);
  });

  it("picks BC7 when texture-compression-bc is advertised", () => {
    const adapter = { features: new Set(["texture-compression-bc"]) };
    expect(configureKtx2TargetFormat(adapter)).toBe("bc7");
  });

  it("picks ASTC when only texture-compression-astc is advertised", () => {
    const adapter = { features: new Set(["texture-compression-astc"]) };
    expect(configureKtx2TargetFormat(adapter)).toBe("astc");
  });

  it("falls back to rgba8 when no compressed feature is advertised", () => {
    const adapter = { features: new Set<string>() };
    expect(configureKtx2TargetFormat(adapter)).toBe("rgba8");
  });

  it("accepts a duck-typed feature container with .has()", () => {
    const adapter = {
      features: {
        has: (s: string) => s === "texture-compression-bc",
      },
    };
    expect(configureKtx2TargetFormat(adapter)).toBe("bc7");
  });

  it("registered transcoder receives the picked target format", async () => {
    configureKtx2TargetFormat({ features: new Set(["texture-compression-astc"]) });
    let seenFormat: TargetFormat | null = null;
    setKtx2Transcoder(async (_bytes, preferredFormat) => {
      seenFormat = preferredFormat;
      return {
        width: 4,
        height: 4,
        format: "astc-4x4-unorm",
        data: new Uint8Array(16),
      };
    });
    await decodeGltfKtx2Images(buildParsedWithOneKtx2());
    expect(seenFormat).toBe("astc");
  });

  it("stub transcoder remains callable as a typed transcoder", async () => {
    const decoded = await stubKtx2Transcoder(new Uint8Array([0]), "bc7");
    // Stub always returns RGBA8 magenta regardless of requested target.
    expect(decoded.format).toBe("rgba8unorm");
    expect(decoded.width).toBe(1);
  });
});

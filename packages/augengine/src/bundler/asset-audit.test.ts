import { describe, expect, it } from "vitest";
import { auditLoadedGltfAsset } from "./index.js";
import type { LoadedGlb } from "../scene/loaders/gltf.js";

describe("asset audit", () => {
  it("reports compression gaps and known GPU byte budgets", () => {
    const loaded: LoadedGlb = {
      parsed: {
        bin: null,
        root: {
          asset: { version: "2.0" },
          extensionsUsed: [],
          images: [{ bufferView: 0, mimeType: "image/png" }],
          bufferViews: [{ buffer: 0, byteLength: 2048 }],
        },
      },
      mesh: null,
      primitives: [{
        primitiveIndex: 0,
        meshIndex: 0,
        meshPrimitiveIndex: 0,
        mesh: {
          vertices: new Float32Array(6000 * 8),
          vertexStride: 8,
          indices: new Uint32Array(12000),
          vertexCount: 6000,
          indexCount: 12000,
          morphTargets: [{ positions: new Float32Array(6000 * 3) }],
        },
      }],
      skin: null,
      animations: [],
    };
    const audit = auditLoadedGltfAsset(loaded, 100_000);
    expect(audit.vertices).toBe(6000);
    expect(audit.morphTargets).toBe(1);
    expect(audit.compression.ktx2Textures).toBe(0);
    expect(audit.estimatedGpuBytes.totalKnown).toBeGreaterThan(audit.estimatedGpuBytes.vertexBuffers);
    expect(audit.recommendations.join(" ")).toContain("KTX2");
    expect(audit.recommendations.join(" ")).toContain("meshopt");
  });

  it("detects native meshopt and ktx2 readiness", () => {
    const loaded: LoadedGlb = {
      parsed: {
        bin: null,
        root: {
          asset: { version: "2.0" },
          extensionsUsed: ["EXT_meshopt_compression"],
          images: [{ uri: "albedo.ktx2", mimeType: "image/ktx2" }],
        },
      },
      mesh: null,
      primitives: [],
      skin: null,
      animations: [],
    };
    const audit = auditLoadedGltfAsset(loaded, 10);
    expect(audit.compression.meshopt).toBe(true);
    expect(audit.compression.ktx2Textures).toBe(1);
    expect(audit.recommendations).not.toContain(expect.stringContaining("KTX2"));
  });
});

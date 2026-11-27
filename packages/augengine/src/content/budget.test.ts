import { describe, it, expect } from "vitest";
import { auditAssetBudget, PLAYABLE_ASSET_BUDGET, suggestedLodForDistance } from "./budget.js";
import type { GltfAssetAudit } from "../bundler/index.js";

function audit(over: Partial<GltfAssetAudit> = {}): GltfAssetAudit {
  return {
    bytes: 1_000_000,
    primitives: 4,
    vertices: 2000,
    indices: 4000,
    morphTargets: 12,
    skins: 1,
    animations: 1,
    extensionsUsed: [],
    extensionsRequired: [],
    compression: { ktx2Textures: 1, meshopt: true, draco: false },
    estimatedGpuBytes: {
      vertexBuffers: 1000,
      indexBuffers: 1000,
      morphDeltaBuffers: 0,
      embeddedTexturePayloads: 0,
      totalKnown: 2000,
    },
    recommendations: [],
    ...over,
  };
}

describe("asset budget", () => {
  it("passes a packed character under the playable cap", () => {
    expect(auditAssetBudget(audit())).toEqual([]);
  });

  it("fails oversized meshes and warns on missing KTX2", () => {
    const fail = auditAssetBudget(audit({
      bytes: PLAYABLE_ASSET_BUDGET.maxBytes + 1,
      vertices: PLAYABLE_ASSET_BUDGET.maxVertices + 1,
      compression: { ktx2Textures: 0, meshopt: false, draco: false },
      estimatedGpuBytes: {
        vertexBuffers: 1,
        indexBuffers: 1,
        morphDeltaBuffers: 0,
        embeddedTexturePayloads: 512_000,
        totalKnown: 512_002,
      },
    }));
    expect(fail.some((w) => w.code === "BYTES" && w.level === "fail")).toBe(true);
    expect(fail.some((w) => w.code === "VERTS")).toBe(true);
    expect(fail.some((w) => w.code === "KTX2")).toBe(true);
  });

  it("suggests a coarser LOD at 40 m than at 2 m", () => {
    expect(suggestedLodForDistance(2)).toBeLessThanOrEqual(suggestedLodForDistance(40));
  });
});

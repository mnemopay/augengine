/**
 * Import-time LOD / memory budget. One set of limits for the browser
 * playable path (Intel UHD floor). Warnings are data, not exceptions —
 * Studio shows them next to the Import button.
 */

import type { GltfAssetAudit } from "../bundler/index.js";
import { selectLod, DEFAULT_LOD_CHAIN, type LodLevel } from "../runtime/world/lod.js";

export interface AssetBudget {
  maxBytes: number;
  maxVertices: number;
  maxPrimitives: number;
  maxMorphTargets: number;
  maxEstimatedGpuBytes: number;
}

/** Floor for the Intel UHD 620 playable path at 1280×800. */
export const PLAYABLE_ASSET_BUDGET: AssetBudget = {
  maxBytes: 8_000_000,
  maxVertices: 80_000,
  maxPrimitives: 48,
  maxMorphTargets: 52,
  maxEstimatedGpuBytes: 32_000_000,
};

export type BudgetLevel = "info" | "warn" | "fail";

export interface BudgetWarning {
  level: BudgetLevel;
  code: string;
  message: string;
}

export function auditAssetBudget(
  audit: GltfAssetAudit,
  budget: AssetBudget = PLAYABLE_ASSET_BUDGET,
): BudgetWarning[] {
  const warnings: BudgetWarning[] = [];
  if (audit.bytes > budget.maxBytes) {
    warnings.push({
      level: "fail",
      code: "BYTES",
      message: `Asset is ${(audit.bytes / 1e6).toFixed(1)} MB; playable budget is ${(budget.maxBytes / 1e6).toFixed(0)} MB. Run gltfpack.`,
    });
  } else if (audit.bytes > budget.maxBytes * 0.6) {
    warnings.push({
      level: "warn",
      code: "BYTES",
      message: `Asset is ${(audit.bytes / 1e6).toFixed(1)} MB — close to the ${(budget.maxBytes / 1e6).toFixed(0)} MB playable cap.`,
    });
  }
  if (audit.vertices > budget.maxVertices) {
    warnings.push({
      level: "fail",
      code: "VERTS",
      message: `${audit.vertices} vertices exceeds ${budget.maxVertices}. Emit LOD1/LOD2 via gltfpack -si.`,
    });
  }
  if (audit.primitives > budget.maxPrimitives) {
    warnings.push({
      level: "warn",
      code: "PRIMS",
      message: `${audit.primitives} primitives — merge meshes or instance repeats.`,
    });
  }
  if (audit.morphTargets > budget.maxMorphTargets) {
    warnings.push({
      level: "fail",
      code: "MORPH",
      message: `${audit.morphTargets} morph targets; v1 cap is Cherry 20 (12 visemes + 8 procedural). ARKit 52 is v2.`,
    });
  }
  if (audit.estimatedGpuBytes.totalKnown > budget.maxEstimatedGpuBytes) {
    warnings.push({
      level: "fail",
      code: "GPU",
      message: `Estimated GPU payload ${(audit.estimatedGpuBytes.totalKnown / 1e6).toFixed(1)} MB exceeds ${(budget.maxEstimatedGpuBytes / 1e6).toFixed(0)} MB.`,
    });
  }
  for (const rec of audit.recommendations) {
    warnings.push({ level: "info", code: "REC", message: rec });
  }
  if (audit.compression.ktx2Textures === 0 && audit.estimatedGpuBytes.embeddedTexturePayloads > 256_000) {
    warnings.push({
      level: "warn",
      code: "KTX2",
      message: "Color textures are uncompressed. Convert to KTX2/BasisU on import.",
    });
  }
  return warnings;
}

/** Suggested LOD for a dropped character at `distanceMeters` from camera. */
export function suggestedLodForDistance(distanceMeters: number, screenHeightPx = 800): LodLevel {
  return selectLod(
    [0, 0, 0],
    [0, 0, distanceMeters],
    0.9,
    {
      ...DEFAULT_LOD_CHAIN,
      focalPx: 0.5 * screenHeightPx / Math.tan(0.5),
    },
  );
}

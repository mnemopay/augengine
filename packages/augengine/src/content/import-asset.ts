/**
 * One import button — glTF/GLB + KTX2.
 *
 * Studio and the CLI (`tools/augengine-asset-pipeline/import.mjs`) both
 * call `importAssetBytes`. GPU upload stays with the host renderer.
 */

import { auditLoadedGltfAsset, type GltfAssetAudit } from "../bundler/index.js";
import { loadGlb, type LoadedGlb } from "../scene/loaders/gltf.js";
import { auditAssetBudget, suggestedLodForDistance, type BudgetWarning } from "./budget.js";
import { isMixamoSkeleton, describeMixamoHandoff } from "./mixamo.js";
import type { Skeleton } from "../characters/retarget.js";

export type ImportKind = "glb" | "gltf" | "ktx2" | "unknown";

const KTX2_MAGIC = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];
const GLB_MAGIC = 0x46546c67;

export interface Ktx2Info {
  pixelWidth: number;
  pixelHeight: number;
  vkFormat: number;
  bytes: number;
}

export interface ImportResult {
  kind: ImportKind;
  filename: string;
  bytes: number;
  warnings: BudgetWarning[];
  glb?: {
    loaded: LoadedGlb;
    audit: GltfAssetAudit;
    mixamo: boolean;
    mixamoNote: string;
    suggestedLodAt40m: number;
  };
  ktx2?: Ktx2Info;
}

export function sniffAssetKind(filename: string, bytes: Uint8Array): ImportKind {
  const lower = filename.toLowerCase();
  if (bytes.length >= 12 && KTX2_MAGIC.every((b, i) => bytes[i] === b)) return "ktx2";
  if (bytes.length >= 4) {
    const magic = bytes[0]! | (bytes[1]! << 8) | (bytes[2]! << 16) | (bytes[3]! << 24);
    if (magic === GLB_MAGIC) return "glb";
  }
  if (lower.endsWith(".ktx2")) return "ktx2";
  if (lower.endsWith(".glb")) return "glb";
  if (lower.endsWith(".gltf")) return "gltf";
  return "unknown";
}

export function parseKtx2Header(bytes: Uint8Array): Ktx2Info {
  if (bytes.length < 80) {
    throw new Error("KTX2: header too short");
  }
  if (!KTX2_MAGIC.every((byte, i) => bytes[i] === byte)) {
    throw new Error("KTX2: invalid magic");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(20, true) === 0 || view.getUint32(24, true) === 0) {
    throw new Error("KTX2: texture dimensions must be positive");
  }
  return {
    vkFormat: view.getUint32(12, true),
    pixelWidth: view.getUint32(20, true),
    pixelHeight: view.getUint32(24, true),
    bytes: bytes.byteLength,
  };
}

function skeletonFromLoaded(loaded: LoadedGlb): Skeleton | null {
  const skin = loaded.skin;
  if (!skin || skin.joints.length === 0) return null;
  const nodes = loaded.parsed.root.nodes ?? [];
  const childToParent = new Map<number, number>();
  for (const [parentIndex, node] of nodes.entries()) {
    for (const child of node.children ?? []) childToParent.set(child, parentIndex);
  }
  const jointSet = new Set(skin.joints);
  return {
    joints: skin.joints.map((nodeIndex, i) => {
      const node = nodes[nodeIndex];
      const parentNode = childToParent.get(nodeIndex);
      const parentJoint = parentNode !== undefined && jointSet.has(parentNode)
        ? skin.joints.indexOf(parentNode)
        : -1;
      return {
        name: node?.name ?? `joint_${i}`,
        parent: parentJoint,
        bindPosition: [0, i === 0 ? 0 : 0.1, 0] as const,
        bindRotation: [0, 0, 0, 1] as const,
      };
    }),
  };
}

export function importAssetBytes(bytes: Uint8Array, filename: string): ImportResult {
  const kind = sniffAssetKind(filename, bytes);
  if (kind === "unknown") {
    return {
      kind,
      filename,
      bytes: bytes.byteLength,
      warnings: [{ level: "fail", code: "KIND", message: `Unsupported asset "${filename}". Drop .glb / .gltf / .ktx2.` }],
    };
  }
  if (kind === "ktx2") {
    const info = parseKtx2Header(bytes);
    const warnings: BudgetWarning[] = [];
    if (info.pixelWidth * info.pixelHeight > 4096 * 4096) {
      warnings.push({
        level: "fail",
        code: "KTX2_SIZE",
        message: `KTX2 is ${info.pixelWidth}×${info.pixelHeight}; cap is 4k for the playable path.`,
      });
    } else if (info.pixelWidth * info.pixelHeight > 2048 * 2048) {
      warnings.push({
        level: "warn",
        code: "KTX2_SIZE",
        message: `KTX2 is ${info.pixelWidth}×${info.pixelHeight} — fine on discrete GPUs, heavy on Intel UHD.`,
      });
    }
    return { kind, filename, bytes: bytes.byteLength, warnings, ktx2: info };
  }
  if (kind === "gltf") {
    return {
      kind,
      filename,
      bytes: bytes.byteLength,
      warnings: [{
        level: "warn",
        code: "GLTF_JSON",
        message: "JSON .gltf needs its .bin/.png siblings. Prefer a packed .glb for the one-button import.",
      }],
    };
  }

  const loaded = loadGlb(bytes);
  const audit = auditLoadedGltfAsset(loaded, bytes.byteLength);
  const warnings = auditAssetBudget(audit);
  const skel = skeletonFromLoaded(loaded);
  const mixamo = skel ? isMixamoSkeleton(skel) : false;
  const mixamoNote = skel ? describeMixamoHandoff(skel) : "No skin — static mesh.";
  return {
    kind: "glb",
    filename,
    bytes: bytes.byteLength,
    warnings,
    glb: {
      loaded,
      audit,
      mixamo,
      mixamoNote,
      suggestedLodAt40m: suggestedLodForDistance(40),
    },
  };
}

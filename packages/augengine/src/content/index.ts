/** Content pipeline — glTF/KTX2 import, budget, Mixamo retarget, Cherry visemes. */
export { bundleGltfFiles, type ImportFile } from "./bundle-gltf.js";

export {
  CHERRY_VISEMES,
  PROCEDURAL_CHANNELS,
  CHERRY_MORPH_COUNT,
  emptyCherryWeights,
  packCherryWeights,
  cherryToArkit,
  proceduralIdleFace,
  type CherryViseme,
  type ProceduralChannel,
  type CherryWeightFrame,
} from "./cherry-visemes.js";

export {
  PROOF_HEAD_ENTITY_ID,
  PROOF_HEAD_GENERATOR,
  PROOF_HEAD_SEED,
  PROOF_HEAD_MESH_KEY,
  PROOF_HEAD_VERTEX_STRIDE,
  PROOF_HEAD_LINE_TEXT,
  PROOF_HEAD_LINE_DURATION,
  PROOF_HEAD_LINE_SPEAK_S,
  PROOF_HEAD_LINE_SWALLOW_S,
  PROOF_HEAD_LINE_BROW_S,
  PROOF_HEAD_LINE_BLINK_S,
  PROOF_HEAD_POSITION,
  proofHeadCherrySlots,
  proofHeadLineFrame,
  proofHeadLineWeights,
  deformProofHead,
  buildProofHeadMesh,
  proofHeadEntity,
  type ProofHeadSlots,
  type ProofHeadRegions,
  type ProofHeadMesh,
} from "./proof-head.js";

export {
  PROOF_BUST_HEAD_POSITION,
  proofBustYaw,
  proofHeadBustEntities,
} from "./proof-head-bust.js";

export {
  PLAYABLE_ASSET_BUDGET,
  auditAssetBudget,
  suggestedLodForDistance,
  type AssetBudget,
  type BudgetLevel,
  type BudgetWarning,
} from "./budget.js";

export {
  MIXAMO_YBOT_JOINTS,
  MIXAMO_TO_ENGINE,
  isMixamoSkeleton,
  retargetMixamoToEngine,
  describeMixamoHandoff,
} from "./mixamo.js";

export {
  sniffAssetKind,
  parseKtx2Header,
  importAssetBytes,
  type ImportKind,
  type ImportResult,
  type Ktx2Info,
} from "./import-asset.js";

/**
 * world — Track 7 (2026-05-29) entry point.
 *
 * `WorldRuntime` glues the four sub-modules (tile manager + LOD + culling
 * + instancing) into a single per-frame `tick(camera)` that the renderer
 * can call alongside its existing `render(scene)`. It is OPT-IN: existing
 * proof-room / smoke-game code paths construct neither a TileManager nor
 * a WorldRuntime and continue to render exactly as before.
 *
 * Bundle cost: this barrel + the four modules add ~12 KB raw / ~5 KB gzip
 * to the engine's `dist/`. No new dependencies. The four sub-modules are
 * pure functions / pure data — no GPU resources owned here, which is what
 * keeps the cost down and the tests easy.
 */

export {
  TileManager,
  tileKey,
  buildUniformTileGrid,
  type TileCoord,
  type TileMeta,
  type TileRecord,
  type TileState,
  type TileFetcher,
  type TileLoadResult,
  type TileSpawnEntity,
  type TileManagerOptions,
  type TileUpdateInput,
} from "./tile-manager.js";

export {
  DEFAULT_LOD_CHAIN,
  selectLod,
  selectLodHysteresis,
  lodMeshKey,
  estimateLodVertexCount,
  type LodLevel,
  type LodChainConfig,
} from "./lod.js";

export {
  extractFrustum,
  aabbInFrustum,
  sphereInFrustum,
  probeOcclusion,
  type Frustum,
  type Plane,
  type AABB,
  type OcclusionCapability,
  type OcclusionResult,
} from "./culling.js";

export {
  INSTANCE_BATCH_THRESHOLD,
  buildInstanceBatches,
  type InstanceCandidate,
  type InstanceBatch,
  type InstancingResult,
} from "./instancing.js";

export {
  WorldRuntime,
  type WorldRuntimeOptions,
  type WorldTickInput,
  type WorldTickResult,
} from "./world-runtime.js";

export {
  StreamedLevel,
  attachStreamedLevel,
  streamMeshLibrary,
  STREAM_LEVEL,
  streamPropId,
  streamSlabId,
  type StreamStats,
} from "./streamed-level.js";

export {
  LEVEL,
  LANDMARKS,
  TILE_SIZE,
  buildTileContent,
  tileCenter,
  type PropSpec,
  type PropDetail,
  type TileContent,
  type KitCatalog,
  type KitCatalogEntry,
} from "./level-content.js";

export { TileNav, type NavBlocker, type TileNavOptions } from "./tile-nav.js";

export {
  TERRAIN,
  terrainHeight,
  terrainMask,
  buildTerrainTile,
  sampleHeight,
  sampleNormal,
  groundUnder,
  terrainCollider,
  terrainTileMesh,
  terrainMeshKey,
  terrainLodHeightfield,
  terrainOccluderDrop,
  terrainSlopeBlockers,
  type TerrainTile,
  type TerrainLod,
} from "./terrain.js";

export {
  SoftwareOcclusion,
  type OcclusionBox,
  type OcclusionStats,
  type SoftwareOcclusionOptions,
} from "./occlusion.js";

export {
  TileActors,
  type NpcSpec,
  type NpcState,
  type AmbientSpec,
  type AmbientVoice,
  type AmbientAudioSink,
  type TileActorsOptions,
  type TileActorStats,
} from "./tile-actors.js";

export { levelNpcs, levelAmbients, npcId } from "./level-actors.js";

export {
  StreamExtras,
  type MeshSink,
  type OcclusionItem,
  type StreamExtrasOptions,
  type StreamExtrasStats,
} from "./stream-extras.js";

export {
  PROBES,
  PROBE_FLOATS,
  PROBES_PER_TILE,
  bakeTileProbes,
  bakeTileLayer,
  bakeProbe,
  bakeBoxes,
  bakeGround,
  propBoxes,
  probeTile,
  probeDirections,
  projectAmbientL1,
  evalAmbientL1,
  openGroundReference,
  type ProbeLight,
  type ProbeLighting,
  type ProbeTileData,
  type ProbeSink,
  type ProbeBox,
  type BakeEnv,
} from "./irradiance-probes.js";

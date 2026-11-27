/**
 * world/WorldRuntime — Track 7 facade.
 *
 * Composes the tile manager, frustum culling, LOD selector, and the
 * instance batcher into one `tick(input)` call. Opt-in: callers that
 * never construct a WorldRuntime see no behaviour change.
 *
 * Per tick:
 *   1. TileManager.update — kick async loads, evict to budget.
 *   2. For each candidate entity (provided by caller as an iterator),
 *      compute the per-entity bound, run frustum culling, pick LOD,
 *      and feed the visible set to the instance batcher.
 *   3. Return `{ batches, singletons, residentTiles }` for the renderer
 *      to consume.
 *
 * The renderer integration is intentionally minimal — `world-runtime.ts`
 * does NOT touch GPU state. That keeps Track 7 strictly additive and
 * means Track 6's encoder + timestamp-query work composes without
 * conflict (we share only `runtime/render/encoder.ts` for muxing — no
 * world code lands there).
 */

import type { EntityId, Mat4, Vec3 } from "../../scene/types.js";
import {
  TileManager,
  type TileManagerOptions,
  type TileRecord,
  type TileUpdateInput,
} from "./tile-manager.js";
import {
  extractFrustum,
  sphereInFrustum,
  type Frustum,
} from "./culling.js";
import {
  selectLod,
  lodMeshKey,
  DEFAULT_LOD_CHAIN,
  type LodChainConfig,
  type LodLevel,
} from "./lod.js";
import {
  buildInstanceBatches,
  INSTANCE_BATCH_THRESHOLD,
  type InstanceBatch,
  type InstanceCandidate,
  type InstancingResult,
} from "./instancing.js";

export interface WorldEntityRef {
  entityId: EntityId;
  /** Mesh handle the renderer uses as the LOD0 base. `lodMeshKey()` is applied for LOD>0. */
  baseMeshKey: string;
  materialKey: string;
  /** World transform — typically `scene.worldTransforms().worldMatrices` slice. */
  worldMatrix: Mat4;
  /** Bounding sphere center in world space. */
  centerWorld: Vec3;
  /** Bounding sphere radius (used for both culling + LOD pixel-error proxy). */
  radius: number;
}

export interface WorldRuntimeOptions {
  /** Tile-manager wiring. */
  tile: TileManagerOptions;
  /** LOD chain. Defaults to `DEFAULT_LOD_CHAIN` (32 / 96 / 256 m). */
  lod?: LodChainConfig;
  /** Instance batching threshold. Defaults to 4. */
  instanceThreshold?: number;
}

export interface WorldTickInput extends TileUpdateInput {
  /** Column-major view-projection used for frustum extraction. */
  viewProj: Mat4;
  /** Optional iterator over candidate entities. WorldRuntime owns no scene state. */
  entities?: Iterable<WorldEntityRef>;
}

export interface WorldTickResult {
  /** Resident tiles after this tick — `renderer.uploadMesh` calls usually mirror this. */
  residentTiles: readonly TileRecord[];
  /** Per-entity culling + LOD decisions. */
  visible: readonly WorldVisibleEntity[];
  /** Instance batches the renderer should draw with `drawIndexed(instanceCount=N)`. */
  batches: readonly InstanceBatch[];
  /** Entities below the batching threshold — draw individually. */
  singletons: readonly InstanceCandidate[];
  /** Counts for diagnostics + perf overlay. */
  stats: WorldTickStats;
}

export interface WorldVisibleEntity {
  ref: WorldEntityRef;
  lod: LodLevel;
  /** Resolved mesh key with LOD suffix applied. */
  meshKey: string;
}

export interface WorldTickStats {
  candidates: number;
  culled: number;
  visible: number;
  batches: number;
  singletons: number;
  residentTiles: number;
  residentBytes: number;
}

export class WorldRuntime {
  readonly tiles: TileManager;
  private lod: LodChainConfig;
  private instanceThreshold: number;
  /** Cached frustum from the latest tick — exposed for tests / debug HUD. */
  lastFrustum: Frustum | null = null;

  constructor(opts: WorldRuntimeOptions) {
    this.tiles = new TileManager(opts.tile);
    this.lod = opts.lod ?? DEFAULT_LOD_CHAIN;
    this.instanceThreshold = opts.instanceThreshold ?? INSTANCE_BATCH_THRESHOLD;
  }

  /**
   * Refresh the LOD config (e.g. on camera resize → focal length changes).
   * Cheap, no-allocation.
   */
  configureLod(cfg: LodChainConfig): void {
    this.lod = cfg;
  }

  tick(input: WorldTickInput): WorldTickResult {
    const residentTiles = this.tiles.update(input);
    const frustum = extractFrustum(input.viewProj);
    this.lastFrustum = frustum;
    const visible: WorldVisibleEntity[] = [];
    const candidates: InstanceCandidate[] = [];
    let total = 0;
    let culled = 0;
    if (input.entities) {
      for (const ref of input.entities) {
        total += 1;
        if (!sphereInFrustum(frustum, ref.centerWorld, ref.radius)) {
          culled += 1;
          continue;
        }
        const lod = selectLod(input.cameraPos, ref.centerWorld, ref.radius, this.lod);
        const meshKey = lodMeshKey(ref.baseMeshKey, lod);
        visible.push({ ref, lod, meshKey });
        candidates.push({
          entityId: ref.entityId,
          meshKey,
          materialKey: ref.materialKey,
          worldMatrix: ref.worldMatrix,
        });
      }
    }
    const inst: InstancingResult = buildInstanceBatches(candidates, this.instanceThreshold);
    return {
      residentTiles,
      visible,
      batches: inst.batches,
      singletons: inst.singletons,
      stats: {
        candidates: total,
        culled,
        visible: visible.length,
        batches: inst.batches.length,
        singletons: inst.singletons.length,
        residentTiles: residentTiles.length,
        residentBytes: this.tiles.residentBytesNow,
      },
    };
  }

  destroy(): void {
    this.tiles.destroy();
    this.lastFrustum = null;
  }
}

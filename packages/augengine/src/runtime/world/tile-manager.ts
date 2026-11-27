/**
 * world/tile-manager — Map3D-scale streaming over an N×N tile grid.
 *
 * Track 7 (2026-05-29). Subdivides a world rectangle into uniform tiles,
 * loads tiles asynchronously near the camera, evicts the least-recently-
 * used residents when the configured memory budget is crossed.
 *
 * Per-tile state machine:
 *
 *     unloaded ──load(prio)──▶ loading ──onReady──▶ resident
 *        ▲                                              │
 *        └────── onEvictDone ◀── evicting ◀── evict() ──┘
 *
 * Streaming priority = (1 − normalized_distance) × screen_coverage_estimate.
 * Tiles closest to the projected camera-forward ray load first so the
 * player sees content arrive ahead of where they're heading, not behind.
 *
 * The manager is environment-agnostic: it dispatches loads through a
 * `TileFetcher` you inject. In production this is a worker that pulls a
 * `.packed.glb` via fetch + meshopt decode; in tests it's a sync stub that
 * resolves immediately with byte-accounting metadata so we can assert
 * residency and eviction order without touching the GPU.
 */

import type { EntityId, Vec3 } from "../../scene/types.js";

export type TileState = "unloaded" | "loading" | "resident" | "evicting";

export interface TileCoord {
  /** Tile X index (0..gridSize-1). */
  x: number;
  /** Tile Z index (0..gridSize-1). */
  z: number;
}

export interface TileMeta {
  coord: TileCoord;
  /** World-space center (assumes ground plane at y=0). */
  centerWorld: Vec3;
  /** World-space half extents of the tile bounding box. */
  halfExtents: Vec3;
  /** Cached payload byte size (manifest-provided; used by the memory accountant). */
  payloadBytes: number;
  /** Identifier the fetcher resolves to bytes. Typically a relative URL. */
  payloadKey: string;
}

export interface TileRecord extends TileMeta {
  state: TileState;
  /** Monotonic counter set each frame the tile is referenced. Drives LRU. */
  lastUsedFrame: number;
  /** Entities spawned into the scene when the tile became resident. */
  spawnedEntities: EntityId[];
  /** Last fetch error, if any. */
  loadError: string | null;
}

export interface TileFetcher {
  /**
   * Resolve when the payload is ready. Implementations are free to use a
   * Worker (transferable buffers) but `TileManager` doesn't require it.
   * Throwing rejects the load and bumps the failure counter.
   */
  fetch(meta: TileMeta, signal: AbortSignal): Promise<TileLoadResult>;
}

export interface TileLoadResult {
  /** Entities (already shaped for `Scene.add`) to register into the scene. */
  entities: TileSpawnEntity[];
  /** Measured residency bytes (may differ from `payloadBytes` after decode). */
  residentBytes: number;
}

/** Minimal shape — caller plugs into their own `Scene.add` adaptor. */
export interface TileSpawnEntity {
  id: EntityId;
  /** Opaque payload the caller's spawn callback re-hydrates. */
  payload: unknown;
}

export interface TileManagerOptions {
  /** World-space origin (x, y, z) of the bottom-left corner of tile (0,0). */
  worldOrigin: Vec3;
  /** Total world extent in the X and Z directions. Defaults to 1024×1024. */
  worldExtent?: { x: number; z: number };
  /** Tiles per side. Default 16 → 16×16 = 256 tiles. */
  gridSize?: number;
  /**
   * Memory budget in bytes for resident tile payloads. Default = 60% of the
   * unloaded-all baseline (the manifest's summed `payloadBytes`).
   * When budget would be crossed, LRU eviction runs until residency fits.
   */
  memoryBudgetBytes: number;
  /** Max tiles loading concurrently. Default 4. */
  loadConcurrency?: number;
  /**
   * Distance (in tile widths) within which a tile is eligible to be loaded.
   * Default 4 (so a 64m tile loads when within 256m of the camera).
   */
  loadRadiusTiles?: number;
  /**
   * Extra reach (in tile widths) for tiles ahead of the camera, scaled by how
   * directly ahead they are (0 behind/sideways, full bonus dead ahead). Lets a
   * fast-moving camera start loads earlier without widening the radius behind
   * it. Default 0.
   */
  forwardRadiusBonusTiles?: number;
  fetcher: TileFetcher;
  /** Hook fired when a tile transitions to resident. Caller adds entities to its Scene. */
  onTileResident?: (tile: TileRecord, payload: TileLoadResult) => void;
  /** Hook fired when a tile transitions to unloaded. Caller removes entities. */
  onTileEvicted?: (tile: TileRecord) => void;
}

export interface TileUpdateInput {
  /** World-space camera position (used for distance + priority). */
  cameraPos: Vec3;
  /** World-space camera forward (normalised; biases tiles ahead of the player). */
  cameraForward: Vec3;
  /** Monotonic frame counter — drives LRU. */
  frameIndex: number;
}

/**
 * The actual coordinator. `update()` is the per-frame entry point; it picks
 * tiles to load, evicts to fit budget, and returns the resident set for the
 * caller (renderer) to consume.
 */
export class TileManager {
  readonly opts: Required<Omit<TileManagerOptions, "fetcher" | "onTileResident" | "onTileEvicted">> & {
    fetcher: TileFetcher;
    onTileResident: TileManagerOptions["onTileResident"];
    onTileEvicted: TileManagerOptions["onTileEvicted"];
  };
  /** key = `${x}|${z}`. */
  private tiles = new Map<string, TileRecord>();
  private inflight = new Map<string, AbortController>();
  /** Total bytes counted against the budget across `resident` tiles. */
  private residentBytes = 0;

  constructor(opts: TileManagerOptions) {
    this.opts = {
      worldOrigin: opts.worldOrigin,
      worldExtent: opts.worldExtent ?? { x: 1024, z: 1024 },
      gridSize: opts.gridSize ?? 16,
      memoryBudgetBytes: opts.memoryBudgetBytes,
      loadConcurrency: opts.loadConcurrency ?? 4,
      loadRadiusTiles: opts.loadRadiusTiles ?? 4,
      forwardRadiusBonusTiles: opts.forwardRadiusBonusTiles ?? 0,
      fetcher: opts.fetcher,
      onTileResident: opts.onTileResident,
      onTileEvicted: opts.onTileEvicted,
    };
  }

  /** Register a tile's metadata. Must be called before that tile can load. */
  registerTile(meta: TileMeta): void {
    const key = tileKey(meta.coord);
    if (this.tiles.has(key)) return;
    this.tiles.set(key, {
      ...meta,
      state: "unloaded",
      lastUsedFrame: -1,
      spawnedEntities: [],
      loadError: null,
    });
  }

  /** Register a batch in one shot. */
  registerTiles(metas: readonly TileMeta[]): void {
    for (const m of metas) this.registerTile(m);
  }

  /** Snapshot of currently resident tiles (read-only — do not mutate). */
  get resident(): readonly TileRecord[] {
    const out: TileRecord[] = [];
    for (const t of this.tiles.values()) if (t.state === "resident") out.push(t);
    return out;
  }

  /** Total payload bytes resident. */
  get residentBytesNow(): number {
    return this.residentBytes;
  }

  /** Sum of all tiles' payloadBytes — the "unloaded-all baseline". */
  get baselineBytes(): number {
    let sum = 0;
    for (const t of this.tiles.values()) sum += t.payloadBytes;
    return sum;
  }

  /** Per-frame update. Returns the resident tile set after the tick. */
  update(input: TileUpdateInput): readonly TileRecord[] {
    const tileSizeX = this.opts.worldExtent.x / this.opts.gridSize;
    const tileSizeZ = this.opts.worldExtent.z / this.opts.gridSize;
    const tileSize = Math.max(tileSizeX, tileSizeZ);
    const loadRadius = this.opts.loadRadiusTiles * tileSize;
    const forwardBonus = this.opts.forwardRadiusBonusTiles * tileSize;

    // Pass 1: scan eligible tiles, score by priority, refresh LRU stamps.
    type Candidate = { tile: TileRecord; priority: number };
    const candidates: Candidate[] = [];
    for (const tile of this.tiles.values()) {
      const dx = tile.centerWorld[0] - input.cameraPos[0];
      const dz = tile.centerWorld[2] - input.cameraPos[2];
      const dist = Math.hypot(dx, dz);
      const dirX = dist > 1e-6 ? dx / dist : 0;
      const dirZ = dist > 1e-6 ? dz / dist : 0;
      const fwdDot = dirX * input.cameraForward[0] + dirZ * input.cameraForward[2];
      const reach = loadRadius + forwardBonus * Math.max(0, fwdDot);
      if (dist > reach) continue;
      if (tile.state === "resident") {
        tile.lastUsedFrame = input.frameIndex;
        continue;
      }
      if (tile.state !== "unloaded") continue;
      // Priority: (1 - normalized_distance) * screen_coverage_estimate
      // screen_coverage_estimate ≈ in-front bias: fwd-aligned tiles score higher.
      const norm = Math.min(1, dist / reach);
      const coverage = 0.5 + 0.5 * Math.max(0, fwdDot); // 0.5 (behind) .. 1 (ahead)
      const priority = (1 - norm) * coverage;
      candidates.push({ tile, priority });
    }

    // Pass 2: kick loads, capped by concurrency.
    candidates.sort((a, b) => b.priority - a.priority);
    let slots = Math.max(0, this.opts.loadConcurrency - this.inflight.size);
    for (const c of candidates) {
      if (slots <= 0) break;
      this.startLoad(c.tile, input.frameIndex);
      slots -= 1;
    }

    // Pass 3: enforce memory budget — LRU eviction.
    if (this.residentBytes > this.opts.memoryBudgetBytes) {
      const evictable = Array.from(this.tiles.values())
        .filter((t) => t.state === "resident")
        .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame);
      for (const t of evictable) {
        if (this.residentBytes <= this.opts.memoryBudgetBytes) break;
        // Never evict a tile that was used this very frame — would thrash.
        if (t.lastUsedFrame === input.frameIndex) continue;
        this.evict(t);
      }
    }

    return this.resident;
  }

  /** Programmatically force a load — used by tests / preloading. */
  forceLoad(coord: TileCoord, frameIndex = 0): void {
    const tile = this.tiles.get(tileKey(coord));
    if (!tile || tile.state !== "unloaded") return;
    this.startLoad(tile, frameIndex);
  }

  /** Programmatic eviction — used by tests. */
  forceEvict(coord: TileCoord): void {
    const tile = this.tiles.get(tileKey(coord));
    if (!tile || tile.state !== "resident") return;
    this.evict(tile);
  }

  /** Cancel everything in flight + clear residency. Caller still owns scene cleanup. */
  destroy(): void {
    for (const ctrl of this.inflight.values()) ctrl.abort();
    this.inflight.clear();
    for (const t of this.tiles.values()) {
      if (t.state === "resident" && this.opts.onTileEvicted) this.opts.onTileEvicted(t);
    }
    this.tiles.clear();
    this.residentBytes = 0;
  }

  private startLoad(tile: TileRecord, frameIndex: number): void {
    tile.state = "loading";
    tile.lastUsedFrame = frameIndex;
    const ctrl = new AbortController();
    const key = tileKey(tile.coord);
    this.inflight.set(key, ctrl);
    this.opts.fetcher.fetch(tile, ctrl.signal).then(
      (result) => {
        this.inflight.delete(key);
        if (ctrl.signal.aborted) return;
        tile.state = "resident";
        tile.spawnedEntities = result.entities.map((e) => e.id);
        this.residentBytes += result.residentBytes;
        if (this.opts.onTileResident) this.opts.onTileResident(tile, result);
      },
      (err: unknown) => {
        this.inflight.delete(key);
        if (ctrl.signal.aborted) return;
        tile.state = "unloaded";
        tile.loadError = err instanceof Error ? err.message : String(err);
      },
    );
  }

  private evict(tile: TileRecord): void {
    tile.state = "evicting";
    if (this.opts.onTileEvicted) this.opts.onTileEvicted(tile);
    this.residentBytes = Math.max(0, this.residentBytes - tile.payloadBytes);
    tile.state = "unloaded";
    tile.spawnedEntities = [];
  }
}

export function tileKey(coord: TileCoord): string {
  return `${coord.x}|${coord.z}`;
}

/**
 * Generate the tile-metadata grid for a uniform N×N world. Centers are
 * computed at ground-plane height (y = worldOrigin.y). `bytesPerTile` is
 * a uniform stub used when the caller doesn't have per-tile payload sizes
 * up front; in practice this comes from the manifest.
 */
export function buildUniformTileGrid(
  origin: Vec3,
  extent: { x: number; z: number },
  gridSize: number,
  bytesPerTile: number,
  keyForCoord: (coord: TileCoord) => string,
): TileMeta[] {
  const tileX = extent.x / gridSize;
  const tileZ = extent.z / gridSize;
  const halfX = tileX / 2;
  const halfZ = tileZ / 2;
  const out: TileMeta[] = [];
  for (let z = 0; z < gridSize; z++) {
    for (let x = 0; x < gridSize; x++) {
      out.push({
        coord: { x, z },
        centerWorld: [origin[0] + x * tileX + halfX, origin[1], origin[2] + z * tileZ + halfZ],
        halfExtents: [halfX, 32, halfZ], // 64 m vertical envelope default
        payloadBytes: bytesPerTile,
        payloadKey: keyForCoord({ x, z }),
      });
    }
  }
  return out;
}

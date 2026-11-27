/**
 * The Loading Range — a 1 km level that streams in 64 m tiles.
 *
 * Gate 2 ("one level, not a room"). Grok's first slice (d5c4b05) put one
 * column and a visual slab per tile on a 256 m yard with one permanent
 * floor collider. This version makes the tile the unit of everything:
 *
 *  - Content: each tile carries its own floor, buildings, pillars, domes and
 *    landmarks (level-content.ts), deterministic per coordinate.
 *  - Collision: the tile floor and every prop collider arrive and leave with
 *    the tile. There is no permanent floor; `guardMove` stops the player
 *    stepping onto a tile that is not resident yet.
 *  - Navigation: a per-tile occupancy grid is built on load and dropped on
 *    evict (tile-nav.ts); paths only cross resident ground.
 *  - LOD: pillars and domes switch between three tessellations and a box
 *    impostor with hysteresis; building detail parts drop out with distance.
 *  - Culling: anything outside the view frustum is hidden, so the renderer
 *    (which draws every visible mesh) issues fewer draws.
 *  - Memory: the tile manager evicts least-recently-used tiles once resident
 *    bytes pass the budget, so entity count stays bounded on any walk.
 *  - Time-slicing: a tile becoming resident or evicted only queues work. Each
 *    tick runs queued steps (floor + nav, one prop, a batch of removals) until
 *    a per-frame time budget is spent, nearest/ahead tiles first. The floor
 *    under or right next to the camera is never deferred, whatever the budget.
 *    Tiles ahead load from further away (forwardRadiusBonusTiles) so a fast
 *    camera does not outrun the queue.
 *
 * The renderer is unchanged: it draws whatever the scene says is visible,
 * and resolves `procedural` geometry to meshes the host uploads once from
 * `streamMeshLibrary()`.
 */
import { Scene } from "../../scene/scene.js";
import { m4Create, m4LookAt, m4Mul, m4Perspective } from "../../scene/math.js";
import { Tag, type Entity, type EntityId, type MeshComponent, type Vec3 } from "../../scene/types.js";
import { cubeMesh, cylinderMesh, sphereMesh, type MeshData } from "../live/webgpu/primitives.js";
import { sphereInFrustum } from "./culling.js";
import { selectLodHysteresis, type LodLevel } from "./lod.js";
import { buildTileContent, LANDMARKS, LEVEL, TILE_SIZE, type KitCatalog, type PropSpec, type TileContent } from "./level-content.js";
import { StreamExtras, type OcclusionItem, type StreamExtrasOptions, type StreamExtrasStats } from "./stream-extras.js";
import { terrainTileBytes } from "./terrain.js";
import { TileNav } from "./tile-nav.js";
import type { TileLoadResult, TileMeta } from "./tile-manager.js";
import { WorldRuntime, type WorldTickResult } from "./world-runtime.js";

/** Rough resident cost per spawned entity (components + GPU uniforms). */
const BYTES_PER_ENTITY = 512;
const NAV_CELL = 1;

export const STREAM_LEVEL = {
  origin: LEVEL.origin,
  extent: LEVEL.extent,
  gridSize: LEVEL.gridSize,
  tileSize: TILE_SIZE,
  /** Tiles within ~112 m of the camera are wanted resident. */
  loadRadiusTiles: 1.75,
  /** Tiles dead ahead are wanted from one tile further out (~176 m). */
  forwardRadiusBonusTiles: 1,
  /** Spawn/despawn work allowed per tick, in milliseconds. */
  frameBudgetMs: 3,
  /** A tile's floor is built immediately (budget or not) when the camera is this close to it. */
  mandatoryFloorDistance: 24,
  /** Entities removed per despawn step. */
  despawnBatch: 24,
  /** Eviction starts once resident bytes pass this many average tiles. */
  budgetTiles: 18,
  spawn: [16, 1.7, LEVEL.roadZ] as Vec3,
  /** Prop LOD steps (metres): finer than 28 m is LOD0, past 110 m is the impostor. */
  lodThresholds: [28, 64, 110] as const,
  lodBand: 0.12,
  roadZ: LEVEL.roadZ,
  /**
   * Occlusion runs every frame while it pays (hides at least `minHidden` draws
   * and `minHiddenFraction` of the visible ones); otherwise it only probes
   * every `probeEvery` frames. Skipping hides nothing, so it can never hide
   * something visible. On the road the pass hid ~2 of ~60 draws for ~1.7 ms
   * of CPU a frame (2026-11-23 captures), so there it mostly probes.
   */
  occlusion: { minHidden: 4, minHiddenFraction: 0.05, probeEvery: 8 },
} as const;

export function streamSlabId(x: number, z: number): EntityId {
  return `stream.${x}.${z}.slab` as EntityId;
}

export function streamPropId(x: number, z: number, key: string): EntityId {
  return `stream.${x}.${z}.${key}` as EntityId;
}

const LOD_GENERATOR: Record<"pillar" | "dome", readonly [string, string, string]> = {
  pillar: ["stream-pillar", "stream-pillar", "stream-pillar"],
  dome: ["stream-dome", "stream-dome", "stream-dome"],
};

/**
 * Meshes the level draws through `procedural` geometry. Unit-sized so the
 * entity scale is the visual size. Upload once:
 * `for (const m of streamMeshLibrary()) renderer.uploadMesh(m.key, m.data, 8)`.
 */
export function streamMeshLibrary(): { key: string; data: MeshData }[] {
  return [
    { key: "proc:stream-pillar:0", data: cylinderMesh(0.5, 1, 32) },
    { key: "proc:stream-pillar:1", data: cylinderMesh(0.5, 1, 12) },
    { key: "proc:stream-pillar:2", data: cylinderMesh(0.5, 1, 6) },
    { key: "proc:stream-dome:0", data: sphereMesh(0.5, 32, 20) },
    { key: "proc:stream-dome:1", data: sphereMesh(0.5, 12, 8) },
    { key: "proc:stream-dome:2", data: sphereMesh(0.5, 6, 4) },
    { key: "proc:stream-impostor:0", data: cubeMesh() },
  ];
}

const TRIS = new Map<string, number>([["primitive:box", 12]]);
for (const m of streamMeshLibrary()) TRIS.set(m.key, m.data.indexCount / 3);

interface PropRuntime {
  tile: string;
  mainId: EntityId;
  shape: PropSpec["shape"];
  /** shape "kit": catalog id (meshes `proc:kit-<id>:<lod>`). */
  kit: string | null;
  center: Vec3;
  radius: number;
  lod: LodLevel | null;
  details: { id: EntityId; maxLod: 0 | 1; radius: number; center: Vec3 }[];
  /** Solid collider core usable as an occluder (PropSpec.occluder). */
  occluder: boolean;
  /** Occlusion inputs, built once at spawn (props never move): what it hides, and its occluder core if solid. */
  occItem: OcclusionItem;
  occBox: OcclusionItem | null;
}

interface SlabRuntime {
  id: EntityId;
  extra: EntityId[];
  center: Vec3;
}

interface SpawnJob {
  key: string;
  x: number;
  z: number;
  content: TileContent;
  floorDone: boolean;
  nextProp: number;
}

export interface StreamedLevelOptions {
  /** Per-tick spawn/despawn budget in ms. Default STREAM_LEVEL.frameBudgetMs. */
  frameBudgetMs?: number;
  /** Clock used for the budget. Default performance.now (Date.now fallback). */
  now?: () => number;
  /**
   * Art kit to build the level from instead of primitives (buildings, trees,
   * street furniture, the water tower). The host uploads each LOD mesh as
   * `proc:kit-<id>:<lod>`, e.g. from @mnemopay/augengine-lod `kitMeshLibrary()`.
   */
  kit?: KitCatalog;
  /** Triangle count per kit mesh key, for `stats.triangles`. Missing keys count 0. */
  kitTriangles?: ReadonlyMap<string, number>;
  /**
   * Terrain, occlusion culling and tile actors (stream-extras.ts). Default
   * on, with no mesh or audio sink; `false` keeps the flat slab floors.
   */
  extras?: StreamExtrasOptions | false;
}

export interface StreamStats {
  residentTiles: number;
  residentBytes: number;
  /** Entities this level currently owns in the scene. */
  streamEntities: number;
  peakStreamEntities: number;
  /** Meshes left visible after culling + LOD — one renderer draw each. */
  draws: number;
  triangles: number;
  culled: number;
  lodCounts: [number, number, number, number];
  navTiles: number;
  /** Resident tiles whose content is still being spawned. */
  pendingSpawnTiles: number;
  /** Evicted entities still waiting to be removed from the scene. */
  pendingDespawnEntities: number;
  /** Time spent on spawn/despawn work in the last tick (ms). */
  lastWorkMs: number;
  /** Steps run in the last tick (forced floors included). */
  lastWorkSteps: number;
  /** Draws hidden by occlusion culling this tick (already excluded from `draws`). */
  occluded: number;
  /** Whether the occlusion pass ran this tick (it is skipped while it does not pay; see STREAM_LEVEL.occlusion). */
  occlusionRan: boolean;
  /** Terrain / occlusion / NPC / audio counters, when extras are on. */
  extras: StreamExtrasStats | null;
  /** Where the last tick's time went (ms), for finding frame spikes. */
  phases: StreamPhases;
}

/** Per-tick time breakdown of StreamedLevel.tick (ms; counts are steps run). */
export interface StreamPhases {
  total: number;
  /** Tile manager: wanted set, fetch starts, evictions. */
  runtime: number;
  floorMs: number;
  floors: number;
  propMs: number;
  props: number;
  despawnMs: number;
  despawns: number;
  /** Tiles finishing: NPCs and emitters brought in (onTileComplete). */
  completeMs: number;
  /** Frustum culling, LOD selection and draw counting (occlusion excluded). */
  visibility: number;
  occlusion: number;
  /** NPC behaviour and ambient audio. */
  actors: number;
}

function emptyPhases(): StreamPhases {
  return {
    total: 0, runtime: 0, floorMs: 0, floors: 0, propMs: 0, props: 0,
    despawnMs: 0, despawns: 0, completeMs: 0, visibility: 0, occlusion: 0, actors: 0,
  };
}

export class StreamedLevel {
  readonly runtime: WorldRuntime;
  readonly nav: TileNav;
  readonly scene: Scene;
  readonly memoryBudgetBytes: number;
  private readonly contents = new Map<string, TileContent>();
  private readonly props = new Map<EntityId, PropRuntime>();
  private readonly slabs = new Map<EntityId, SlabRuntime>();
  private readonly owned = new Set<EntityId>();
  private readonly view = m4Create();
  private readonly proj = m4Create();
  private readonly viewProj = m4Create();
  private tail: Promise<void> = Promise.resolve();
  private peak = 0;
  private readonly spawnJobs = new Map<string, SpawnJob>();
  /** Evicted tile key → entity ids still in the scene (already hidden). */
  private readonly despawnJobs = new Map<string, EntityId[]>();
  private readonly frameBudgetMs: number;
  private readonly now: () => number;
  private readonly kitTriangles: ReadonlyMap<string, number>;
  private lastWorkMs = 0;
  private lastWorkSteps = 0;
  private lastTickAt: number | null = null;
  /** Terrain floors, occlusion and tile actors; null when disabled. */
  readonly extras: StreamExtras | null;
  last: WorldTickResult | null = null;
  stats: StreamStats = {
    residentTiles: 0, residentBytes: 0, streamEntities: 0, peakStreamEntities: 0,
    draws: 0, triangles: 0, culled: 0, lodCounts: [0, 0, 0, 0], navTiles: 0,
    pendingSpawnTiles: 0, pendingDespawnEntities: 0, lastWorkMs: 0, lastWorkSteps: 0,
    occluded: 0, occlusionRan: false, extras: null, phases: emptyPhases(),
  };
  /** Phase timings being collected for the current tick. */
  private phases: StreamPhases = emptyPhases();
  /** Reused each frame for the occlusion pass's inputs (avoids per-frame garbage). */
  private readonly occItems: OcclusionItem[] = [];
  /** Occlusion gating: did the last pass pay, and frames since the last pass. */
  private occPays = true;
  private occSince = 0;

  constructor(scene: Scene, opts: StreamedLevelOptions = {}) {
    this.scene = scene;
    this.frameBudgetMs = opts.frameBudgetMs ?? STREAM_LEVEL.frameBudgetMs;
    this.now = opts.now ?? (typeof performance !== "undefined" ? () => performance.now() : () => Date.now());
    this.kitTriangles = opts.kitTriangles ?? new Map();
    this.nav = new TileNav({
      originX: LEVEL.origin[0], originZ: LEVEL.origin[2], tileSize: TILE_SIZE, cell: NAV_CELL, agentRadius: 0.4,
    });
    this.extras = opts.extras === false
      ? null
      : new StreamExtras(scene, this.nav, { ...(opts.extras ?? {}), contentAt: (x, z) => this.contents.get(`${x}|${z}`) });
    const metas: TileMeta[] = [];
    let total = 0;
    for (let z = 0; z < LEVEL.gridSize; z++) {
      for (let x = 0; x < LEVEL.gridSize; x++) {
        const content = buildTileContent(x, z, opts.kit);
        this.contents.set(`${x}|${z}`, content);
        const bytes = tileBytes(content) + (this.extras ? terrainTileBytes() : 0);
        total += bytes;
        metas.push({
          coord: { x, z },
          centerWorld: content.center,
          halfExtents: [TILE_SIZE / 2, 32, TILE_SIZE / 2],
          payloadBytes: bytes,
          payloadKey: `stream/${x}/${z}`,
        });
      }
    }
    this.memoryBudgetBytes = Math.round((total / metas.length) * STREAM_LEVEL.budgetTiles);
    this.runtime = new WorldRuntime({
      tile: {
        worldOrigin: LEVEL.origin,
        worldExtent: LEVEL.extent,
        gridSize: LEVEL.gridSize,
        memoryBudgetBytes: this.memoryBudgetBytes,
        loadConcurrency: 4,
        loadRadiusTiles: STREAM_LEVEL.loadRadiusTiles,
        forwardRadiusBonusTiles: STREAM_LEVEL.forwardRadiusBonusTiles,
        fetcher: {
          fetch: (meta) => {
            // The tile's terrain work happens here, spread over idle tasks,
            // so the tile is resident only once its floor is cheap to place.
            const prep = this.extras ? this.extras.prepare(meta.coord.x, meta.coord.z, NAV_CELL) : Promise.resolve();
            const work = prep.then(() => this.payload(meta));
            this.tail = this.tail.then(() => work.then(() => undefined, () => undefined));
            return work;
          },
        },
        onTileResident: (tile) => this.queueSpawn(tile.coord.x, tile.coord.z),
        onTileEvicted: (tile) => this.queueDespawn(tile.coord.x, tile.coord.z),
      },
    });
    this.runtime.tiles.registerTiles(metas);
  }

  /** Landmark names by tile, for HUDs and tests. */
  static landmarks(): Readonly<Record<string, string>> {
    return LANDMARKS;
  }

  tick(cameraPos: Vec3, cameraForward: Vec3, frameIndex: number): WorldTickResult {
    const target: Vec3 = [cameraPos[0] + cameraForward[0], cameraPos[1] + cameraForward[1], cameraPos[2] + cameraForward[2]];
    m4LookAt(this.view, cameraPos, target, [0, 1, 0]);
    m4Perspective(this.proj, (80 * Math.PI) / 180, 16 / 10, 0.1, 400);
    m4Mul(this.viewProj, this.proj, this.view);
    this.phases = emptyPhases();
    const t0 = this.now();
    const result = this.runtime.tick({ cameraPos, cameraForward, frameIndex, viewProj: this.viewProj });
    const t1 = this.now();
    this.phases.runtime = t1 - t0;
    this.runWork(cameraPos, cameraForward, this.frameBudgetMs);
    const t2 = this.now();
    this.applyVisibility(cameraPos);
    const t3 = this.now();
    this.phases.visibility = t3 - t2 - this.phases.occlusion;
    if (this.extras) {
      const dt = this.lastTickAt === null ? 1 / 60 : Math.min(0.25, Math.max(0, (t3 - this.lastTickAt) / 1000));
      this.lastTickAt = t3;
      this.extras.update(dt, cameraPos);
      this.stats.extras = { ...this.extras.stats };
    }
    const t4 = this.now();
    this.phases.actors = t4 - t3;
    this.phases.total = t4 - t0;
    this.stats.phases = this.phases;
    this.last = result;
    return result;
  }

  /**
   * Let the tile fetches started by tick() resolve. Their content is queued,
   * not spawned: the next tick() spawns it within the frame budget, or call
   * flush() to spawn everything now.
   */
  async settle(): Promise<void> {
    // Tile fetches resolve in later tasks (terrain is prepared in idle time),
    // and the tile manager starts queued fetches as earlier ones finish, so
    // wait until no new fetch joined the chain while we waited.
    let seen: Promise<void>;
    do {
      seen = this.tail;
      await seen;
      await Promise.resolve();
      await Promise.resolve();
    } while (seen !== this.tail);
  }

  /** Run every queued spawn/despawn step now, ignoring the budget (teleports, tests). */
  flush(): void {
    this.runWork(null, null, Infinity);
  }

  /** True when the ground under (x, z) is a resident tile (its floor collider exists). */
  groundReady(x: number, z: number): boolean {
    const tx = Math.floor((x - LEVEL.origin[0]) / TILE_SIZE);
    const tz = Math.floor((z - LEVEL.origin[2]) / TILE_SIZE);
    if (tx < 0 || tz < 0 || tx >= LEVEL.gridSize || tz >= LEVEL.gridSize) return false;
    return this.nav.hasTile(tx, tz);
  }

  /**
   * Keep a move on loaded ground: if `next` lands on a tile that is not
   * resident (or off the map), the horizontal part of the move is undone.
   */
  guardMove(prev: Vec3, next: Vec3): Vec3 {
    if (this.groundReady(next[0], next[2])) return next;
    if (this.groundReady(next[0], prev[2])) return [next[0], next[1], prev[2]];
    if (this.groundReady(prev[0], next[2])) return [prev[0], next[1], next[2]];
    return [prev[0], next[1], prev[2]];
  }

  destroy(): void {
    this.runtime.destroy();
    this.extras?.destroy();
    for (const id of this.owned) this.scene.remove(id);
    this.owned.clear();
    this.props.clear();
    this.slabs.clear();
    this.spawnJobs.clear();
    this.despawnJobs.clear();
  }

  // ── tiles ────────────────────────────────────────────────────────────

  private payload(meta: TileMeta): TileLoadResult {
    return { residentBytes: meta.payloadBytes, entities: [] };
  }

  private queueSpawn(x: number, z: number): void {
    const key = `${x}|${z}`;
    const content = this.contents.get(key);
    if (!content) return;
    // Streamed back in before its removal finished: finish removing first so
    // the ids are free (spawn would otherwise skip ids still in the scene).
    const leftovers = this.despawnJobs.get(key);
    if (leftovers) {
      for (const id of leftovers) this.drop(id);
      this.despawnJobs.delete(key);
    }
    this.spawnJobs.set(key, { key, x, z, content, floorDone: false, nextProp: 0 });
  }

  private queueDespawn(x: number, z: number): void {
    const key = `${x}|${z}`;
    this.spawnJobs.delete(key);
    const ids: EntityId[] = [];
    const slabId = streamSlabId(x, z);
    const slab = this.slabs.get(slabId);
    if (slab) {
      ids.push(slab.id, ...slab.extra);
      this.slabs.delete(slabId);
    }
    for (const [id, p] of this.props) {
      if (p.tile !== key) continue;
      ids.push(id, ...p.details.map((d) => d.id));
      this.props.delete(id);
    }
    this.nav.removeTile(x, z);
    // Hidden now (no draw), removed from the scene a batch at a time.
    for (const id of ids) setVisible(this.scene, id, false);
    // NPC state saved, emitters fading, terrain meshes freed (the floor is hidden).
    this.extras?.onTileEvicted(x, z);
    if (ids.length > 0) this.despawnJobs.set(key, [...(this.despawnJobs.get(key) ?? []), ...ids]);
  }

  /** Floor, road and nav grid: what makes the tile walkable. */
  private spawnFloor(job: SpawnJob): void {
    const { x, z, content } = job;
    const slabId = streamSlabId(x, z);
    const extra: EntityId[] = [];
    this.add(this.extras ? this.extras.floorEntity(slabId, x, z) : slabEntity(slabId, x, z, content.center));
    if (content.road) {
      const roadId = `stream.${x}.${z}.road` as EntityId;
      this.add(roadEntity(roadId, content.center));
      extra.push(roadId);
    }
    this.slabs.set(slabId, { id: slabId, extra, center: content.center });
    // Nav blocks every prop footprint now, before the props themselves spawn,
    // plus any terrain too steep to walk.
    this.nav.addTile(x, z, [
      ...content.props.filter((p) => p.collider).map((p) => ({
        minX: p.center[0] - p.collider![0], maxX: p.center[0] + p.collider![0],
        minZ: p.center[2] - p.collider![2], maxZ: p.center[2] + p.collider![2],
      })),
      ...(this.extras ? this.extras.slopeBlockers(x, z, NAV_CELL) : []),
    ]);
    this.extras?.onFloor(x, z);
    job.floorDone = true;
  }

  /** The job's floor and every prop are in: drop the job, bring in its actors. */
  private finishIfDone(job: SpawnJob): void {
    if (!job.floorDone || job.nextProp < job.content.props.length) return;
    this.spawnJobs.delete(job.key);
    this.extras?.onTileComplete(job.x, job.z);
  }

  private spawnProp(job: SpawnJob): void {
    const raw = job.content.props[job.nextProp++]!;
    // On terrain, props stand on the lowest ground under their footprint.
    const p = this.extras ? this.extras.liftProp(raw) : raw;
    const mainId = streamPropId(job.x, job.z, p.key);
    this.add(propEntity(mainId, p));
    const details = p.details.map((d, k) => {
      const id = `${mainId}.d${k}` as EntityId;
      const center: Vec3 = [p.center[0] + d.offset[0], p.center[1] + d.offset[1], p.center[2] + d.offset[2]];
      this.add(detailEntity(id, center, d.size, d.color, d.emissive));
      return { id, maxLod: d.maxLod, radius: Math.hypot(...d.size) / 2, center };
    });
    const radius = Math.hypot(p.size[0], p.size[1], p.size[2]) / 2;
    // Only props flagged as solid (buildings, primitive or kit) occlude, using
    // their box collider (world half extents) as the conservative core.
    const half = p.occluder === true && p.collider ? p.collider : null;
    const solid = !!half && half[1] * 2 >= 4 && Math.min(half[0], half[2]) * 2 >= 4;
    this.props.set(mainId, {
      tile: job.key, mainId, shape: p.shape, kit: p.kit ?? null, center: p.center,
      radius, lod: null, details,
      occluder: p.occluder === true,
      // Details sit on and around the prop: bound them with its sphere.
      occItem: { ids: [mainId, ...details.map((d) => d.id)], box: { center: p.center, halfExtents: [radius, radius, radius] }, occluder: false },
      occBox: solid && half ? { ids: [], box: { center: p.center, halfExtents: [half[0], half[1], half[2]] }, occluder: true } : null,
    });
  }

  private despawnBatch(): boolean {
    const first = this.despawnJobs.entries().next();
    if (first.done) return false;
    const [key, ids] = first.value;
    for (const id of ids.splice(0, STREAM_LEVEL.despawnBatch)) this.drop(id);
    if (ids.length === 0) this.despawnJobs.delete(key);
    return true;
  }

  /**
   * Run queued work. `cam` null means no camera (flush): everything runs.
   * Order per step: any unfinished floor (nearest/ahead first), then the next
   * prop of the nearest tile; removals get at least one batch per tick so an
   * evict backlog cannot grow without bound.
   */
  private runWork(cam: Vec3 | null, fwd: Vec3 | null, budgetMs: number): void {
    const start = this.now();
    let steps = 0;
    const score = (job: SpawnJob): number => {
      if (!cam) return 0;
      const dx = job.content.center[0] - cam[0];
      const dz = job.content.center[2] - cam[2];
      const dist = Math.hypot(dx, dz);
      const ahead = fwd && dist > 1e-6 ? (dx * fwd[0] + dz * fwd[2]) / dist : 0;
      return dist * (1 - 0.35 * Math.max(0, ahead));
    };
    // 1. Floors the camera is on or about to step onto: never deferred.
    const ph = this.phases;
    const timed = (kind: "floor" | "prop" | "despawn" | "complete", fn: () => void): void => {
      const t = this.now();
      fn();
      const ms = this.now() - t;
      if (kind === "floor") { ph.floorMs += ms; ph.floors += 1; }
      else if (kind === "prop") { ph.propMs += ms; ph.props += 1; }
      else if (kind === "complete") ph.completeMs += ms;
      else { ph.despawnMs += ms; ph.despawns += 1; }
    };
    if (cam) {
      const half = TILE_SIZE / 2;
      for (const job of this.spawnJobs.values()) {
        if (job.floorDone) continue;
        const ox = Math.max(0, Math.abs(cam[0] - job.content.center[0]) - half);
        const oz = Math.max(0, Math.abs(cam[2] - job.content.center[2]) - half);
        if (Math.hypot(ox, oz) <= STREAM_LEVEL.mandatoryFloorDistance) {
          timed("floor", () => this.spawnFloor(job));
          steps += 1;
        }
      }
    }
    // 2. Budgeted steps.
    let removedThisTick = false;
    for (;;) {
      if (steps > 0 && this.now() - start >= budgetMs) break;
      let best: SpawnJob | null = null;
      let bestScore = Infinity;
      let bestFloor = false;
      for (const job of this.spawnJobs.values()) {
        const needsFloor = !job.floorDone;
        const s = score(job);
        // Any pending floor outranks any prop.
        if ((needsFloor && !bestFloor) || (needsFloor === bestFloor && s < bestScore)) {
          best = job; bestScore = s; bestFloor = needsFloor;
        }
      }
      if (best) {
        const job = best;
        if (!job.floorDone) timed("floor", () => this.spawnFloor(job));
        else if (job.nextProp < job.content.props.length) timed("prop", () => this.spawnProp(job));
        timed("complete", () => this.finishIfDone(job));
        steps += 1;
        continue;
      }
      let removed = false;
      timed("despawn", () => { removed = this.despawnBatch(); });
      if (removed) { removedThisTick = true; steps += 1; continue; }
      ph.despawns -= 1;
      break;
    }
    if (!removedThisTick) {
      let removed = false;
      timed("despawn", () => { removed = this.despawnBatch(); });
      if (removed) steps += 1;
      else ph.despawns -= 1;
    }
    this.lastWorkMs = this.now() - start;
    this.lastWorkSteps = steps;
  }

  private pendingDespawnCount(): number {
    let n = 0;
    for (const ids of this.despawnJobs.values()) n += ids.length;
    return n;
  }

  private add(e: Entity): void {
    if (this.scene.has(e.id)) return;
    this.scene.add(e);
    this.owned.add(e.id);
  }

  private drop(id: EntityId): void {
    if (this.scene.has(id)) this.scene.remove(id);
    this.owned.delete(id);
  }

  // ── per-frame culling + LOD ──────────────────────────────────────────

  private applyVisibility(cam: Vec3): void {
    const frustum = this.runtime.lastFrustum;
    let draws = 0, triangles = 0, culled = 0;
    const lodCounts: [number, number, number, number] = [0, 0, 0, 0];
    // 1. Frustum culling + LOD selection.
    for (const slab of this.slabs.values()) {
      const inView = !frustum || sphereInFrustum(frustum, slab.center, TILE_SIZE * 0.72);
      setVisible(this.scene, slab.id, inView);
      for (const id of slab.extra) setVisible(this.scene, id, inView);
    }
    this.extras?.selectTerrainLods(cam, (id) => this.scene.get(id)?.components[Tag.Mesh]?.visible === true);
    for (const p of this.props.values()) {
      const dist = Math.hypot(cam[0] - p.center[0], cam[1] - p.center[1], cam[2] - p.center[2]);
      const lod = selectLodHysteresis(p.lod, dist, STREAM_LEVEL.lodThresholds, STREAM_LEVEL.lodBand);
      p.lod = lod;
      const inView = !frustum || sphereInFrustum(frustum, p.center, p.radius);
      const entity = this.scene.get(p.mainId);
      const mesh = entity?.components[Tag.Mesh];
      if (p.shape === "kit") {
        // The kit bakes its own LOD3 impostor, so every level is the kit's mesh.
        setGeometry(mesh, `kit-${p.kit}`, lod);
      } else if (p.shape !== "box") {
        const generator = lod === 3 ? "stream-impostor" : LOD_GENERATOR[p.shape][lod];
        setGeometry(mesh, generator, lod === 3 ? 0 : lod);
      }
      setVisible(this.scene, p.mainId, inView);
      for (const d of p.details) setVisible(this.scene, d.id, inView && lod <= d.maxLod);
    }
    // 2. Occlusion: props certainly behind nearer buildings or hills are hidden.
    let occluded = 0;
    let occlusionRan = false;
    const gate = STREAM_LEVEL.occlusion;
    this.occSince += 1;
    if (this.extras?.occlusion && frustum && (this.occPays || this.occSince >= gate.probeEvery)) {
      occlusionRan = true;
      this.occSince = 0;
      const items = this.occItems;
      items.length = 0;
      for (const p of this.props.values()) {
        if (!this.scene.get(p.mainId)?.components[Tag.Mesh]?.visible) continue;
        items.push(p.occItem);
        if (p.occBox) items.push(p.occBox);
      }
      // The same camera the frustum came from (set in tick()).
      const to = this.now();
      occluded = this.extras.occlusionPass(this.viewProj, cam, items);
      this.phases.occlusion = this.now() - to;
      let visibleDraws = 0;
      for (const it of items) visibleDraws += it.ids.length;
      this.occPays = occluded >= gate.minHidden && occluded >= gate.minHiddenFraction * visibleDraws;
    }
    // 3. Count what is left to draw.
    for (const slab of this.slabs.values()) {
      if (this.scene.get(slab.id)?.components[Tag.Mesh]?.visible) {
        draws += 1;
        if (!this.extras) triangles += 12; // terrain triangles come from extras
      } else culled += 1;
      for (const id of slab.extra) {
        if (this.scene.get(id)?.components[Tag.Mesh]?.visible) { draws += 1; triangles += 12; } else culled += 1;
      }
    }
    if (this.extras) triangles += this.extras.stats.terrainTriangles;
    for (const p of this.props.values()) {
      const lod = p.lod ?? 0;
      if (this.scene.get(p.mainId)?.components[Tag.Mesh]?.visible) {
        draws += 1;
        lodCounts[lod] += 1;
        triangles += p.shape === "kit"
          ? this.kitTriangles.get(`proc:kit-${p.kit}:${lod}`) ?? 0
          : TRIS.get(propMeshKey(p.shape, lod)) ?? 0;
      } else culled += 1;
      for (const d of p.details) {
        if (this.scene.get(d.id)?.components[Tag.Mesh]?.visible) { draws += 1; triangles += 12; } else culled += 1;
      }
    }
    this.peak = Math.max(this.peak, this.owned.size);
    this.stats = {
      phases: this.phases,
      occluded,
      occlusionRan,
      extras: this.extras ? { ...this.extras.stats } : null,
      residentTiles: this.runtime.tiles.resident.length,
      residentBytes: this.runtime.tiles.residentBytesNow,
      streamEntities: this.owned.size,
      peakStreamEntities: this.peak,
      draws, triangles, culled, lodCounts,
      navTiles: this.nav.tileCount,
      pendingSpawnTiles: this.spawnJobs.size,
      pendingDespawnEntities: this.pendingDespawnCount(),
      lastWorkMs: this.lastWorkMs,
      lastWorkSteps: this.lastWorkSteps,
    };
  }
}

export function attachStreamedLevel(scene: Scene, opts: StreamedLevelOptions = {}): StreamedLevel {
  return new StreamedLevel(scene, opts);
}

function tileBytes(c: TileContent): number {
  const entities = 1 + (c.road ? 1 : 0) + c.props.reduce((n, p) => n + 1 + p.details.length, 0);
  return entities * BYTES_PER_ENTITY + (TILE_SIZE / NAV_CELL) ** 2;
}

function setVisible(scene: Scene, id: EntityId, visible: boolean): boolean {
  const mesh = scene.get(id)?.components[Tag.Mesh];
  if (!mesh) return false;
  if (mesh.visible !== visible) mesh.visible = visible;
  return visible;
}

function slabEntity(id: EntityId, x: number, z: number, center: Vec3): Entity {
  const shade = 0.15 + ((x + z) % 2) * 0.035;
  return {
    id,
    name: `Tile ${x},${z} floor`,
    components: {
      [Tag.Transform]: { position: [center[0], -0.25, center[2]], rotation: [0, 0, 0, 1], scale: [TILE_SIZE, 0.5, TILE_SIZE], parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: [shade, shade + 0.015, shade + 0.025], metallic: 0.02, roughness: 0.9 },
        visible: true,
      },
      // World-space half extents (ColliderComponent): the floor top is y = 0.
      [Tag.Collider]: { shape: "box", extents: [TILE_SIZE / 2, 0.25, TILE_SIZE / 2] },
    },
  };
}

function roadEntity(id: EntityId, center: Vec3): Entity {
  return {
    id,
    name: "Road",
    components: {
      [Tag.Transform]: { position: [center[0], 0.01, LEVEL.roadZ], rotation: [0, 0, 0, 1], scale: [TILE_SIZE, 0.02, LEVEL.roadHalfWidth * 2], parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: [0.07, 0.07, 0.08], metallic: 0, roughness: 0.75 },
        visible: true,
      },
    },
  };
}

/** Mesh key a non-kit prop draws at `lod` (matches the geometry applyVisibility sets). */
function propMeshKey(shape: PropSpec["shape"], lod: LodLevel): string {
  if (shape === "box" || shape === "kit") return "primitive:box";
  if (lod === 3) return "proc:stream-impostor:0";
  return `proc:${LOD_GENERATOR[shape][lod]}:${lod}`;
}

function setGeometry(mesh: MeshComponent | undefined, generator: string, seed: number): void {
  if (!mesh) return;
  const g = mesh.geometry;
  if (g.kind !== "procedural" || g.generator !== generator || g.seed !== seed) {
    mesh.geometry = { kind: "procedural", generator, seed };
  }
}

function propEntity(id: EntityId, p: PropSpec): Entity {
  const kit = p.shape === "kit";
  const geometry = p.shape === "box"
    ? { kind: "primitive" as const, shape: "box" as const }
    : { kind: "procedural" as const, generator: kit ? `kit-${p.kit}` : LOD_GENERATOR[p.shape as "pillar" | "dome"][0], seed: 0 };
  const yaw = p.yaw ?? 0;
  const e: Entity = {
    id,
    name: p.landmark ?? p.key,
    components: {
      [Tag.Transform]: {
        position: p.center,
        rotation: [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
        // Kit meshes are modelled at real size; primitives are unit meshes scaled to size.
        scale: kit ? [1, 1, 1] : p.size,
        parent: null,
      },
      [Tag.Mesh]: {
        geometry,
        material: {
          kind: "pbr",
          // Kit colour lives in the mesh's vertex colours; the material base stays white.
          baseColor: p.color,
          metallic: p.metallic ?? (p.shape === "pillar" ? 0.35 : 0.08),
          roughness: p.roughness ?? 0.6,
          ...(p.emissive ? { emissive: p.emissive } : {}),
        },
        visible: true,
      },
    },
  };
  if (p.collider) e.components[Tag.Collider] = { shape: "box", extents: p.collider };
  return e;
}

function detailEntity(id: EntityId, center: Vec3, size: Vec3, color: Vec3, emissive?: Vec3): Entity {
  return {
    id,
    name: "detail",
    components: {
      [Tag.Transform]: { position: center, rotation: [0, 0, 0, 1], scale: size, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: color, metallic: 0.3, roughness: 0.5, ...(emissive ? { emissive } : {}) },
        visible: true,
      },
    },
  };
}

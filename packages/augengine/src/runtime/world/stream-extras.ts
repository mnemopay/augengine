/**
 * world/stream-extras — terrain, occlusion and tile actors for The Loading Range.
 *
 * Kept beside StreamedLevel rather than inside it so the level's tile
 * scheduling (budgeted spawning, instancing) stays readable. StreamedLevel
 * calls in at four points:
 *
 *  - spawn:   `floorEntity` (the tile's terrain, replacing the flat slab),
 *             `slopeBlockers` for the tile's nav and `onFloor` (meshes up)
 *             with the floor; `liftProp` for every prop; `onTileComplete`
 *             (NPCs and emitters in) once the last prop is placed.
 *  - evict:   `onTileEvicted` (meshes released, NPC state saved, fades out).
 *
 * With `probes` set, each tile's irradiance probes (irradiance-probes.ts) are
 * baked in `prepare` (one idle task per layer), handed to the probe sink with
 * the floor and invalidated on evict.
 *  - frame:   `selectTerrainLods` and `occlusionPass` after frustum culling,
 *             then `update` for NPCs and ambient audio.
 */
import type { Scene } from "../../scene/scene.js";
import { Tag, type Entity, type EntityId, type Mat4, type Vec3 } from "../../scene/types.js";
import type { MeshData } from "../live/webgpu/primitives.js";
import { selectLodHysteresis, type LodLevel } from "./lod.js";
import { levelAmbients, levelNpcs } from "./level-actors.js";
import { TILE_SIZE, type PropSpec, type TileContent } from "./level-content.js";
import {
  bakeBoxes,
  bakeGround,
  bakeTileLayer,
  PROBE_FLOATS,
  PROBES,
  PROBES_PER_TILE,
  probeTile,
  type ProbeLighting,
  type ProbeSink,
} from "./irradiance-probes.js";
import { SoftwareOcclusion, type OcclusionBox } from "./occlusion.js";
import {
  buildTerrainTile,
  groundUnder,
  sampleHeight,
  TERRAIN,
  terrainCollider,
  terrainGenerator,
  terrainHeight,
  terrainLodTriangles,
  terrainMeshKey,
  terrainNormals,
  terrainOccluderDrop,
  terrainSlopeBlockers,
  terrainTileCenter,
  terrainTileMesh,
  type TerrainLod,
  type TerrainTile,
} from "./terrain.js";
import type { NavBlocker, TileNav } from "./tile-nav.js";
import { TileActors, type AmbientAudioSink } from "./tile-actors.js";

/** Where the level's per-tile meshes go (the renderer, in the game). */
export interface MeshSink {
  upload(key: string, data: MeshData): void;
  release?(key: string): void;
}

export interface StreamExtrasOptions {
  meshSink?: MeshSink;
  audio?: AmbientAudioSink;
  /** CPU occlusion culling. Default true. */
  occlusion?: boolean;
  /** Persona NPCs and ambient emitters. Default true. */
  actors?: boolean;
  /** Bake and stream per-tile irradiance probes to this sink, under this lighting. */
  probes?: { sink: ProbeSink; lighting: ProbeLighting };
  /** Tile content lookup for the probe bake (StreamedLevel supplies it). */
  contentAt?: (tx: number, tz: number) => TileContent | undefined;
}

/** One thing the occlusion pass may hide: the ids drawn for it and their bounds. */
export interface OcclusionItem {
  ids: readonly EntityId[];
  box: OcclusionBox;
  /** Solid enough to hide things behind it (big buildings). */
  occluder: boolean;
}

export interface StreamExtrasStats {
  terrainTiles: number;
  terrainTriangles: number;
  terrainLods: [number, number, number];
  meshesResident: number;
  occlusionTested: number;
  occluded: number;
  occluderTriangles: number;
  npcs: number;
  voices: number;
}

type TerrainOccluder = { vertices: Float32Array; indices: Uint32Array; offset: Vec3 };

interface TerrainRuntime {
  tile: TerrainTile;
  id: EntityId;
  lod: LodLevel | null;
  /** Lowered LOD2 top surface used as an occluder (prepared, or built on first use). */
  occluder?: TerrainOccluder;
}

/**
 * Everything a tile's terrain needs that is too slow for a frame (~13 ms of
 * height/normal/mesh work per tile on a laptop CPU), computed while the tile
 * is still loading: `prepare` runs each part in its own idle task, and the
 * tile only becomes resident once it is done.
 */
interface PreparedTerrain {
  tile: TerrainTile;
  meshes?: [MeshData, MeshData, MeshData];
  slopes?: { navCell: number; blockers: NavBlocker[] };
  occluder?: TerrainOccluder;
  /** Irradiance probes, baked one layer per idle task; `probeLayers` counts layers done. */
  probes?: Float32Array;
  probeLayers?: number;
}

const OCCLUDER_RANGE = 180;
const MAX_BOX_OCCLUDERS = 48;
/** Prepared-but-not-resident tiles kept around (a tile that streams back soon skips the work). */
const MAX_PREPARED_IDLE = 24;

interface IdleDeadlineLike { didTimeout: boolean; timeRemaining(): number }
type RequestIdle = (cb: (d: IdleDeadlineLike) => void, o: { timeout: number }) => void;

/** Idle time a preparation part wants before it starts (parts take ~1-4 ms). */
const IDLE_NEEDED_MS = 3;
/** Re-queues before a part runs regardless, so a busy host cannot starve tile loads. */
const MAX_IDLE_DEFERS = 4;

/**
 * Run `fn` in its own task, in idle time when the host has it (so it lands
 * between frames). With requestIdleCallback, a part that finds less than
 * IDLE_NEEDED_MS of the idle period left waits for the next one (a few times
 * at most); the 250 ms timeout only forces it when the host stays busy.
 */
const nextTask: (fn: () => void) => void =
  typeof (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback === "function"
    ? (fn) => {
        const ric = (globalThis as unknown as { requestIdleCallback: RequestIdle }).requestIdleCallback;
        let defers = 0;
        const attempt = (d: IdleDeadlineLike) => {
          if (!d.didTimeout && d.timeRemaining() < IDLE_NEEDED_MS && defers < MAX_IDLE_DEFERS) {
            defers += 1;
            ric(attempt, { timeout: 250 });
            return;
          }
          fn();
        };
        ric(attempt, { timeout: 250 });
      }
    : (fn) => setTimeout(fn, 0);

const inTask = <T>(fn: () => T): Promise<T> =>
  new Promise<T>((resolve, reject) => nextTask(() => { try { resolve(fn()); } catch (e) { reject(e); } }));

/** One idle-task run of terrain preparation (diagnostics: where off-frame CPU went). */
export interface PrepTask {
  /** performance.now() when it started. */
  at: number;
  ms: number;
  kind: "heights" | "meshes" | "slopes" | "occluder" | "probes";
  tile: string;
}

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class StreamExtras {
  readonly scene: Scene;
  readonly nav: TileNav;
  readonly actors: TileActors | null;
  readonly occlusion: SoftwareOcclusion | null;
  private readonly meshSink: MeshSink | undefined;
  private readonly probes: StreamExtrasOptions["probes"];
  private readonly contentAt: StreamExtrasOptions["contentAt"];
  /** Tiles whose probes the sink currently holds. */
  private readonly probeTiles = new Set<string>();
  private readonly terrain = new Map<string, TerrainRuntime>();
  private readonly prepared = new Map<string, PreparedTerrain>();
  private readonly uploaded = new Set<string>();
  /** Last terrain-preparation tasks (newest last, capped), for spike diagnosis. */
  readonly prepLog: PrepTask[] = [];
  private readonly nearScratch: OcclusionItem[] = [];
  private readonly distScratch = new Map<OcclusionItem, number>();
  stats: StreamExtrasStats = {
    terrainTiles: 0, terrainTriangles: 0, terrainLods: [0, 0, 0], meshesResident: 0,
    occlusionTested: 0, occluded: 0, occluderTriangles: 0, npcs: 0, voices: 0,
  };

  constructor(scene: Scene, nav: TileNav, opts: StreamExtrasOptions = {}) {
    this.scene = scene;
    this.nav = nav;
    this.meshSink = opts.meshSink;
    this.probes = opts.probes;
    this.contentAt = opts.contentAt;
    this.occlusion = opts.occlusion === false ? null : new SoftwareOcclusion();
    this.actors = opts.actors === false
      ? null
      : new TileActors(scene, {
          npcsFor: levelNpcs,
          ambientsFor: levelAmbients,
          groundY: (x, z) => this.groundAt(x, z),
          walkable: (x, z) => nav.isWalkable(x, z),
          ...(opts.audio ? { audio: opts.audio } : {}),
        });
  }

  /** Ground height under (x, z): the resident tile's collider surface, else the analytic height. */
  groundAt(x: number, z: number): number {
    const t = this.terrain.get(`${Math.floor(x / TILE_SIZE)}|${Math.floor(z / TILE_SIZE)}`);
    return (t && sampleHeight(t.tile, x, z)) ?? terrainHeight(x, z);
  }

  /**
   * Do the tile's slow terrain work ahead of time, one part per idle task:
   * heights, then normals + the three LOD meshes, then nav slope blockers,
   * then the hill occluder. Call while the tile loads (StreamedLevel's fetch);
   * the floor step then only places the entity.
   */
  async prepare(tx: number, tz: number, navCell: number): Promise<void> {
    const key = `${tx}|${tz}`;
    const task = <T>(kind: PrepTask["kind"], fn: () => T): Promise<T> => inTask(() => {
      const at = now();
      const out = fn();
      this.prepLog.push({ at, ms: now() - at, kind, tile: key });
      if (this.prepLog.length > 400) this.prepLog.splice(0, this.prepLog.length - 400);
      return out;
    });
    let p = this.prepared.get(key);
    if (!p) {
      const tile = await task("heights", () => buildTerrainTile(tx, tz));
      p = { tile };
      this.prepared.set(key, p);
      this.trimPrepared();
    }
    const prep = p;
    if (!prep.meshes) {
      prep.meshes = await task("meshes", () => {
        const normals = terrainNormals(prep.tile);
        return [0, 1, 2].map((lod) => terrainTileMesh(prep.tile, lod as TerrainLod, normals)) as [MeshData, MeshData, MeshData];
      });
    }
    if (!prep.slopes || prep.slopes.navCell !== navCell) {
      prep.slopes = await task("slopes", () => ({ navCell, blockers: terrainSlopeBlockers(prep.tile, navCell) }));
    }
    if (!prep.occluder && prep.tile.maxHeight >= 3) {
      prep.occluder = await task("occluder", () => buildTerrainOccluder(prep.tile));
    }
    const probes = this.probes, contentAt = this.contentAt;
    if (probes && contentAt) {
      const out = (prep.probes ??= new Float32Array(PROBES_PER_TILE * PROBE_FLOATS));
      let done: number = prep.probeLayers ?? 0;
      if (done < PROBES.layers) {
        const env = await task("probes", () => ({ boxes: bakeBoxes(tx, tz, contentAt), lighting: probes.lighting, ground: bakeGround(tx, tz) }));
        // One row of four probes per idle task (~2–5 ms each on a laptop).
        while (done < PROBES.layers) {
          const layer = done;
          for (let row = 0; row < PROBES.perTileAxis; row++) {
            await task("probes", () => bakeTileLayer(tx, tz, layer, env, out, [row, row + 1]));
          }
          done = layer + 1;
          prep.probeLayers = done;
        }
      }
    }
  }

  /** Drop the oldest prepared tiles that are not resident, beyond the idle cap. */
  private trimPrepared(): void {
    let idle = 0;
    for (const k of this.prepared.keys()) if (!this.terrain.has(k)) idle += 1;
    for (const k of this.prepared.keys()) {
      if (idle <= MAX_PREPARED_IDLE) break;
      if (this.terrain.has(k)) continue;
      this.prepared.delete(k);
      idle -= 1;
    }
  }

  /** The tile's floor: terrain mesh + heightfield collider, under the given id. */
  floorEntity(id: EntityId, tx: number, tz: number): Entity {
    const prep = this.prepared.get(`${tx}|${tz}`);
    const tile = prep?.tile ?? buildTerrainTile(tx, tz);
    this.terrain.set(`${tx}|${tz}`, { tile, id, lod: null, ...(prep?.occluder ? { occluder: prep.occluder } : {}) });
    const shade = 0.16 + ((tx + tz) % 2) * 0.02;
    return {
      id,
      name: `Tile ${tx},${tz} terrain`,
      components: {
        [Tag.Transform]: { position: terrainTileCenter(tile), rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Mesh]: {
          geometry: { kind: "procedural", generator: terrainGenerator(tx, tz), seed: 2 },
          material: { kind: "pbr", baseColor: [shade * 0.9, shade + 0.06, shade * 0.75], metallic: 0, roughness: 0.95 },
          visible: true,
        },
        [Tag.Collider]: terrainCollider(tile),
      },
    };
  }

  /** The prop, standing on the ground: lifted to the lowest point under its footprint. */
  liftProp(p: PropSpec): PropSpec {
    const hx = p.size[0] / 2, hz = p.size[2] / 2;
    const lift = groundUnder(p.center[0] - hx, p.center[0] + hx, p.center[2] - hz, p.center[2] + hz);
    if (lift === 0) return p;
    return { ...p, center: [p.center[0], p.center[1] + lift, p.center[2]] };
  }

  /** Too-steep cells of the tile, for its nav grid. */
  slopeBlockers(tx: number, tz: number, navCell: number): NavBlocker[] {
    const prep = this.prepared.get(`${tx}|${tz}`);
    if (prep?.slopes && prep.slopes.navCell === navCell) return prep.slopes.blockers;
    const t = this.terrain.get(`${tx}|${tz}`);
    return terrainSlopeBlockers(t ? t.tile : buildTerrainTile(tx, tz), navCell);
  }

  /** With the tile's floor: upload its terrain meshes (prepared ahead when possible). */
  onFloor(tx: number, tz: number): void {
    const t = this.terrain.get(`${tx}|${tz}`);
    if (t && this.meshSink) {
      const prepMeshes = this.prepared.get(`${tx}|${tz}`)?.meshes;
      const normals = prepMeshes ? null : terrainNormals(t.tile);
      for (const lod of [0, 1, 2] as TerrainLod[]) {
        const key = terrainMeshKey(tx, tz, lod);
        if (this.uploaded.has(key)) continue;
        this.meshSink.upload(key, prepMeshes ? prepMeshes[lod] : terrainTileMesh(t.tile, lod, normals!));
        this.uploaded.add(key);
      }
    }
    const prep = this.prepared.get(`${tx}|${tz}`);
    if (this.probes && prep?.probes && prep.probeLayers === PROBES.layers) {
      this.probes.sink.setTile(probeTile(tx, tz, prep.probes));
      this.probeTiles.add(`${tx}|${tz}`);
    }
    this.stats.meshesResident = this.uploaded.size;
  }

  /** Once the tile's props are all in: bring in its residents and emitters. */
  onTileComplete(tx: number, tz: number): void {
    this.actors?.onTileResident(tx, tz);
  }

  /** The tile is leaving: save its NPCs, fade its emitters, release its meshes. */
  onTileEvicted(tx: number, tz: number): void {
    this.actors?.onTileEvicted(tx, tz);
    this.terrain.delete(`${tx}|${tz}`);
    if (this.probes && this.probeTiles.delete(`${tx}|${tz}`)) this.probes.sink.clearTile(tx, tz);
    // Keep its prepared data for a while: a tile that streams straight back skips the work.
    this.trimPrepared();
    if (this.meshSink?.release) {
      for (const lod of [0, 1, 2] as TerrainLod[]) {
        const key = terrainMeshKey(tx, tz, lod);
        if (this.uploaded.delete(key)) this.meshSink.release(key);
      }
    }
    this.stats.meshesResident = this.uploaded.size;
  }

  /** Pick each resident terrain tile's LOD (with hysteresis) by camera distance. */
  selectTerrainLods(cam: Vec3, visible: (id: EntityId) => boolean): void {
    let tris = 0;
    const lods: [number, number, number] = [0, 0, 0];
    for (const t of this.terrain.values()) {
      const cx = t.tile.x0 + TILE_SIZE / 2, cz = t.tile.z0 + TILE_SIZE / 2;
      // Distance to the tile's nearest point, so the tile you stand on is always LOD0.
      const dx = Math.max(0, Math.abs(cam[0] - cx) - TILE_SIZE / 2);
      const dz = Math.max(0, Math.abs(cam[2] - cz) - TILE_SIZE / 2);
      const dist = Math.hypot(dx, dz);
      // lod.ts works on 3 thresholds; the third is beyond any resident tile.
      const lod = Math.min(2, selectLodHysteresis(t.lod, dist, [TERRAIN.lodThresholds[0], TERRAIN.lodThresholds[1], 1e9], 0.1)) as TerrainLod;
      t.lod = lod;
      const mesh = this.scene.get(t.id)?.components[Tag.Mesh];
      if (mesh) {
        const g = mesh.geometry;
        if (g.kind !== "procedural" || g.seed !== lod) mesh.geometry = { kind: "procedural", generator: terrainGenerator(t.tile.tx, t.tile.tz), seed: lod };
      }
      if (visible(t.id)) {
        tris += terrainLodTriangles(lod);
        lods[lod] += 1;
      }
    }
    this.stats.terrainTiles = this.terrain.size;
    this.stats.terrainTriangles = tris;
    this.stats.terrainLods = lods;
  }

  /**
   * Hide items that are certainly behind nearer buildings or hills. Only items
   * already visible after frustum culling should be passed in. Returns how
   * many entity draws it hid.
   */
  occlusionPass(viewProj: Mat4, cam: Vec3, items: readonly OcclusionItem[]): number {
    const occ = this.occlusion;
    if (!occ) return 0;
    occ.begin(viewProj);
    // Occluders, nearest first: the closest big buildings… (scratch arrays are
    // reused across frames: this runs every frame and used to allocate four
    // arrays and an object per candidate).
    const near = this.nearScratch;
    const dist = this.distScratch;
    near.length = 0;
    for (const it of items) {
      if (!it.occluder) continue;
      const d = Math.hypot(it.box.center[0] - cam[0], it.box.center[2] - cam[2]);
      if (d >= OCCLUDER_RANGE) continue;
      dist.set(it, d);
      near.push(it);
    }
    near.sort((a, b) => dist.get(a)! - dist.get(b)!);
    const n = Math.min(near.length, MAX_BOX_OCCLUDERS);
    for (let k = 0; k < n; k++) occ.addBox(near[k]!.box);
    dist.clear();
    // …and the hills, from their coarse (LOD2) triangles lowered by the tile's
    // worst overhang above any drawn LOD, so they never hide more than the
    // ground actually drawn does.
    for (const t of this.terrain.values()) {
      if (t.tile.maxHeight < 3) continue;
      // Same range cut as the box occluders: far hills cost raster time for little gain.
      const dx = Math.max(0, Math.abs(cam[0] - (t.tile.x0 + TILE_SIZE / 2)) - TILE_SIZE / 2);
      const dz = Math.max(0, Math.abs(cam[2] - (t.tile.z0 + TILE_SIZE / 2)) - TILE_SIZE / 2);
      if (Math.hypot(dx, dz) > OCCLUDER_RANGE) continue;
      const occluder = this.terrainOccluder(t);
      occ.addTriangles(occluder.vertices, occluder.indices, occluder.offset, 8);
    }
    let hidden = 0;
    for (const it of items) {
      if (it.ids.length === 0 || !occ.isOccluded(it.box)) continue;
      for (const id of it.ids) {
        const mesh = this.scene.get(id)?.components[Tag.Mesh];
        if (mesh && mesh.visible) {
          mesh.visible = false;
          hidden += 1;
        }
      }
    }
    this.stats.occlusionTested = occ.stats.tested;
    this.stats.occluded = occ.stats.occluded;
    this.stats.occluderTriangles = occ.stats.occluderTriangles;
    return hidden;
  }

  private terrainOccluder(t: TerrainRuntime): TerrainOccluder {
    if (!t.occluder) t.occluder = buildTerrainOccluder(t.tile);
    return t.occluder;
  }

  /** NPC behaviour and ambient fades. */
  update(dt: number, listener: Vec3): void {
    if (!this.actors) return;
    this.actors.update(dt, listener);
    this.stats.npcs = this.actors.stats.npcs;
    this.stats.voices = this.actors.stats.voices;
  }

  destroy(): void {
    this.actors?.destroy();
    if (this.meshSink?.release) for (const key of this.uploaded) this.meshSink.release(key);
    this.uploaded.clear();
    if (this.probes) for (const k of this.probeTiles) {
      const [x, z] = k.split("|").map(Number);
      this.probes.sink.clearTile(x!, z!);
    }
    this.probeTiles.clear();
    this.terrain.clear();
    this.prepared.clear();
  }
}

/** The tile's LOD2 top surface, lowered so it never hides more than the drawn ground does. */
function buildTerrainOccluder(tile: TerrainTile): TerrainOccluder {
  const m = terrainTileMesh(tile, 2);
  const topTris = (TERRAIN.cells / TERRAIN.lodSteps[2]) ** 2 * 2;
  const c = terrainTileCenter(tile);
  return { vertices: m.vertices, indices: m.indices.slice(0, topTris * 3), offset: [c[0], c[1] - terrainOccluderDrop(tile), c[2]] };
}

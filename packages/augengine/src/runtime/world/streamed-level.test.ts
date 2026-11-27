import { describe, expect, it } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type Vec3 } from "../../scene/types.js";
import { KinematicCharacterController } from "../../physics/character-controller.js";
import { buildTileContent, LANDMARKS, LEVEL, TILE_SIZE, type KitCatalog } from "./level-content.js";
import { selectLodHysteresis } from "./lod.js";
import { STREAM_LEVEL, StreamedLevel, streamMeshLibrary, streamSlabId } from "./streamed-level.js";
import { TileNav } from "./tile-nav.js";
import { terrainLodTriangles } from "./terrain.js";

function emptyScene(): Scene {
  return new Scene({ schemaVersion: 1, source: { kind: "authored", path: "test:stream-level" }, entities: [] });
}

const lastFrame = new WeakMap<StreamedLevel, number>();

async function at(level: StreamedLevel, pos: Vec3, frame: number, fwd: Vec3 = [1, 0, 0]): Promise<void> {
  // Tiles load a few at a time and each prepares its terrain over idle tasks,
  // so a single tick no longer brings in every wanted tile: tick until the
  // resident set stops changing. Frame indices only ever increase.
  let f = Math.max(frame, (lastFrame.get(level) ?? 0) + 1);
  let before = -1;
  for (let k = 0; k < 12 && level.stats.residentTiles !== before; k++) {
    before = level.stats.residentTiles;
    level.tick(pos, fwd, f++);
    await level.settle();
    // Content is time-sliced across ticks; these tests check what a tile
    // contains once it has fully arrived, so drain the queue.
    level.flush();
    level.tick(pos, fwd, f++);
  }
  lastFrame.set(level, f);
}

describe("streamed level: tiles carry content", () => {
  it("is deterministic per tile and keeps props off the road", () => {
    const a = buildTileContent(3, 8);
    const b = buildTileContent(3, 8);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    for (let x = 0; x < LEVEL.gridSize; x++) {
      for (const p of buildTileContent(x, 8).props) {
        if (p.landmark || !p.collider) continue;
        const halfZ = p.collider[2];
        expect(Math.abs(p.center[2] - LEVEL.roadZ) - halfZ, `${x} ${p.key} clear of road`).toBeGreaterThan(LEVEL.roadHalfWidth);
      }
    }
    expect(Object.keys(LANDMARKS).length).toBe(5);
  });

  it("spawns floor, props and colliders with the tile and removes them on evict", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene);
    await at(level, STREAM_LEVEL.spawn, 1);
    expect(scene.has(streamSlabId(0, 8))).toBe(true);
    const slab = scene.get(streamSlabId(0, 8))!;
    // The floor is the tile's terrain: a heightfield spanning the tile.
    const floor = slab.components[Tag.Collider]!;
    expect(floor.shape).toBe("heightfield");
    expect([floor.heightfield!.sizeX, floor.heightfield!.sizeZ]).toEqual([TILE_SIZE, TILE_SIZE]);
    const gate = [...scene.entities.values()].find((e) => e.name === "West Gate");
    expect(gate?.components[Tag.Collider]).toBeTruthy();
    expect(level.nav.hasTile(0, 8)).toBe(true);

    // Walk to the far east end: the start tile and everything on it must be gone.
    let f = 10;
    for (let x = 16; x <= 1000; x += 16) await at(level, [x, 1.7, LEVEL.roadZ], (f += 2));
    expect(scene.has(streamSlabId(0, 8))).toBe(false);
    expect([...scene.entities.values()].some((e) => e.name === "West Gate")).toBe(false);
    expect(level.nav.hasTile(0, 8)).toBe(false);
    expect([...scene.entities.values()].some((e) => e.name === "East Beacon")).toBe(true);
    level.destroy();
    expect(scene.entities.size).toBe(0);
  });

  it("keeps memory and entity count bounded over the whole 1 km walk", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene);
    let f = 1;
    let maxEntities = 0, maxTiles = 0, maxBytes = 0, evictions = 0, prevTiles = new Set<string>();
    // East along the road, then back west one row north: ~2 km of walking.
    const path: Vec3[] = [];
    for (let x = 16; x <= 1008; x += 8) path.push([x, 1.7, LEVEL.roadZ]);
    for (let x = 1008; x >= 16; x -= 8) path.push([x, 1.7, LEVEL.roadZ + 70]);
    for (const p of path) {
      await at(level, p, (f += 2));
      const tiles = new Set(level.runtime.tiles.resident.map((t) => `${t.coord.x}|${t.coord.z}`));
      for (const t of prevTiles) if (!tiles.has(t)) evictions += 1;
      prevTiles = tiles;
      maxEntities = Math.max(maxEntities, level.stats.streamEntities);
      maxTiles = Math.max(maxTiles, level.stats.residentTiles);
      maxBytes = Math.max(maxBytes, level.stats.residentBytes);
      expect(level.groundReady(p[0], p[2]), `ground under ${p[0]},${p[2]}`).toBe(true);
    }
    const allEntities = Array.from({ length: LEVEL.gridSize ** 2 }, (_, i) => buildTileContent(i % 16, Math.floor(i / 16)))
      .reduce((n, c) => n + 1 + (c.road ? 1 : 0) + c.props.reduce((m, p) => m + 1 + p.details.length, 0), 0);
    console.log(JSON.stringify({ maxEntities, allEntities, maxTiles, maxBytes, budget: level.memoryBudgetBytes, evictions }));
    // The budget may be overshot by at most one concurrent load batch (4 tiles).
    expect(maxTiles).toBeLessThanOrEqual(STREAM_LEVEL.budgetTiles + 4);
    expect(maxBytes).toBeLessThanOrEqual(level.memoryBudgetBytes * 1.35);
    expect(maxEntities).toBeLessThan(allEntities * 0.15);
    expect(evictions).toBeGreaterThan(20);
    level.destroy();
  });
});

describe("streamed level: LOD and culling", () => {
  it("hysteresis stops flicker at a threshold and steps over big jumps correctly", () => {
    const t = [28, 64, 110] as const;
    expect(selectLodHysteresis(null, 29, t)).toBe(1);
    // Parked just past 28 m while drawn at LOD0: stays LOD0 until 28 * 1.12.
    expect(selectLodHysteresis(0, 29, t)).toBe(0);
    expect(selectLodHysteresis(0, 32, t)).toBe(1);
    // Drawn at LOD1 at 27 m: stays LOD1 until inside 28 * 0.88.
    expect(selectLodHysteresis(1, 27, t)).toBe(1);
    expect(selectLodHysteresis(1, 24, t)).toBe(0);
    // A teleport from close to 70 m lands on LOD1 (70 < 64 * 1.12), not LOD0 or LOD2.
    expect(selectLodHysteresis(0, 70, t)).toBe(1);
    expect(selectLodHysteresis(0, 200, t)).toBe(3);
  });

  it("drops tessellation and detail parts with distance and hides what is behind the camera", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene);
    const pos: Vec3 = [16, 1.7, LEVEL.roadZ];
    await at(level, pos, 1, [1, 0, 0]);
    const east = { ...level.stats };
    await at(level, pos, 3, [-1, 0, 0]); // face the map edge: almost nothing in view
    const west = { ...level.stats };
    console.log(JSON.stringify({ east, west }));
    expect(east.lodCounts[0]).toBeGreaterThan(0);
    expect(east.lodCounts[2] + east.lodCounts[3]).toBeGreaterThan(0);
    expect(west.draws).toBeLessThan(east.draws * 0.6);
    expect(west.triangles).toBeLessThan(east.triangles * 0.6);

    // Same scene with LOD forced to 0 and no culling: the work we avoided.
    let fullDraws = 0, fullTris = 0;
    const tris = new Map(streamMeshLibrary().map((m) => [m.key, m.data.indexCount / 3]));
    for (const e of scene.entities.values()) {
      const m = e.components[Tag.Mesh];
      if (!m) continue;
      fullDraws += 1;
      const g = m.geometry;
      if (g.kind === "procedural" && g.generator.startsWith("stream-terrain")) {
        fullTris += terrainLodTriangles(0); // terrain at full detail
        continue;
      }
      fullTris += g.kind === "procedural" ? tris.get(`proc:${g.generator === "stream-impostor" ? (e.name.includes("dome") ? "stream-dome" : "stream-pillar") : g.generator}:0`) ?? 12 : 12;
    }
    console.log(JSON.stringify({ fullDraws, fullTris, eastDraws: east.draws, eastTris: east.triangles }));
    expect(east.draws).toBeLessThan(fullDraws);
    expect(east.triangles).toBeLessThan(fullTris);
    level.destroy();
  });
});

describe("streamed level: collision and navigation", () => {
  it("guardMove refuses to step onto an unloaded tile or off the map", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene);
    await at(level, STREAM_LEVEL.spawn, 1);
    const here: Vec3 = [16, 1.7, LEVEL.roadZ];
    expect(level.guardMove(here, [18, 1.7, LEVEL.roadZ])).toEqual([18, 1.7, LEVEL.roadZ]);
    // Off the west edge of the map.
    expect(level.guardMove([1, 1.7, 512], [-1, 1.5, 513])).toEqual([1, 1.5, 513]);
    // Tile (10, 8) is far away and not resident.
    expect(level.groundReady(10 * 64 + 5, 512)).toBe(false);
    expect(level.guardMove(here, [10 * 64 + 5, 1.7, 512])).toEqual(here);
    level.destroy();
  });

  it("the player walks the road on streamed floors and never falls through", async () => {
    const scene = emptyScene();
    scene.add({
      id: "player" as never,
      name: "player",
      components: {
        [Tag.Transform]: { position: [...STREAM_LEVEL.spawn] as Vec3, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.9 },
      },
    });
    const level = new StreamedLevel(scene);
    const controller = new KinematicCharacterController(scene, { entityId: "player" as never, moveSpeed: 8 });
    let minY = Infinity;
    for (let frame = 0; frame < 60 * 30; frame++) {
      const tx = scene.get("player" as never)!.components[Tag.Transform]!;
      const prev: Vec3 = [...tx.position] as Vec3;
      level.tick(prev, [1, 0, 0], frame);
      if (frame % 4 === 0) await level.settle();
      controller.update(1 / 60, [1, 0, 0]);
      const next = scene.get("player" as never)!.components[Tag.Transform]!.position;
      const guarded = level.guardMove(prev, next);
      scene.setLocalTransform("player" as never, guarded, tx.rotation, tx.scale);
      if (frame > 30) minY = Math.min(minY, guarded[1]);
    }
    const end = scene.get("player" as never)!.components[Tag.Transform]!.position;
    console.log(JSON.stringify({ endX: end[0], minY }));
    // 30 s at 8 m/s along an open road: well past the first tiles.
    expect(end[0]).toBeGreaterThan(200);
    expect(minY).toBeGreaterThan(1.2); // capsule centre stays on the floor (1.3)
    level.destroy();
    // 1,800 simulated frames, each tile's terrain prepared over idle tasks.
  }, 30_000);

  it("nav grids exist only for resident tiles and paths route around colliders", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene);
    await at(level, [96, 1.7, LEVEL.roadZ], 1);
    await level.settle();
    level.flush();
    expect(level.nav.tileCount).toBe(level.runtime.tiles.resident.length);
    const path = level.nav.findPath([20, LEVEL.roadZ], [150, LEVEL.roadZ]);
    expect(path).not.toBeNull();
    for (const [x, z] of path!) expect(level.nav.isWalkable(x, z)).toBe(true);
    // A goal on an unloaded tile has no path.
    expect(level.nav.findPath([20, LEVEL.roadZ], [900, LEVEL.roadZ])).toBeNull();
    level.destroy();
  });

  it("TileNav blocks inflated footprints and refuses corner cuts", () => {
    const nav = new TileNav({ originX: 0, originZ: 0, tileSize: 16, cell: 1, agentRadius: 0.4 });
    nav.addTile(0, 0, [{ minX: 4, maxX: 12, minZ: 4, maxZ: 12 }]);
    expect(nav.isWalkable(2, 2)).toBe(true);
    expect(nav.isWalkable(8, 8)).toBe(false);
    expect(nav.isWalkable(3.7, 8)).toBe(false); // inside the 0.4 m inflation
    const path = nav.findPath([2, 8], [14, 8])!;
    expect(path.every(([x, z]) => nav.isWalkable(x, z))).toBe(true);
    expect(Math.min(...path.map((p) => p[1])) < 4 || Math.max(...path.map((p) => p[1])) > 12).toBe(true);
    nav.removeTile(0, 0);
    expect(nav.isWalkable(2, 2)).toBe(false);
  });
});

describe("streamed level: time-sliced spawning", () => {
  /** A clock that advances 1 ms every time it is read. */
  function tickingClock(): () => number {
    let t = 0;
    return () => (t += 1);
  }

  it("never spends more than the per-frame budget on queued content", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene, { frameBudgetMs: 3, now: tickingClock() });
    const pos: Vec3 = [16, 1.7, LEVEL.roadZ];
    level.tick(pos, [1, 0, 0], 1);
    await level.settle();
    let queued = 0;
    let frame = 2;
    let maxSteps = 0;
    // Stand still and let the queue drain one budgeted tick at a time.
    for (; frame < 400; frame++) {
      level.tick(pos, [1, 0, 0], frame);
      // Forced floors (camera standing on / next to them) are outside the budget.
      maxSteps = Math.max(maxSteps, level.stats.lastWorkSteps);
      queued = Math.max(queued, level.stats.pendingSpawnTiles);
      if (level.stats.pendingSpawnTiles === 0) break;
    }
    const sliced = [...scene.entities.keys()].sort();
    // Same content as spawning it all at once.
    const ref = new StreamedLevel(emptyScene());
    ref.tick(pos, [1, 0, 0], 1);
    await ref.settle();
    ref.flush();
    const all = [...ref.scene.entities.keys()].sort();
    console.log(JSON.stringify({ queued, ticks: frame - 1, maxSteps, entities: sliced.length }));
    expect(queued).toBeGreaterThan(2);
    expect(level.stats.pendingSpawnTiles).toBe(0);
    expect(frame - 1).toBeGreaterThan(10); // it really was spread over many ticks
    // 3 budgeted steps + 1 guaranteed removal batch + the forced floors (<= 4 tiles touch the camera).
    expect(maxSteps).toBeLessThanOrEqual(3 + 1 + 4);
    expect(sliced).toEqual(all);
    level.destroy();
    ref.destroy();
  });

  it("builds every pending floor before any prop, nearest first", async () => {
    const scene = emptyScene();
    // One step per tick.
    const level = new StreamedLevel(scene, { frameBudgetMs: 1, now: tickingClock() });
    const pos: Vec3 = [96, 1.7, LEVEL.roadZ];
    level.tick(pos, [1, 0, 0], 1);
    await level.settle();
    const floorOrder: string[] = [];
    let propsBeforeAllFloors = 0;
    for (let frame = 2; frame < 200 && (frame === 2 || level.stats.pendingSpawnTiles > 0); frame++) {
      const before = new Set(level.runtime.tiles.resident.filter((t) => level.nav.hasTile(t.coord.x, t.coord.z)).map((t) => `${t.coord.x}|${t.coord.z}`));
      const propsBefore = [...scene.entities.keys()].filter((k) => !k.endsWith(".slab") && !k.endsWith(".road")).length;
      level.tick(pos, [1, 0, 0], frame);
      for (const t of level.runtime.tiles.resident) {
        const k = `${t.coord.x}|${t.coord.z}`;
        if (level.nav.hasTile(t.coord.x, t.coord.z) && !before.has(k)) floorOrder.push(k);
      }
      const propsAfter = [...scene.entities.keys()].filter((k) => !k.endsWith(".slab") && !k.endsWith(".road")).length;
      const floorsPending = level.runtime.tiles.resident.some((t) => !level.nav.hasTile(t.coord.x, t.coord.z));
      if (propsAfter > propsBefore && floorsPending) propsBeforeAllFloors += 1;
      await level.settle(); // more tiles arrive mid-drain, as in a real run
    }
    const dist = (k: string) => {
      const [x, z] = k.split("|").map(Number) as [number, number];
      return Math.hypot(x * 64 + 32 - pos[0], z * 64 + 32 - pos[2]);
    };
    console.log(JSON.stringify({ floorOrder: floorOrder.map((k) => [k, Math.round(dist(k))]) }));
    expect(floorOrder.length).toBeGreaterThan(5);
    expect(propsBeforeAllFloors).toBe(0);
    // Floors that were budgeted (not forced) arrive in non-decreasing distance, allowing for the ahead bias.
    const ds = floorOrder.map(dist);
    for (let i = 1; i < ds.length; i++) expect(ds[i]! + 1).toBeGreaterThanOrEqual(ds[i - 1]! * 0.6);
    level.destroy();
  });

  it("the ground under a 240 m/s camera is always built, even at one step per tick", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene, { frameBudgetMs: 1, now: tickingClock() });
    let missing = 0;
    let maxPending = 0;
    let frame = 1;
    // Level start: the first tiles arrive before the player is deployed.
    level.tick([16, 1.7, LEVEL.roadZ], [1, 0, 0], frame++);
    await level.settle();
    for (let x = 16; x <= 1000; x += 4) {
      level.tick([x, 1.7, LEVEL.roadZ], [1, 0, 0], frame++);
      if (!level.groundReady(x, LEVEL.roadZ)) missing += 1;
      maxPending = Math.max(maxPending, level.stats.pendingSpawnTiles);
      // Fetches resolve between frames, as rAF would allow.
      await level.settle();
    }
    console.log(JSON.stringify({ missing, maxPending, entities: level.stats.streamEntities }));
    expect(missing).toBe(0);
    // Removals keep up: entity count stays bounded on the run.
    expect(level.stats.peakStreamEntities).toBeLessThan(900);
    level.destroy();
  });

  it("a tile evicted and streamed back before its removal finished spawns cleanly", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene, { frameBudgetMs: 1, now: tickingClock() });
    const pos: Vec3 = [16, 1.7, LEVEL.roadZ];
    await at(level, pos, 1);
    const before = [...scene.entities.keys()].filter((k) => k.startsWith("stream.0.8.")).sort();
    expect(before.length).toBeGreaterThan(1);
    level.runtime.tiles.forceEvict({ x: 0, z: 8 });
    expect(level.groundReady(16, LEVEL.roadZ)).toBe(false);
    // Evicted entities are hidden right away, removed later.
    expect(before.every((id) => scene.get(id as never)?.components[Tag.Mesh]?.visible !== true)).toBe(true);
    level.runtime.tiles.forceLoad({ x: 0, z: 8 }, 5);
    await level.settle();
    level.flush();
    const after = [...scene.entities.keys()].filter((k) => k.startsWith("stream.0.8.")).sort();
    expect(after).toEqual(before);
    expect(level.groundReady(16, LEVEL.roadZ)).toBe(true);
    level.destroy();
    expect(scene.entities.size).toBe(0);
  });
});

// ── art kit ─────────────────────────────────────────────────────────────


/** Stand-in for @mnemopay/augengine-lod's kitCatalog(): the engine only needs ids, kinds and sizes. */
const KIT: KitCatalog = [
  { id: "building-a", kind: "building", size: [10, 13, 8], metallic: 0.05, roughness: 0.78 },
  { id: "building-b", kind: "building", size: [14, 17, 9], metallic: 0.05, roughness: 0.78 },
  { id: "tree-a", kind: "tree", size: [4.4, 7, 4.4], metallic: 0, roughness: 0.9 },
  { id: "lamp", kind: "lamp", size: [1.5, 5, 0.4], metallic: 0.5, roughness: 0.45 },
  { id: "bench", kind: "bench", size: [1.9, 0.9, 0.6], metallic: 0.1, roughness: 0.7 },
  { id: "barrier", kind: "barrier", size: [3, 0.8, 0.6], metallic: 0, roughness: 0.85 },
  { id: "water-tower", kind: "water-tower", size: [13, 27, 13], metallic: 0.4, roughness: 0.5 },
];

describe("streamed level: art kit", () => {
  it("without a kit the level is unchanged (primitives only)", () => {
    for (let x = 0; x < LEVEL.gridSize; x += 3) {
      expect(JSON.stringify(buildTileContent(x, 8, undefined))).toBe(JSON.stringify(buildTileContent(x, 8)));
      expect(buildTileContent(x, 8).props.every((p) => p.shape !== "kit")).toBe(true);
    }
  });

  it("builds tiles from the kit: grounded, deterministic, street furniture on the road, kit water tower", () => {
    expect(JSON.stringify(buildTileContent(5, 8, KIT))).toBe(JSON.stringify(buildTileContent(5, 8, KIT)));
    let lamps = 0;
    for (let x = 0; x < LEVEL.gridSize; x++) {
      for (const z of [3, 8]) {
        const t = buildTileContent(x, z, KIT);
        for (const p of t.props) {
          if (p.landmark && p.shape !== "kit") continue;
          expect(p.shape, `${x},${z} ${p.key}`).toBe("kit");
          expect(KIT.some((e) => e.id === p.kit)).toBe(true);
          // Centre-positioned at half height: the base sits on the floor (y = 0).
          expect(p.center[1]).toBeCloseTo(p.size[1] / 2, 6);
          expect(p.collider).not.toBeNull();
          expect(p.collider![1]).toBeCloseTo(p.size[1] / 2, 6);
          if (p.key.startsWith("lamp")) lamps += 1;
          // Filler (not street furniture) stays off the road.
          if (/^(building|tree)\./.test(p.key) && t.road) {
            expect(Math.abs(p.center[2] - LEVEL.roadZ) - p.collider![2]).toBeGreaterThan(LEVEL.roadHalfWidth);
          }
        }
        if (z !== 8) expect(t.props.some((p) => p.key.startsWith("lamp"))).toBe(false);
      }
    }
    expect(lamps).toBeGreaterThan(LEVEL.gridSize * 4); // both sides, every 16 m, minus landmark clearances
    const tower = buildTileContent(4, 8, KIT).props.find((p) => p.landmark === "Water Tower");
    expect(tower?.kit).toBe("water-tower");
    // Primitive landmarks stay (gate, spire, ring, beacon are bespoke shapes).
    expect(buildTileContent(0, 8, KIT).props.some((p) => p.landmark === "West Gate" && p.shape !== "kit")).toBe(true);
  });

  it("quarter-turned kit props swap footprint so the axis-aligned collider still fits", () => {
    for (let x = 0; x < LEVEL.gridSize; x++) {
      for (const p of buildTileContent(x, 2, KIT).props.filter((q) => q.key.startsWith("building."))) {
        const e = KIT.find((k) => k.id === p.kit)!;
        const turned = Math.abs(Math.sin(p.yaw ?? 0)) > 0.5;
        expect(p.size[0]).toBeCloseTo(turned ? e.size[2] : e.size[0], 6);
        expect(p.collider![0] * 2).toBeCloseTo(p.size[0], 6);
      }
    }
  });

  it("streams kit props with per-LOD kit meshes and counts their triangles", async () => {
    const scene = emptyScene();
    const tris = new Map<string, number>();
    for (const e of KIT) [1000, 250, 60, 8].forEach((n, lod) => tris.set(`proc:kit-${e.id}:${lod}`, n));
    const level = new StreamedLevel(scene, { kit: KIT, kitTriangles: tris });
    await at(level, STREAM_LEVEL.spawn, 1);
    const kitEntities = [...scene.entities.values()].filter((e) => {
      const g = e.components[Tag.Mesh]?.geometry;
      return g?.kind === "procedural" && g.generator.startsWith("kit-");
    });
    expect(kitEntities.length).toBeGreaterThan(20);
    for (const e of kitEntities) {
      expect(e.components[Tag.Transform]!.scale).toEqual([1, 1, 1]);
      const g = e.components[Tag.Mesh]!.geometry;
      expect(g.kind === "procedural" && g.seed >= 0 && g.seed <= 3).toBe(true);
    }
    // Every visible kit draw contributes one of the mapped counts, so totals are sums of them.
    expect(level.stats.triangles).toBeGreaterThan(0);
    // Far-away kit props reach the kit's own LOD3, not the shared box impostor.
    // From mid-map, tiles ahead load out to ~176 m, past the 110 m LOD3 step.
    const mid: Vec3 = [512, 1.7, LEVEL.roadZ];
    await at(level, mid, 20);
    const far = [...scene.entities.values()].find((e) => {
      const g = e.components[Tag.Mesh]?.geometry;
      if (!(g?.kind === "procedural" && g.generator.startsWith("kit-")) || !e.components[Tag.Mesh]!.visible) return false;
      const p = e.components[Tag.Transform]!.position;
      return Math.hypot(p[0] - mid[0], p[2] - mid[2]) > STREAM_LEVEL.lodThresholds[2] * (1 + STREAM_LEVEL.lodBand);
    });
    expect(far, "a kit prop beyond the LOD3 threshold").toBeTruthy();
    const gFar = far!.components[Tag.Mesh]!.geometry;
    expect(gFar.kind === "procedural" && gFar.generator.startsWith("kit-") && gFar.seed === 3).toBe(true);
    // Street lamps stream in with the road tile, each with a pole collider.
    const lamp = kitEntities.find((e) => e.name.startsWith("lamp"));
    expect(lamp?.components[Tag.Collider]?.extents).toEqual([0.2, 2.5, 0.2]);
    level.destroy();
    expect(scene.entities.size).toBe(0);
  });

  it("only solid buildings are occluders: domes, the open water tower, trees and street props never are", () => {
    let kitBuildings = 0, primBuildings = 0;
    for (let x = 0; x < LEVEL.gridSize; x++) {
      for (const p of buildTileContent(x, 8, KIT).props) {
        const kind = KIT.find((e) => e.id === p.kit)?.kind;
        expect(p.occluder === true, `${x} ${p.key}`).toBe(kind === "building");
        if (kind === "building") kitBuildings += 1;
      }
      for (const p of buildTileContent(x, 8).props) {
        expect(p.occluder === true, `${x} ${p.key}`).toBe(p.key.startsWith("building."));
        if (p.key.startsWith("building.")) primBuildings += 1;
      }
    }
    expect(kitBuildings).toBeGreaterThan(5);
    expect(primBuildings).toBeGreaterThan(5);
  });

  it("occlusion only runs every frame while it pays; skipped frames hide nothing", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene, { kit: KIT });
    // Facing the map edge from the west gate: almost nothing to hide.
    const pos: Vec3 = [16, 1.7, LEVEL.roadZ];
    await at(level, pos, 1, [-1, 0, 0]);
    let f = (lastFrame.get(level) ?? 0) + 1;
    const ran: boolean[] = [];
    for (let k = 0; k < 24; k++) {
      level.tick(pos, [-1, 0, 0], f++);
      ran.push(level.stats.occlusionRan);
      if (!level.stats.occlusionRan) expect(level.stats.occluded).toBe(0);
    }
    const runs = ran.filter(Boolean).length;
    expect(runs, `ran on ${runs} of 24 ticks`).toBeLessThanOrEqual(Math.ceil(24 / STREAM_LEVEL.occlusion.probeEvery) + 1);
    expect(runs).toBeGreaterThan(0);

    // Among the buildings, looking along the row: it pays, so it runs every tick.
    let paying = 0, tested = 0;
    for (let x = 24; x <= 600 && paying === 0; x += 24) {
      const p: Vec3 = [x, 1.7, LEVEL.roadZ - 96];
      await at(level, p, f, [1, 0, 0]);
      f = (lastFrame.get(level) ?? f) + 1;
      level.tick(p, [1, 0, 0], f++);
      if (level.stats.occluded >= STREAM_LEVEL.occlusion.minHidden) {
        // Paid this tick: the next ticks at the same spot must all run.
        for (let k = 0; k < 6; k++) {
          level.tick(p, [1, 0, 0], f++);
          tested += 1;
          if (level.stats.occlusionRan) paying += 1;
        }
      }
    }
    expect(tested, "found a spot among buildings where occlusion pays").toBeGreaterThan(0);
    expect(paying).toBe(tested);
    level.destroy();
  });

  it("kit buildings occlude what is behind them as the camera walks the level", async () => {
    const scene = emptyScene();
    const level = new StreamedLevel(scene, { kit: KIT });
    let maxOccluded = 0;
    let f = 1;
    // Walk a row off the road, among the buildings, looking along the row.
    for (let x = 24; x <= 600; x += 24) {
      await at(level, [x, 1.7, LEVEL.roadZ - 96], (f += 2), [1, 0, 0]);
      maxOccluded = Math.max(maxOccluded, level.stats.occluded);
    }
    console.log(JSON.stringify({ maxOccluded }));
    expect(maxOccluded).toBeGreaterThan(0);
    level.destroy();
  });
});

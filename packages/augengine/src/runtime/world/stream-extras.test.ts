import { describe, expect, it } from "vitest";
import { Scene } from "../../scene/scene.js";
import { m4Create, m4LookAt, m4Mul, m4Perspective } from "../../scene/math.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../../scene/types.js";
import { buildTileContent, LEVEL, TILE_SIZE } from "./level-content.js";
import type { ProbeSink, ProbeTileData } from "./irradiance-probes.js";
import { StreamExtras, type MeshSink } from "./stream-extras.js";
import { terrainHeight, terrainMeshKey } from "./terrain.js";
import { TileNav } from "./tile-nav.js";

function emptyScene(): Scene {
  return new Scene({ schemaVersion: 1, source: { kind: "authored", path: "test:stream-extras" }, entities: [] });
}

class FakeMeshes implements MeshSink {
  live = new Set<string>();
  uploads = 0;
  releases = 0;
  upload(key: string): void {
    if (this.live.has(key)) throw new Error(`double upload ${key}`);
    this.live.add(key);
    this.uploads += 1;
  }
  release(key: string): void {
    if (!this.live.delete(key)) throw new Error(`release of unknown ${key}`);
    this.releases += 1;
  }
}

function nav(): TileNav {
  return new TileNav({ originX: LEVEL.origin[0], originZ: LEVEL.origin[2], tileSize: TILE_SIZE, cell: 1, agentRadius: 0.4 });
}

function load(scene: Scene, ex: StreamExtras, n: TileNav, tx: number, tz: number): EntityId {
  const id = `stream.${tx}.${tz}.slab` as EntityId;
  scene.add(ex.floorEntity(id, tx, tz));
  n.addTile(tx, tz, ex.slopeBlockers(tx, tz, 1));
  ex.onFloor(tx, tz);
  ex.onTileComplete(tx, tz);
  return id;
}

function unload(scene: Scene, ex: StreamExtras, n: TileNav, tx: number, tz: number): void {
  scene.remove(`stream.${tx}.${tz}.slab` as EntityId);
  n.removeTile(tx, tz);
  ex.onTileEvicted(tx, tz);
}

describe("stream extras: terrain meshes follow tile residency", () => {
  it("uploads three LODs per resident tile and releases them on evict", () => {
    const scene = emptyScene(), n = nav(), meshes = new FakeMeshes();
    const ex = new StreamExtras(scene, n, { meshSink: meshes, actors: false, occlusion: false });
    load(scene, ex, n, 4, 3);
    load(scene, ex, n, 5, 3);
    expect(meshes.live.size).toBe(6);
    expect(meshes.live.has(terrainMeshKey(4, 3, 2))).toBe(true);
    unload(scene, ex, n, 4, 3);
    expect(meshes.live.size).toBe(3);
    expect(meshes.releases).toBe(3);
  });

  it("keeps the mesh count bounded over a long walk", () => {
    const scene = emptyScene(), n = nav(), meshes = new FakeMeshes();
    const ex = new StreamExtras(scene, n, { meshSink: meshes, actors: false, occlusion: false });
    const resident = new Set<string>();
    let peak = 0;
    for (let x = 8; x <= 1016; x += 8) {
      const want = new Set<string>();
      for (let tz = 6; tz <= 10; tz++) for (let tx = 0; tx < LEVEL.gridSize; tx++) {
        if (Math.hypot(tx + 0.5 - x / TILE_SIZE, tz + 0.5 - LEVEL.roadZ / TILE_SIZE) <= 1.75) want.add(`${tx}|${tz}`);
      }
      for (const k of [...resident]) if (!want.has(k)) { resident.delete(k); const [a, b] = k.split("|").map(Number) as [number, number]; unload(scene, ex, n, a, b); }
      for (const k of want) if (!resident.has(k)) { resident.add(k); const [a, b] = k.split("|").map(Number) as [number, number]; load(scene, ex, n, a, b); }
      peak = Math.max(peak, meshes.live.size);
      expect(meshes.live.size).toBe(resident.size * 3);
    }
    expect(peak).toBeLessThanOrEqual(13 * 3);
    expect(meshes.uploads - meshes.releases).toBe(meshes.live.size);
  });

  it("switches a tile to coarser LODs as the camera moves away", () => {
    const scene = emptyScene(), n = nav();
    const ex = new StreamExtras(scene, n, { actors: false, occlusion: false });
    const id = load(scene, ex, n, 8, 8);
    const seed = () => (scene.get(id)!.components[Tag.Mesh]!.geometry as { seed: number }).seed;
    ex.selectTerrainLods([8 * 64 + 32, 2, 512], () => true);
    expect(seed()).toBe(0);
    ex.selectTerrainLods([8 * 64 + 32 + 32 + 160, 2, 512], () => true);
    expect(seed()).toBe(1);
    ex.selectTerrainLods([8 * 64 + 32 + 32 + 400, 2, 512], () => true);
    expect(seed()).toBe(2);
  });
});

describe("stream extras: props stand on the ground", () => {
  it("lifts a prop to the lowest ground under its footprint", () => {
    const ex = new StreamExtras(emptyScene(), nav(), { actors: false, occlusion: false });
    // Find a hilly spot.
    let spot: Vec3 = [0, 0, 0];
    for (let x = 32; x < 1000; x += 16) for (let z = 32; z < 400; z += 16) if (terrainHeight(x, z) > spot[1]) spot = [x, terrainHeight(x, z), z];
    const lifted = ex.liftProp({ key: "b", shape: "box", center: [spot[0], 5, spot[2]], size: [8, 10, 8], color: [1, 1, 1], collider: [4, 5, 4], details: [] });
    const base = lifted.center[1] - 5;
    for (const [dx, dz] of [[-4, -4], [4, -4], [-4, 4], [4, 4], [0, 0]]) expect(base).toBeLessThanOrEqual(terrainHeight(spot[0] + dx, spot[2] + dz) + 1e-9);
    expect(base).toBeGreaterThan(0);
  });
});

describe("stream extras: occlusion pass hides only what is behind", () => {
  it("hides a pillar behind a building and counts it", () => {
    const scene = emptyScene();
    const ex = new StreamExtras(scene, nav(), { actors: false });
    const box = (id: string, center: Vec3, size: Vec3): Entity => ({
      id: id as EntityId, name: id,
      components: {
        [Tag.Transform]: { position: center, rotation: [0, 0, 0, 1], scale: size, parent: null },
        [Tag.Mesh]: { geometry: { kind: "primitive", shape: "box" }, material: { kind: "pbr", baseColor: [1, 1, 1], metallic: 0, roughness: 1 }, visible: true },
      },
    });
    scene.add(box("wall", [20, 8, 0], [2, 16, 30]));
    scene.add(box("behind", [40, 3, 0], [2, 6, 2]));
    scene.add(box("beside", [40, 3, 30], [2, 6, 2]));
    const view = m4Create(), proj = m4Create(), vp = m4Create();
    m4LookAt(view, [0, 1.7, 0], [1, 1.7, 0], [0, 1, 0]);
    m4Perspective(proj, (80 * Math.PI) / 180, 16 / 10, 0.1, 400);
    m4Mul(vp, proj, view);
    const hidden = ex.occlusionPass(vp, [0, 1.7, 0], [
      { ids: ["wall" as EntityId], box: { center: [20, 8, 0], halfExtents: [1, 8, 15] }, occluder: true },
      { ids: ["behind" as EntityId], box: { center: [40, 3, 0], halfExtents: [1, 3, 1] }, occluder: false },
      { ids: ["beside" as EntityId], box: { center: [40, 3, 30], halfExtents: [1, 3, 1] }, occluder: false },
    ]);
    expect(hidden).toBe(1);
    expect(scene.get("behind" as EntityId)!.components[Tag.Mesh]!.visible).toBe(false);
    expect(scene.get("beside" as EntityId)!.components[Tag.Mesh]!.visible).toBe(true);
    expect(scene.get("wall" as EntityId)!.components[Tag.Mesh]!.visible).toBe(true);
  });
});

describe("stream extras: irradiance probes follow tile residency", () => {
  class FakeProbes implements ProbeSink {
    live = new Map<string, ProbeTileData>();
    cleared: string[] = [];
    setTile(t: ProbeTileData): void { this.live.set(`${t.tx}|${t.tz}`, t); }
    clearTile(tx: number, tz: number): void { this.live.delete(`${tx}|${tz}`); this.cleared.push(`${tx}|${tz}`); }
  }
  const lighting = { lights: [{ dir: [0.408, -0.816, -0.408] as Vec3, ratio: [11.4, 9.9, 8.2] as Vec3 }] };

  it("bakes probes in prepare (idle tasks), hands them over with the floor and clears them on evict", async () => {
    const scene = emptyScene(), n = nav(), probes = new FakeProbes();
    const ex = new StreamExtras(scene, n, {
      actors: false, occlusion: false,
      probes: { sink: probes, lighting },
      contentAt: (x, z) => buildTileContent(x, z),
    });
    // No prepare: no probes (the floor still places; the renderer falls back to the sky ambient).
    load(scene, ex, n, 2, 2);
    expect(probes.live.size).toBe(0);
    await ex.prepare(4, 3, 1);
    expect(ex.prepLog.filter((t) => t.kind === "probes").length).toBe(9); // setup + 4 rows × 2 layers
    load(scene, ex, n, 4, 3);
    const t = probes.live.get("4|3")!;
    expect(t.gi0).toBe(16);
    expect(t.probes[15]).toBe(1);
    unload(scene, ex, n, 4, 3);
    expect(probes.live.size).toBe(0);
    expect(probes.cleared).toEqual(["4|3"]);
    // Streamed straight back: the prepared probes are reused, not rebaked.
    await ex.prepare(4, 3, 1);
    expect(ex.prepLog.filter((t) => t.kind === "probes").length).toBe(9);
    load(scene, ex, n, 4, 3);
    expect(probes.live.has("4|3")).toBe(true);
    ex.destroy();
    expect(probes.live.size).toBe(0);
  }, 20000);
});

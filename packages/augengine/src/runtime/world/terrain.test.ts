import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../../scene/types.js";
import { PhysicsSystem } from "../../physics/system.js";
import { sampleHeightfield } from "../../physics/heightfield.js";
import { raycast } from "../../physics/raycast.js";
import { KinematicCharacterController } from "../../physics/character-controller.js";
import { LEVEL, TILE_SIZE } from "./level-content.js";
import {
  buildTerrainTile,
  sampleHeight,
  sampleNormal,
  TERRAIN,
  terrainCollider,
  terrainHeight,
  terrainLodHeightfield,
  terrainLodTriangles,
  terrainOccluderDrop,
  terrainSlopeBlockers,
  terrainTileCenter,
  terrainTileMesh,
  type TerrainLod,
  type TerrainTile,
} from "./terrain.js";

const N = TERRAIN.cells;
const at = (t: TerrainTile, i: number, j: number) => t.samples[j * (N + 1) + i]!;

function emptyScene(): Scene {
  return new Scene({ schemaVersion: 1, source: { kind: "authored", path: "test:terrain" }, entities: [] });
}

function addTerrain(scene: Scene, tile: TerrainTile): EntityId {
  const id = `terrain.${tile.tx}.${tile.tz}` as EntityId;
  scene.add({
    id,
    name: id,
    components: {
      [Tag.Transform]: { position: terrainTileCenter(tile), rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Collider]: terrainCollider(tile),
    },
  });
  return id;
}

/** The hilliest tile in a band away from the road, for the walking tests. */
function hilliest(): TerrainTile {
  let best: TerrainTile | null = null;
  for (let tz = 1; tz <= 4; tz++) {
    for (let tx = 1; tx < LEVEL.gridSize - 1; tx++) {
      const t = buildTerrainTile(tx, tz);
      if (!best || t.maxHeight - t.minHeight > best.maxHeight - best.minHeight) best = t;
    }
  }
  return best!;
}

let physics: PhysicsSystem | null = null;
afterEach(() => {
  physics?.dispose();
  physics = null;
});

describe("terrain: one height function, seamless tiles", () => {
  it("shares identical samples on every tile edge", () => {
    for (let tz = 0; tz < LEVEL.gridSize - 1; tz += 3) {
      for (let tx = 0; tx < LEVEL.gridSize - 1; tx += 3) {
        const a = buildTerrainTile(tx, tz), right = buildTerrainTile(tx + 1, tz), up = buildTerrainTile(tx, tz + 1);
        for (let k = 0; k <= N; k++) {
          expect(at(a, N, k)).toBe(at(right, 0, k));
          expect(at(a, k, N)).toBe(at(up, k, 0));
        }
      }
    }
  });

  it("answers the same height from both tiles on a seam", () => {
    const a = buildTerrainTile(5, 3), b = buildTerrainTile(6, 3);
    const seamX = b.x0;
    for (let k = 0; k < 200; k++) {
      const z = a.z0 + (k / 199) * TILE_SIZE;
      expect(sampleHeight(a, seamX, z)).toBeCloseTo(sampleHeight(b, seamX, z)!, 5);
    }
  });

  it("keeps the road and landmark plazas flat and puts real hills elsewhere", () => {
    for (let x = 0; x <= LEVEL.extent.x; x += 8) {
      expect(terrainHeight(x, LEVEL.roadZ)).toBe(0);
      expect(terrainHeight(x, LEVEL.roadZ + LEVEL.roadHalfWidth)).toBe(0);
    }
    let peak = 0;
    for (let z = 0; z < LEVEL.extent.z; z += 16) for (let x = 0; x < LEVEL.extent.x; x += 16) peak = Math.max(peak, terrainHeight(x, z));
    expect(peak).toBeGreaterThan(12);
  });

  it("is deterministic", () => {
    expect(Array.from(buildTerrainTile(7, 2).samples)).toEqual(Array.from(buildTerrainTile(7, 2).samples));
  });
});

describe("terrain: mesh LODs and skirts", () => {
  const tile = hilliest();

  it("puts LOD0 vertices exactly on the samples and winds the top upward", () => {
    for (const lod of [0, 1, 2] as TerrainLod[]) {
      const m = terrainTileMesh(tile, lod);
      expect(m.indexCount / 3).toBe(terrainLodTriangles(lod));
      expect(m.vertices.every(Number.isFinite)).toBe(true);
      const q = N / TERRAIN.lodSteps[lod];
      const topTris = q * q * 2;
      for (let t = 0; t < topTris; t++) {
        const p = [0, 1, 2].map((k) => m.indices[t * 3 + k]! * 8);
        const [a, b, c] = p.map((o) => [m.vertices[o]!, m.vertices[o + 1]!, m.vertices[o + 2]!]) as [number[], number[], number[]];
        const ux = b[0]! - a[0]!, uy = b[1]! - a[1]!, uz = b[2]! - a[2]!;
        const vx = c[0]! - a[0]!, vy = c[1]! - a[1]!, vz = c[2]! - a[2]!;
        expect(uz * vx - ux * vz, `tri ${t} faces up`).toBeGreaterThan(0);
        void uy; void vy;
      }
    }
    const m0 = terrainTileMesh(tile, 0);
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) expect(m0.vertices[(j * (N + 1) + i) * 8 + 1]).toBe(at(tile, i, j));
  });

  it("hangs a skirt below every border vertex", () => {
    for (const lod of [0, 1, 2] as TerrainLod[]) {
      const m = terrainTileMesh(tile, lod);
      const q = N / TERRAIN.lodSteps[lod];
      const top = (q + 1) * (q + 1);
      expect(m.vertexCount).toBe(top + 4 * (q + 1));
      for (let v = top; v < m.vertexCount; v++) {
        const x = m.vertices[v * 8]!, y = m.vertices[v * 8 + 1]!, z = m.vertices[v * 8 + 2]!;
        expect(Math.abs(Math.abs(x) - TILE_SIZE / 2) < 1e-4 || Math.abs(Math.abs(z) - TILE_SIZE / 2) < 1e-4).toBe(true);
        const gh = sampleHeight(tile, x + tile.x0 + TILE_SIZE / 2, z + tile.z0 + TILE_SIZE / 2)!;
        expect(y).toBeCloseTo(gh - TERRAIN.skirt, 4);
      }
    }
  });

  it("never opens a crack deeper than the skirt between any two LODs on any edge", () => {
    let worst = 0;
    for (let tz = 0; tz < LEVEL.gridSize; tz++) {
      for (let tx = 0; tx < LEVEL.gridSize; tx++) {
        const t = buildTerrainTile(tx, tz);
        for (const step of [2, 4]) {
          // Along each edge: fine samples vs the coarse edge's linear interpolation.
          for (const edge of [0, 1, 2, 3]) {
            for (let k = 0; k <= N; k++) {
              const sample = (kk: number) => edge === 0 ? at(t, kk, 0) : edge === 1 ? at(t, N, kk) : edge === 2 ? at(t, kk, N) : at(t, 0, kk);
              const k0 = Math.floor(k / step) * step, k1 = Math.min(N, k0 + step);
              const f = k1 === k0 ? 0 : (k - k0) / (k1 - k0);
              const coarse = sample(k0) + (sample(k1) - sample(k0)) * f;
              worst = Math.max(worst, Math.abs(sample(k) - coarse));
            }
          }
        }
      }
    }
    expect(worst).toBeLessThan(TERRAIN.skirt);
  });
});

describe("terrain: hill occluders never rise above the drawn ground", () => {
  it("lowered LOD2 stays at or below every drawn LOD at random points", () => {
    let seed = 99;
    const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
    for (const [tx, tz] of [[hilliest().tx, hilliest().tz], [3, 2], [11, 13], [6, 5]] as [number, number][]) {
      const tile = buildTerrainTile(tx, tz);
      const drop = terrainOccluderDrop(tile);
      const lods = ([0, 1, 2] as TerrainLod[]).map((l) => terrainLodHeightfield(tile, l));
      for (let k = 0; k < 2000; k++) {
        const x = tile.x0 + rand() * TILE_SIZE, z = tile.z0 + rand() * TILE_SIZE;
        const [h0, h1, h2] = lods.map((hf) => sampleHeightfield(hf, tile.x0, tile.z0, 0, x, z)!.height) as [number, number, number];
        expect(h2 - drop).toBeLessThanOrEqual(Math.min(h0, h1, h2) + 1e-6);
      }
    }
  });
});

describe("terrain: collider agrees with the visual surface", () => {
  const tile = hilliest();

  it("Rapier's heightfield matches sampleHeight at random points", async () => {
    physics = new PhysicsSystem();
    await physics.init();
    const scene = emptyScene();
    addTerrain(scene, tile);
    physics.stepFixed(scene, 1 / 60);
    const world = (physics as unknown as { world: { castRay(ray: unknown, max: number, solid: boolean): { timeOfImpact: number } | null } }).world;
    const R = await import("@dimforge/rapier3d-deterministic-compat");
    let worst = 0;
    let seed = 7;
    const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
    for (let k = 0; k < 300; k++) {
      const x = tile.x0 + 0.01 + rand() * (TILE_SIZE - 0.02), z = tile.z0 + 0.01 + rand() * (TILE_SIZE - 0.02);
      const hit = world.castRay(new R.Ray({ x, y: 500, z }, { x: 0, y: -1, z: 0 }), 1000, true);
      expect(hit, `ray at ${x},${z}`).not.toBeNull();
      worst = Math.max(worst, Math.abs(500 - hit!.timeOfImpact - sampleHeight(tile, x, z)!));
    }
    expect(worst).toBeLessThan(2e-3);
  });

  it("the engine raycast hits the same surface", () => {
    const scene = emptyScene();
    addTerrain(scene, tile);
    for (let k = 0; k < 40; k++) {
      const x = tile.x0 + 1 + (k * 1.53) % (TILE_SIZE - 2), z = tile.z0 + 1 + (k * 2.71) % (TILE_SIZE - 2);
      const hit = raycast(scene, [x, 300, z], [0, -1, 0], { maxDistance: 600 });
      expect(hit).not.toBeNull();
      expect(hit!.point[1]).toBeCloseTo(sampleHeight(tile, x, z)!, 2);
      expect(hit!.normal[1]).toBeCloseTo(sampleNormal(tile, x, z)![1], 2);
    }
  });

  it("blocks only steep cells for nav", () => {
    const blockers = terrainSlopeBlockers(tile, 1);
    for (const b of blockers) {
      const n = sampleNormal(tile, (b.minX + b.maxX) / 2, (b.minZ + b.maxZ) / 2)!;
      // A blocker's centre cell is steep; flat road tiles produce none.
      expect(n[1]).toBeLessThan(TERRAIN.maxWalkNormalY + 0.05);
    }
    expect(terrainSlopeBlockers(buildTerrainTile(3, 8), 1).filter((b) => Math.abs((b.minZ + b.maxZ) / 2 - LEVEL.roadZ) < LEVEL.roadHalfWidth)).toHaveLength(0);
  });
});

describe("terrain: the player walks hills across tile edges", () => {
  /** The triangulated ground (what the collider is), over the 3x3 patch. */
  function patchGround(center: TerrainTile): (x: number, z: number) => number {
    const tiles: TerrainTile[] = [];
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) tiles.push(buildTerrainTile(center.tx + dx, center.tz + dz));
    return (x, z) => {
      for (const t of tiles) {
        const h = sampleHeight(t, x, z);
        if (h !== null) return h;
      }
      throw new Error(`walked off the test patch at ${x},${z}`);
    };
  }

  function setup(center: TerrainTile, start: Vec3) {
    const scene = emptyScene();
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) addTerrain(scene, buildTerrainTile(center.tx + dx, center.tz + dz));
    scene.add({
      id: "player" as EntityId,
      name: "player",
      components: {
        [Tag.Transform]: { position: start, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.5 },
      },
    });
    const ctl = new KinematicCharacterController(scene, { entityId: "player" as EntityId, moveSpeed: 6, snapDistance: 0.6 });
    return { scene, ctl };
  }

  it("never ends a frame below the ground while crossing seams in every direction", () => {
    const tile = hilliest();
    const ground = patchGround(tile);
    const cx = tile.x0 + TILE_SIZE / 2, cz = tile.z0 + TILE_SIZE / 2;
    let seamsCrossed = 0;
    for (const dir of [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0.7, 0, 0.7]] as Vec3[]) {
      const start: Vec3 = [cx, ground(cx, cz) + 3, cz];
      const { scene, ctl } = setup(tile, start);
      let worstSink = 0;
      // 9 s at <= 6 m/s from the centre stays inside the 3x3 patch (96 m to its edge).
      for (let f = 0; f < 60 * 9; f++) {
        ctl.update(1 / 60, dir, false);
        const p = scene.get("player" as EntityId)!.components[Tag.Transform]!.position;
        // Capsule bottom = centre - halfHeight - radius.
        worstSink = Math.max(worstSink, ground(p[0], p[2]) - (p[1] - 0.9));
      }
      const end = scene.get("player" as EntityId)!.components[Tag.Transform]!.position;
      if (Math.floor((end[0] - LEVEL.origin[0]) / TILE_SIZE) !== tile.tx || Math.floor((end[2] - LEVEL.origin[2]) / TILE_SIZE) !== tile.tz) seamsCrossed += 1;
      // It went somewhere (or was stopped by a steep face).
      expect(Math.hypot(end[0] - start[0], end[2] - start[2])).toBeGreaterThan(5);
      expect(worstSink, `dir ${dir}`).toBeLessThan(0.05);
    }
    expect(seamsCrossed).toBeGreaterThanOrEqual(3);
  });

  it("survives long frames without tunnelling", () => {
    const tile = hilliest();
    const ground = patchGround(tile);
    // Start just inside the tile's west edge and run west across the seam.
    const cx = tile.x0 + 2, cz = tile.z0 + TILE_SIZE / 2;
    const { scene, ctl } = setup(tile, [cx, ground(cx, cz) + 1, cz]);
    for (let f = 0; f < 80; f++) {
      ctl.update(0.1, [-1, 0, 0], false);
      const p = scene.get("player" as EntityId)!.components[Tag.Transform]!.position;
      expect(p[1] - 0.9).toBeGreaterThan(ground(p[0], p[2]) - 0.05);
    }
    expect(scene.get("player" as EntityId)!.components[Tag.Transform]!.position[0]).toBeLessThan(tile.x0);
  });
});

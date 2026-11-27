/**
 * Tile-manager unit tests — LRU eviction order, memory accounting, and
 * load-priority math. No GPU / no fetch — uses an injected stub fetcher.
 */

import { describe, it, expect } from "vitest";
import {
  TileManager,
  buildUniformTileGrid,
  type TileFetcher,
  type TileMeta,
  type TileLoadResult,
} from "./tile-manager.js";
import type { EntityId, Vec3 } from "../../scene/types.js";

function makeStubFetcher(): TileFetcher {
  return {
    async fetch(meta: TileMeta): Promise<TileLoadResult> {
      return {
        entities: [{ id: `e_${meta.coord.x}_${meta.coord.z}` as EntityId, payload: meta }],
        residentBytes: meta.payloadBytes,
      };
    },
  };
}

function makeFailingFetcher(): TileFetcher {
  return {
    async fetch(): Promise<TileLoadResult> {
      throw new Error("simulated fetch failure");
    },
  };
}

describe("tile-manager / grid construction", () => {
  it("builds an N×N grid with centered tiles", () => {
    const grid = buildUniformTileGrid(
      [0, 0, 0],
      { x: 1024, z: 1024 },
      16,
      4096,
      (c) => `tile_${c.x}_${c.z}.glb`,
    );
    expect(grid).toHaveLength(256);
    // First tile center is half a tile inset from origin.
    expect(grid[0]!.centerWorld).toEqual([32, 0, 32]);
    // Last tile center.
    const last = grid[255]!;
    expect(last.centerWorld[0]).toBeCloseTo(1024 - 32);
    expect(last.centerWorld[2]).toBeCloseTo(1024 - 32);
    expect(last.payloadKey).toBe("tile_15_15.glb");
  });
});

describe("tile-manager / LRU eviction", () => {
  it("evicts least-recently-used resident tiles when over budget", async () => {
    const fetcher = makeStubFetcher();
    const grid = buildUniformTileGrid([0, 0, 0], { x: 256, z: 256 }, 4, 1000, (c) => `${c.x}_${c.z}`);
    const mgr = new TileManager({
      worldOrigin: [0, 0, 0],
      worldExtent: { x: 256, z: 256 },
      gridSize: 4,
      memoryBudgetBytes: 3000, // fits exactly 3 tiles
      loadConcurrency: 16,
      loadRadiusTiles: 1, // tight — anything past 1 tile width is "far"
      fetcher,
    });
    mgr.registerTiles(grid);
    // Force-load 4 tiles in distinct frames so LRU order is deterministic.
    mgr.forceLoad({ x: 0, z: 0 }, 1);
    mgr.forceLoad({ x: 1, z: 0 }, 2);
    mgr.forceLoad({ x: 2, z: 0 }, 3);
    mgr.forceLoad({ x: 3, z: 0 }, 4);
    await new Promise((r) => setTimeout(r, 0));
    expect(mgr.resident).toHaveLength(4);
    expect(mgr.residentBytesNow).toBe(4000);
    // Drive update at frame 5 from way outside the load radius — no
    // resident tile gets its lastUsedFrame refreshed, so LRU = (0,0)
    // is the oldest and evicts first.
    mgr.update({
      cameraPos: [10000, 0, 10000],
      cameraForward: [1, 0, 0],
      frameIndex: 5,
    });
    expect(mgr.residentBytesNow).toBeLessThanOrEqual(3000);
    const residentCoords = mgr.resident.map((t) => `${t.coord.x},${t.coord.z}`).sort();
    expect(residentCoords).not.toContain("0,0"); // evicted first
  });

  it("never evicts tiles touched in the current frame", async () => {
    const fetcher = makeStubFetcher();
    const mgr = new TileManager({
      worldOrigin: [0, 0, 0],
      worldExtent: { x: 256, z: 256 },
      gridSize: 4,
      memoryBudgetBytes: 1000, // only fits 1
      loadConcurrency: 16,
      loadRadiusTiles: 8,
      fetcher,
    });
    const grid = buildUniformTileGrid([0, 0, 0], { x: 256, z: 256 }, 4, 1000, (c) => `${c.x}_${c.z}`);
    mgr.registerTiles(grid);
    mgr.forceLoad({ x: 0, z: 0 }, 1);
    await new Promise((r) => setTimeout(r, 0));
    // Park the camera right on top of tile (0,0). Even though we're over
    // budget, the tile is the only candidate and is touched this frame.
    mgr.update({ cameraPos: [32, 0, 32], cameraForward: [1, 0, 0], frameIndex: 2 });
    expect(mgr.resident.some((t) => t.coord.x === 0 && t.coord.z === 0)).toBe(true);
  });
});

describe("tile-manager / load failure", () => {
  it("resets a tile to unloaded + records loadError on fetch rejection", async () => {
    const mgr = new TileManager({
      worldOrigin: [0, 0, 0],
      worldExtent: { x: 256, z: 256 },
      gridSize: 4,
      memoryBudgetBytes: 100000,
      loadConcurrency: 4,
      loadRadiusTiles: 8,
      fetcher: makeFailingFetcher(),
    });
    const grid = buildUniformTileGrid([0, 0, 0], { x: 256, z: 256 }, 4, 1000, (c) => `${c.x}_${c.z}`);
    mgr.registerTiles(grid);
    mgr.forceLoad({ x: 0, z: 0 }, 1);
    await new Promise((r) => setTimeout(r, 0));
    const tile = mgr.resident;
    expect(tile).toHaveLength(0); // no residents on failure
    // Re-scan all known tiles for the error stamp.
    const grid2 = (mgr as unknown as { tiles: Map<string, { state: string; loadError: string | null }> }).tiles;
    const t = grid2.get("0|0")!;
    expect(t.state).toBe("unloaded");
    expect(t.loadError).toContain("simulated");
  });
});

describe("tile-manager / load priority", () => {
  it("prefers tiles ahead of the camera over tiles behind", async () => {
    const order: string[] = [];
    const fetcher: TileFetcher = {
      async fetch(meta: TileMeta): Promise<TileLoadResult> {
        order.push(`${meta.coord.x},${meta.coord.z}`);
        return { entities: [], residentBytes: meta.payloadBytes };
      },
    };
    const mgr = new TileManager({
      worldOrigin: [0, 0, 0],
      worldExtent: { x: 256, z: 256 },
      gridSize: 4,
      memoryBudgetBytes: 1_000_000,
      loadConcurrency: 1, // serialize so order is deterministic
      loadRadiusTiles: 8,
      fetcher,
    });
    const grid = buildUniformTileGrid([0, 0, 0], { x: 256, z: 256 }, 4, 1000, (c) => `${c.x}_${c.z}`);
    mgr.registerTiles(grid);
    // Place the camera in the center facing +x. Tiles to the +x side
    // should kick off before tiles to the -x side.
    const cameraPos: Vec3 = [128, 0, 128];
    const cameraForward: Vec3 = [1, 0, 0];
    // Tick once — only one load fires due to concurrency=1.
    mgr.update({ cameraPos, cameraForward, frameIndex: 1 });
    await new Promise((r) => setTimeout(r, 0));
    expect(order).toHaveLength(1);
    const [x, z] = order[0]!.split(",").map(Number);
    // The first load should be the tile straddling the camera or directly
    // ahead — i.e. centered on x=160 (tile column 2).
    expect(x).toBeGreaterThanOrEqual(1);
    // Should be near vertical center.
    expect(Math.abs((z ?? 0) - 1.5)).toBeLessThanOrEqual(2);
  });
});

describe("tile-manager / forward reach", () => {
  it("loads tiles ahead from further out than tiles behind, by forwardRadiusBonusTiles", async () => {
    const grid = buildUniformTileGrid([0, 0, 0], { x: 1024, z: 1024 }, 16, 1000, (c) => `${c.x}_${c.z}`);
    const run = async (bonus: number) => {
      const mgr = new TileManager({
        worldOrigin: [0, 0, 0],
        worldExtent: { x: 1024, z: 1024 },
        gridSize: 16,
        memoryBudgetBytes: 1e9,
        loadConcurrency: 64,
        loadRadiusTiles: 1.75,
        forwardRadiusBonusTiles: bonus,
        fetcher: makeStubFetcher(),
      });
      mgr.registerTiles(grid);
      mgr.update({ cameraPos: [512, 0, 512], cameraForward: [1, 0, 0], frameIndex: 1 });
      await new Promise((r) => setTimeout(r, 0));
      const xs = mgr.resident.map((t) => t.centerWorld[0] - 512);
      mgr.destroy();
      return { ahead: Math.max(...xs), behind: Math.min(...xs) };
    };
    const plain = await run(0);
    const boosted = await run(1);
    // Without the bonus the reach is symmetric (~112 m).
    expect(plain.ahead).toBe(-plain.behind);
    // With it, a tile a full extra width ahead is loaded; behind is unchanged.
    expect(boosted.ahead).toBeGreaterThanOrEqual(plain.ahead + 64);
    expect(boosted.behind).toBe(plain.behind);
  });
});

import { describe, expect, it } from "vitest";
import type { Vec3 } from "../../scene/types.js";
import { buildTileContent, LEVEL, TILE_SIZE } from "./level-content.js";
import {
  bakeBoxes,
  bakeGround,
  bakeProbe,
  bakeTileLayer,
  bakeTileProbes,
  evalAmbientL1,
  openGroundReference,
  PROBE_FLOATS,
  PROBES,
  PROBES_PER_TILE,
  probeDirections,
  projectAmbientL1,
  type ProbeBox,
  type ProbeLighting,
} from "./irradiance-probes.js";
import { ProbeRing, sampleProbeRing } from "../live/webgpu/probe-ring.js";

const LIGHTING: ProbeLighting = {
  lights: [
    { dir: [0.408, -0.816, -0.408], ratio: [11.4, 9.9, 8.2] },
    { dir: [-0.873, -0.436, 0.218], ratio: [2.2, 2.5, 3.1] },
  ],
};

describe("L1 projection", () => {
  const dirs = probeDirections(256);

  it("a constant radiance projects to a constant ambient equal to it", () => {
    const rad = new Float32Array(dirs.length).fill(0.7);
    const k = new Float32Array(12);
    projectAmbientL1(dirs, rad, k, 0);
    for (const n of [[0, 1, 0], [1, 0, 0], [0, -1, 0], [0.6, 0, 0.8]] as Vec3[]) {
      const a = evalAmbientL1(k, 0, n);
      expect(a[0]).toBeCloseTo(0.7, 2);
      expect(a[2]).toBeCloseTo(0.7, 2);
    }
  });

  it("radiance L = d·y gives the cosine-convolved irradiance (2/3)·n.y", () => {
    // E(n)/π for L(ω) = ω_y is exactly (2/3) n_y (only the L1 band, A1/π = 2/3).
    const rad = new Float32Array(dirs.length);
    for (let i = 0; i < dirs.length / 3; i++) rad[i * 3] = rad[i * 3 + 1] = rad[i * 3 + 2] = dirs[i * 3 + 1]!;
    const k = new Float32Array(12);
    projectAmbientL1(dirs, rad, k, 0);
    expect(evalAmbientL1(k, 0, [0, 1, 0])[1]).toBeCloseTo(2 / 3, 2);
    expect(evalAmbientL1(k, 0, [0, -1, 0])[1]).toBeCloseTo(-2 / 3, 2);
    expect(evalAmbientL1(k, 0, [1, 0, 0])[1]).toBeCloseTo(0, 2);
  });
});

describe("probe bake", () => {
  it("an open-ground probe averages exactly 1 after normalisation, brighter facing the sky", () => {
    const ref = openGroundReference(LIGHTING);
    const k = new Float32Array(12);
    bakeProbe({ boxes: [], lighting: LIGHTING, ground: () => 0 }, [0, PROBES.heightAboveGround, LEVEL.roadZ + 40], k, 0);
    for (let c = 0; c < 3; c++) expect(k[c * 4]! / ref[c]!).toBeCloseTo(1, 5);
    const up = evalAmbientL1(k, 0, [0, 1, 0]);
    const down = evalAmbientL1(k, 0, [0, -1, 0]);
    expect(up[2]).toBeGreaterThan(down[2]);
  });

  it("a wall on one side occludes the sky and bounces its own colour", () => {
    // A red wall 3 m east of the probe; the sun travels +x, so the wall face the probe sees is sunlit.
    const lighting: ProbeLighting = { lights: [{ dir: [0.6, -0.6, 0], ratio: [10, 10, 10] }] };
    const wall: ProbeBox = { min: [3, 0, -20], max: [4, 20, 20], albedo: [0.8, 0.1, 0.1] };
    const open = new Float32Array(12), walled = new Float32Array(12);
    const env = { lighting, ground: () => 0 };
    bakeProbe({ ...env, boxes: [] }, [0, 1.5, 0], open, 0);
    bakeProbe({ ...env, boxes: [wall] }, [0, 1.5, 0], walled, 0);
    const eastOpen = evalAmbientL1(open, 0, [1, 0, 0]);
    const eastWalled = evalAmbientL1(walled, 0, [1, 0, 0]);
    // Facing the wall: less blue (sky blocked), relatively more red (bounce).
    expect(eastWalled[2]).toBeLessThan(eastOpen[2]);
    expect(eastWalled[0] / eastWalled[2]).toBeGreaterThan(eastOpen[0] / eastOpen[2]);
  });

  it("a probe inside a building is lifted above its roof instead of baking black", () => {
    const roof: ProbeBox = { min: [-10, 0, -10], max: [10, 6, 10], albedo: [0.4, 0.4, 0.4] };
    const k = new Float32Array(PROBES_PER_TILE * PROBE_FLOATS);
    // Use the tile-layer path so the lift runs: tile (0,0)'s first probe sits at (8, ·, 8).
    const t = bakeTileProbes(0, 0, { boxes: [{ ...roof, min: [0, 0, 0], max: [16, 6, 16] }], lighting: LIGHTING, ground: () => 0 });
    k.set(t.probes);
    expect(k[0]).toBeGreaterThan(0.3); // not black
    expect(k[15]).toBe(1);
  });

  it("writes meta (global column, ground height, valid) for every probe of a real tile; one row fits an idle task", () => {
    const contentAt = (x: number, z: number) =>
      x >= 0 && z >= 0 && x < LEVEL.gridSize && z < LEVEL.gridSize ? buildTileContent(x, z) : undefined;
    const boxes = bakeBoxes(3, 8, contentAt);
    const env = { boxes, lighting: LIGHTING, ground: bakeGround(3, 8) };
    bakeTileProbes(3, 8, env); // warm-up (JIT)
    const tile = bakeTileProbes(3, 8, env);
    expect(tile.probes.length).toBe(PROBES_PER_TILE * PROBE_FLOATS);
    for (let k = 0; k < PROBES.layers; k++) {
      for (let b = 0; b < 4; b++) {
        for (let a = 0; a < 4; a++) {
          const o = ((k * 4 + b) * 4 + a) * PROBE_FLOATS;
          expect(tile.probes[o + 12]).toBe(3 * 4 + a);
          expect(tile.probes[o + 13]).toBe(8 * 4 + b);
          expect(tile.probes[o + 15]).toBe(1);
          for (let q = 0; q < 12; q++) expect(Number.isFinite(tile.probes[o + q]!)).toBe(true);
        }
      }
    }
    // StreamExtras bakes one row of four probes per idle task.
    const rows: number[] = [];
    const out = new Float32Array(PROBES_PER_TILE * PROBE_FLOATS);
    for (let k = 0; k < PROBES.layers; k++) {
      for (let r = 0; r < 4; r++) {
        const t0 = performance.now();
        bakeTileLayer(3, 8, k, env, out, [r, r + 1]);
        rows.push(performance.now() - t0);
      }
    }
    rows.sort((x, y) => x - y);
    // eslint-disable-next-line no-console
    console.info(`[probes] tile 3,8 (${boxes.length} boxes), warm row tasks ms: median ${rows[4]!.toFixed(1)}, max ${rows[7]!.toFixed(1)}`);
    expect(rows[4]!).toBeLessThan(25); // generous: shared CI / laptop under load
  });
});

describe("probe ring + sampler", () => {
  const grid = { spacing: PROBES.spacing, heightAboveGround: PROBES.heightAboveGround, layerSpacing: PROBES.layerSpacing };
  const contentAt = (x: number, z: number) =>
    x >= 0 && z >= 0 && x < LEVEL.gridSize && z < LEVEL.gridSize ? buildTileContent(x, z) : undefined;

  it("blends across a tile seam with no step", () => {
    const ring = new ProbeRing();
    for (const tx of [2, 3]) ring.setTile(bakeTileProbes(tx, 8, { boxes: bakeBoxes(tx, 8, contentAt), lighting: LIGHTING }));
    const seamX = 3 * TILE_SIZE;
    const n: Vec3 = [0, 1, 0];
    const z = 8 * TILE_SIZE + 20;
    const left = sampleProbeRing(ring.data, [seamX - 0.01, 2, z], n, grid);
    const right = sampleProbeRing(ring.data, [seamX + 0.01, 2, z], n, grid);
    expect(left.coverage).toBe(1);
    expect(right.coverage).toBe(1);
    for (let c = 0; c < 3; c++) expect(Math.abs(left.rgb[c]! - right.rgb[c]!)).toBeLessThan(0.01);
    // And it is continuous along the whole crossing, not just at the line.
    let maxStep = 0;
    let prev = sampleProbeRing(ring.data, [seamX - 16, 2, z], n, grid).rgb;
    for (let x = seamX - 15.5; x <= seamX + 16; x += 0.5) {
      const cur = sampleProbeRing(ring.data, [x, 2, z], n, grid).rgb;
      maxStep = Math.max(maxStep, ...cur.map((v, c) => Math.abs(v - prev[c]!)));
      prev = cur;
    }
    expect(maxStep).toBeLessThan(0.08);
  });

  it("a cleared neighbour fades coverage instead of reading stale slots, and aliased tiles are rejected", () => {
    const ring = new ProbeRing();
    const a = bakeTileProbes(2, 8, { boxes: [], lighting: LIGHTING, ground: () => 0 });
    const b = bakeTileProbes(3, 8, { boxes: [], lighting: LIGHTING, ground: () => 0 });
    ring.setTile(a);
    ring.setTile(b);
    ring.clearTile(b.gi0, b.gj0, b.perAxis, b.layers);
    const inB = sampleProbeRing(ring.data, [3 * TILE_SIZE + 40, 2, 8 * TILE_SIZE + 20], [0, 1, 0], grid);
    expect(inB.coverage).toBe(0);
    // Tile 11 aliases tile 3 in a 32-column ring (8 tiles × 4 columns): its slots hold tile 11, so tile 3 reads nothing.
    ring.setTile(bakeTileProbes(11, 8, { boxes: [], lighting: LIGHTING, ground: () => 0 }));
    expect(sampleProbeRing(ring.data, [3 * TILE_SIZE + 40, 2, 8 * TILE_SIZE + 20], [0, 1, 0], grid).coverage).toBe(0);
    expect(sampleProbeRing(ring.data, [11 * TILE_SIZE + 40, 2, 8 * TILE_SIZE + 20], [0, 1, 0], grid).coverage).toBe(1);
    // Clearing tile 3 now must not wipe tile 11's slots.
    expect(ring.clearTile(b.gi0, b.gj0, b.perAxis, b.layers)).toEqual([]);
  });

  it("follows the ground: a point 1.5 m over a hill reads layer 0", () => {
    const ring = new ProbeRing();
    const t = bakeTileProbes(5, 5, { boxes: [], lighting: LIGHTING, ground: () => 10 });
    // Make layer 1 distinguishable.
    for (let i = 16; i < 32; i++) t.probes[i * PROBE_FLOATS] = 5;
    ring.setTile(t);
    const p = sampleProbeRing(ring.data, [5 * TILE_SIZE + 24, 10 + PROBES.heightAboveGround, 5 * TILE_SIZE + 24], [0, 0, 0], grid);
    expect(p.rgb[0]).toBeLessThan(2);
    const high = sampleProbeRing(ring.data, [5 * TILE_SIZE + 24, 10 + PROBES.heightAboveGround + PROBES.layerSpacing, 5 * TILE_SIZE + 24], [0, 0, 0], grid);
    expect(high.rgb[0]).toBeCloseTo(5, 3);
  });
});

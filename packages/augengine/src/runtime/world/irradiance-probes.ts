/**
 * world/irradiance-probes — baked per-tile ambient for The Loading Range.
 *
 * Every 64 m tile carries a 4 × 4 × 2 grid of irradiance probes (16 m apart,
 * two layers: 1.5 m and 13.5 m above the ground under each column). Each
 * probe stores L0 + L1 spherical harmonics of the light arriving at it, per
 * colour channel, already convolved with the cosine lobe and divided by π —
 * the same "albedo × ambient" quantity the material shaders use — and
 * expressed RELATIVE to the scene's flat ambient: an unoccluded probe on open
 * ground averages 1. The renderer multiplies by the frame ambient, so the
 * scene keeps the brightness it was tuned at and the probes add direction,
 * occlusion and colour bleeding.
 *
 * Baked on the CPU while a tile loads (StreamExtras.prepare: one idle task for
 * the box list + height grid, then one per row of four probes), streamed and
 * released with the tile. The approximation, honestly:
 *
 *  - 64 fixed directions per probe (spherical Fibonacci), one ray each,
 *    projected with equal weights 4π/64. No importance sampling, no noise
 *    between probes (the directions are identical), so it is stable but
 *    band-limited — L1 cannot hold sharp contrast anyway.
 *  - Geometry is the tile's and its 8 neighbours' props as axis-aligned boxes
 *    (buildings at their visual size, trees as a canopy box, domes as a box
 *    inside the dome, thin street furniture skipped; only boxes within 64 m
 *    of the probe) plus the terrain height, read from an 8 m grid over the 3 × 3
 *    tiles (bilinear) and marched in 12 steps up to 64 m, refined by bisection.
 *  - A ray that escapes sees a neutral sky dome (zenith 1, horizon 0.8 —
 *    the same dome ibl.ts uses when no sky gradient is set).
 *  - A ray that hits a surface returns albedo × (1 + Σ ratio·max(n·L, 0)/π ×
 *    sun visibility): the surface's own ambient is assumed unoccluded, and
 *    sun visibility is ONE shadow ray against the boxes (terrain does not
 *    shadow terrain). One bounce only.
 *  - Albedos are guesses per prop kind (the kit's colours live in vertex
 *    colours the bake does not read).
 *  - A probe that lands inside a box is lifted to 0.5 m above its roof.
 *
 * Moving lights or a time-of-day change would need a rebake: probes are tied
 * to the ProbeLighting they were baked with.
 */
import type { Vec3 } from "../../scene/types.js";
import { LEVEL, TILE_SIZE, type PropSpec, type TileContent } from "./level-content.js";
import { groundUnder, terrainHeight } from "./terrain.js";

export const PROBES = {
  perTileAxis: 4,
  /** Metres between probe columns (TILE_SIZE / perTileAxis). */
  spacing: TILE_SIZE / 4,
  layers: 2,
  /** Layer 0 height above the ground under the column; layer k adds k × layerSpacing. */
  heightAboveGround: 1.5,
  layerSpacing: 12,
  directions: 64,
  maxDistance: 64,
  terrainSteps: 12,
  /** Terrain height grid used by the bake (metres between samples). */
  heightCell: 8,
} as const;

/** Floats per probe: r (k0, kx, ky, kz), g (…), b (…), meta (gi, gj, groundY, valid). */
export const PROBE_FLOATS = 16;
export const PROBES_PER_TILE = PROBES.perTileAxis * PROBES.perTileAxis * PROBES.layers;

export interface ProbeLight {
  /** Direction the light travels (from the light toward the scene). */
  dir: Vec3;
  /** Light radiance divided by the scene's flat ambient, per channel. */
  ratio: Vec3;
}

export interface ProbeLighting {
  /** Directional lights; the first is treated as the sun (shadow ray). */
  lights: ProbeLight[];
}

/** One tile's probes, ready for the renderer (ProbeSink.setTile). */
export interface ProbeTileData {
  tx: number;
  tz: number;
  /** Global probe column index of the tile's first column (tx × perTileAxis). */
  gi0: number;
  gj0: number;
  perAxis: number;
  layers: number;
  /** PROBES_PER_TILE × PROBE_FLOATS, ordered (layer, b, a): index = (k·perAxis + b)·perAxis + a. */
  probes: Float32Array;
}

export interface ProbeSink {
  setTile(tile: ProbeTileData): void;
  clearTile(tx: number, tz: number): void;
}

/** An occluding / bouncing box. */
export interface ProbeBox {
  min: Vec3;
  max: Vec3;
  albedo: Vec3;
}

const Y00 = 0.28209479177387814;
const Y1 = 0.4886025119029199;
const TERRAIN_ALBEDO: Vec3 = [0.15, 0.21, 0.12];
const ROAD_ALBEDO: Vec3 = [0.07, 0.07, 0.08];
const KIT_BUILDING_ALBEDO: Vec3 = [0.42, 0.4, 0.37];
const CANOPY_ALBEDO: Vec3 = [0.12, 0.2, 0.08];

/** Spherical Fibonacci directions (unit vectors, flat [x, y, z]*n). */
export function probeDirections(n: number = PROBES.directions): Float32Array {
  const out = new Float32Array(n * 3);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const a = i * golden;
    out[i * 3] = Math.cos(a) * r;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = Math.sin(a) * r;
  }
  return out;
}

const DIRS = probeDirections();

/**
 * Project radiance samples along `dirs` onto L0 + L1 and convolve with the
 * clamped cosine, divided by π: ambient(n) = k0 + kx·n.x + ky·n.y + kz·n.z.
 * `radiance` is flat rgb per direction. Writes 12 floats (r4, g4, b4) at `offset`.
 */
export function projectAmbientL1(dirs: Float32Array, radiance: Float32Array, out: Float32Array, offset: number): void {
  const n = dirs.length / 3;
  const w = (4 * Math.PI) / n;
  for (let c = 0; c < 3; c++) {
    let c0 = 0, cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      const L = radiance[i * 3 + c]!;
      c0 += L * Y00 * w;
      cx += L * Y1 * dirs[i * 3]! * w;
      cy += L * Y1 * dirs[i * 3 + 1]! * w;
      cz += L * Y1 * dirs[i * 3 + 2]! * w;
    }
    // A0/π = 1, A1/π = 2/3; Y_1m(n) = Y1 · n_axis.
    const o = offset + c * 4;
    out[o] = c0 * Y00;
    out[o + 1] = (2 / 3) * Y1 * cx;
    out[o + 2] = (2 / 3) * Y1 * cy;
    out[o + 3] = (2 / 3) * Y1 * cz;
  }
}

/** Evaluate a packed L1 ambient (12 floats at offset) for normal n. */
export function evalAmbientL1(k: Float32Array, offset: number, n: Vec3): Vec3 {
  const e = (c: number) => {
    const o = offset + c * 4;
    return k[o]! + k[o + 1]! * n[0] + k[o + 2]! * n[1] + k[o + 3]! * n[2];
  };
  return [e(0), e(1), e(2)];
}

function skyDome(y: number): number {
  // Neutral dome (ibl.ts default): horizon 0.8 → zenith 1 above the horizon.
  return 0.8 + 0.2 * Math.max(0, Math.min(1, y));
}

/** Boxes a prop contributes to the bake (none for thin street furniture). */
export function propBoxes(p: PropSpec, lift = 0): ProbeBox[] {
  const cx = p.center[0], cy = p.center[1] + lift, cz = p.center[2];
  const box = (hx: number, hy: number, hz: number, oy: number, albedo: Vec3): ProbeBox => ({
    min: [cx - hx, cy + oy - hy, cz - hz],
    max: [cx + hx, cy + oy + hy, cz + hz],
    albedo,
  });
  const [w, h, d] = p.size;
  if (p.shape === "kit") {
    const k = p.key.split(".")[0];
    if (k === "lamp" || k === "bench" || k === "barrier") return [];
    if (k === "tree") return [box(w * 0.35, h * 0.3, d * 0.35, h * 0.15, CANOPY_ALBEDO)];
    if (k === "tower") return [box(w * 0.35, h * 0.2, d * 0.35, h * 0.3, [0.6, 0.6, 0.62])];
    return [box(w / 2, h / 2, d / 2, 0, KIT_BUILDING_ALBEDO)];
  }
  if (p.shape === "dome") return [box(w * 0.35, h * 0.3, d * 0.35, 0, p.color)];
  if (p.shape === "pillar" && w < 1.5) return [];
  return [box(w / 2, h / 2, d / 2, 0, p.color)];
}

/**
 * Boxes for baking tile (tx, tz): its own props and its 8 neighbours', each
 * lifted onto the terrain the way StreamExtras.liftProp places them.
 */
export function bakeBoxes(tx: number, tz: number, contentAt: (x: number, z: number) => TileContent | undefined, onTerrain = true): ProbeBox[] {
  const out: ProbeBox[] = [];
  for (let dz = -1; dz <= 1; dz++) {
    for (let dx = -1; dx <= 1; dx++) {
      const c = contentAt(tx + dx, tz + dz);
      if (!c) continue;
      for (const p of c.props) {
        const lift = onTerrain
          ? groundUnder(p.center[0] - p.size[0] / 2, p.center[0] + p.size[0] / 2, p.center[2] - p.size[2] / 2, p.center[2] + p.size[2] / 2)
          : 0;
        out.push(...propBoxes(p, lift));
      }
    }
  }
  return out;
}

/** Nearest box hit along o + t·d for t in (0, tMax): returns t and the face normal axis/sign. */
function hitBoxes(boxes: readonly ProbeBox[], o: Vec3, d: Vec3, tMax: number): { t: number; box: ProbeBox; axis: number; sign: number } | null {
  let best: { t: number; box: ProbeBox; axis: number; sign: number } | null = null;
  let bestT = tMax;
  for (const b of boxes) {
    let t0 = 0, t1 = bestT, axis = -1, sign = 0;
    let miss = false;
    for (let a = 0; a < 3; a++) {
      const da = d[a]!, oa = o[a]!;
      if (Math.abs(da) < 1e-9) {
        if (oa < b.min[a]! || oa > b.max[a]!) { miss = true; break; }
        continue;
      }
      let ta = (b.min[a]! - oa) / da, tb = (b.max[a]! - oa) / da;
      let s = -1;
      if (ta > tb) { const tmp = ta; ta = tb; tb = tmp; s = 1; }
      if (ta > t0) { t0 = ta; axis = a; sign = s; }
      if (tb < t1) t1 = tb;
      if (t0 > t1) { miss = true; break; }
    }
    if (miss || axis < 0 || t0 <= 1e-4 || t0 >= bestT) continue;
    bestT = t0;
    best = { t: t0, box: b, axis, sign };
  }
  return best;
}

function insideBox(boxes: readonly ProbeBox[], p: Vec3): ProbeBox | null {
  for (const b of boxes) {
    if (p[0] > b.min[0] && p[0] < b.max[0] && p[1] > b.min[1] && p[1] < b.max[1] && p[2] > b.min[2] && p[2] < b.max[2]) return b;
  }
  return null;
}

/**
 * Terrain height over tile (tx, tz) and its 8 neighbours from a
 * `PROBES.heightCell` grid (bilinear), the analytic height outside it. The
 * bake asks for heights thousands of times; the analytic function is ~1.3 µs
 * a call on a laptop, the grid ~0.05.
 */
export function bakeGround(tx: number, tz: number, height: (x: number, z: number) => number = terrainHeight): (x: number, z: number) => number {
  const cell = PROBES.heightCell;
  const x0 = (tx - 1) * TILE_SIZE, z0 = (tz - 1) * TILE_SIZE;
  const n = Math.round((3 * TILE_SIZE) / cell) + 1;
  const h = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) h[j * n + i] = height(x0 + i * cell, z0 + j * cell);
  return (x, z) => {
    const fx = (x - x0) / cell, fz = (z - z0) / cell;
    if (fx < 0 || fz < 0 || fx >= n - 1 || fz >= n - 1) return height(x, z);
    const i = Math.floor(fx), j = Math.floor(fz);
    const u = fx - i, v = fz - j;
    const o = j * n + i;
    return (h[o]! * (1 - u) + h[o + 1]! * u) * (1 - v) + (h[o + n]! * (1 - u) + h[o + n + 1]! * u) * v;
  };
}

export interface BakeEnv {
  boxes: readonly ProbeBox[];
  lighting: ProbeLighting;
  /** Ground height at (x, z). Default: the analytic terrain. */
  ground?: (x: number, z: number) => number;
}

function groundAlbedo(z: number): Vec3 {
  return Math.abs(z - LEVEL.roadZ) < LEVEL.roadHalfWidth ? ROAD_ALBEDO : TERRAIN_ALBEDO;
}

/** Radiance of a lit surface hit at p with normal n (relative to the flat ambient). */
function surfaceRadiance(env: BakeEnv, p: Vec3, n: Vec3, albedo: Vec3, out: Float32Array, o: number): void {
  let r = 1, g = 1, b = 1;
  const lights = env.lighting.lights;
  for (let li = 0; li < lights.length; li++) {
    const L = lights[li]!;
    const toL: Vec3 = [-L.dir[0], -L.dir[1], -L.dir[2]];
    const ndl = n[0] * toL[0] + n[1] * toL[1] + n[2] * toL[2];
    if (ndl <= 0) continue;
    let vis = 1;
    if (li === 0) {
      const so: Vec3 = [p[0] + n[0] * 0.05, p[1] + n[1] * 0.05, p[2] + n[2] * 0.05];
      if (hitBoxes(env.boxes, so, toL, PROBES.maxDistance)) vis = 0;
    }
    const k = (ndl * vis) / Math.PI;
    r += L.ratio[0] * k; g += L.ratio[1] * k; b += L.ratio[2] * k;
  }
  out[o] = albedo[0] * r; out[o + 1] = albedo[1] * g; out[o + 2] = albedo[2] * b;
}

/** Trace one probe: radiance per direction, then L1-project into out[offset..offset+12). */
export function bakeProbe(envIn: BakeEnv, pos: Vec3, out: Float32Array, offset: number, scratch?: Float32Array): void {
  const ground = envIn.ground ?? terrainHeight;
  // Only boxes the rays (or a hit's shadow ray) can reach.
  const reach = PROBES.maxDistance * 1.5;
  const near = envIn.boxes.filter((b) =>
    Math.max(b.min[0] - pos[0], pos[0] - b.max[0], 0) < reach && Math.max(b.min[2] - pos[2], pos[2] - b.max[2], 0) < reach);
  const env: BakeEnv = { ...envIn, boxes: near };
  const n = DIRS.length / 3;
  const rad = scratch && scratch.length >= n * 3 ? scratch : new Float32Array(n * 3);
  const maxD = PROBES.maxDistance;
  for (let i = 0; i < n; i++) {
    const d: Vec3 = [DIRS[i * 3]!, DIRS[i * 3 + 1]!, DIRS[i * 3 + 2]!];
    // Terrain: march, then bisect the first step that went under the ground.
    let tGround = Infinity;
    if (d[1] < -1e-3) {
      const step = maxD / PROBES.terrainSteps;
      let prev = 0;
      for (let s = 1; s <= PROBES.terrainSteps; s++) {
        const t = s * step;
        const x = pos[0] + d[0] * t, y = pos[1] + d[1] * t, z = pos[2] + d[2] * t;
        if (y < ground(x, z)) {
          let lo = prev, hi = t;
          for (let it = 0; it < 4; it++) {
            const m = (lo + hi) / 2;
            if (pos[1] + d[1] * m < ground(pos[0] + d[0] * m, pos[2] + d[2] * m)) hi = m; else lo = m;
          }
          tGround = hi;
          break;
        }
        prev = t;
      }
      // Past the march on flat-ish ground: the ray still lands somewhere.
      if (!Number.isFinite(tGround)) tGround = Math.min(maxD * 4, (pos[1] - ground(pos[0], pos[2]) + 0.01) / -d[1]);
    }
    const hb = hitBoxes(env.boxes, pos, d, Math.min(maxD, tGround));
    if (hb) {
      const p: Vec3 = [pos[0] + d[0] * hb.t, pos[1] + d[1] * hb.t, pos[2] + d[2] * hb.t];
      const nrm: [number, number, number] = [0, 0, 0];
      nrm[hb.axis] = hb.sign;
      surfaceRadiance(env, p, nrm, hb.box.albedo, rad, i * 3);
    } else if (Number.isFinite(tGround)) {
      const p: Vec3 = [pos[0] + d[0] * tGround, pos[1] + d[1] * tGround, pos[2] + d[2] * tGround];
      const e = 0.5;
      const hx = ground(p[0] + e, p[2]) - ground(p[0] - e, p[2]);
      const hz = ground(p[0], p[2] + e) - ground(p[0], p[2] - e);
      const len = Math.hypot(hx, 2 * e, hz);
      surfaceRadiance(env, p, [-hx / len, (2 * e) / len, -hz / len], groundAlbedo(p[2]), rad, i * 3);
    } else {
      const s = skyDome(d[1]);
      rad[i * 3] = s; rad[i * 3 + 1] = s; rad[i * 3 + 2] = s;
    }
  }
  projectAmbientL1(DIRS, rad, out, offset);
}

const openCache = new Map<string, Float32Array>();

/**
 * L0 of an unoccluded probe 1.5 m over open flat ground under `lighting` —
 * the normalisation that makes open ground average exactly 1 per channel.
 */
export function openGroundReference(lighting: ProbeLighting): Vec3 {
  const key = JSON.stringify(lighting);
  let k = openCache.get(key);
  if (!k) {
    k = new Float32Array(12);
    bakeProbe({ boxes: [], lighting, ground: () => 0 }, [0, PROBES.heightAboveGround, LEVEL.roadZ + 40], k, 0);
    openCache.set(key, k);
  }
  return [k[0]!, k[4]!, k[8]!];
}

/**
 * Bake one layer of tile (tx, tz)'s probes into `out` (a PROBES_PER_TILE ×
 * PROBE_FLOATS array; other layers untouched). Split by layer so each half
 * fits one idle task.
 */
export function bakeTileLayer(tx: number, tz: number, layer: number, env: BakeEnv, out: Float32Array, rows: [number, number] = [0, PROBES.perTileAxis]): void {
  const ground = env.ground ?? terrainHeight;
  const per = PROBES.perTileAxis;
  const ref = openGroundReference(env.lighting);
  const scratch = new Float32Array(DIRS.length);
  for (let b = rows[0]; b < rows[1]; b++) {
    for (let a = 0; a < per; a++) {
      const gi = tx * per + a, gj = tz * per + b;
      const x = (gi + 0.5) * PROBES.spacing, z = (gj + 0.5) * PROBES.spacing;
      const g = ground(x, z);
      const pos: [number, number, number] = [x, g + PROBES.heightAboveGround + layer * PROBES.layerSpacing, z];
      for (let guard = 0; guard < 4; guard++) {
        const inside = insideBox(env.boxes, pos);
        if (!inside) break;
        pos[1] = inside.max[1] + 0.5;
      }
      const o = ((layer * per + b) * per + a) * PROBE_FLOATS;
      bakeProbe(env, pos, out, o, scratch);
      for (let c = 0; c < 3; c++) {
        const s = ref[c]! > 1e-6 ? 1 / ref[c]! : 0;
        for (let q = 0; q < 4; q++) out[o + c * 4 + q]! *= s;
      }
      out[o + 12] = gi;
      out[o + 13] = gj;
      out[o + 14] = g;
      out[o + 15] = 1;
    }
  }
}

/** Bake every layer of a tile (tests, synchronous hosts). */
export function bakeTileProbes(tx: number, tz: number, envIn: BakeEnv): ProbeTileData {
  const probes = new Float32Array(PROBES_PER_TILE * PROBE_FLOATS);
  const env = envIn.ground ? envIn : { ...envIn, ground: bakeGround(tx, tz) };
  for (let k = 0; k < PROBES.layers; k++) bakeTileLayer(tx, tz, k, env, probes);
  return probeTile(tx, tz, probes);
}

export function probeTile(tx: number, tz: number, probes: Float32Array): ProbeTileData {
  return { tx, tz, gi0: tx * PROBES.perTileAxis, gj0: tz * PROBES.perTileAxis, perAxis: PROBES.perTileAxis, layers: PROBES.layers, probes };
}

/**
 * world/level-content — what each tile of The Loading Range contains.
 *
 * Pure and deterministic: the same tile coordinate always yields the same
 * props, so a tile that is evicted and streamed back in is identical, and
 * tests can assert on content without a renderer. Props carry their own
 * world-space collider extents (ColliderComponent convention) and the
 * detail parts that only draw at close range.
 */
import type { Vec3 } from "../../scene/types.js";

export const LEVEL = {
  origin: [0, 0, 0] as Vec3,
  /** 1024 x 1024 m, 16 x 16 tiles of 64 m. */
  extent: { x: 1024, z: 1024 },
  gridSize: 16,
  /** The walk route runs west to east along this z, through every landmark. */
  roadZ: 512,
  roadHalfWidth: 7,
} as const;

export const TILE_SIZE = LEVEL.extent.x / LEVEL.gridSize;

export type PropShape = "box" | "pillar" | "dome" | "kit";

/**
 * A baked art-kit asset the level may place instead of primitives (e.g.
 * `kitCatalog()` from @mnemopay/augengine-lod). Meshes are centred on their
 * bounds vertically and on their own footprint origin horizontally, drawn at
 * scale 1 via `{ kind: "procedural", generator: "kit-<id>", seed: <lod> }`.
 */
export interface KitCatalogEntry {
  id: string;
  kind: "building" | "lamp" | "bench" | "barrier" | "tree" | "water-tower";
  /** Full extents [x, height, z] in metres. */
  size: Vec3;
  metallic: number;
  roughness: number;
}

export type KitCatalog = readonly KitCatalogEntry[];

export interface PropDetail {
  /** Offset from the prop centre, world-space. */
  offset: Vec3;
  size: Vec3;
  color: Vec3;
  /** Highest LOD level at which this part still draws (0 = closest only). */
  maxLod: 0 | 1;
  emissive?: Vec3;
}

export interface PropSpec {
  /** Stable within the tile: `${kind}.${index}`. */
  key: string;
  shape: PropShape;
  /** World-space centre. */
  center: Vec3;
  /** Visual size (box: full size; pillar/dome: [diameter, height, diameter]). */
  size: Vec3;
  color: Vec3;
  emissive?: Vec3;
  /** World-space half extents of the box collider, or null for no collider. */
  collider: Vec3 | null;
  details: PropDetail[];
  landmark?: string;
  /** shape "kit": the catalog asset id. */
  kit?: string;
  /** Rotation about +Y in radians (colliders stay axis-aligned: kit props only use symmetric turns). */
  yaw?: number;
  metallic?: number;
  roughness?: number;
  /**
   * True when the collider box is solid and inside the visual (buildings), so
   * occlusion culling may use it to hide what is behind. Domes, open frames
   * (the water tower's legs), trees and street props are never occluders.
   */
  occluder?: boolean;
}

export interface TileContent {
  x: number;
  z: number;
  center: Vec3;
  props: PropSpec[];
  /** True when the road crosses this tile. */
  road: boolean;
}

/** Named landmarks on the route, keyed by tile coordinate. */
export const LANDMARKS: Readonly<Record<string, string>> = {
  "0|8": "West Gate",
  "4|8": "Water Tower",
  "8|8": "Signal Spire",
  "12|8": "Monolith Ring",
  "15|8": "East Beacon",
};

/** mulberry32 — tiny deterministic PRNG. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function tileCenter(x: number, z: number): Vec3 {
  return [LEVEL.origin[0] + (x + 0.5) * TILE_SIZE, 0, LEVEL.origin[2] + (z + 0.5) * TILE_SIZE];
}

export function roadCrossesTile(z: number): boolean {
  const z0 = LEVEL.origin[2] + z * TILE_SIZE;
  return LEVEL.roadZ + LEVEL.roadHalfWidth > z0 && LEVEL.roadZ - LEVEL.roadHalfWidth < z0 + TILE_SIZE;
}

const STONE: Vec3 = [0.46, 0.44, 0.4];
const RUST: Vec3 = [0.62, 0.34, 0.14];
const STEEL: Vec3 = [0.38, 0.42, 0.46];

/**
 * Content of tile (x, z). Without `kit` the level is built from primitives
 * (boxes, pillars, domes). With a kit, buildings come from the kit's
 * buildings, pillars and domes become trees, road tiles get street lamps,
 * benches and barriers, and the Water Tower landmark uses the kit tower.
 * Deterministic per (x, z, kit).
 */
export function buildTileContent(x: number, z: number, kit?: KitCatalog): TileContent {
  const center = tileCenter(x, z);
  const road = roadCrossesTile(z);
  const rand = rng((x * 73856093) ^ (z * 19349663) ^ 0x5eed);
  const props: PropSpec[] = [];
  const kitBy = kit && kit.length > 0 ? groupKit(kit) : null;
  const landmark = LANDMARKS[`${x}|${z}`];
  if (landmark) {
    const tower = kitBy?.get("water-tower")?.[0];
    if (landmark === "Water Tower" && tower) props.push(kitProp("tower", tower, [center[0], center[2]], LEVEL.roadZ - 16, { landmark }));
    else props.push(...landmarkProps(landmark, center));
  }
  if (kitBy && road) props.push(...streetFurniture(x, center, kitBy, rand, props));

  // Filler: buildings, pillars and domes, kept off the road and off landmarks.
  const count = landmark ? 3 : 6 + Math.floor(rand() * 5);
  let attempts = 0;
  while (props.filter((p) => !p.landmark && !STREET.has(p.key.split(".")[0]!)).length < count && attempts++ < 60) {
    const px = center[0] + (rand() - 0.5) * (TILE_SIZE - 10);
    const pz = center[2] + (rand() - 0.5) * (TILE_SIZE - 10);
    const pick = rand();
    const spec = kitBy
      ? pick < 0.55
        ? kitPick("building", props.length, px, pz, kitBy, rand)
        : kitPick("tree", props.length, px, pz, kitBy, rand)
      : pick < 0.55
        ? building(props.length, px, pz, rand)
        : pick < 0.85
          ? pillar(props.length, px, pz, rand)
          : dome(props.length, px, pz, rand);
    if (!spec) break;
    const r = Math.max(spec.size[0], spec.size[2]) / 2;
    if (road && Math.abs(pz - LEVEL.roadZ) < LEVEL.roadHalfWidth + r + 1) continue;
    if (props.some((p) => Math.hypot(p.center[0] - px, p.center[2] - pz) < r + Math.max(p.size[0], p.size[2]) / 2 + 2)) continue;
    props.push(spec);
  }
  return { x, z, center, props, road };
}

function building(i: number, x: number, z: number, rand: () => number): PropSpec {
  const w = 5 + rand() * 7;
  const d = 5 + rand() * 7;
  const h = 4 + rand() * 14;
  const color: Vec3 = rand() < 0.5 ? STONE : [0.3 + rand() * 0.1, 0.32, 0.35];
  return {
    key: `building.${i}`,
    shape: "box",
    center: [x, h / 2, z],
    size: [w, h, d],
    color,
    collider: [w / 2, h / 2, d / 2],
    occluder: true,
    details: [
      // Roof plant: visible to LOD1.
      { offset: [w * 0.15, h / 2 + 0.6, -d * 0.1], size: [w * 0.35, 1.2, d * 0.35], color: STEEL, maxLod: 1 },
      // Window band and an antenna: closest LOD only.
      { offset: [0, h * 0.15, d / 2 + 0.05], size: [w * 0.8, h * 0.12, 0.1], color: [0.12, 0.16, 0.2], maxLod: 0, emissive: [0.05, 0.08, 0.1] },
      { offset: [-w * 0.3, h / 2 + 2, d * 0.3], size: [0.15, 4, 0.15], color: STEEL, maxLod: 0 },
    ],
  };
}

function pillar(i: number, x: number, z: number, rand: () => number): PropSpec {
  const dia = 1.2 + rand() * 1.8;
  const h = 6 + rand() * 12;
  return {
    key: `pillar.${i}`,
    shape: "pillar",
    center: [x, h / 2, z],
    size: [dia, h, dia],
    color: RUST,
    collider: [dia / 2, h / 2, dia / 2],
    details: [{ offset: [0, h / 2 + 0.25, 0], size: [dia * 1.4, 0.5, dia * 1.4], color: STEEL, maxLod: 0 }],
  };
}

function dome(i: number, x: number, z: number, rand: () => number): PropSpec {
  const dia = 6 + rand() * 6;
  return {
    key: `dome.${i}`,
    shape: "dome",
    center: [x, dia * 0.3, z],
    size: [dia, dia, dia],
    color: [0.55, 0.57, 0.6],
    collider: [dia * 0.4, dia * 0.3, dia * 0.4],
    details: [{ offset: [0, dia * 0.55, 0], size: [0.6, 1.2, 0.6], color: STEEL, maxLod: 0 }],
  };
}

// ── kit placement ──────────────────────────────────────────────────────

/** Key prefixes of road-side furniture (not counted as filler). */
const STREET = new Set(["lamp", "bench", "barrier"]);

function groupKit(kit: KitCatalog): Map<KitCatalogEntry["kind"], KitCatalogEntry[]> {
  const by = new Map<KitCatalogEntry["kind"], KitCatalogEntry[]>();
  for (const e of kit) by.set(e.kind, [...(by.get(e.kind) ?? []), e]);
  return by;
}

/** World-space collider half extents for a kit asset: what you actually bump into. */
function kitCollider(e: KitCatalogEntry): Vec3 {
  const [w, h, d] = e.size;
  switch (e.kind) {
    case "building": return [w / 2, h / 2, d / 2];
    case "tree": return [0.25, h / 2, 0.25]; // trunk only: foliage is above head height
    case "lamp": return [0.2, h / 2, 0.2]; // the pole
    case "bench": return [w / 2, h / 2, d / 2];
    case "barrier": return [w / 2, h / 2, d / 2];
    case "water-tower": return [w * 0.35, h / 2, d * 0.35]; // the leg footprint
  }
}

function kitProp(key: string, e: KitCatalogEntry, xz: [number, number], z: number | null, extra: Partial<PropSpec> = {}): PropSpec {
  return {
    key,
    shape: "kit",
    kit: e.id,
    center: [xz[0], e.size[1] / 2, z ?? xz[1]],
    size: e.size,
    color: [1, 1, 1],
    collider: kitCollider(e),
    occluder: e.kind === "building",
    details: [],
    metallic: e.metallic,
    roughness: e.roughness,
    ...extra,
  };
}

function kitPick(
  kind: "building" | "tree",
  i: number,
  x: number,
  z: number,
  by: Map<KitCatalogEntry["kind"], KitCatalogEntry[]>,
  rand: () => number,
): PropSpec | null {
  const list = by.get(kind) ?? by.get("building") ?? null;
  if (!list || list.length === 0) return null;
  const e = list[Math.floor(rand() * list.length)]!;
  // Quarter turns keep the axis-aligned collider correct for any footprint.
  const yaw = Math.floor(rand() * 4) * (Math.PI / 2);
  const turned = yaw % Math.PI !== 0;
  const spec = kitProp(`${kind}.${i}`, e, [x, z], null, { yaw });
  if (turned) {
    spec.size = [e.size[2], e.size[1], e.size[0]];
    if (spec.collider) spec.collider = [spec.collider[2], spec.collider[1], spec.collider[0]];
  }
  return spec;
}

/** Lamps both sides of the road every 16 m, plus the odd bench and barrier. */
function streetFurniture(
  tx: number,
  c: Vec3,
  by: Map<KitCatalogEntry["kind"], KitCatalogEntry[]>,
  rand: () => number,
  existing: PropSpec[],
): PropSpec[] {
  const out: PropSpec[] = [];
  const lamp = by.get("lamp")?.[0];
  const bench = by.get("bench")?.[0];
  const barrier = by.get("barrier")?.[0];
  const edge = LEVEL.roadHalfWidth + 1.2;
  const clear = (x: number, z: number, r: number) =>
    ![...existing, ...out].some((p) => Math.hypot(p.center[0] - x, p.center[2] - z) < r + Math.max(p.size[0], p.size[2]) / 2 + 0.5);
  const x0 = c[0] - TILE_SIZE / 2;
  if (lamp) {
    for (let k = 0; k < TILE_SIZE / 16; k++) {
      const x = x0 + 8 + k * 16;
      // North side (z < road) arm points +z at the road; south side points -z.
      if (clear(x, LEVEL.roadZ - edge, 0.5)) out.push(kitProp(`lamp.n${k}`, lamp, [x, LEVEL.roadZ - edge], null, { yaw: -Math.PI / 2 }));
      if (clear(x + 8, LEVEL.roadZ + edge, 0.5)) out.push(kitProp(`lamp.s${k}`, lamp, [x + 8, LEVEL.roadZ + edge], null, { yaw: Math.PI / 2 }));
    }
  }
  if (bench && rand() < 0.6) {
    const x = x0 + 12 + rand() * (TILE_SIZE - 24);
    const z = LEVEL.roadZ - edge - 1.2;
    // The bench faces +z (toward the road) as built.
    if (clear(x, z, 1)) out.push(kitProp(`bench.${tx}`, bench, [x, z], null));
  }
  if (barrier && rand() < 0.35) {
    const x = x0 + 10 + rand() * (TILE_SIZE - 20);
    const z = LEVEL.roadZ + edge + 0.9;
    if (clear(x, z, 1.6)) out.push(kitProp(`barrier.${tx}`, barrier, [x, z], null));
  }
  return out;
}

function landmarkProps(name: string, c: Vec3): PropSpec[] {
  const tag = { landmark: name };
  const roadZ = LEVEL.roadZ;
  switch (name) {
    case "West Gate":
      return [
        { key: "gate.n", shape: "pillar", center: [c[0], 9, roadZ - 10], size: [3, 18, 3], color: STONE, collider: [1.5, 9, 1.5], details: [], ...tag },
        { key: "gate.s", shape: "pillar", center: [c[0], 9, roadZ + 10], size: [3, 18, 3], color: STONE, collider: [1.5, 9, 1.5], details: [], ...tag },
        { key: "gate.beam", shape: "box", center: [c[0], 19, roadZ], size: [3, 2, 24], color: STONE, collider: null, details: [], ...tag },
      ];
    case "Water Tower":
      return [
        { key: "tower.leg", shape: "pillar", center: [c[0], 10, roadZ - 16], size: [2.4, 20, 2.4], color: STEEL, collider: [1.2, 10, 1.2], details: [], ...tag },
        { key: "tower.tank", shape: "dome", center: [c[0], 23, roadZ - 16], size: [12, 12, 12], color: [0.7, 0.72, 0.74], collider: null, details: [], ...tag },
      ];
    case "Signal Spire":
      return [
        {
          key: "spire", shape: "pillar", center: [c[0], 30, roadZ - 14], size: [3.5, 60, 3.5], color: [0.2, 0.22, 0.25],
          collider: [1.75, 30, 1.75], emissive: [0.02, 0.02, 0.03],
          details: [{ offset: [0, 31, 0], size: [2.4, 2.4, 2.4], color: [1, 0.25, 0.12], maxLod: 1, emissive: [1.2, 0.3, 0.1] }],
          ...tag,
        },
      ];
    case "Monolith Ring": {
      const out: PropSpec[] = [];
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2;
        const mx = c[0] + Math.cos(a) * 18;
        const mz = roadZ + 26 + Math.sin(a) * 12;
        out.push({ key: `monolith.${k}`, shape: "box", center: [mx, 6, mz], size: [2, 12, 4], color: [0.1, 0.1, 0.12], collider: [1, 6, 2], details: [], ...tag });
      }
      return out;
    }
    case "East Beacon":
      return [
        {
          key: "beacon", shape: "dome", center: [c[0] + 10, 4, roadZ], size: [8, 8, 8], color: [0.85, 0.6, 0.2],
          collider: [3.2, 2.4, 3.2], emissive: [0.6, 0.35, 0.08], details: [], ...tag,
        },
      ];
    default:
      return [];
  }
}

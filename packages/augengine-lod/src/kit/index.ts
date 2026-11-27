/**
 * kit — an original, procedurally generated city kit for The Loading Range.
 *
 * Everything here is generated from code: no third-party models, textures or
 * downloads. Assets are unit-free metres, origin at the base centre (y = 0 is
 * the ground), so a prop entity can be placed by its footprint centre with
 * scale [1, 1, 1]. Per-part colour is carried in vertex colours (the engine
 * multiplies them into the material base colour), so each asset is one mesh
 * and one draw.
 */
import { MeshBuilder, rng } from "../builder.js";
import type { KitMaterial, KitMesh, Rgba, Vec3 } from "../mesh.js";

export interface KitAsset {
  /** Stable id, also the mesh-key stem: `proc:kit-<id>:<lod>`. */
  id: string;
  kind: "building" | "lamp" | "bench" | "barrier" | "tree" | "water-tower";
  mesh: KitMesh;
  material: KitMaterial;
  /** Footprint size [x, height, z] in metres, for placement and colliders. */
  size: Vec3;
}

/** LOD3 impostor style per kind: solid shapes get a box, thin/leafy ones crossed cards. */
export function impostorFor(kind: KitAsset["kind"]): "box" | "cross" {
  return kind === "tree" || kind === "lamp" ? "cross" : "box";
}

const rgba = (r: number, g: number, b: number, a = 1): Rgba => [r, g, b, a];

/** Facade palettes: concrete, brick, sandstone, slate. */
const WALLS: Rgba[] = [rgba(0.62, 0.6, 0.56), rgba(0.55, 0.3, 0.22), rgba(0.74, 0.64, 0.48), rgba(0.36, 0.39, 0.43)];
const TRIM = rgba(0.82, 0.8, 0.76);
const GLASS_DARK = rgba(0.08, 0.11, 0.15);
const GLASS_LIT = rgba(0.95, 0.78, 0.45);
const FRAME = rgba(0.2, 0.2, 0.22);
const ROOF = rgba(0.24, 0.24, 0.26);
const METAL = rgba(0.5, 0.52, 0.55);
const DARK_METAL = rgba(0.16, 0.17, 0.19);
const WOOD = rgba(0.48, 0.32, 0.18);
const BARK = rgba(0.32, 0.22, 0.14);

export interface BuildingParams {
  width: number;
  depth: number;
  floors: number;
  /** 0..3 — facade palette. */
  palette: number;
  seed: number;
}

/**
 * Mid-rise block: plinth, facade with a window grid per floor (some lit),
 * floor ledges, a cornice, a flat roof with parapet and rooftop plant.
 */
export function building(p: BuildingParams): KitAsset {
  const r = rng(p.seed);
  const b = new MeshBuilder();
  const floorH = 3.2;
  const plinthH = 0.6;
  const h = plinthH + p.floors * floorH;
  const hw = p.width / 2;
  const hd = p.depth / 2;
  const wall = WALLS[p.palette % WALLS.length]!;

  b.box([-hw - 0.15, 0, -hd - 0.15], [hw + 0.15, plinthH, hd + 0.15], TRIM, { ny: false });
  b.box([-hw, plinthH, -hd], [hw, h, hd], wall, { ny: false, py: false });

  // Ledges between floors and a heavier cornice on top.
  for (let f = 1; f < p.floors; f++) {
    const y = plinthH + f * floorH;
    b.box([-hw - 0.08, y - 0.08, -hd - 0.08], [hw + 0.08, y + 0.08, hd + 0.08], TRIM, { ny: false });
  }
  // Cornice as a rim (4 pieces), not a slab: a slab hidden under the roof deck
  // gives the simplifier stacked layers to collapse into, which shows as stripes.
  const ch = 0.35, cw = 0.6;
  b.box([-hw - 0.3, h, -hd - 0.3], [hw + 0.3, h + ch, -hd - 0.3 + cw], TRIM);
  b.box([-hw - 0.3, h, hd + 0.3 - cw], [hw + 0.3, h + ch, hd + 0.3], TRIM);
  b.box([-hw - 0.3, h, -hd - 0.3 + cw], [-hw - 0.3 + cw, h + ch, hd + 0.3 - cw], TRIM);
  b.box([hw + 0.3 - cw, h, -hd - 0.3 + cw], [hw + 0.3, h + ch, hd + 0.3 - cw], TRIM);

  // Windows on all four facades: recessed glass with a frame sill.
  const facades: { axis: "x" | "z"; sign: 1 | -1; span: number }[] = [
    { axis: "z", sign: 1, span: p.width },
    { axis: "z", sign: -1, span: p.width },
    { axis: "x", sign: 1, span: p.depth },
    { axis: "x", sign: -1, span: p.depth },
  ];
  for (const f of facades) {
    const cols = Math.max(1, Math.floor((f.span - 1) / 2.4));
    const pitch = f.span / cols;
    for (let fl = 0; fl < p.floors; fl++) {
      const y0 = plinthH + fl * floorH + 0.9;
      const y1 = y0 + 1.6;
      for (let c = 0; c < cols; c++) {
        const u0 = -f.span / 2 + c * pitch + pitch * 0.22;
        const u1 = u0 + pitch * 0.56;
        const lit = r() < 0.28;
        window(b, f.axis, f.sign, f.axis === "z" ? hd : hw, u0, u1, y0, y1, lit ? GLASS_LIT : GLASS_DARK);
      }
    }
  }

  // Roof: deck, parapet, plant boxes, a small water tank on some roofs.
  // Roof deck fills the rim's opening down to the wall top, so there is no gap and no hidden layer.
  b.box([-hw + 0.3, h, -hd + 0.3], [hw - 0.3, h + 0.4, hd - 0.3], ROOF, { ny: false });
  const top = h + 0.4;
  const pw = 0.25, ph = 0.8;
  b.box([-hw, top, -hd], [hw, top + ph, -hd + pw], TRIM, { ny: false });
  b.box([-hw, top, hd - pw], [hw, top + ph, hd], TRIM, { ny: false });
  b.box([-hw, top, -hd + pw], [-hw + pw, top + ph, hd - pw], TRIM, { ny: false });
  b.box([hw - pw, top, -hd + pw], [hw, top + ph, hd - pw], TRIM, { ny: false });
  const units = 1 + Math.floor(r() * 3);
  for (let i = 0; i < units; i++) {
    const ux = (r() - 0.5) * (p.width - 3);
    const uz = (r() - 0.5) * (p.depth - 3);
    b.boxAt([ux, top + 0.6, uz], [1.4, 1.2, 1.0], METAL, { ny: false });
    b.boxAt([ux, top + 1.25, uz], [1.0, 0.1, 0.7], DARK_METAL, { ny: false });
  }
  if (r() < 0.45) waterTank(b, [(r() - 0.5) * (p.width - 4), top, (r() - 0.5) * (p.depth - 4)], 1.1 + r() * 0.5);

  return {
    id: `building-${p.palette}-${p.floors}-${p.seed}`,
    kind: "building",
    mesh: b.build(),
    material: { baseColor: [1, 1, 1], metallic: 0.05, roughness: 0.78 },
    size: [p.width + 0.6, top + ph, p.depth + 0.6],
  };
}

/** A window on a facade: dark/lit glass slightly proud of the wall plus a sill. */
function window(b: MeshBuilder, axis: "x" | "z", sign: 1 | -1, face: number, u0: number, u1: number, y0: number, y1: number, glass: Rgba): void {
  const off = face + 0.02;
  if (axis === "z") {
    const z = sign * off;
    if (sign > 0) b.quad([u0, y0, z], [u1, y0, z], [u1, y1, z], [u0, y1, z], glass);
    else b.quad([u1, y0, z], [u0, y0, z], [u0, y1, z], [u1, y1, z], glass);
    b.box([u0 - 0.08, y0 - 0.12, sign > 0 ? face : -face - 0.18], [u1 + 0.08, y0, sign > 0 ? face + 0.18 : -face], FRAME, { ny: false });
  } else {
    const x = sign * off;
    if (sign > 0) b.quad([x, y0, u1], [x, y0, u0], [x, y1, u0], [x, y1, u1], glass);
    else b.quad([x, y0, u0], [x, y0, u1], [x, y1, u1], [x, y1, u0], glass);
    b.box([sign > 0 ? face : -face - 0.18, y0 - 0.12, u0 - 0.08], [sign > 0 ? face + 0.18 : -face, y0, u1 + 0.08], FRAME, { ny: false });
  }
}

function waterTank(b: MeshBuilder, base: Vec3, radius: number): void {
  const legH = 1.6;
  for (const [dx, dz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
    b.boxAt([base[0] + dx * radius * 0.6, base[1] + legH / 2, base[2] + dz * radius * 0.6], [0.14, legH, 0.14], DARK_METAL, { ny: false });
  }
  b.cylinder([base[0], base[1] + legH, base[2]], radius, radius, radius * 1.6, 10, WOOD, { bottom: true });
  b.cylinder([base[0], base[1] + legH + radius * 1.6, base[2]], radius * 1.05, 0, radius * 0.6, 10, DARK_METAL);
}

/** Street lamp: base, tapered pole, arm and a lit head. */
export function lampPost(seed = 1): KitAsset {
  const b = new MeshBuilder();
  b.cylinder([0, 0, 0], 0.18, 0.16, 0.4, 8, DARK_METAL);
  b.cylinder([0, 0.4, 0], 0.09, 0.06, 4.6, 8, DARK_METAL);
  b.boxAt([0.55, 4.9, 0], [1.2, 0.08, 0.08], DARK_METAL, { ny: true });
  b.boxAt([1.1, 4.82, 0], [0.5, 0.16, 0.3], DARK_METAL);
  b.boxAt([1.1, 4.72, 0], [0.42, 0.04, 0.24], rgba(1, 0.86, 0.55));
  return { id: `lamp-${seed}`, kind: "lamp", mesh: b.build(), material: { baseColor: [1, 1, 1], metallic: 0.5, roughness: 0.45 }, size: [1.5, 5.0, 0.4] };
}

/** Park bench: two cast legs, three seat slats, two backrest slats. */
export function bench(seed = 1): KitAsset {
  const b = new MeshBuilder();
  for (const x of [-0.75, 0.75]) {
    b.boxAt([x, 0.22, 0], [0.08, 0.44, 0.5], DARK_METAL);
    b.boxAt([x, 0.62, -0.22], [0.08, 0.5, 0.06], DARK_METAL);
  }
  for (const z of [-0.16, 0, 0.16]) b.boxAt([0, 0.46, z], [1.8, 0.05, 0.12], WOOD);
  for (const y of [0.62, 0.78]) b.boxAt([0, y, -0.25], [1.8, 0.1, 0.04], WOOD);
  return { id: `bench-${seed}`, kind: "bench", mesh: b.build(), material: { baseColor: [1, 1, 1], metallic: 0.1, roughness: 0.7 }, size: [1.9, 0.9, 0.6] };
}

/** Jersey road barrier with hazard stripes. */
export function barrier(seed = 1): KitAsset {
  const b = new MeshBuilder();
  const len = 3;
  const prof: [number, number][] = [[-0.3, 0], [0.3, 0], [0.12, 0.25], [0.1, 0.8], [-0.1, 0.8], [-0.12, 0.25]];
  const body = rgba(0.78, 0.77, 0.74);
  for (let i = 0; i < prof.length; i++) {
    const a = prof[i]!, c = prof[(i + 1) % prof.length]!;
    if (i === 0) continue; // underside, never seen
    // Profile runs counter-clockwise in (z, y); this order makes each side face point outward.
    b.quad([-len / 2, a[1], a[0]], [len / 2, a[1], a[0]], [len / 2, c[1], c[0]], [-len / 2, c[1], c[0]], body);
  }
  for (const x of [-len / 2, len / 2]) {
    const s = x < 0 ? -1 : 1;
    const pts = prof.map(([z, y]): Vec3 => [x, y, z]);
    for (let i = 1; i < pts.length - 1; i++) {
      if (s > 0) b.tri(pts[0]!, pts[i + 1]!, pts[i]!, body);
      else b.tri(pts[0]!, pts[i]!, pts[i + 1]!, body);
    }
  }
  // Stripes on both slopes, slightly proud of the surface.
  for (let k = 0; k < 4; k++) {
    const x0 = -len / 2 + 0.25 + k * 0.7;
    const col = k % 2 === 0 ? rgba(0.95, 0.55, 0.08) : rgba(0.95, 0.95, 0.92);
    b.quad([x0, 0.3, 0.135], [x0 + 0.45, 0.3, 0.135], [x0 + 0.45, 0.7, 0.115], [x0, 0.7, 0.115], col);
    b.quad([x0 + 0.45, 0.3, -0.135], [x0, 0.3, -0.135], [x0, 0.7, -0.115], [x0 + 0.45, 0.7, -0.115], col);
  }
  return { id: `barrier-${seed}`, kind: "barrier", mesh: b.build(), material: { baseColor: [1, 1, 1], metallic: 0, roughness: 0.85 }, size: [len, 0.8, 0.6] };
}

/** Low-poly tree: tapered trunk and 2-3 jittered foliage clusters. */
export function tree(seed: number): KitAsset {
  const r = rng(seed);
  const b = new MeshBuilder();
  const trunkH = 2.4 + r() * 1.4;
  b.cylinder([0, 0, 0], 0.22, 0.13, trunkH, 7, BARK);
  const greens: Rgba[] = [rgba(0.22, 0.42, 0.18), rgba(0.3, 0.5, 0.2), rgba(0.18, 0.36, 0.2)];
  const clusters = 2 + Math.floor(r() * 2);
  let top = 0;
  for (let i = 0; i < clusters; i++) {
    const rad = 1.3 + r() * 0.9;
    const y = trunkH + i * 0.9 + rad * 0.5;
    const ox = (r() - 0.5) * 0.9, oz = (r() - 0.5) * 0.9;
    b.icosphere([ox, y, oz], [rad, rad * 0.85, rad], 1, greens[i % greens.length]!, r);
    top = Math.max(top, y + rad * 1.1);
  }
  return { id: `tree-${seed}`, kind: "tree", mesh: b.build(), material: { baseColor: [1, 1, 1], metallic: 0, roughness: 0.9 }, size: [4.4, top, 4.4] };
}

/** Landmark water tower (replaces the pillar + dome of the Water Tower tile). */
export function waterTowerLandmark(): KitAsset {
  const b = new MeshBuilder();
  const legs = 4, spread = 4.2, legH = 17;
  for (let i = 0; i < legs; i++) {
    const a = (i / legs) * Math.PI * 2 + Math.PI / 4;
    b.cylinder([Math.cos(a) * spread, 0, Math.sin(a) * spread], 0.35, 0.3, legH, 8, METAL);
  }
  for (const y of [5, 10, 15]) {
    b.boxAt([0, y, spread * 0.7], [spread * 1.45, 0.18, 0.18], DARK_METAL);
    b.boxAt([0, y, -spread * 0.7], [spread * 1.45, 0.18, 0.18], DARK_METAL);
    b.boxAt([spread * 0.7, y, 0], [0.18, 0.18, spread * 1.45], DARK_METAL);
    b.boxAt([-spread * 0.7, y, 0], [0.18, 0.18, spread * 1.45], DARK_METAL);
  }
  b.cylinder([0, legH, 0], 6, 6, 7, 20, rgba(0.72, 0.74, 0.76), { bottom: true });
  b.cylinder([0, legH + 7, 0], 6.3, 0.4, 3, 20, rgba(0.55, 0.2, 0.15));
  b.cylinder([0, legH + 3.2, 0], 6.05, 6.05, 0.6, 20, rgba(0.9, 0.9, 0.88), { top: false });
  return { id: "water-tower", kind: "water-tower", mesh: b.build(), material: { baseColor: [1, 1, 1], metallic: 0.4, roughness: 0.5 }, size: [13, legH + 10, 13] };
}

/** The whole kit with fixed seeds, so outputs are reproducible. */
export function buildKit(): KitAsset[] {
  const out: KitAsset[] = [];
  const specs: BuildingParams[] = [
    { width: 10, depth: 8, floors: 3, palette: 0, seed: 11 },
    { width: 12, depth: 10, floors: 5, palette: 1, seed: 23 },
    { width: 8, depth: 8, floors: 2, palette: 2, seed: 37 },
    { width: 14, depth: 9, floors: 4, palette: 3, seed: 41 },
    { width: 9, depth: 12, floors: 6, palette: 0, seed: 53 },
    { width: 11, depth: 11, floors: 3, palette: 2, seed: 67 },
  ];
  for (const s of specs) out.push(building(s));
  out.push(lampPost(), bench(), barrier(), tree(3), tree(7), tree(19), waterTowerLandmark());
  return out;
}

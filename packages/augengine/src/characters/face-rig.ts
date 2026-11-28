/**
 * Face rig for generated characters: a "face patch" transplanted onto the
 * generated head, carrying the 52 ARKit blendshapes.
 *
 * Image-to-3D heads (TRELLIS) have ~150 low-resolution face vertices and the
 * mouth and eyes are only painted on the texture. Deforming that mesh in place
 * gives a stretched chin, not a mouth. So instead:
 *
 *   1. find the face on the mesh. Preferred: landmarks from a face-mesh
 *      detector run on a front orthographic render (MediaPipe in the Studio),
 *      passed in as `opts.landmarks` via `faceLandmarksFromMeshPoints`, which
 *      also gives the face outline the patch is cut to. Fallback (heuristic,
 *      can misplace features on generated heads): a front depth + colour map
 *      gives the nose (depth peak), the chin (where the profile turns under),
 *      the eyes (two dark blobs) and the mouth line (dark/red row); fixed head
 *      proportions fill in anything that isn't found;
 *   2. lay a regular grid over the face and shrink-wrap it onto the generated
 *      surface (orthographic rays along -Z). Each grid vertex copies the
 *      surface's position, normal and texture coordinate, so at rest the patch
 *      looks exactly like the generated face (same texture, same shape);
 *   3. the grid has the topology the source lacks: the lips are split along
 *      the mouth line, a dark mouth cavity and a tongue sit behind them, and
 *      a collapsed eyelid sheet sits on each upper eye edge;
 *   4. author the 52 ARKit shapes as region deformers in face-local units
 *      (D = chin-to-eye distance), faded to zero at the patch border so it
 *      never tears away from the rest of the head.
 *
 * The caller removes the source triangles the patch covers
 * (`coveredTriangles`), binds the patch to the head bone, and exports the
 * targets with their ARKit names. Shape order is `FACE_RIG_SHAPES`, which is
 * the audio2face / Audio2Face-3D channel order, so the live lip-sync path
 * (`renderer.setBlendshapeWeights`, which maps by index) drives the patch
 * directly. Other tools read the names.
 *
 * Pure math: no three.js, no GPU. Tested in face-rig.test.ts.
 */

/**
 * The 52 ARKit blendshapes in the order the engine's lip-sync emits them
 * (`@mnemopay/augengine-audio2face` `ARKIT_SHAPES`, the NVIDIA Audio2Face-3D
 * channel order). Same names as `ARKIT_52`, different order.
 */
export const FACE_RIG_SHAPES = [
  "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
  "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
  "jawForward", "jawLeft", "jawRight", "jawOpen",
  "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
  "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
  "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
  "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
] as const;

export type FaceRigShape = (typeof FACE_RIG_SHAPES)[number];

export const FACE_RIG_INDEX: Readonly<Record<FaceRigShape, number>> = Object.freeze(
  Object.fromEntries(FACE_RIG_SHAPES.map((s, i) => [s, i])) as Record<FaceRigShape, number>,
);

/**
 * Shapes the patch cannot really do: painted eyes have no eyeball to turn,
 * so the eyeLook* shapes only slide the painted eye a little. Listed so the
 * Studio and docs can say so.
 */
export const FACE_RIG_APPROXIMATE: readonly FaceRigShape[] = [
  "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
  "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
];

/** 52 weights in `FACE_RIG_SHAPES` order from a name → weight map (unknown names ignored). */
export function faceRigWeights(named: Partial<Record<string, number>>, out = new Float32Array(FACE_RIG_SHAPES.length)): Float32Array {
  out.fill(0);
  for (const [name, w] of Object.entries(named)) {
    const i = (FACE_RIG_INDEX as Record<string, number>)[name];
    if (i !== undefined && w !== undefined) out[i] = Math.max(0, Math.min(1, w));
  }
  return out;
}

/**
 * Re-order 52 `FACE_RIG_SHAPES`-order weights for a mesh whose morph targets
 * are named in another order (VRoid, MetaHuman, Apple's alphabetical list…).
 * Names are matched case-insensitively, ignoring any "prefix." / "prefix:".
 * Targets with no ARKit name get 0.
 */
export function remapFaceRigWeights(weights: ArrayLike<number>, targetNames: readonly string[], out = new Float32Array(targetNames.length)): Float32Array {
  const byLower = new Map(FACE_RIG_SHAPES.map((s, i) => [s.toLowerCase(), i]));
  for (let t = 0; t < targetNames.length; t++) {
    const i = byLower.get(targetNames[t]!.replace(/^.*[.:|]/, "").toLowerCase());
    out[t] = i === undefined ? 0 : (weights[i] ?? 0);
  }
  return out;
}

// ─── visemes → named shapes (for scripted lines and previews) ───────────

/** Mouth shapes per viseme (Oculus-style labels as used by the engine's Cherry set). */
export const VISEME_SHAPES: Readonly<Record<string, Partial<Record<FaceRigShape, number>>>> = {
  sil: {},
  PP: { mouthClose: 0.55, mouthPressLeft: 0.35, mouthPressRight: 0.35, jawOpen: 0.08 },
  FF: { mouthRollLower: 0.55, mouthUpperUpLeft: 0.2, mouthUpperUpRight: 0.2, jawOpen: 0.1 },
  TH: { tongueOut: 0.45, jawOpen: 0.22 },
  DD: { jawOpen: 0.3, mouthStretchLeft: 0.12, mouthStretchRight: 0.12 },
  kk: { jawOpen: 0.32, mouthStretchLeft: 0.1, mouthStretchRight: 0.1 },
  CH: { mouthFunnel: 0.5, jawOpen: 0.2 },
  SS: { mouthStretchLeft: 0.3, mouthStretchRight: 0.3, mouthSmileLeft: 0.15, mouthSmileRight: 0.15, jawOpen: 0.1 },
  nn: { jawOpen: 0.18 },
  RR: { mouthPucker: 0.35, jawOpen: 0.2 },
  aa: { jawOpen: 0.72, mouthLowerDownLeft: 0.2, mouthLowerDownRight: 0.2 },
  E: { jawOpen: 0.38, mouthStretchLeft: 0.35, mouthStretchRight: 0.35 },
  I: { jawOpen: 0.2, mouthSmileLeft: 0.32, mouthSmileRight: 0.32 },
  ih: { jawOpen: 0.2, mouthSmileLeft: 0.32, mouthSmileRight: 0.32 },
  O: { jawOpen: 0.45, mouthFunnel: 0.55 },
  oh: { jawOpen: 0.45, mouthFunnel: 0.55 },
  ou: { mouthPucker: 0.7, jawOpen: 0.15 },
};

/** Procedural channel names (engine Cherry set) → ARKit names. */
const PROCEDURAL_SHAPES: Readonly<Record<string, readonly FaceRigShape[]>> = {
  browInnerUp: ["browInnerUp"],
  browOuterUpL: ["browOuterUpLeft"],
  browOuterUpR: ["browOuterUpRight"],
  browDown: ["browDownLeft", "browDownRight"],
  eyeLookIn: ["eyeLookInLeft", "eyeLookInRight"],
  eyeLookOut: ["eyeLookOutLeft", "eyeLookOutRight"],
  blinkL: ["eyeBlinkLeft"],
  blinkR: ["eyeBlinkRight"],
};

/**
 * A viseme frame (viseme name → weight, procedural channel → weight) as 52
 * `FACE_RIG_SHAPES` weights. Overlapping visemes add and clamp at 1.
 */
export function visemeFrameToFaceRig(
  frame: { visemes: Partial<Record<string, number>>; procedural?: Partial<Record<string, number>> },
  out = new Float32Array(FACE_RIG_SHAPES.length),
): Float32Array {
  out.fill(0);
  const add = (shape: FaceRigShape, w: number) => {
    const i = FACE_RIG_INDEX[shape];
    out[i] = Math.min(1, (out[i] ?? 0) + w);
  };
  for (const [v, w] of Object.entries(frame.visemes)) {
    if (!w || w <= 0) continue;
    for (const [shape, k] of Object.entries(VISEME_SHAPES[v] ?? {})) add(shape as FaceRigShape, w * (k ?? 0));
  }
  for (const [c, w] of Object.entries(frame.procedural ?? {})) {
    if (!w || w <= 0) continue;
    for (const shape of PROCEDURAL_SHAPES[c] ?? []) add(shape, w);
  }
  return out;
}

// ─── source mesh + face detection ───────────────────────────────────────

/** Model-space mesh: metres, Y up, feet on y = 0, facing +Z (the normalised character). */
export interface FaceSourceMesh {
  positions: Float32Array;
  indices: Uint32Array | Uint16Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  /** RGBA per vertex. */
  colors?: Float32Array;
}

/** Reads the base-colour texture: u, v in [0, 1] (glTF convention, v down) → rgb in [0, 1]. */
export type FaceTextureSampler = (u: number, v: number) => readonly [number, number, number];

export interface HeadBox {
  minX: number; maxX: number;
  /** Neck (narrowest point under the head). */
  minY: number;
  /** Top of the head (hair included). */
  maxY: number;
  minZ: number; maxZ: number;
  /** Face midline x. */
  centerX: number;
}

export interface FaceLandmarks {
  centerX: number;
  chinY: number;
  mouthY: number;
  noseY: number;
  eyeY: number;
  browY: number;
  /** Distance of each eye centre from the midline. */
  eyeX: number;
  eyeHalfW: number;
  eyeHalfH: number;
  mouthHalfW: number;
  /**
   * Half-width of the face (cheek to cheek) when a detector measured it; the
   * patch then spans the whole jaw so jawOpen fades out at the jawline, not
   * mid-cheek. Omitted by the heuristic.
   */
  faceHalfW?: number;
  /**
   * Face silhouette (closed polygon, model x / y) when a detector found it.
   * The patch is cut to this outline, grown by `OUTLINE_GROW`, so it never
   * reaches past the cheeks and jaw onto hair, ears or a head wrap.
   */
  outline?: { x: number; y: number }[];
  /** Where each landmark came from: the mesh/texture, or head proportions. */
  found: { chin: boolean; nose: boolean; eyes: boolean; mouth: boolean };
}

const smoothstep = (a: number, b: number, x: number): number => {
  if (a === b) return x < a ? 0 : 1;
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  return s[clamp(Math.floor(p * (s.length - 1)), 0, s.length - 1)]!;
}

/**
 * Where the head is: the narrowest horizontal slice in the upper body is the
 * neck; everything above it (within the torso's width) is the head.
 */
export function findHeadBox(mesh: FaceSourceMesh): HeadBox {
  const P = mesh.positions;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 1; i < P.length; i += 3) { minY = Math.min(minY, P[i]!); maxY = Math.max(maxY, P[i]!); }
  const H = Math.max(1e-6, maxY - minY);
  const lim = 0.16 * H;
  // Width profile above 70 % of the height, arms excluded by |x| < 0.16 H.
  const slices = 48;
  const y0 = minY + 0.7 * H;
  const halfW: number[][] = Array.from({ length: slices }, () => []);
  for (let i = 0; i < P.length; i += 3) {
    const x = P[i]!, y = P[i + 1]!;
    if (y < y0 || Math.abs(x) > lim) continue;
    const s = Math.min(slices - 1, Math.floor(((y - y0) / (maxY - y0)) * slices));
    halfW[s]!.push(Math.abs(x));
  }
  const widthAt = halfW.map((xs) => (xs.length >= 3 ? percentile(xs, 0.9) : Infinity));
  // Neck: narrowest slice between 76 % of the height and a little under the top.
  let neckS = -1, best = Infinity;
  for (let s = 0; s < slices; s++) {
    const y = y0 + ((s + 0.5) / slices) * (maxY - y0);
    if (y < minY + 0.76 * H || y > maxY - 0.07 * H) continue;
    if (widthAt[s]! < best) { best = widthAt[s]!; neckS = s; }
  }
  // A long neck is a run of near-equal slices: the head starts at its top.
  while (neckS >= 0 && neckS + 1 < slices && widthAt[neckS + 1]! <= best * 1.08 && y0 + ((neckS + 1.5) / slices) * (maxY - y0) <= maxY - 0.07 * H) neckS++;
  const neckY = neckS >= 0 ? y0 + ((neckS + 0.5) / slices) * (maxY - y0) : minY + 0.845 * H;
  const xs: number[] = [], zs: number[] = [];
  for (let i = 0; i < P.length; i += 3) {
    if (P[i + 1]! < neckY || Math.abs(P[i]!) > lim) continue;
    xs.push(P[i]!); zs.push(P[i + 2]!);
  }
  const minX = percentile(xs, 0.01), maxX = percentile(xs, 0.99);
  return { minX, maxX, minY: neckY, maxY, minZ: percentile(zs, 0.01), maxZ: percentile(zs, 0.995), centerX: (minX + maxX) / 2 };
}

interface Hit { z: number; tri: number; b0: number; b1: number; b2: number }

/** Frontmost-surface lookup along -Z, bucketed in XY. */
export class OrthoCaster {
  private readonly cell: number;
  private readonly buckets = new Map<number, number[]>();
  private readonly ox: number;
  private readonly oy: number;

  constructor(private readonly mesh: FaceSourceMesh, region: { minX: number; maxX: number; minY: number; maxY: number }) {
    const P = mesh.positions, I = mesh.indices;
    this.cell = Math.max(1e-4, Math.max(region.maxX - region.minX, region.maxY - region.minY) / 40);
    this.ox = region.minX; this.oy = region.minY;
    for (let t = 0; t + 2 < I.length; t += 3) {
      const a = I[t]! * 3, b = I[t + 1]! * 3, c = I[t + 2]! * 3;
      const x0 = Math.min(P[a]!, P[b]!, P[c]!), x1 = Math.max(P[a]!, P[b]!, P[c]!);
      const y0 = Math.min(P[a + 1]!, P[b + 1]!, P[c + 1]!), y1 = Math.max(P[a + 1]!, P[b + 1]!, P[c + 1]!);
      if (x1 < region.minX || x0 > region.maxX || y1 < region.minY || y0 > region.maxY) continue;
      for (let gy = this.key1(y0, this.oy); gy <= this.key1(y1, this.oy); gy++) {
        for (let gx = this.key1(x0, this.ox); gx <= this.key1(x1, this.ox); gx++) {
          const k = gy * 4096 + gx;
          let list = this.buckets.get(k);
          if (!list) { list = []; this.buckets.set(k, list); }
          list.push(t / 3);
        }
      }
    }
  }

  private key1(v: number, o: number): number { return Math.floor((v - o) / this.cell) + 1024; }

  /** Frontmost hit at (x, y), or null. */
  cast(x: number, y: number): Hit | null {
    const list = this.buckets.get(this.key1(y, this.oy) * 4096 + this.key1(x, this.ox));
    if (!list) return null;
    const P = this.mesh.positions, I = this.mesh.indices;
    let best: Hit | null = null;
    for (const tri of list) {
      const a = I[tri * 3]! * 3, b = I[tri * 3 + 1]! * 3, c = I[tri * 3 + 2]! * 3;
      const bc = bary2(x, y, P[a]!, P[a + 1]!, P[b]!, P[b + 1]!, P[c]!, P[c + 1]!);
      if (!bc) continue;
      const [b0, b1, b2] = bc;
      const e = -1e-7;
      if (b0 < e || b1 < e || b2 < e) continue;
      const z = b0 * P[a + 2]! + b1 * P[b + 2]! + b2 * P[c + 2]!;
      if (!best || z > best.z) best = { z, tri, b0, b1, b2 };
    }
    return best;
  }

  /** Barycentrics of (x, y) in triangle `tri`'s XY projection — may be outside [0, 1]. */
  baryIn(tri: number, x: number, y: number): [number, number, number] | null {
    const P = this.mesh.positions, I = this.mesh.indices;
    const a = I[tri * 3]! * 3, b = I[tri * 3 + 1]! * 3, c = I[tri * 3 + 2]! * 3;
    return bary2(x, y, P[a]!, P[a + 1]!, P[b]!, P[b + 1]!, P[c]!, P[c + 1]!);
  }

  /** Interpolate a per-vertex attribute (stride `n`) at barycentrics. */
  attr(arr: Float32Array, n: number, tri: number, b0: number, b1: number, b2: number): number[] {
    const I = this.mesh.indices;
    const a = I[tri * 3]! * n, b = I[tri * 3 + 1]! * n, c = I[tri * 3 + 2]! * n;
    const out: number[] = [];
    for (let k = 0; k < n; k++) out.push(b0 * arr[a + k]! + b1 * arr[b + k]! + b2 * arr[c + k]!);
    return out;
  }

  /** Geometric normal of triangle `tri`. */
  faceNormal(tri: number): [number, number, number] {
    const P = this.mesh.positions, I = this.mesh.indices;
    const a = I[tri * 3]! * 3, b = I[tri * 3 + 1]! * 3, c = I[tri * 3 + 2]! * 3;
    const ux = P[b]! - P[a]!, uy = P[b + 1]! - P[a + 1]!, uz = P[b + 2]! - P[a + 2]!;
    const vx = P[c]! - P[a]!, vy = P[c + 1]! - P[a + 1]!, vz = P[c + 2]! - P[a + 2]!;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    return nz >= 0 ? [nx / l, ny / l, nz / l] : [-nx / l, -ny / l, -nz / l];
  }
}

function bary2(px: number, py: number, ax: number, ay: number, bx: number, by: number, cx: number, cy: number): [number, number, number] | null {
  const d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(d) < 1e-14) return null;
  const b0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d;
  const b1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d;
  return [b0, b1, 1 - b0 - b1];
}

/** Front depth + colour of the head on a regular grid. */
export interface FaceMaps {
  w: number; h: number;
  x0: number; y0: number; dx: number; dy: number;
  /** NaN where nothing is hit. Row-major, row 0 = bottom (y0). */
  depth: Float32Array;
  /** Luminance 0..1, NaN when there is no colour source. */
  lum: Float32Array;
  /** Redness r − (g + b)/2. */
  red: Float32Array;
}

function colourAt(mesh: FaceSourceMesh, caster: OrthoCaster, hit: Hit, sampler?: FaceTextureSampler): readonly [number, number, number] | null {
  if (sampler && mesh.uvs) {
    const [u, v] = caster.attr(mesh.uvs, 2, hit.tri, hit.b0, hit.b1, hit.b2);
    return sampler(u!, v!);
  }
  if (mesh.colors) {
    const [r, g, b] = caster.attr(mesh.colors, 4, hit.tri, hit.b0, hit.b1, hit.b2);
    return [r!, g!, b!];
  }
  return null;
}

export function faceMaps(mesh: FaceSourceMesh, box: HeadBox, caster: OrthoCaster, sampler?: FaceTextureSampler, res = 72): FaceMaps {
  const w = res, h = res;
  const dx = (box.maxX - box.minX) / w, dy = (box.maxY - box.minY) / h;
  const depth = new Float32Array(w * h).fill(NaN), lum = new Float32Array(w * h).fill(NaN), red = new Float32Array(w * h).fill(NaN);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const x = box.minX + (i + 0.5) * dx, y = box.minY + (j + 0.5) * dy;
      const hit = caster.cast(x, y);
      if (!hit) continue;
      depth[j * w + i] = hit.z;
      const c = colourAt(mesh, caster, hit, sampler);
      if (c) {
        lum[j * w + i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        red[j * w + i] = c[0] - (c[1] + c[2]) / 2;
      }
    }
  }
  return { w, h, x0: box.minX, y0: box.minY, dx, dy, depth, lum, red };
}

/** Landmarks from head proportions alone. */
export function proportionalLandmarks(box: HeadBox): FaceLandmarks {
  const hh = box.maxY - box.minY;
  const hw = (box.maxX - box.minX) / 2;
  const chinY = box.minY + 0.1 * hh;
  const eyeY = chinY + 0.44 * (box.maxY - chinY);
  const D = eyeY - chinY;
  const eyeX = 0.42 * hw;
  const eyeHalfW = 0.19 * hw;
  return {
    centerX: box.centerX, chinY, eyeY, mouthY: chinY + 0.3 * D, noseY: chinY + 0.55 * D, browY: eyeY + 0.24 * D,
    eyeX, eyeHalfW, eyeHalfH: 0.5 * eyeHalfW, mouthHalfW: 0.85 * eyeX,
    found: { chin: false, nose: false, eyes: false, mouth: false },
  };
}

/**
 * Landmarks from the depth/colour maps, falling back to proportions for
 * anything not found (or found out of order).
 */
export function detectFaceLandmarks(maps: FaceMaps, box: HeadBox): FaceLandmarks {
  const L = proportionalLandmarks(box);
  const { w, h, dx, dy, x0, y0 } = maps;
  const hw = (box.maxX - box.minX) / 2;
  const colOf = (x: number) => clamp(Math.round((x - x0) / dx - 0.5), 0, w - 1);
  const rowOf = (y: number) => clamp(Math.round((y - y0) / dy - 0.5), 0, h - 1);
  const yOf = (j: number) => y0 + (j + 0.5) * dy;
  const xOf = (i: number) => x0 + (i + 0.5) * dx;
  const at = (arr: Float32Array, i: number, j: number) => (i < 0 || j < 0 || i >= w || j >= h ? NaN : arr[j * w + i]!);
  const ci = colOf(box.centerX);
  const band = Math.max(1, Math.round((0.08 * hw) / dx));
  const centreDepth = (j: number) => {
    let s = 0, n = 0;
    for (let i = ci - band; i <= ci + band; i++) { const z = at(maps.depth, i, j); if (Number.isFinite(z)) { s += z; n++; } }
    return n ? s / n : NaN;
  };

  // Nose: the depth peak of the centre column, 25–80 % up the head.
  let noseJ = -1, noseZ = -Infinity;
  for (let j = rowOf(box.minY + 0.25 * (box.maxY - box.minY)); j <= rowOf(box.minY + 0.55 * (box.maxY - box.minY)); j++) {
    const z = centreDepth(j);
    if (z > noseZ) { noseZ = z; noseJ = j; }
  }
  if (noseJ >= 0) { L.noseY = yOf(noseJ); L.found.nose = true; }

  // Chin: below the nose, where the profile turns under (largest dz/dy), then
  // up to where the slope eases — the chin point.
  if (L.found.nose) {
    let maxSlope = 0, slopeJ = -1;
    const top = rowOf(L.noseY - 0.3 * (L.noseY - box.minY));
    for (let j = 1; j <= top; j++) {
      const a = centreDepth(j - 1), b = centreDepth(j + 1);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const s = (b - a) / (2 * dy);
      if (s > maxSlope) { maxSlope = s; slopeJ = j; }
    }
    if (slopeJ >= 0 && maxSlope > 0.6) {
      let j = slopeJ;
      while (j + 1 <= top) {
        const a = centreDepth(j - 1), b = centreDepth(j + 1);
        if (!Number.isFinite(a) || !Number.isFinite(b) || (b - a) / (2 * dy) < 0.3 * maxSlope) break;
        j++;
      }
      L.chinY = yOf(j);
      L.found.chin = true;
    }
  }
  const D0 = Math.max(1e-6, L.noseY - L.chinY);

  // Colour features need a colour source.
  const hasColour = maps.lum.some((v) => Number.isFinite(v));
  if (hasColour) {
    const lumS = (i: number, j: number) => {
      let s = 0, n = 0;
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) { const v = at(maps.lum, i + a, j + b); if (Number.isFinite(v)) { s += v; n++; } }
      return n >= 5 ? s / n : NaN;
    };
    // Skin reference: median luminance of the cheeks/nose area.
    const skinVals: number[] = [];
    for (let j = rowOf(L.chinY + 0.3 * D0); j <= rowOf(L.noseY); j++) {
      for (let i = colOf(box.centerX - 0.5 * hw); i <= colOf(box.centerX + 0.5 * hw); i++) {
        const v = at(maps.lum, i, j); if (Number.isFinite(v)) skinVals.push(v);
      }
    }
    const skin = percentile(skinVals, 0.5);

    // Eyes: darkest blob each side, above the nose.
    const eyeOn = (side: 1 | -1) => {
      let best = Infinity, bi = -1, bj = -1;
      const surround = Math.max(2, Math.round(0.15 * D0 / dy));
      for (let j = rowOf(L.noseY + 0.15 * D0); j <= rowOf(L.noseY + 1.1 * D0); j++) {
        for (let i = colOf(box.centerX + side * 0.14 * hw); side > 0 ? i <= colOf(box.centerX + 0.78 * hw) : i >= colOf(box.centerX - 0.78 * hw); i += side) {
          const v = lumS(i, j);
          // Prefer a dark eye surrounded by skin over a dark hat or hairline.
          const above = lumS(i, j + surround), below = lumS(i, j - surround);
          const score = v - Math.min(above, below);
          if (score < best) { best = score; bi = i; bj = j; }
        }
      }
      return { lum: bi >= 0 ? lumS(bi, bj) : Infinity, x: bi >= 0 ? xOf(bi) : NaN, y: bj >= 0 ? yOf(bj) : NaN, i: bi, j: bj };
    };
    const eL = eyeOn(1), eR = eyeOn(-1);
    const contrast = skin - Math.max(eL.lum, eR.lum);
    if (contrast > 0.12 && Math.abs(eL.y - eR.y) < 0.3 * D0) {
      L.eyeY = (eL.y + eR.y) / 2;
      L.eyeX = clamp((Math.abs(eL.x - box.centerX) + Math.abs(eR.x - box.centerX)) / 2, 0.2 * hw, 0.7 * hw);
      // Width: dark run along the eye row around each minimum.
      const thr = skin - 0.5 * contrast;
      const run = (e: typeof eL) => {
        let a = e.i, b = e.i;
        while (a - 1 >= 0 && lumS(a - 1, e.j) < thr) a--;
        while (b + 1 < w && lumS(b + 1, e.j) < thr) b++;
        let c = e.j, d = e.j;
        while (c - 1 >= 0 && lumS(e.i, c - 1) < thr) c--;
        while (d + 1 < h && lumS(e.i, d + 1) < thr) d++;
        return { hw: ((b - a + 1) * dx) / 2, hh: ((d - c + 1) * dy) / 2 };
      };
      const rL = run(eL), rR = run(eR);
      L.eyeHalfW = clamp(Math.max((rL.hw + rR.hw) / 2 * 1.35, 0.12 * hw), 0.12 * hw, 0.3 * hw);
      L.eyeHalfH = clamp((rL.hh + rR.hh) / 2 * 1.2, 0.35 * L.eyeHalfW, 0.8 * L.eyeHalfW);
      L.found.eyes = true;
    }

    // Mouth: the darkest / reddest row between chin and nose, near the midline.
    let bestScore = -Infinity, mouthJ = -1;
    for (let j = rowOf(L.chinY + 0.12 * D0); j <= rowOf(L.noseY - 0.15 * D0); j++) {
      let s = 0, n = 0;
      for (let i = colOf(box.centerX - 0.18 * hw); i <= colOf(box.centerX + 0.18 * hw); i++) {
        const v = lumS(i, j), r = at(maps.red, i, j);
        if (!Number.isFinite(v)) continue;
        s += skin - v + 0.6 * (Number.isFinite(r) ? r : 0);
        n++;
      }
      if (n && s / n > bestScore) { bestScore = s / n; mouthJ = j; }
    }
    if (mouthJ >= 0 && bestScore > 0.04) {
      L.mouthY = yOf(mouthJ);
      const score = (i: number) => { const v = lumS(i, mouthJ); const r = at(maps.red, i, mouthJ); return Number.isFinite(v) ? skin - v + 0.6 * (Number.isFinite(r) ? r : 0) : -1; };
      let a = ci, b = ci;
      while (a - 1 >= 0 && score(a - 1) > 0.45 * bestScore) a--;
      while (b + 1 < w && score(b + 1) > 0.45 * bestScore) b++;
      L.mouthHalfW = clamp(((b - a + 1) * dx) / 2, 0.22 * hw, 0.6 * hw);
      L.found.mouth = true;
    }
  }

  // Keep the order chin < mouth < nose < eyes; re-derive what breaks it.
  const D = Math.max(1e-6, L.eyeY - L.chinY);
  if (!(L.chinY < L.mouthY && L.mouthY < L.noseY && L.noseY < L.eyeY)) {
    if (!(L.noseY < L.eyeY)) { L.eyeY = L.noseY + 0.8 * D0; L.found.eyes = false; }
    if (!(L.chinY < L.mouthY && L.mouthY < L.noseY)) { L.mouthY = L.chinY + 0.55 * (L.noseY - L.chinY); L.found.mouth = false; }
  }
  if (!L.found.mouth) L.mouthHalfW = 0.85 * L.eyeX;
  L.browY = L.eyeY + 0.24 * Math.max(D, L.eyeY - L.chinY);
  L.centerX = box.centerX;
  return L;
}

// ─── landmarks from a face-mesh detector (MediaPipe Face Landmarker) ────

/**
 * Indices into the 478-point MediaPipe face mesh (Face Landmarker v2,
 * Apache-2.0) used to place the patch. "Left/Right" are the subject's own
 * sides, like ARKit: the character's left eye is on the right of a front
 * view, at +X in model space.
 */
export const FACE_MESH_POINTS = {
  rightEyeOuter: 33, rightEyeInner: 133, rightEyeUpper: 159, rightEyeLower: 145,
  leftEyeOuter: 263, leftEyeInner: 362, leftEyeUpper: 386, leftEyeLower: 374,
  rightBrow: 105, leftBrow: 334,
  noseTip: 1,
  mouthRight: 61, mouthLeft: 291,
  upperLipInner: 13, lowerLipInner: 14,
  chin: 152,
  /** Between the eyes, top of the nose bridge. */
  bridge: 168,
  /** Face oval at cheekbone height (subject's right / left). */
  rightCheek: 234, leftCheek: 454,
  /** Face oval at the jaw angle (subject's right / left). */
  rightJaw: 172, leftJaw: 397,
} as const;

/** The face-mesh silhouette, in order around the face (MediaPipe FACEMESH_FACE_OVAL). */
export const FACE_MESH_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
] as const;

/** Radial growth of the detected face outline before the patch is cut to it. */
export const OUTLINE_GROW = 1.18;

/**
 * Face landmarks from detector points already mapped into model space
 * (x, y of each point: an orthographic front view along −Z maps image
 * pixels straight to model x/y). `points` is indexed like the MediaPipe face
 * mesh (`FACE_MESH_POINTS`); only those indices are read.
 *
 * `onSurface(x, y)` — optional — says whether the head surface is under a
 * point (a ray along −Z hits the mesh); every feature point must be, so a
 * detection that landed on the background or a hat brim is refused.
 *
 * Returns null when the points are missing, out of anatomical order (chin <
 * mouth < nose < eyes < brows, eyes either side of the midline), or off the
 * surface — the caller then falls back to `detectFaceLandmarks`.
 */
export function faceLandmarksFromMeshPoints(
  points: ReadonlyArray<{ x: number; y: number } | undefined>,
  box: HeadBox,
  onSurface?: (x: number, y: number) => boolean,
): FaceLandmarks | null {
  const K = FACE_MESH_POINTS;
  const p = (i: number) => points[i];
  // The face-oval points sit on the silhouette, so only the inner features must hit the surface.
  const oval = new Set<number>([K.rightCheek, K.leftCheek, K.rightJaw, K.leftJaw]);
  for (const i of Object.values(K)) {
    const q = p(i);
    if (!q || !Number.isFinite(q.x) || !Number.isFinite(q.y)) return null;
    if (onSurface && !oval.has(i) && !onSurface(q.x, q.y)) return null;
  }
  const P = (i: number) => p(i)!;
  const mid = (a: number, b: number) => ({ x: (P(a).x + P(b).x) / 2, y: (P(a).y + P(b).y) / 2 });
  const rEye = mid(K.rightEyeOuter, K.rightEyeInner), lEye = mid(K.leftEyeOuter, K.leftEyeInner);
  const rOpen = Math.abs(P(K.rightEyeUpper).y - P(K.rightEyeLower).y), lOpen = Math.abs(P(K.leftEyeUpper).y - P(K.leftEyeLower).y);
  const rW = Math.hypot(P(K.rightEyeOuter).x - P(K.rightEyeInner).x, P(K.rightEyeOuter).y - P(K.rightEyeInner).y);
  const lW = Math.hypot(P(K.leftEyeOuter).x - P(K.leftEyeInner).x, P(K.leftEyeOuter).y - P(K.leftEyeInner).y);
  const mouth = mid(K.upperLipInner, K.lowerLipInner);
  // Midline: halfway between the eyes, averaged with the mouth and chin so a
  // slightly turned head still gets a centred patch.
  const centerX = 0.5 * ((rEye.x + lEye.x) / 2) + 0.25 * mouth.x + 0.25 * P(K.chin).x;
  const eyeY = (rEye.y + lEye.y) / 2;
  const eyeX = Math.abs(lEye.x - rEye.x) / 2;
  // Generated textures paint the eye as a soft dark blob a little larger than
  // the detector's lid contour; size the lids to cover the blob.
  const eyeHalfW = 1.2 * (rW + lW) / 4;
  const chinY = P(K.chin).y;
  const noseY = P(K.noseTip).y;
  const mouthY = mouth.y;
  const browY = (P(K.rightBrow).y + P(K.leftBrow).y) / 2;
  const mouthHalfW = Math.abs(P(K.mouthLeft).x - P(K.mouthRight).x) / 2;
  const hw = (box.maxX - box.minX) / 2;
  const faceHalfW = Math.min(
    (Math.abs(P(K.leftCheek).x - centerX) + Math.abs(P(K.rightCheek).x - centerX)) / 2,
    (Math.abs(P(K.leftJaw).x - centerX) + Math.abs(P(K.rightJaw).x - centerX)) / 2 * 1.15,
  );

  // Anatomical sanity: the detector can lock onto a face-like pattern elsewhere.
  if (!(chinY < mouthY && mouthY < noseY && noseY < eyeY && eyeY < browY)) return null;
  if (!(lEye.x > centerX && rEye.x < centerX)) return null;
  if (!(eyeHalfW > 0 && eyeX > eyeHalfW * 0.6 && mouthHalfW > 0)) return null;
  if (Math.abs(lEye.y - rEye.y) > 0.5 * (eyeY - chinY)) return null;
  if (chinY < box.minY - 0.1 * (box.maxY - box.minY) || browY > box.maxY || eyeX > hw) return null;

  // Lid opening → eyelid sheet height; the blink deformer and lid sheet span
  // eyeY + eyeHalfH (upper lid) down to eyeY − 0.55·eyeHalfH.
  const open = (rOpen + lOpen) / 2;
  return {
    centerX,
    chinY,
    mouthY,
    noseY,
    eyeY,
    browY,
    eyeX,
    eyeHalfW,
    eyeHalfH: clamp(1.1 * open, 0.45 * eyeHalfW, 0.8 * eyeHalfW),
    mouthHalfW,
    faceHalfW: clamp(faceHalfW, mouthHalfW * 1.5, hw),
    ...(FACE_MESH_OVAL.every((i) => p(i) && Number.isFinite(p(i)!.x) && Number.isFinite(p(i)!.y))
      ? { outline: FACE_MESH_OVAL.map((i) => ({ x: p(i)!.x, y: p(i)!.y })) }
      : {}),
    found: { chin: true, nose: true, eyes: true, mouth: true },
  };
}

// ─── the patch ─────────────────────────────────────────────────────────

export interface FaceRigOptions {
  /** Grid columns / rows before the landmark rows/columns are inserted. */
  columns?: number;
  rows?: number;
  sampler?: FaceTextureSampler;
  /** Use these instead of detecting. */
  landmarks?: FaceLandmarks;
  headBox?: HeadBox;
}

export interface FaceRigResult {
  /** Patch geometry in the source mesh's model space. */
  positions: Float32Array;
  normals: Float32Array;
  uvs?: Float32Array;
  colors?: Float32Array;
  indices: Uint32Array;
  /** `FACE_RIG_SHAPES.length` relative position deltas, 3 floats per vertex. */
  morphPositions: Float32Array[];
  morphNames: readonly FaceRigShape[];
  /** 1 for each source triangle the patch replaces (remove it from the body). */
  coveredTriangles: Uint8Array;
  /** 1 for each source vertex under the patch (bind it 100 % to the head bone). */
  coveredVertices: Uint8Array;
  landmarks: FaceLandmarks;
  headBox: HeadBox;
  /** Chin-to-eye distance, the unit the shapes are authored in. */
  scale: number;
  stats: { vertices: number; triangles: number; seamFixed: number; misses: number; covered: number };
}

type Kind = "face" | "lid" | "cavity" | "tongue";
interface Meta {
  kind: Kind;
  /** Face-local rest position (x relative to the midline). */
  x: number; y: number; z: number;
  /** +1 upper lip side, −1 lower (jaw) side, 0 mouth corner. */
  lip: number;
  /** Fade to the patch border (face vertices). */
  mask: number;
  /** Eye side for lids (+1 = character's left = +x). */
  side: number;
  /** Lid row 0..1 (0 = top edge). */
  lidT: number;
  /** Lid: rest → closed delta. */
  close?: [number, number, number];
}

/**
 * Build the face patch for a normalised character mesh. Returns null when
 * the head can't be found (no surface in front of the head box).
 */
export function buildFaceRig(mesh: FaceSourceMesh, opts: FaceRigOptions = {}): FaceRigResult | null {
  const box = opts.headBox ?? findHeadBox(mesh);
  const hh = box.maxY - box.minY;
  const hw = (box.maxX - box.minX) / 2;
  if (!(hh > 0) || !(hw > 0)) return null;
  // Generous margin: the neck estimate can sit above a long chin (it does on
  // the TRELLIS fixture), and the patch must reach below the chin.
  const pad = 0.3 * hh;
  const caster = new OrthoCaster(mesh, { minX: box.minX - pad, maxX: box.maxX + pad, minY: box.minY - pad, maxY: box.maxY + pad });
  const L = opts.landmarks ?? detectFaceLandmarks(faceMaps(mesh, box, caster, opts.sampler), box);
  const cx = L.centerX;
  const D = Math.max(1e-6, L.eyeY - L.chinY);
  const eps = 0.006 * D;

  // Grid extent.
  const Wp = Math.min(0.92 * hw, Math.max(L.eyeX + 1.9 * L.eyeHalfW, 1.9 * L.mouthHalfW, L.faceHalfW ?? 0));
  // Bottom edge below the chin (not clipped at the neck estimate, which can be
  // above the chin): the jaw needs patch below it to swing into.
  const yBot = Math.max(box.minY - 0.25 * hh, L.chinY - 0.24 * D);
  const yTop = Math.min(box.maxY - 0.04 * hh, L.browY + 0.32 * D);
  const NU = opts.columns ?? 30, NV = opts.rows ?? 34;
  const merge = (uniform: number[], special: number[], cell: number) => {
    const keep = uniform.filter((u) => special.every((s) => Math.abs(u - s) > 0.35 * cell));
    return [...new Set([...keep, ...special].map((v) => Math.round(v * 1e7) / 1e7))].sort((a, b) => a - b);
  };
  const cellX = (2 * Wp) / NU, cellY = (yTop - yBot) / NV;
  const xs = merge(
    Array.from({ length: NU + 1 }, (_, i) => -Wp + i * cellX),
    [0, -L.mouthHalfW, L.mouthHalfW, -(L.eyeX - L.eyeHalfW), L.eyeX - L.eyeHalfW, -(L.eyeX + L.eyeHalfW), L.eyeX + L.eyeHalfW].filter((v) => Math.abs(v) < Wp - 0.5 * cellX),
    cellX,
  );
  const ys = merge(
    Array.from({ length: NV + 1 }, (_, j) => yBot + j * cellY),
    [L.mouthY, L.eyeY].filter((v) => v > yBot + 0.5 * cellY && v < yTop - 0.5 * cellY),
    cellY,
  );
  const nx = xs.length, ny = ys.length;
  const rm = ys.findIndex((y) => Math.abs(y - L.mouthY) < 1e-6);

  const pos: number[] = [], nrm: number[] = [], uv: number[] = [], col: number[] = [];
  const meta: Meta[] = [];
  const hitTri: number[] = [];
  const useUv = !!mesh.uvs;
  const useCol = !useUv && !!mesh.colors;
  let misses = 0;

  const surface = (x: number, y: number) => caster.cast(cx + x, y);
  const pushVertex = (p: [number, number, number], n: [number, number, number], t: number[] | null, c: number[] | null, m: Meta, tri: number): number => {
    pos.push(...p); nrm.push(...n);
    if (useUv) uv.push(t?.[0] ?? 0, t?.[1] ?? 0);
    if (useCol) col.push(c?.[0] ?? 0.5, c?.[1] ?? 0.5, c?.[2] ?? 0.5, 1);
    meta.push(m); hitTri.push(tri);
    return meta.length - 1;
  };
  const normalAt = (hit: Hit): [number, number, number] => {
    if (mesh.normals) {
      const [a, b, c] = caster.attr(mesh.normals, 3, hit.tri, hit.b0, hit.b1, hit.b2);
      const l = Math.hypot(a!, b!, c!) || 1;
      return [a! / l, b! / l, c! / l];
    }
    return caster.faceNormal(hit.tri);
  };
  const uvAt = (hit: Hit) => (useUv ? caster.attr(mesh.uvs!, 2, hit.tri, hit.b0, hit.b1, hit.b2) : null);
  const colAt = (hit: Hit) => (useCol ? caster.attr(mesh.colors!, 4, hit.tri, hit.b0, hit.b1, hit.b2) : null);

  // Face grid.
  const grid = new Int32Array(nx * ny).fill(-1);
  const lower = new Int32Array(nx).fill(-1);
  const zFront = surface(0, L.noseY)?.z ?? box.maxZ;
  /** Grid points left out for being far behind the face (see below). */
  const behind = new Uint8Array(nx * ny);
  /** Where jawOpen moves the skin by more than 30 % (below the mouth, inside the jaw's reach). */
  const jawZone = (x: number, y: number) => y < L.mouthY && 1 - smoothstep(L.mouthHalfW * 1.05, Wp, Math.abs(x)) > 0.3;
  // Detected outline, grown about the nose: grid points outside it are left
  // out (treated like misses, so the border fade starts at the outline).
  const outline = L.outline && L.outline.length >= 3
    ? L.outline.map((q) => ({
      x: (q.x - cx) * OUTLINE_GROW,
      // …and further down around the jaw, so the chin moves with the jaw and
      // the stretch happens under it, on the neck, rather than across the chin.
      y: L.noseY + (q.y - L.noseY) * OUTLINE_GROW - 0.3 * D * clamp((L.mouthY - q.y) / Math.max(1e-6, L.mouthY - L.chinY), 0, 1),
    }))
    : null;
  const inOutline = (x: number, y: number) => {
    if (!outline) return true;
    let inside = false;
    for (let a = 0, b = outline.length - 1; a < outline.length; b = a++) {
      const A = outline[a]!, B = outline[b]!;
      if ((A.y > y) !== (B.y > y) && x < ((B.x - A.x) * (y - A.y)) / (B.y - A.y) + A.x) inside = !inside;
    }
    return inside;
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = xs[i]!, y = ys[j]!;
      if (!inOutline(x, y)) continue;
      const hit = surface(x, y);
      if (!hit) { misses++; continue; }
      // Far behind the face front (the neck behind the jaw, a hanging earring,
      // the back of a head wrap): not face. Left out like a miss.
      // Under the jaw allow deeper (the neck right under the chin is the wall
      // the jaw swings over); further back than that is the far side of the neck.
      if (hit.z < zFront - (jawZone(x, y) ? 1.2 : 0.9) * D) { misses++; behind[j * nx + i] = 1; continue; }
      const n = normalAt(hit);
      const p: [number, number, number] = [cx + x + n[0] * eps, y + n[1] * eps, hit.z + n[2] * eps];
      const inMouth = j === rm && Math.abs(x) < L.mouthHalfW - 1e-9;
      const lip = j === rm ? (inMouth ? 1 : 0) : y > L.mouthY ? 1 : -1;
      const m: Meta = { kind: "face", x, y, z: hit.z, lip, mask: 0, side: 0, lidT: 0 };
      grid[j * nx + i] = pushVertex(p, n, uvAt(hit), colAt(hit), m, hit.tri);
      if (inMouth) lower[i] = pushVertex(p, n, uvAt(hit), colAt(hit), { ...m, lip: -1 }, hit.tri);
    }
  }

  // Border fade: grid distance (in cells) to the rectangle edge or a miss.
  const dist = new Float32Array(nx * ny).fill(Infinity);
  const queue: number[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i;
    // Under the jaw the patch ends where the surface drops away behind the
    // chin; that edge must swing with the jaw, so it does not start a fade.
    if (grid[k]! < 0) { if (!(behind[k] && jawZone(xs[i]!, ys[j]!))) { dist[k] = 0; queue.push(k); } }
    else if (i === 0 || j === 0 || i === nx - 1 || j === ny - 1) { dist[k] = 0; queue.push(k); }
  }
  for (let q = 0; q < queue.length; q++) {
    const k = queue[q]!, i = k % nx, j = (k - i) / nx;
    for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const ii = i + a, jj = j + b;
      if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
      const kk = jj * nx + ii;
      if (dist[kk]! > dist[k]! + 1) { dist[kk] = dist[k]! + 1; queue.push(kk); }
    }
  }
  const FADE = 3;
  for (let k = 0; k < nx * ny; k++) {
    const v = grid[k]!;
    if (v < 0) continue;
    const m = smoothstep(0, FADE, dist[k]!);
    meta[v]!.mask = m;
    const i = k % nx;
    if ((k - i) / nx === rm && lower[i]! >= 0) meta[lower[i]!]!.mask = m;
  }

  const idx: number[] = [];
  const CLIFF = 0.2 * D;
  /** Grid cells left out as cliffs (index j * nx + i of the cell's lower-left corner). */
  const cliff = new Uint8Array(nx * ny);
  const tri = (a: number, b: number, c: number) => { if (a >= 0 && b >= 0 && c >= 0) idx.push(a, b, c); };
  for (let j = 0; j + 1 < ny; j++) {
    for (let i = 0; i + 1 < nx; i++) {
      const top = (ii: number) => (j + 1 === rm && lower[ii]! >= 0 ? lower[ii]! : grid[(j + 1) * nx + ii]!);
      const v00 = grid[j * nx + i]!, v10 = grid[j * nx + i + 1]!, v11 = top(i + 1), v01 = top(i);
      if (v00 < 0 || v10 < 0 || v11 < 0 || v01 < 0) continue;
      // A cliff — the side of the face against a head wrap or hair, the
      // underside of the chin — projects to one thin cell spanning
      // centimetres of depth, and its texture smears across atlas charts
      // into streaks seen from the side. Leave the cell out and keep the
      // source surface under it (see maskAt below) — except where the jaw
      // swings (below the mouth, where jawOpen moves it > 30 %): a free edge there
      // would open a jagged gap along the jawline when the mouth opens.
      const z0 = meta[v00]!.z, z1 = meta[v10]!.z, z2 = meta[v11]!.z, z3 = meta[v01]!.z;
      const jawCell = jawZone(Math.max(Math.abs(xs[i]!), Math.abs(xs[i + 1]!)), ys[j + 1]!);
      const zTop = Math.max(z0, z1, z2, z3);
      if (zTop - Math.min(z0, z1, z2, z3) > CLIFF) {
        if (!jawCell) { cliff[j * nx + i] = 1; continue; }
        // Kept for the jaw: give its far corners (on the neck behind) the
        // texture of the front corner, so the wall reads as skin under the
        // jaw instead of streaks of whatever atlas charts lie between.
        const front = [v00, v10, v11, v01][[z0, z1, z2, z3].indexOf(zTop)]!;
        const skinCopy = (v: number) => {
          if (meta[v]!.z > zTop - 0.5 * CLIFF) return v;
          return pushVertex([pos[v * 3]!, pos[v * 3 + 1]!, pos[v * 3 + 2]!], [nrm[v * 3]!, nrm[v * 3 + 1]!, nrm[v * 3 + 2]!],
            useUv ? [uv[front * 2]!, uv[front * 2 + 1]!] : null, useCol ? col.slice(front * 4, front * 4 + 4) : null, { ...meta[v]! }, hitTri[front]!);
        };
        const a = skinCopy(v00), b = skinCopy(v10), c = skinCopy(v11), d = skinCopy(v01);
        tri(a, b, c); tri(a, c, d);
        continue;
      }
      tri(v00, v10, v11); tri(v00, v11, v01);
    }
  }
  const faceTriEnd = idx.length;

  // Eyelid sheets: collapsed on the upper eye edge at rest; closed covers the eye.
  const KC = 9, KR = 4;
  const lift = 0.012 * D;
  const eyeEdge = (u: number) => Math.sqrt(Math.max(0, 1 - u * u));
  for (const side of [1, -1] as const) {
    const ids: number[] = [];
    for (let r = 0; r < KR; r++) {
      const t = r / (KR - 1);
      for (let c = 0; c < KC; c++) {
        const u = -1 + (2 * c) / (KC - 1);
        const x = side * L.eyeX + u * L.eyeHalfW;
        const e = eyeEdge(u);
        const yT = L.eyeY + L.eyeHalfH * e;
        const yB = L.eyeY - 0.55 * L.eyeHalfH * e;
        const rest = surface(x, yT);
        const yC = yT - (yT - yB) * t;
        const shut = surface(x, yC) ?? rest;
        const skin = surface(x, yT + L.eyeHalfH * (0.55 - 0.42 * t) + 0.04 * D * (1 - e)) ?? rest;
        if (!rest || !shut) { ids.push(-1); continue; }
        const n = normalAt(rest);
        const p: [number, number, number] = [cx + x + n[0] * lift, yT + n[1] * lift, rest.z + n[2] * lift];
        const ns = normalAt(shut);
        const q: [number, number, number] = [cx + x + ns[0] * lift * (1 + 0.6 * e), yC + ns[1] * lift, shut.z + ns[2] * lift * (1 + 0.6 * e)];
        const m: Meta = { kind: "lid", x, y: yT, z: rest.z, lip: 1, mask: 1, side, lidT: t, close: [q[0] - p[0], q[1] - p[1], q[2] - p[2]] };
        ids.push(pushVertex(p, n, skin ? uvAt(skin) : uvAt(rest), skin ? colAt(skin) : colAt(rest), m, rest.tri));
      }
    }
    for (let r = 0; r + 1 < KR; r++) {
      for (let c = 0; c + 1 < KC; c++) {
        const tl = ids[r * KC + c]!, tr = ids[r * KC + c + 1]!, bl = ids[(r + 1) * KC + c]!, br = ids[(r + 1) * KC + c + 1]!;
        tri(bl, br, tr); tri(bl, tr, tl);
      }
    }
  }

  // Mouth cavity (dark) and tongue behind the lips.
  const mouthHit = surface(0, L.mouthY);
  const darkUv = useUv && opts.sampler ? darkestUv(opts.sampler) : mouthHit ? uvAt(mouthHit) : null;
  const lipHit = surface(0, L.mouthY - 0.05 * D) ?? mouthHit;
  const lipUv = lipHit ? uvAt(lipHit) : null;
  const zMouth = mouthHit?.z ?? box.maxZ;
  // The cavity and tongue are flat-ish sheets behind the lips; where the face
  // recedes (the lower lip and chin slope back) a flat sheet would poke through
  // the closed mouth, so keep every inner vertex at least `gap` behind the surface.
  const behindSurface = (x: number, y: number, z: number, gap: number) => {
    const h = surface(x, y);
    return h ? Math.min(z, h.z - gap) : z;
  };
  if (mouthHit) {
    const CC = 9, CR = 6;
    const ids: number[] = [];
    for (let r = 0; r < CR; r++) {
      const y = L.mouthY + 0.04 * D - (0.16 * D * r) / (CR - 1);
      for (let c = 0; c < CC; c++) {
        const u = -1 + (2 * c) / (CC - 1);
        const x = u * 1.08 * L.mouthHalfW;
        const z = behindSurface(x, y, zMouth - 0.045 * D - 0.05 * D * u * u, 0.035 * D);
        const m: Meta = { kind: "cavity", x, y, z, lip: y < L.mouthY ? -1 : 1, mask: 1, side: 0, lidT: 0 };
        ids.push(pushVertex([cx + x, y, z], [0, 0, 1], darkUv, useCol ? [0.07, 0.03, 0.03, 1] : null, m, mouthHit.tri));
      }
    }
    for (let r = 0; r + 1 < CR; r++) for (let c = 0; c + 1 < CC; c++) {
      const tl = ids[r * CC + c]!, tr = ids[r * CC + c + 1]!, bl = ids[(r + 1) * CC + c]!, br = ids[(r + 1) * CC + c + 1]!;
      tri(bl, br, tr); tri(bl, tr, tl);
    }
    const TC = 5, TR = 3;
    const tids: number[] = [];
    for (let r = 0; r < TR; r++) {
      const t = r / (TR - 1);
      for (let c = 0; c < TC; c++) {
        const u = -1 + (2 * c) / (TC - 1);
        const x = u * 0.55 * L.mouthHalfW * (1 - 0.25 * t);
        const y = L.mouthY - 0.045 * D - 0.015 * D * Math.abs(u);
        const z = behindSurface(x, y, zMouth - 0.035 * D - 0.045 * D * (1 - t) - 0.02 * D * u * u, 0.03 * D);
        const m: Meta = { kind: "tongue", x, y, z, lip: -1, mask: 1, side: 0, lidT: t };
        tids.push(pushVertex([cx + x, y, z], [0, 0.3, 0.95], lipUv, useCol ? [0.55, 0.22, 0.24, 1] : null, m, mouthHit.tri));
      }
    }
    for (let r = 0; r + 1 < TR; r++) for (let c = 0; c + 1 < TC; c++) {
      const back0 = tids[r * TC + c]!, back1 = tids[r * TC + c + 1]!, tip0 = tids[(r + 1) * TC + c]!, tip1 = tids[(r + 1) * TC + c + 1]!;
      tri(back0, tip0, tip1); tri(back0, tip1, back1);
    }
  }

  // Texture-atlas seams: a face triangle whose UVs jump (corners in different
  // charts) gets its own vertices with UVs extrapolated inside one chart.
  let seamFixed = 0;
  if (useUv) {
    const ratios: number[] = [];
    const ratio = (t: number) => {
      let r = 0;
      for (let e = 0; e < 3; e++) {
        const a = idx[t + e]!, b = idx[t + ((e + 1) % 3)]!;
        const duv = Math.hypot(uv[a * 2]! - uv[b * 2]!, uv[a * 2 + 1]! - uv[b * 2 + 1]!);
        const dxy = Math.hypot(pos[a * 3]! - pos[b * 3]!, pos[a * 3 + 1]! - pos[b * 3 + 1]!, pos[a * 3 + 2]! - pos[b * 3 + 2]!);
        r = Math.max(r, duv / Math.max(dxy, 1e-9));
      }
      return r;
    };
    for (let t = 0; t < faceTriEnd; t += 3) ratios.push(ratio(t));
    const med = percentile(ratios, 0.5);
    for (let t = 0; t < faceTriEnd; t += 3) {
      if (ratios[t / 3]! <= 5 * med) continue;
      const vs = [idx[t]!, idx[t + 1]!, idx[t + 2]!];
      const ccx = (meta[vs[0]!]!.x + meta[vs[1]!]!.x + meta[vs[2]!]!.x) / 3;
      const ccy = (meta[vs[0]!]!.y + meta[vs[1]!]!.y + meta[vs[2]!]!.y) / 3;
      const hit = surface(ccx, ccy);
      if (!hit) continue;
      for (let e = 0; e < 3; e++) {
        const v = vs[e]!;
        const b = caster.baryIn(hit.tri, cx + meta[v]!.x, meta[v]!.y);
        if (!b) continue;
        const t2 = caster.attr(mesh.uvs!, 2, hit.tri, b[0], b[1], b[2]);
        const copy = pushVertex([pos[v * 3]!, pos[v * 3 + 1]!, pos[v * 3 + 2]!], [nrm[v * 3]!, nrm[v * 3 + 1]!, nrm[v * 3 + 2]!], t2, null, { ...meta[v]! }, hit.tri);
        idx[t + e] = copy;
      }
      seamFixed++;
    }
  }

  // Shapes.
  const nV = meta.length;
  const morphPositions = FACE_RIG_SHAPES.map(() => new Float32Array(nV * 3));
  const pivot = { y: L.mouthY + 0.34 * D, z: zMouth - 0.95 * D };
  for (let v = 0; v < nV; v++) {
    const m = meta[v]!;
    for (let s = 0; s < FACE_RIG_SHAPES.length; s++) {
      const d = shapeDelta(FACE_RIG_SHAPES[s]!, m, L, D, Wp, yBot, pivot);
      if (!d) continue;
      const k = m.kind === "face" ? m.mask : 1;
      morphPositions[s]![v * 3] = d[0] * k;
      morphPositions[s]![v * 3 + 1] = d[1] * k;
      morphPositions[s]![v * 3 + 2] = d[2] * k;
    }
  }

  // Source triangles / vertices under the patch.
  const P = mesh.positions;
  const nSrc = P.length / 3;
  const coveredVertices = new Uint8Array(nSrc);
  const strong = new Uint8Array(nSrc);
  const maskAt = (x: number, y: number): number => {
    // Bilinear on the (non-uniform) grid.
    if (x <= xs[0]! || x >= xs[nx - 1]! || y <= ys[0]! || y >= ys[ny - 1]!) return 0;
    let i = 0; while (i + 2 < nx && xs[i + 1]! < x) i++;
    let j = 0; while (j + 2 < ny && ys[j + 1]! < y) j++;
    // Under a cliff cell, or a cell with a corner left out, the patch has no
    // surface (or a free edge that moves with the jaw): keep the source there.
    if (cliff[j * nx + i]) return 0;
    if (grid[j * nx + i]! < 0 || grid[j * nx + i + 1]! < 0 || grid[(j + 1) * nx + i]! < 0 || grid[(j + 1) * nx + i + 1]! < 0) return 0;
    const fx = (x - xs[i]!) / (xs[i + 1]! - xs[i]!), fy = (y - ys[j]!) / (ys[j + 1]! - ys[j]!);
    const g = (ii: number, jj: number) => { const v = grid[jj * nx + ii]!; return v >= 0 ? meta[v]!.mask : 0; };
    return (g(i, j) * (1 - fx) + g(i + 1, j) * fx) * (1 - fy) + (g(i, j + 1) * (1 - fx) + g(i + 1, j + 1) * fx) * fy;
  };
  let covered = 0;
  for (let v = 0; v < nSrc; v++) {
    const x = P[v * 3]! - cx, y = P[v * 3 + 1]!, z = P[v * 3 + 2]!;
    if (x < -Wp || x > Wp || y < yBot || y > yTop) continue;
    const hit = surface(x, y);
    if (!hit || Math.abs(z - hit.z) > 0.05 * D) continue;
    coveredVertices[v] = 1;
    if (maskAt(x, y) > 0.6) strong[v] = 1;
  }
  const I = mesh.indices;
  const coveredTriangles = new Uint8Array(I.length / 3);
  for (let t = 0; t < coveredTriangles.length; t++) {
    if (strong[I[t * 3]!] && strong[I[t * 3 + 1]!] && strong[I[t * 3 + 2]!]) { coveredTriangles[t] = 1; covered++; }
  }

  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    ...(useUv ? { uvs: new Float32Array(uv) } : {}),
    ...(useCol ? { colors: new Float32Array(col) } : {}),
    indices: new Uint32Array(idx),
    morphPositions,
    morphNames: FACE_RIG_SHAPES,
    coveredTriangles,
    coveredVertices,
    landmarks: L,
    headBox: box,
    scale: D,
    stats: { vertices: nV, triangles: idx.length / 3, seamFixed, misses, covered },
  };
}

/** UV of the darkest small region of the texture (the cavity colour). */
function darkestUv(sampler: FaceTextureSampler): number[] {
  const N = 96;
  const lum = new Float32Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const c = sampler((i + 0.5) / N, (j + 0.5) / N);
    lum[j * N + i] = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  let best = Infinity, bi = 0, bj = 0;
  for (let j = 2; j < N - 2; j++) for (let i = 2; i < N - 2; i++) {
    let mx = 0;
    for (let b = -2; b <= 2; b++) for (let a = -2; a <= 2; a++) mx = Math.max(mx, lum[(j + b) * N + i + a]!);
    if (mx < best) { best = mx; bi = i; bj = j; }
  }
  return [(bi + 0.5) / N, (bj + 0.5) / N];
}

const g2 = (dx: number, dy: number, rx: number, ry: number) => Math.exp(-((dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)));

/**
 * Delta of one ARKit shape at one patch vertex, face-local, before the border
 * fade. Exported for tests. Left/Right are the character's own sides
 * (character faces +Z, so its left is +X).
 */
export function shapeDelta(
  shape: FaceRigShape,
  m: Pick<Meta, "kind" | "x" | "y" | "z" | "lip" | "side" | "lidT" | "close">,
  L: FaceLandmarks,
  D: number,
  Wp: number,
  yBot: number,
  pivot: { y: number; z: number },
): [number, number, number] | null {
  const { x, y } = m;
  const ax = Math.abs(x);
  const sx = Math.sign(x);
  const mw = L.mouthHalfW;

  // Lids only blink / squint.
  if (m.kind === "lid") {
    const c = m.close!;
    const own = (m.side > 0 && shape.endsWith("Left")) || (m.side < 0 && shape.endsWith("Right"));
    if (!own) return null;
    if (shape.startsWith("eyeBlink")) return [c[0], c[1], c[2]];
    if (shape.startsWith("eyeSquint")) return [c[0] * 0.3, c[1] * 0.3, c[2] * 0.3];
    return null;
  }

  // Jaw side weight: lower face, fading out sideways and toward the neck.
  const jawW = (): number => {
    if (m.kind === "cavity" || m.kind === "tongue") return m.lip < 0 ? 1 : 0;
    if (m.lip > 0) return 0;
    const h = 1 - smoothstep(mw * 1.05, Wp, ax);
    const v = y < L.chinY ? smoothstep(yBot, L.chinY, y) : 1;
    if (m.lip === 0) return 0.45 * (1 - smoothstep(mw, 1.5 * mw, ax)) * h * v;
    // Past the mouth corners the skin is not split, so the cheek must stretch:
    // the jaw's pull grows with depth below the mouth line, over a run that
    // lengthens away from the corner. Without this the row under the mouth
    // line drops a full jaw-height while the row above stays — a shear line
    // from each corner out to the edge of the patch.
    const out = Math.max(0, ax - mw);
    const ramp = out > 0 ? smoothstep(0, 0.08 * D + 1.2 * out, L.mouthY - y) : 1;
    // Inside the mouth the lower lip opens less toward the corners (a lens,
    // not a letterbox); the taper fades out below the lip.
    const lipT = 1 - smoothstep(0, 0.18 * D, L.mouthY - y);
    const taper = out > 0 ? 1 : 1 - lipT * 0.55 * smoothstep(0.3 * mw, mw, ax);
    return h * v * ramp * taper;
  };
  const rotJaw = (angle: number): [number, number, number] => {
    const dy = y - pivot.y, dz = m.z - pivot.z;
    const c = Math.cos(angle), s = Math.sin(angle);
    return [0, dy * c - dz * s - dy, dy * s + dz * c - dz];
  };
  // Lips: a band around the mouth line, narrowing past the corners.
  const lipBand = g2(0, y - L.mouthY, 1, 0.1 * D) * (1 - smoothstep(mw * 0.85, mw * 1.6, ax));
  const upperLip = m.lip > 0 ? lipBand : m.lip === 0 ? 0.5 * lipBand : 0;
  const lowerLip = m.lip < 0 ? lipBand : m.lip === 0 ? 0.5 * lipBand : 0;
  const corner = (side: number) => (m.kind === "face" ? g2(x - side * mw, y - L.mouthY, 0.42 * D, 0.26 * D) : 0);
  const sideOf = (s: FaceRigShape) => (s.endsWith("Left") ? 1 : s.endsWith("Right") ? -1 : 0);
  const halfSide = (side: number) => smoothstep(-0.25 * mw, 0.35 * mw, side * x);

  if (m.kind === "cavity" || m.kind === "tongue") {
    switch (shape) {
      case "jawOpen": return m.lip < 0 ? rotJaw(0.3) : null;
      case "jawForward": return m.lip < 0 ? [0, 0, 0.04 * D] : null;
      case "jawLeft": return m.lip < 0 ? [0.05 * D, 0, 0] : null;
      case "jawRight": return m.lip < 0 ? [-0.05 * D, 0, 0] : null;
      case "tongueOut": return m.kind === "tongue" ? [0, 0.012 * D + 0.01 * D * m.lidT, 0.12 * D * (0.5 + 0.5 * m.lidT)] : null;
      default: return null;
    }
  }

  const side = sideOf(shape);
  switch (shape) {
    case "jawOpen": { const w = jawW(); return w ? rotJaw(0.3 * w) : null; }
    case "jawForward": { const w = jawW(); return w ? [0, 0, 0.045 * D * w] : null; }
    case "jawLeft": case "jawRight": { const w = jawW(); return w ? [side * 0.055 * D * w, 0, 0] : null; }
    case "mouthClose": {
      // Lips meet while the jaw is open: the lower lip rides back up by what jawOpen drops it.
      const w = jawW();
      if (lowerLip > 0 && w) { const j = rotJaw(0.3 * w); return [0, -j[1] * lowerLip * 0.95, -j[2] * lowerLip]; }
      return upperLip > 0 ? [0, -0.015 * D * upperLip, 0] : null;
    }
    case "mouthFunnel": {
      const b = Math.max(upperLip, lowerLip);
      if (!b) return null;
      return [-x * 0.22 * b, (m.lip > 0 ? 0.025 : -0.035) * D * b, 0.05 * D * b];
    }
    case "mouthPucker": {
      const b = Math.max(upperLip, lowerLip);
      return b ? [-x * 0.38 * b, 0, 0.07 * D * b] : null;
    }
    case "mouthLeft": case "mouthRight": {
      const b = g2(x, y - L.mouthY, 1.5 * mw, 0.2 * D);
      return [side * 0.28 * mw * b, 0, 0];
    }
    case "mouthSmileLeft": case "mouthSmileRight": {
      const b = corner(side);
      return b > 1e-4 ? [side * 0.05 * D * b, 0.075 * D * b, -0.02 * D * b] : null;
    }
    case "mouthFrownLeft": case "mouthFrownRight": {
      const b = corner(side);
      return b > 1e-4 ? [side * 0.01 * D * b, -0.06 * D * b, 0] : null;
    }
    case "mouthDimpleLeft": case "mouthDimpleRight": {
      const b = corner(side);
      return b > 1e-4 ? [side * 0.025 * D * b, 0, -0.035 * D * b] : null;
    }
    case "mouthStretchLeft": case "mouthStretchRight": {
      const b = corner(side);
      return b > 1e-4 ? [side * 0.07 * D * b, -0.012 * D * b, -0.01 * D * b] : null;
    }
    case "mouthRollLower": return lowerLip ? [0, 0.012 * D * lowerLip, -0.035 * D * lowerLip] : null;
    case "mouthRollUpper": return upperLip ? [0, -0.012 * D * upperLip, -0.035 * D * upperLip] : null;
    case "mouthShrugLower": {
      const chin = m.lip < 0 ? g2(x, y - (L.mouthY + L.chinY) / 2, 1.3 * mw, 0.35 * D) : 0;
      return chin > 1e-4 ? [0, 0.035 * D * chin, 0.01 * D * chin] : null;
    }
    case "mouthShrugUpper": return upperLip ? [0, 0.03 * D * upperLip, 0.005 * D * upperLip] : null;
    case "mouthPressLeft": case "mouthPressRight": {
      const h = halfSide(side);
      if (upperLip) return [0, -0.012 * D * upperLip * h, 0];
      if (lowerLip) return [0, 0.012 * D * lowerLip * h, 0];
      return null;
    }
    case "mouthLowerDownLeft": case "mouthLowerDownRight": return lowerLip ? [0, -0.05 * D * lowerLip * halfSide(side), 0.005 * D * lowerLip] : null;
    case "mouthUpperUpLeft": case "mouthUpperUpRight": return upperLip ? [0, 0.045 * D * upperLip * halfSide(side), 0.004 * D * upperLip] : null;
    case "browDownLeft": case "browDownRight": {
      const b = g2(x - side * L.eyeX * 0.85, y - L.browY, 0.85 * L.eyeX, 0.2 * D);
      return b > 1e-4 ? [-side * 0.012 * D * b, -0.045 * D * b, 0.004 * D * b] : null;
    }
    case "browInnerUp": {
      const b = g2(ax - 0.35 * L.eyeX, y - L.browY, 0.45 * L.eyeX, 0.2 * D);
      return b > 1e-4 ? [0, 0.055 * D * b, 0] : null;
    }
    case "browOuterUpLeft": case "browOuterUpRight": {
      const b = g2(x - side * L.eyeX * 1.25, y - L.browY, 0.5 * L.eyeX, 0.2 * D);
      return b > 1e-4 ? [0, 0.05 * D * b, 0] : null;
    }
    case "cheekPuff": {
      const b = g2(ax - 1.45 * mw, y - (L.mouthY + 0.35 * (L.noseY - L.mouthY)), 0.55 * mw, 0.28 * D);
      return b > 1e-4 ? [sx * 0.03 * D * b, 0, 0.05 * D * b] : null;
    }
    case "cheekSquintLeft": case "cheekSquintRight": {
      const b = g2(x - side * L.eyeX, y - (L.eyeY - 0.35 * (L.eyeY - L.noseY) - 0.1 * D), 0.6 * L.eyeX, 0.18 * D);
      return b > 1e-4 ? [0, 0.03 * D * b, 0.008 * D * b] : null;
    }
    case "noseSneerLeft": case "noseSneerRight": {
      const b = g2(x - side * 0.3 * L.eyeX, y - L.noseY, 0.3 * L.eyeX, 0.16 * D);
      return b > 1e-4 ? [0, 0.03 * D * b, 0.006 * D * b] : null;
    }
    case "tongueOut": return null;
    default: break;
  }

  // Eyes: the painted eye can only move with the surface around it.
  if (shape.startsWith("eye")) {
    const eSide = side;
    const ex = x - eSide * L.eyeX, ey = y - L.eyeY;
    const inside = g2(ex, ey, 1.1 * L.eyeHalfW, 1.2 * L.eyeHalfH);
    const above = g2(ex, y - (L.eyeY + 1.4 * L.eyeHalfH), 1.3 * L.eyeHalfW, 0.9 * L.eyeHalfH);
    const below = g2(ex, y - (L.eyeY - 1.3 * L.eyeHalfH), 1.2 * L.eyeHalfW, 0.8 * L.eyeHalfH);
    const look = 0.14;
    switch (shape) {
      case "eyeBlinkLeft": case "eyeBlinkRight": {
        // Compress the painted eye into the closing crease beneath the lid.
        // Underlying geometry must follow even where the lid is occluded.
        const creaseY = L.eyeY - 0.35 * L.eyeHalfH;
        return x * eSide > 0 && inside > 1e-4
          ? [0, (creaseY - y) * inside, 0.012 * D * inside] : null;
      }
      case "eyeSquintLeft": case "eyeSquintRight": return below > 1e-4 ? [0, 0.35 * L.eyeHalfH * below, 0.004 * D * below] : null;
      case "eyeWideLeft": case "eyeWideRight": return above > 1e-4 ? [0, 0.3 * L.eyeHalfH * above, 0] : null;
      case "eyeLookUpLeft": case "eyeLookUpRight": return inside > 1e-4 ? [0, look * L.eyeHalfH * inside, 0] : null;
      case "eyeLookDownLeft": case "eyeLookDownRight": return inside > 1e-4 ? [0, -look * L.eyeHalfH * inside, 0] : null;
      // "In" turns toward the nose: −x for the left (+x) eye.
      case "eyeLookInLeft": case "eyeLookInRight": return inside > 1e-4 ? [-eSide * look * L.eyeHalfW * inside, 0, 0] : null;
      case "eyeLookOutLeft": case "eyeLookOutRight": return inside > 1e-4 ? [eSide * look * L.eyeHalfW * inside, 0, 0] : null;
      default: return null;
    }
  }
  return null;
}

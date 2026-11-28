import jpeg from "jpeg-js";
import { describe, expect, it } from "vitest";
import { bakeLods } from "./bake.js";
import { createImageDecoder, decodeImageNode, decodeJpeg, decodeWebp } from "./decode.js";
import { textureImage, type LodMaterialSet } from "./material.js";
import { triangleCount, type KitMesh } from "./mesh.js";
import { encodePngStored, imageSize, isJpeg, isWebp, type RgbaImage } from "./png.js";
import { SEAM_EDGE, SEAM_INTERIOR, SEAM_JUNCTION, seamClasses, uvIslandBreaks, uvIslands, weldUvSeams } from "./seams.js";

/** 8x8 lossless WebP: yellow/blue 2x2 checker (made with @jsquash/webp's encoder). */
const WEBP_CHECKER = Uint8Array.from(atob("UklGRiIAAABXRUJQVlA4TBUAAAAvB8ABAA8w//O///MfPPgDfET/4wEA"), (c) => c.charCodeAt(0));

function checker(n: number, a: number[], b: number[]): RgbaImage {
  const rgba = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const c = (x < n / 2) !== (y < n / 2) ? a : b;
    rgba.set([c[0]!, c[1]!, c[2]!, 255], (y * n + x) * 4);
  }
  return { width: n, height: n, rgba };
}

function jpegChecker(n: number): Uint8Array {
  const img = checker(n, [255, 255, 0], [0, 0, 255]);
  return new Uint8Array(jpeg.encode({ data: img.rgba, width: n, height: n }, 95).data);
}

const near = (c: ArrayLike<number>, want: number[], tol: number) => want.every((w, k) => Math.abs(c[k]! - w) <= tol);

describe("image decoding (Node)", () => {
  it("decodes JPEG with jpeg-js: the checker's colours come back", () => {
    const bytes = jpegChecker(32);
    expect(isJpeg(bytes)).toBe(true);
    expect(imageSize(bytes, "image/jpeg")).toEqual({ width: 32, height: 32 });
    const img = decodeJpeg(bytes)!;
    expect(img.width).toBe(32);
    const px = (x: number, y: number) => img.rgba.subarray((y * 32 + x) * 4, (y * 32 + x) * 4 + 4);
    expect(near(px(4, 4), [0, 0, 255, 255], 24)).toBe(true); // blue quadrant
    expect(near(px(28, 4), [255, 255, 0, 255], 24)).toBe(true); // yellow quadrant
    expect(decodeImageNode({ bytes, mimeType: "image/jpeg" })!.width).toBe(32);
  });

  it("recognises formats by magic bytes, not the declared mimeType", () => {
    const bytes = jpegChecker(16);
    expect(decodeImageNode({ bytes, mimeType: "image/png" })!.width).toBe(16);
    expect(decodeJpeg(new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]))).toBeNull(); // truncated: null, no throw
  });

  it("decodes WebP (lossless) exactly, and reads its size from the header", async () => {
    expect(isWebp(WEBP_CHECKER)).toBe(true);
    expect(imageSize(WEBP_CHECKER, "image/webp")).toEqual({ width: 8, height: 8 });
    const img = (await decodeWebp(WEBP_CHECKER))!;
    expect([img.width, img.height]).toEqual([8, 8]);
    expect(Array.from(img.rgba.subarray(0, 4))).toEqual([0, 0, 255, 255]);
    expect(Array.from(img.rgba.subarray(7 * 4, 8 * 4))).toEqual([255, 255, 0, 255]);
    expect(await decodeWebp(new Uint8Array(20))).toBeNull();
  });

  it("createImageDecoder covers PNG, JPEG and WebP synchronously; KTX2 stays null", async () => {
    const ktx = new Uint8Array(64);
    ktx.set([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);
    const images = [{ bytes: WEBP_CHECKER, mimeType: "image/webp" }, { bytes: jpegChecker(8), mimeType: "image/jpeg" }, { bytes: ktx, mimeType: "image/ktx2" }];
    const dec = await createImageDecoder({ materials: [], textures: [], samplers: [], images, extensionsUsed: [], extensionsRequired: [] });
    expect(dec(images[0]!)!.width).toBe(8);
    expect(dec(images[1]!)!.width).toBe(8);
    expect(dec(images[2]!)).toBeNull();
  });

  it("textureImage follows EXT_texture_webp when the texture has no fallback source", () => {
    const set = { textures: [{ extensions: { EXT_texture_webp: { source: 3 } } }, { source: 1, extensions: { EXT_texture_webp: { source: 2 } } }] } as unknown as LodMaterialSet;
    expect(textureImage(set, 0)).toBe(3);
    expect(textureImage(set, 1)).toBe(1); // the fallback is what every loader can read
  });
});

/** A 2 x 4 x 2 box, sides textured with image 0 through `texture`, top plain red. */
function tallBox(images: LodMaterialSet["images"], texture: any): { mesh: KitMesh; set: LodMaterialSet } {
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], side: number[] = [], top: number[] = [];
  const quad = (pts: number[][], n: number[], into: number[]) => {
    const b = P.length / 3;
    pts.forEach((p, i) => { P.push(...p); N.push(...n); UV.push(i === 1 || i === 2 ? 1 : 0, i >= 2 ? 1 : 0); C.push(1, 1, 1, 1); });
    into.push(b, b + 1, b + 2, b, b + 2, b + 3);
  };
  quad([[1, 0, 1], [1, 0, -1], [1, 4, -1], [1, 4, 1]], [1, 0, 0], side);
  quad([[-1, 0, -1], [-1, 0, 1], [-1, 4, 1], [-1, 4, -1]], [-1, 0, 0], side);
  quad([[-1, 0, 1], [1, 0, 1], [1, 4, 1], [-1, 4, 1]], [0, 0, 1], side);
  quad([[1, 0, -1], [-1, 0, -1], [-1, 4, -1], [1, 4, -1]], [0, 0, -1], side);
  quad([[-1, 4, 1], [1, 4, 1], [1, 4, -1], [-1, 4, -1]], [0, 1, 0], top);
  return {
    mesh: { positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV), colors: Float32Array.from(C), indices: Uint32Array.from([...side, ...top]), groups: [{ material: 0, start: 0, count: side.length }, { material: 1, start: side.length, count: top.length }] },
    set: {
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }, { pbrMetallicRoughness: { baseColorFactor: [0.9, 0.1, 0.1, 1] } }],
      textures: [texture], samplers: [], images, extensionsUsed: [], extensionsRequired: [],
    },
  };
}

function sideCellColours(atlas: RgbaImage): number[][] {
  const out: number[][] = [];
  for (let y = 2; y < 30; y += 4) for (let x = 2; x < 30; x += 4) out.push(Array.from(atlas.rgba.subarray((y * atlas.width + x) * 4, (y * atlas.width + x) * 4 + 4)));
  return out;
}

describe("impostor atlas from JPEG and WebP base colours", () => {
  for (const [label, images, texture] of [
    ["JPEG", [{ bytes: jpegChecker(64), mimeType: "image/jpeg" }], { source: 0 }],
    ["WebP (EXT_texture_webp, no fallback)", [{ bytes: WEBP_CHECKER, mimeType: "image/webp" }], { extensions: { EXT_texture_webp: { source: 0 } } }],
  ] as const) {
    it(`${label}: the side views carry the texture's colours, not white`, async () => {
      const { mesh, set } = tallBox([...images], texture);
      const r = await bakeLods(mesh, { materials: set, impostor: "box", impostorTexture: "atlas", decodeImage: await createImageDecoder(set), atlasTile: 32 });
      expect(r.warnings).toEqual([]);
      const side = sideCellColours(r.impostor!.atlas!);
      expect(side.some((c) => c[0]! > 200 && c[1]! > 200 && c[2]! < 70)).toBe(true); // yellow
      expect(side.some((c) => c[0]! < 70 && c[1]! < 70 && c[2]! > 200)).toBe(true); // blue
      expect(side.some((c) => c[0]! > 240 && c[1]! > 240 && c[2]! > 240)).toBe(false); // no white fallback
    });
  }

  it("a large texture is box-filtered before sampling (still the right colours)", async () => {
    const big = new Uint8Array(jpeg.encode({ data: checker(1024, [255, 255, 0], [0, 0, 255]).rgba, width: 1024, height: 1024 }, 90).data);
    const { mesh, set } = tallBox([{ bytes: big, mimeType: "image/jpeg" }], { source: 0 });
    const r = await bakeLods(mesh, { materials: set, impostor: "box", impostorTexture: "atlas", decodeImage: decodeImageNode, atlasTile: 32 });
    const side = sideCellColours(r.impostor!.atlas!);
    expect(side.some((c) => c[0]! > 200 && c[1]! > 200 && c[2]! < 70)).toBe(true);
    expect(side.some((c) => c[0]! < 70 && c[1]! < 70 && c[2]! > 200)).toBe(true);
  });
});

/**
 * A UV sphere whose UV map is cut into `cols` x `rows` islands (each its own
 * cell of the atlas, with a gutter): the shape of image-to-3D output, where
 * a large share of vertices sits on a seam.
 */
function choppedSphere(seg = 48, rings = 32, cols = 6, rows = 4): KitMesh {
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [];
  const cw = seg / cols, ch = rings / rows;
  for (let iy = 0; iy < rows; iy++) for (let ix = 0; ix < cols; ix++) {
    const base = P.length / 3;
    for (let r = 0; r <= ch; r++) for (let s = 0; s <= cw; s++) {
      const R = iy * ch + r, S = ix * cw + s;
      const th = (R / rings) * Math.PI, ph = (S / seg) * Math.PI * 2;
      const x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
      P.push(x, y, z); N.push(x, y, z); C.push(1, 1, 1, 1);
      UV.push((ix + 0.05 + 0.9 * (s / cw)) / cols, (iy + 0.05 + 0.9 * (r / ch)) / rows);
    }
    for (let r = 0; r < ch; r++) for (let s = 0; s < cw; s++) {
      const a = base + r * (cw + 1) + s, b = a + cw + 1;
      I.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  return { positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV), colors: Float32Array.from(C), indices: Uint32Array.from(I) };
}

const oneTexture: LodMaterialSet = {
  materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
  textures: [{ source: 0 }], samplers: [], images: [{ bytes: new Uint8Array(0), mimeType: "image/png" }], extensionsUsed: [], extensionsRequired: [],
};

describe("UV seams", () => {
  it("seamClasses: two UV wedges = seam edge, three = junction, a normal-only split is not a seam", () => {
    // Four vertices at one position: UVs A, B, C, and A again with another normal.
    const m: KitMesh = {
      positions: Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0]),
      normals: Float32Array.from([0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
      uvs: Float32Array.from([0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.1, 0.1, 0.6, 0.1, 0.2, 0.2, 0.2, 0.2]),
      colors: new Float32Array(32).fill(1),
      indices: new Uint32Array(0),
    };
    const { classes, stats } = seamClasses(m, [0, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(classes)).toEqual([SEAM_JUNCTION, SEAM_JUNCTION, SEAM_JUNCTION, SEAM_JUNCTION, SEAM_EDGE, SEAM_EDGE, SEAM_INTERIOR, SEAM_INTERIOR]);
    expect(stats).toEqual({ positions: 3, seamPositions: 2, junctions: 1 });
    // A material boundary with matching UVs is a junction too.
    expect(seamClasses(m, [0, 0, 0, 0, 0, 0, 0, 1]).classes[6]).toBe(SEAM_JUNCTION);
  });

  it("weldUvSeams snaps wedges closer than the tolerance and leaves real seams alone", () => {
    const m: KitMesh = {
      positions: Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0]),
      normals: new Float32Array(9), colors: new Float32Array(12).fill(1),
      uvs: Float32Array.from([0.25, 0.25, 0.25 + 1e-5, 0.25, 0.75, 0.25]),
      indices: Uint32Array.from([0, 1, 2]),
    };
    const { mesh, snapped } = weldUvSeams(m, () => 1 / 2048);
    expect(snapped).toBe(1);
    expect(Array.from(mesh.uvs)).toEqual([0.25, 0.25, 0.25, 0.25, 0.75, 0.25]);
    expect(m.uvs[2]).toBeCloseTo(0.25 + 1e-5, 7); // input untouched
  });

  it("a sub-texel UV split (exporter rounding) no longer blocks simplification", async () => {
    // A heightfield whose middle column is split in UV by 0.002 (a 64 px texture: 1/8 texel).
    const set: LodMaterialSet = { ...oneTexture, images: [{ bytes: encodePngStored(checker(64, [9, 9, 9], [200, 200, 200])), mimeType: "image/png" }] };
    const n = 24, P: number[] = [], UV: number[] = [], I: number[] = [];
    const vert = (i: number, j: number, du: number) => { P.push(i, Math.sin(i * 0.4) * Math.cos(j * 0.3), j); UV.push(i / n + du, j / n); return P.length / 3 - 1; };
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const du = i >= n / 2 ? 0.002 : 0; // right half's UVs are nudged: a fake seam at i = n/2
      const a = vert(i, j, du), b = vert(i + 1, j, du), c = vert(i + 1, j + 1, du), d = vert(i, j + 1, du);
      I.push(a, b, c, a, c, d);
    }
    const m: KitMesh = { positions: Float32Array.from(P), normals: new Float32Array(P.length).map((_, k) => (k % 3 === 1 ? 1 : 0)), uvs: Float32Array.from(UV), colors: new Float32Array((P.length / 3) * 4).fill(1), indices: Uint32Array.from(I) };
    const welded = await bakeLods(m, { materials: set, impostor: false, shading: "smooth" });
    const unwelded = await bakeLods(m, { materials: set, impostor: false, shading: "smooth", seamWeld: false });
    expect(welded.seams!.snapped).toBeGreaterThan(0);
    expect(welded.seams!.seamPositions).toBe(0);
    expect(unwelded.seams!.seamPositions).toBeGreaterThan(0);
    expect(welded.levels[2]!.triangles).toBeLessThanOrEqual(unwelded.levels[2]!.triangles);
  });

  it("preserve (default): a heavily cut atlas reduces >= 60% per level with zero island-spanning triangles", async () => {
    const src = choppedSphere();
    const r = await bakeLods(src, { materials: oneTexture, impostor: false });
    expect(r.seams!.mode).toBe("preserve");
    expect(r.seams!.seamPositions / r.seams!.positions).toBeGreaterThan(0.15);
    const t = r.levels.map((l) => l.triangles);
    expect(t[1]!).toBeLessThanOrEqual(t[0]! * 0.4);
    expect(t[2]!).toBeLessThanOrEqual(t[1]! * 0.4);
    expect(r.levels.slice(1).map((l) => l.uvIslandBreaks)).toEqual([0, 0]);
    expect(r.warnings).toEqual([]);
  });

  it("output UVs stay inside their own island (each corner's UV is in the same atlas cell as its triangle's)", async () => {
    const r = await bakeLods(choppedSphere(), { materials: oneTexture, impostor: false });
    for (const l of r.levels.slice(1)) {
      const m = l.mesh;
      for (let t = 0; t < m.indices.length; t += 3) {
        const cell = (v: number) => `${Math.floor(m.uvs[v * 2]! * 6)},${Math.floor(m.uvs[v * 2 + 1]! * 4)}`;
        const c = cell(m.indices[t]!);
        expect(cell(m.indices[t + 1]!)).toBe(c);
        expect(cell(m.indices[t + 2]!)).toBe(c);
      }
    }
  });

  it("lock pins only junctions: every junction survives, ordinary seam vertices may go", async () => {
    const src = choppedSphere();
    const r = await bakeLods(src, { materials: oneTexture, impostor: false, uvSeams: "lock", seamWeld: false });
    const f = (x: number) => (Math.round(x * 1e4) / 1e4 + 0).toFixed(4); // + 0 turns -0 into 0
    const key = (p: Float32Array, v: number) => `${f(p[v * 3]!)},${f(p[v * 3 + 1]!)},${f(p[v * 3 + 2]!)}`;
    const { classes } = seamClasses(src, new Uint32Array(src.positions.length / 3));
    const junction = new Set<string>(), edge = new Set<string>();
    classes.forEach((c, v) => { if (c === SEAM_JUNCTION) junction.add(key(src.positions, v)); else if (c === SEAM_EDGE) edge.add(key(src.positions, v)); });
    // Poles are degenerate in the source (dropped before baking): skip them.
    const pole = (k: string) => Math.abs(Math.abs(Number(k.split(",")[1])) - 1) < 1e-3;
    const lod2 = r.levels[2]!.mesh;
    const have = new Set<string>();
    for (let v = 0; v < lod2.positions.length / 3; v++) have.add(key(lod2.positions, v));
    expect([...junction].filter((k) => !pole(k) && !have.has(k))).toEqual([]);
    expect([...edge].filter((k) => !have.has(k)).length).toBeGreaterThan(0);
    expect(r.levels.slice(1).map((l) => l.uvIslandBreaks)).toEqual([0, 0]);
  });

  it("permissive crosses seams: fewer triangles, island breaks counted and warned about", async () => {
    const src = choppedSphere(96, 64, 24, 16); // 384 islands of 4 x 4 quads
    const keep = await bakeLods(src, { materials: oneTexture, impostor: false, ratios: [0.25, 0.02] });
    const perm = await bakeLods(src, { materials: oneTexture, impostor: false, ratios: [0.25, 0.02], uvSeams: "permissive" });
    expect(perm.levels[2]!.triangles).toBeLessThan(keep.levels[2]!.triangles);
    expect(perm.levels[2]!.uvIslandBreaks!).toBeGreaterThan(0);
    expect(perm.warnings.join(" ")).toMatch(/span two UV islands/);
  });

  it("uvIslands / uvIslandBreaks: one break per triangle that mixes islands", () => {
    const m = choppedSphere(12, 8, 2, 2);
    const isl = uvIslands(m, new Uint32Array(m.positions.length / 3));
    expect(new Set(isl).size).toBe(4);
    expect(uvIslandBreaks(m.indices, isl)).toBe(0);
    // Stitch a triangle across two islands.
    expect(uvIslandBreaks(Uint32Array.from([0, 1, m.positions.length / 3 - 1]), isl)).toBe(1);
    expect(triangleCount(m)).toBe(12 * 8 * 2);
  });
});

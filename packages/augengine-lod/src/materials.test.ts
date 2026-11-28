import { inflateSync, deflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { bakeLods } from "./bake.js";
import { readGlb, withImpostorMaterial, writeLodGlb } from "./gltf.js";
import { building } from "./kit/index.js";
import { buildManifest, lodKey } from "./manifest.js";
import type { LodMaterialSet } from "./material.js";
import { groupsOf, isFlatShaded, triangleCount, type KitMesh } from "./mesh.js";
import { decodePng, encodePngStored, encodePngWith, imageSize, type RgbaImage } from "./png.js";
import { decodeImageNode, writeBaked } from "./index.js";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const inflate = (d: Uint8Array) => new Uint8Array(inflateSync(d));

/** 2x2-cell checker, `n` px, colours a / b (sRGB bytes). */
function checker(n: number, a: number[], b: number[]): RgbaImage {
  const rgba = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const c = (x < n / 2) !== (y < n / 2) ? a : b;
    rgba.set([c[0]!, c[1]!, c[2]!, 255], (y * n + x) * 4);
  }
  return { width: n, height: n, rgba };
}

/**
 * UV sphere with authored smooth normals and a UV seam at u = 0/1. The top
 * half uses material 0 (textured), the bottom half material 1 (plain red).
 */
function texturedSphere(seg = 32, rings = 24): KitMesh {
  const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], top: number[] = [], bot: number[] = [];
  for (let r = 0; r <= rings; r++) for (let s = 0; s <= seg; s++) {
    const th = (r / rings) * Math.PI, ph = (s / seg) * Math.PI * 2;
    const x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
    P.push(x, y, z); N.push(x, y, z); UV.push(s / seg, r / rings); C.push(1, 1, 1, 1);
  }
  for (let r = 0; r < rings; r++) for (let s = 0; s < seg; s++) {
    const a = r * (seg + 1) + s, b = a + seg + 1;
    (r < rings / 2 ? top : bot).push(a, a + 1, b, b, a + 1, b + 1);
  }
  return {
    positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV), colors: Float32Array.from(C),
    indices: Uint32Array.from([...top, ...bot]),
    groups: [{ material: 0, start: 0, count: top.length }, { material: 1, start: top.length, count: bot.length }],
  };
}

function sphereSet(): LodMaterialSet {
  return {
    materials: [
      { name: "painted", pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 0.7 }, normalTexture: { index: 1, scale: 0.8 }, occlusionTexture: { index: 1 }, emissiveTexture: { index: 0 }, emissiveFactor: [0.2, 0.1, 0] },
      { name: "red", pbrMetallicRoughness: { baseColorFactor: [0.9, 0.1, 0.1, 1], metallicFactor: 0, roughnessFactor: 0.5 } },
    ],
    textures: [{ source: 0, sampler: 0 }, { source: 1, sampler: 0 }],
    samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 }],
    images: [
      { bytes: encodePngStored(checker(16, [255, 255, 0], [0, 0, 255])), mimeType: "image/png" },
      { bytes: encodePngStored(checker(8, [128, 128, 255], [128, 128, 255])), mimeType: "image/png" },
    ],
    extensionsUsed: [],
    extensionsRequired: [],
  };
}

describe("png", () => {
  it("round-trips through stored and zlib-deflated encodings", () => {
    const img = checker(10, [10, 200, 30], [250, 5, 99]);
    for (const png of [encodePngStored(img), encodePngWith(img, (d) => new Uint8Array(deflateSync(d)))]) {
      const back = decodePng(png, inflate)!;
      expect(back.width).toBe(10);
      expect(Array.from(back.rgba)).toEqual(Array.from(img.rgba));
      expect(imageSize(png, "image/png")).toEqual({ width: 10, height: 10 });
    }
  });
});

describe("materials carried through the bake", () => {
  it("readGlb keeps materials, textures, image bytes and groups (no factor folding)", () => {
    const set = sphereSet();
    const glb = writeLodGlb([{ name: "s", lod: 0, mesh: texturedSphere() }], set);
    const back = readGlb(glb);
    expect(back.materials.materials).toEqual(set.materials);
    expect(back.materials.textures).toEqual(set.textures);
    expect(back.materials.samplers).toEqual(set.samplers);
    expect(back.materials.images.map((i) => Array.from(i.bytes))).toEqual(set.images.map((i) => Array.from(i.bytes)));
    expect(groupsOf(back.mesh).map((g) => g.material)).toEqual([0, 1]);
    expect(back.hasTextures).toBe(true);
    expect(back.hasNormals).toBe(true);
    expect(back.shading).toBe("smooth");
    // Colours stay white: the red material's factor lives in the material, not the vertices.
    expect(Math.min(...back.mesh.colors)).toBeCloseTo(1, 5);
  });

  it("\"fold\" reproduces 0.1: one material, factors in vertex colours, a warning about textures", () => {
    const glb = writeLodGlb([{ name: "s", lod: 0, mesh: texturedSphere() }], sphereSet());
    const back = readGlb(glb, { materials: "fold" });
    expect(back.mesh.groups).toBeUndefined();
    expect(back.materials.materials).toHaveLength(1);
    expect(Math.min(...back.mesh.colors.filter((_, i) => i % 4 === 1))).toBeCloseTo(0.1, 5);
    expect(back.warnings.join(" ")).toMatch(/textures are not carried/);
  });

  it("LOD1/LOD2 keep both materials, UVs in range, and the authored smooth normals", async () => {
    const src = texturedSphere();
    const r = await bakeLods(src, { materials: sphereSet() });
    expect(r.shading).toBe("smooth");
    // Authored normal per position, to compare against.
    const authored = new Map<string, number[]>();
    const key = (p: Float32Array, v: number) => `${p[v * 3]!.toFixed(4)},${p[v * 3 + 1]!.toFixed(4)},${p[v * 3 + 2]!.toFixed(4)}`;
    for (let v = 0; v < src.positions.length / 3; v++) authored.set(key(src.positions, v), [src.normals[v * 3]!, src.normals[v * 3 + 1]!, src.normals[v * 3 + 2]!]);
    for (const l of r.levels.slice(1, 3)) {
      expect(l.triangles).toBeLessThan(r.levels[0]!.triangles);
      expect(groupsOf(l.mesh).map((g) => g.material)).toEqual([0, 1]);
      for (let v = 0; v < l.mesh.positions.length / 3; v++) {
        const u = l.mesh.uvs[v * 2]!, vv = l.mesh.uvs[v * 2 + 1]!;
        expect(u).toBeGreaterThanOrEqual(-1e-6); expect(u).toBeLessThanOrEqual(1 + 1e-6);
        expect(vv).toBeGreaterThanOrEqual(-1e-6); expect(vv).toBeLessThanOrEqual(1 + 1e-6);
        const a = authored.get(key(l.mesh.positions, v))!;
        expect(a).toBeDefined(); // simplification only collapses onto existing vertices
        expect(l.mesh.normals[v * 3]! * a[0]! + l.mesh.normals[v * 3 + 1]! * a[1]! + l.mesh.normals[v * 3 + 2]! * a[2]!).toBeGreaterThan(0.9999);
      }
    }
  });

  it("keeps the UV seam closed: seam vertices keep both UVs, no triangle wraps across u = 0/1", async () => {
    const src = texturedSphere();
    const r = await bakeLods(src, { materials: sphereSet() });
    expect(r.seams!.mode).toBe("preserve");
    const f = (x: number) => (Math.round(x * 1e4) / 1e4 + 0).toFixed(4); // + 0 turns -0 into 0
    for (const l of r.levels.slice(1, 3)) {
      const m = l.mesh;
      expect(l.uvIslandBreaks).toBe(0);
      const uAt = new Map<string, Set<string>>();
      for (let v = 0; v < m.positions.length / 3; v++) {
        // Seam meridian (z = 0, x > 0). The rings next to the poles are skipped: the pole fans are
        // degenerate in the source itself (a ring of coincident pole vertices), so not a clean seam.
        const onSeam = Math.abs(m.positions[v * 3 + 2]!) < 1e-5 && m.positions[v * 3]! > 1e-3 && Math.abs(m.positions[v * 3 + 1]!) < 0.9;
        if (!onSeam) continue;
        const k = `${f(m.positions[v * 3]!)},${f(m.positions[v * 3 + 1]!)}`;
        if (!uAt.has(k)) uAt.set(k, new Set());
        uAt.get(k)!.add(m.uvs[v * 2]!.toFixed(3));
      }
      expect(uAt.size).toBeGreaterThan(0);
      for (const [k, us] of uAt) expect([k, ...[...us].sort()]).toEqual([k, "0.000", "1.000"]);
      for (let t = 0; t < m.indices.length; t += 3) {
        const u = [0, 1, 2].map((k) => m.uvs[m.indices[t + k]! * 2]!);
        expect(Math.max(...u) - Math.min(...u)).toBeLessThan(0.5);
      }
    }
  });

  it("a flat (faceted) source stays flat; LOD normals are face normals", async () => {
    const src = building({ width: 10, depth: 8, floors: 3, palette: 1, seed: 4 }).mesh;
    const r = await bakeLods(src);
    expect(r.shading).toBe("flat");
    for (const l of r.levels.slice(0, 3)) expect(isFlatShaded(l.mesh)).toBe(true);
  });

  it("the written GLB stores each image once and every level points at the shared materials", async () => {
    const set = sphereSet();
    const r = await bakeLods(texturedSphere(), { materials: set, impostor: "box", impostorTexture: "atlas", decodeImage: decodeImageNode });
    const { set: out, material: imp } = withImpostorMaterial(set, r.impostor!);
    const glb = writeLodGlb(r.levels.map((l) => ({ name: lodKey("s", l.level), lod: l.level, mesh: l.kind === "impostor" ? { ...l.mesh, groups: [{ material: imp, start: 0, count: l.mesh.indices.length }] } : l.mesh })), out);
    const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + new DataView(glb.buffer).getUint32(12, true))));
    expect(json.images).toHaveLength(3); // two source images + the atlas — not one copy per level
    expect(json.meshes.map((m: any) => m.primitives.map((p: any) => p.material))).toEqual([[0, 1], [0, 1], [0, 1], [2]]);
    expect(json.materials[0].normalTexture).toEqual({ index: 1, scale: 0.8 });
    expect(json.materials[0].occlusionTexture).toEqual({ index: 1 });
    expect(json.materials[0].emissiveTexture).toEqual({ index: 0 });
    expect(json.meshes[0].primitives[0].attributes.COLOR_0).toBeUndefined(); // white colours are not written
  });
});

describe("impostor colours", () => {
  /** A 2 x 4 x 2 box: sides textured blue/yellow checker, top plain red. */
  function tallBox(): { mesh: KitMesh; set: LodMaterialSet } {
    const set = sphereSet();
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
      set,
    };
  }

  it("atlas: side views show the texture's colours, the top view the red material", async () => {
    const { mesh, set } = tallBox();
    const r = await bakeLods(mesh, { materials: set, impostor: "box", impostorTexture: "atlas", decodeImage: decodeImageNode, atlasTile: 32 });
    const atlas = r.impostor!.atlas!;
    expect(r.impostor!.layout).toEqual({ tile: 32, cols: 3, rows: 2, faces: ["+x", "-x", "+z", "-z", "+y"] });
    expect(atlas.width).toBe(96);
    const px = (x: number, y: number) => Array.from(atlas.rgba.subarray((y * atlas.width + x) * 4, (y * atlas.width + x) * 4 + 4));
    // +x cell (0,0): quadrants of the checker -> yellow and blue both present.
    const cell = (cx: number, cy: number) => { const out: number[][] = []; for (let y = 2; y < 30; y += 4) for (let x = 2; x < 30; x += 4) out.push(px(cx * 32 + x, cy * 32 + y)); return out; };
    const side = cell(0, 0);
    expect(side.some((c) => c[0]! > 200 && c[1]! > 200 && c[2]! < 60)).toBe(true); // yellow
    expect(side.some((c) => c[0]! < 60 && c[1]! < 60 && c[2]! > 200)).toBe(true); // blue
    // +y cell (1,1): the red material (0.9, 0.1, 0.1 linear -> sRGB ~ 243, 89, 89).
    for (const c of cell(1, 1)) { expect(c[0]!).toBeGreaterThan(230); expect(c[1]!).toBeLessThan(110); expect(c[3]).toBe(255); }
    expect(r.warnings).toEqual([]);
  });

  it("vertex: each face gets its view's mean colour; no decoder falls back to factors with a warning", async () => {
    const { mesh, set } = tallBox();
    const r = await bakeLods(mesh, { materials: set, impostor: "box", impostorTexture: "vertex", decodeImage: decodeImageNode });
    const imp = r.levels[3]!.mesh;
    expect(r.impostor!.atlas).toBeNull();
    const faceColor = (f: number) => Array.from(imp.colors.subarray(f * 16, f * 16 + 3));
    const side = faceColor(0), topc = faceColor(4);
    expect(side[2]!).toBeGreaterThan(0.25); // blue + yellow average, linear
    expect(topc[0]!).toBeCloseTo(0.9, 1);
    expect(topc[1]!).toBeCloseTo(0.1, 1);
    const noDecoder = await bakeLods(mesh, { materials: set, impostor: "box", impostorTexture: "vertex" });
    expect(noDecoder.warnings.join(" ")).toMatch(/not sampled/);
  });
});

describe("KTX2 / Basis passthrough", () => {
  it("keeps KHR_texture_basisu textures and their bytes as-is", async () => {
    const ktx = new Uint8Array(64);
    ktx.set([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);
    new DataView(ktx.buffer).setUint32(20, 256, true);
    new DataView(ktx.buffer).setUint32(24, 128, true);
    const set: LodMaterialSet = {
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0, extensions: { KHR_texture_transform: { scale: [2, 2] } } } } }],
      textures: [{ extensions: { KHR_texture_basisu: { source: 0 } } }],
      samplers: [],
      images: [{ bytes: ktx, mimeType: "image/ktx2" }],
      extensionsUsed: ["KHR_texture_basisu", "KHR_texture_transform"],
      extensionsRequired: ["KHR_texture_basisu"],
    };
    const mesh = texturedSphere();
    const one = { ...mesh, groups: [{ material: 0, start: 0, count: mesh.indices.length }] };
    const back = readGlb(writeLodGlb([{ name: "k", lod: 0, mesh: one }], set));
    expect(back.materials.textures).toEqual(set.textures);
    expect(back.materials.extensionsUsed).toEqual(set.extensionsUsed);
    expect(back.materials.extensionsRequired).toEqual(["KHR_texture_basisu"]);
    expect(Array.from(back.materials.images[0]!.bytes)).toEqual(Array.from(ktx));
    const r = await bakeLods(back.mesh, { materials: back.materials, impostor: "box", impostorTexture: "atlas", decodeImage: decodeImageNode });
    expect(r.warnings.join(" ")).toMatch(/image\/ktx2/);
    const m = buildManifest("k", "k", "k.lods.glb", r, back.material, { set: back.materials });
    expect(m.textures.images[0]).toMatchObject({ mimeType: "image/ktx2", width: 256, height: 128 });
  });
});

describe("normals", () => {
  it("sources without normals get flat normals by default and smooth ones on request", () => {
    const s = texturedSphere();
    const noNormals = writeLodGlb([{ name: "n", lod: 0, mesh: s }], sphereSet());
    // Strip NORMAL from the JSON chunk.
    const len = new DataView(noNormals.buffer).getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(noNormals.subarray(20, 20 + len)));
    for (const m of json.meshes) for (const p of m.primitives) delete p.attributes.NORMAL;
    let jb = new TextEncoder().encode(JSON.stringify(json));
    const pad = (4 - (jb.length % 4)) % 4;
    if (pad) { const t = new Uint8Array(jb.length + pad).fill(0x20); t.set(jb); jb = t; }
    const rest = noNormals.subarray(20 + len);
    const glb = new Uint8Array(20 + jb.length + rest.length);
    glb.set(noNormals.subarray(0, 20)); glb.set(jb, 20); glb.set(rest, 20 + jb.length);
    const dv = new DataView(glb.buffer); dv.setUint32(8, glb.length, true); dv.setUint32(12, jb.length, true);
    expect(readGlb(glb).shading).toBe("flat");
    const smooth = readGlb(glb, { missingNormals: "smooth" });
    expect(smooth.shading).toBe("smooth");
    expect(smooth.hasNormals).toBe(false);
  });
});

describe("writeBaked (files)", () => {
  it("writes a v2 manifest with primitives, materials, texture memory and an atlas impostor", async () => {
    const dir = mkdtempSync(join(tmpdir(), "lodmat-"));
    const set = sphereSet();
    const { written } = await writeBaked("ball", texturedSphere(), { baseColor: [1, 1, 1], metallic: 0, roughness: 0.6 }, dir, { materials: set, impostor: "box" });
    const m = written.manifest;
    expect(m.format).toBe("augengine-lod/2");
    expect(m.shading).toBe("smooth");
    expect(m.lods.map((l) => l.mesh)).toEqual(["ball", "ball#lod1", "ball#lod2", "ball#lod3"]);
    expect(m.lods[1]!.primitives.map((p) => p.material)).toEqual([0, 1]);
    expect(m.lods[1]!.primitives.reduce((s, p) => s + p.triangles, 0)).toBe(m.lods[1]!.triangles);
    expect(m.lods[3]!.primitives).toEqual([{ primitive: 0, material: 2, triangles: 10 }]);
    expect(m.materials[0]!.textures).toMatchObject({ baseColor: { image: 0 }, normal: { image: 1 }, occlusion: { image: 1 }, emissive: { image: 0 } });
    expect(m.impostor).toMatchObject({ kind: "box", texture: "atlas", material: 2, atlas: { image: 2, tile: 64, cols: 3, rows: 2 } });
    expect(m.textures.images).toHaveLength(3);
    expect(m.textures.gpuBytes).toBe([16 * 16, 8 * 8, 192 * 128].reduce((sum, px) => sum + Math.round(px * 4 * (4 / 3)), 0));
    const glb = readGlb(new Uint8Array(readFileSync(written.glbPath)));
    expect(triangleCount(glb.mesh)).toBe(m.lods.reduce((s, l) => s + l.triangles, 0));
    const onDisk = JSON.parse(readFileSync(written.manifestPath, "utf8"));
    expect(onDisk.format).toBe("augengine-lod/2");
  });
});

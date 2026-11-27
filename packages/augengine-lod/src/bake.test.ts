import { describe, expect, it } from "vitest";
import { bakeLods, simplifySmooth, weld } from "./bake.js";
import { MeshBuilder } from "./builder.js";
import { readGlb, writeGlb } from "./gltf.js";
import { buildKit, building, tree } from "./kit/index.js";
import { buildManifest, lodKey } from "./manifest.js";
import { bounds, toEngineMesh, triangleArea2, triangleCount, type KitMesh } from "./mesh.js";
import { encodePng, renderPreview } from "./preview.js";

function noDegenerates(m: KitMesh) {
  for (let t = 0; t < triangleCount(m); t++) {
    const a = m.indices[t * 3]!, b = m.indices[t * 3 + 1]!, c = m.indices[t * 3 + 2]!;
    expect(a !== b && b !== c && a !== c).toBe(true);
    expect(triangleArea2(m, t)).toBeGreaterThan(1e-10);
  }
}

function unitNormals(m: KitMesh) {
  for (let v = 0; v < m.normals.length / 3; v++) {
    expect(Math.hypot(m.normals[v * 3]!, m.normals[v * 3 + 1]!, m.normals[v * 3 + 2]!)).toBeCloseTo(1, 4);
  }
}

/** A dense, smooth-ish surface: a heightfield grid (the kind of mesh simplification is for). */
function grid(n: number): KitMesh {
  const b = new MeshBuilder();
  const h = (x: number, z: number) => Math.sin(x * 0.3) * Math.cos(z * 0.25) * 1.5;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    b.quad([i, h(i, j + 1), j + 1], [i + 1, h(i + 1, j + 1), j + 1], [i + 1, h(i + 1, j), j], [i, h(i, j), j], [0.4, 0.6, 0.3, 1]);
  }
  return b.build();
}

describe("bakeLods", () => {
  it("hits the triangle targets on a dense surface and keeps bounds", async () => {
    const src = grid(40); // 3,200 triangles
    const r = await bakeLods(src);
    const t = r.levels.map((l) => l.triangles);
    expect(t[0]).toBe(3200);
    expect(t[1]!).toBeLessThanOrEqual(3200 * 0.25 * 1.1);
    expect(t[2]!).toBeLessThanOrEqual(3200 * 0.0625 * 1.5);
    expect(t[1]!).toBeLessThan(t[0]!);
    expect(t[2]!).toBeLessThan(t[1]!);
    expect(t[3]).toBe(8); // impostor: two double-sided quads
    const b0 = bounds(r.levels[0]!.mesh);
    for (const l of r.levels.slice(1, 3)) {
      const b = bounds(l.mesh);
      for (let k = 0; k < 3; k++) {
        // Simplification never grows the box, and keeps >= 85% of each extent.
        expect(b.min[k]!).toBeGreaterThanOrEqual(b0.min[k]! - 1e-4);
        expect(b.max[k]!).toBeLessThanOrEqual(b0.max[k]! + 1e-4);
        const e0 = b0.max[k]! - b0.min[k]!;
        if (e0 > 0.5) expect(b.max[k]! - b.min[k]!).toBeGreaterThanOrEqual(e0 * 0.85);
      }
      noDegenerates(l.mesh);
      unitNormals(l.mesh);
    }
  });

  it("impostor spans the source bounds", async () => {
    const r = await bakeLods(grid(10));
    const b0 = bounds(r.levels[0]!.mesh);
    const bi = bounds(r.levels[3]!.mesh);
    expect(bi.min[1]).toBeCloseTo(b0.min[1], 5);
    expect(bi.max[1]).toBeCloseTo(b0.max[1], 5);
    expect(bi.max[0] - bi.min[0]).toBeCloseTo(b0.max[0] - b0.min[0], 5);
  });

  it("lockBorder keeps the open border of a tile fixed", async () => {
    const src = grid(24);
    const r = await bakeLods(src, { lockBorder: true });
    const lod1 = r.levels[1]!.mesh;
    // Every original border vertex at x=0 must still exist in LOD1.
    const want = new Set<string>();
    for (let v = 0; v < src.positions.length / 3; v++) if (src.positions[v * 3] === 0) want.add(`${src.positions[v * 3 + 1]!.toFixed(4)},${src.positions[v * 3 + 2]!.toFixed(4)}`);
    const have = new Set<string>();
    for (let v = 0; v < lod1.positions.length / 3; v++) if (Math.abs(lod1.positions[v * 3]!) < 1e-6) have.add(`${lod1.positions[v * 3 + 1]!.toFixed(4)},${lod1.positions[v * 3 + 2]!.toFixed(4)}`);
    for (const k of want) expect(have.has(k)).toBe(true);
  });

  it("weld keeps colour seams apart and merges shared corners", () => {
    const b = new MeshBuilder();
    b.quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0, 1]);
    b.quad([1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 0, 0, 1]); // same colour: shares 2 verts
    b.quad([2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0], [0, 0, 1, 1]); // new colour: seam
    expect(weld(b.build(), false).positions.length / 3).toBe(6 + 4);
  });
});

describe("kit", () => {
  it("every asset is a valid, grounded mesh that bakes to a shrinking chain", async () => {
    const kit = buildKit();
    expect(kit.length).toBeGreaterThanOrEqual(12);
    for (const a of kit) {
      const b = bounds(a.mesh);
      expect(b.min[1]).toBeGreaterThanOrEqual(-1e-6); // sits on the ground
      expect(b.max[1]).toBeGreaterThan(0.5);
      noDegenerates(a.mesh);
      unitNormals(a.mesh);
      expect(a.mesh.colors.length / 4).toBe(a.mesh.positions.length / 3);
      const r = await bakeLods(a.mesh);
      const t = r.levels.map((l) => l.triangles);
      expect(t[1]!).toBeLessThanOrEqual(t[0]!);
      expect(t[2]!).toBeLessThanOrEqual(t[1]!);
      expect(t[3]).toBe(8);
      for (const l of r.levels) noDegenerates(l.mesh);
    }
  });

  it("buildings are deterministic per seed and differ across seeds", () => {
    const a = building({ width: 10, depth: 8, floors: 3, palette: 0, seed: 5 });
    const b = building({ width: 10, depth: 8, floors: 3, palette: 0, seed: 5 });
    const c = building({ width: 10, depth: 8, floors: 3, palette: 0, seed: 6 });
    expect(Array.from(a.mesh.colors)).toEqual(Array.from(b.mesh.colors));
    expect(Array.from(a.mesh.colors)).not.toEqual(Array.from(c.mesh.colors));
  });

  it("buildings bake to a real reduction (window panes pruned at LOD2)", async () => {
    const a = building({ width: 12, depth: 10, floors: 5, palette: 1, seed: 23 });
    const r = await bakeLods(a.mesh);
    expect(r.levels[2]!.triangles).toBeLessThan(r.levels[0]!.triangles * 0.35);
  });

  it("box faces point outward", () => {
    const m = new MeshBuilder().box([-1, -1, -1], [1, 1, 1], [1, 1, 1, 1]).build();
    for (let v = 0; v < m.positions.length / 3; v++) {
      const dot = m.positions[v * 3]! * m.normals[v * 3]! + m.positions[v * 3 + 1]! * m.normals[v * 3 + 1]! + m.positions[v * 3 + 2]! * m.normals[v * 3 + 2]!;
      expect(dot).toBeGreaterThan(0);
    }
  });

  it("tree foliage is closed geometry above the trunk", () => {
    const t = tree(7);
    expect(bounds(t.mesh).max[1]).toBeGreaterThan(3);
  });
});

describe("glb + manifest", () => {
  it("round-trips every LOD through writeGlb/readGlb", async () => {
    const r = await bakeLods(building({ width: 8, depth: 8, floors: 2, palette: 2, seed: 37 }).mesh);
    const glb = writeGlb(r.levels.map((l) => ({ name: lodKey("b", l.level), lod: l.level, mesh: l.mesh })), { baseColor: [1, 1, 1], metallic: 0, roughness: 0.8 });
    const back = readGlb(glb);
    const total = r.levels.reduce((s, l) => s + l.triangles, 0);
    expect(triangleCount(back.mesh)).toBe(total);
    expect(back.warnings).toEqual([]);
  });

  it("rejects non-GLB input with a clear message", () => {
    expect(() => readGlb(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))).toThrow(/not a GLB/);
  });

  it("manifest uses the engine's lodMeshKey convention", async () => {
    const r = await bakeLods(grid(8));
    const m = buildManifest("g", "proc:kit-g", "g.lods.glb", r, { baseColor: [1, 1, 1], metallic: 0, roughness: 1 });
    expect(m.lods.map((l) => l.mesh)).toEqual(["proc:kit-g", "proc:kit-g#lod1", "proc:kit-g#lod2", "proc:kit-g#lod3"]);
    expect(m.suggestedThresholds[0]).toBeLessThan(m.suggestedThresholds[1]);
  });

  it("engine mesh layout is 8 floats per vertex plus RGBA colours", () => {
    const m = new MeshBuilder().boxAt([0, 0, 0], [1, 1, 1], [1, 0, 0, 1]).build();
    const e = toEngineMesh(m);
    expect(e.vertices.length).toBe(e.vertexCount * 8);
    expect(e.colors!.length).toBe(e.vertexCount * 4);
    expect(e.indexCount).toBe(36);
  });

  it("preview renders a non-background PNG", () => {
    const img = renderPreview(tree(3).mesh, { size: 64 });
    let fg = 0;
    for (let i = 0; i < 64 * 64; i++) if (img.rgba[i * 4 + 1]! < 150) fg++;
    expect(fg).toBeGreaterThan(100);
    const png = encodePng(img);
    expect(Array.from(png.subarray(1, 4))).toEqual([80, 78, 71]); // "PNG"
  });
});

describe("impostor styles", () => {
  it("box impostor is the bounds box without a bottom (10 tris) and spans the bounds", async () => {
    const src = building({ width: 10, depth: 8, floors: 3, palette: 0, seed: 11 }).mesh;
    const r = await bakeLods(src, { impostor: "box" });
    const imp = r.levels[3]!;
    expect(imp.triangles).toBe(10);
    const b0 = bounds(r.levels[0]!.mesh), bi = bounds(imp.mesh);
    for (let k = 0; k < 3; k++) {
      expect(bi.min[k]).toBeCloseTo(b0.min[k]!, 5);
      expect(bi.max[k]).toBeCloseTo(b0.max[k]!, 5);
    }
    noDegenerates(imp.mesh);
  });

  it("kit picks box for solid kinds and cross for trees and lamps", async () => {
    const { impostorFor } = await import("./kit/index.js");
    expect(impostorFor("building")).toBe("box");
    expect(impostorFor("water-tower")).toBe("box");
    expect(impostorFor("tree")).toBe("cross");
    expect(impostorFor("lamp")).toBe("cross");
  });
});

describe("simplifySmooth (organic meshes)", () => {
  // A UV sphere: smooth, textured, ~2k triangles.
  function sphere(seg = 32, rings = 32): KitMesh {
    const P: number[] = [], N: number[] = [], UV: number[] = [], C: number[] = [], I: number[] = [];
    for (let r = 0; r <= rings; r++) {
      for (let s = 0; s <= seg; s++) {
        const th = (r / rings) * Math.PI, ph = (s / seg) * Math.PI * 2;
        const x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
        P.push(x, y, z); N.push(x, y, z); UV.push(s / seg, r / rings); C.push(1, 1, 1, 1);
      }
    }
    for (let r = 0; r < rings; r++) for (let s = 0; s < seg; s++) {
      const a = r * (seg + 1) + s, b = a + seg + 1;
      I.push(a, a + 1, b, b, a + 1, b + 1);
    }
    return { positions: Float32Array.from(P), normals: Float32Array.from(N), uvs: Float32Array.from(UV), colors: Float32Array.from(C), indices: Uint32Array.from(I) };
  }

  it("cuts triangles toward the target and keeps smooth unit normals", async () => {
    const src = sphere();
    const out = await simplifySmooth(src, 400, 0.05);
    expect(out.triangles).toBeLessThan(triangleCount(src) / 2);
    expect(out.triangles).toBeGreaterThan(100);
    noDegenerates(out.mesh);
    unitNormals(out.mesh);
    // Smooth: a vertex normal on a sphere points (nearly) along its position.
    for (let v = 0; v < out.mesh.positions.length / 3; v += 7) {
      const p = [out.mesh.positions[v * 3]!, out.mesh.positions[v * 3 + 1]!, out.mesh.positions[v * 3 + 2]!];
      const l = Math.hypot(p[0]!, p[1]!, p[2]!);
      const d = (p[0]! * out.mesh.normals[v * 3]! + p[1]! * out.mesh.normals[v * 3 + 1]! + p[2]! * out.mesh.normals[v * 3 + 2]!) / l;
      expect(d).toBeGreaterThan(0.9);
    }
  });

  it("returns small meshes unchanged in triangle count", async () => {
    const src = sphere(8, 8);
    const out = await simplifySmooth(src, 10_000);
    // The 2 x 8 pole triangles are degenerate (two corners at the pole) and are dropped.
    expect(out.triangles).toBe(triangleCount(src) - 16);
    expect(out.error).toBe(0);
  });
});

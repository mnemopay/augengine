import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadGltfIntoRenderer, loadLodIntoRenderer } from "./gltf.js";
import { loadGlb } from "../../scene/loaders/gltf.js";
import { loadGltfIntoRenderer as fromRoot } from "../../index.js";
import { Scene } from "../../scene/scene.js";
import { Tag } from "../../scene/types.js";

// Real textured glTF fixture shared with the repo's e2e suite.
const CUBE = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..", "..", "e2e", "fixtures", "textured-cube.glb"),
);

function fakeRenderer() {
  const meshes: Array<{ key: string; stride: number | undefined; vertexCount: number }> = [];
  const textures: Array<{ key: string; slots: string[] }> = [];
  return {
    meshes,
    textures,
    uploadMesh(key: string, data: { vertexCount: number }, stride?: 6 | 8 | 16) {
      meshes.push({ key, stride, vertexCount: data.vertexCount });
    },
    async uploadPbrTextures(key: string, sources: Record<string, unknown>) {
      textures.push({ key, slots: Object.keys(sources) });
      return true;
    },
  };
}
const okFetch = (bytes: Uint8Array) => async () => ({
  ok: true, status: 200, statusText: "OK",
  arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
});

describe("loadGltfIntoRenderer", () => {
  it("is exported from the root barrel", () => {
    expect(fromRoot).toBe(loadGltfIntoRenderer);
  });

  it("uploads every primitive under gltf:<url>#n, its textures, and adds entities that reference those keys", async () => {
    const r = fakeRenderer();
    const scene = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [] });
    const out = await loadGltfIntoRenderer(r, scene, "/cube.glb", { idPrefix: "cube", position: [1, 2, 3], fetch: okFetch(CUBE) });

    expect(out.meshKeys.length).toBe(out.loaded.primitives.length);
    expect(out.meshKeys[0]).toBe("gltf:/cube.glb#0");
    expect(r.meshes.map((m) => m.key)).toEqual(out.meshKeys);
    expect(r.meshes[0]!.vertexCount).toBeGreaterThan(0);
    expect(r.textures.length).toBeGreaterThan(0);
    expect(r.textures[0]!.slots).toContain("baseColor");

    const root = scene.get(out.instance!.rootEntityId)!;
    expect(root.components[Tag.Transform]!.position).toEqual([1, 2, 3]);
    const meshEntity = scene.get(out.instance!.meshEntityIds[0]!)!;
    const geom = meshEntity.components[Tag.Mesh]!.geometry as { kind: string; url: string; primitiveIndex?: number };
    expect(geom.kind).toBe("gltf");
    expect(`gltf:${geom.url}#${geom.primitiveIndex ?? 0}`).toBe(out.meshKeys[0]);
  });

  it("scene = null uploads meshes only; skipTextures skips texture upload", async () => {
    const r = fakeRenderer();
    const out = await loadGltfIntoRenderer(r, null, "/cube.glb", { fetch: okFetch(CUBE), skipTextures: true });
    expect(out.instance).toBeNull();
    expect(r.meshes.length).toBeGreaterThan(0);
    expect(r.textures).toEqual([]);
  });

  it("throws on an HTTP error", async () => {
    const r = fakeRenderer();
    await expect(
      loadGltfIntoRenderer(r, null, "/missing.glb", {
        fetch: async () => ({ ok: false, status: 404, statusText: "Not Found", arrayBuffer: async () => new ArrayBuffer(0) }),
      }),
    ).rejects.toThrow("404");
    expect(r.meshes).toEqual([]);
  });
});

describe("loadLodIntoRenderer", () => {
  it.each(["augengine-lod/1", "augengine-lod/2"])("resolves textured primitives for %s", async format => {
    const renderer = fakeRenderer();
    const mesh = loadGlb(CUBE).parsed.root.meshes![0]!.name!;
    const levels = await loadLodIntoRenderer(renderer, "/cube.glb", { format, lods:[{level:0, mesh}] }, {fetch:okFetch(CUBE)});
    expect(levels.get(0)).toContain("gltf:/cube.glb#0");
    expect(renderer.textures[0]!.slots).toContain("baseColor");
  });
});

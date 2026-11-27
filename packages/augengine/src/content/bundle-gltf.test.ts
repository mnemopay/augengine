import { describe, it, expect } from "vitest";
import { bundleGltfFiles, type ImportFile } from "./bundle-gltf.js";
import { loadGlb } from "../scene/loaders/gltf.js";

function file(name: string, value: string | Uint8Array): ImportFile {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  return { name: name.split("/").pop()!, webkitRelativePath: name, arrayBuffer: async () => bytes.slice().buffer as ArrayBuffer };
}
function model(uri: string) {
  return { asset: { version: "2.0" }, buffers: [{ byteLength: 36, uri }], bufferViews: [{ buffer: 0, byteLength: 36 }], accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" }], meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }] };
}
const triangle = new Uint8Array(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer);
describe("JSON glTF companion-file import", () => {
  it("packs selected companion buffers into a GLB usable by the engine", async () => {
    const source = file("asset/scene.gltf", JSON.stringify(model("meshes/triangle.bin")));
    const result = await bundleGltfFiles(source, [source, file("asset/meshes/triangle.bin", triangle)]);
    expect(loadGlb(result.bytes).mesh!.vertexCount).toBe(3);
    expect(result.dependencies).toEqual(["asset/meshes/triangle.bin"]);
  });
  it("embeds external images and data-URI buffers", async () => {
    const root = { ...model("data:application/octet-stream;base64," + btoa(String.fromCharCode(...triangle))), images: [{ uri: "albedo.png" }] };
    const source = file("scene.gltf", JSON.stringify(root));
    const result = await bundleGltfFiles(source, [source, file("albedo.png", new Uint8Array([137, 80, 78, 71]))]);
    const loaded = loadGlb(result.bytes);
    expect(loaded.parsed.root.images![0]!.uri).toBeUndefined();
    expect(loaded.parsed.root.images![0]!.mimeType).toBe("image/png");
    expect(loaded.mesh!.vertexCount).toBe(3);
  });
  it("reports missing companions instead of importing a broken model", async () => {
    const source = file("scene.gltf", JSON.stringify(model("missing.bin")));
    await expect(bundleGltfFiles(source, [source])).rejects.toThrow("Missing glTF dependency");
  });
  it("refuses undersized buffers and references outside the selected folder", async () => {
    const source = file("scene.gltf", JSON.stringify(model("../secret.bin")));
    await expect(bundleGltfFiles(source, [source])).rejects.toThrow("escapes");
    const truncated = file("scene.gltf", JSON.stringify(model("tiny.bin")));
    await expect(bundleGltfFiles(truncated, [truncated, file("tiny.bin", new Uint8Array(1))])).rejects.toThrow("shorter");
  });
});

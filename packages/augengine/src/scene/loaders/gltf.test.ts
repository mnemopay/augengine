/**
 * GLB loader tests — synthesize a minimal GLB in-memory and verify
 * round-trip parsing + accessor reading + mesh extraction.
 */

import { describe, it, expect } from "vitest";
import {
  parseGlb,
  readAccessor,
  extractPrimitive,
  extractSkin,
  loadGlb,
} from "./gltf.js";

/**
 * Build a minimal valid GLB:
 *   - One mesh with one primitive (4 vertices, 6 indices for a quad)
 *   - Positions (FLOAT vec3)
 *   - Indices (UINT16 SCALAR)
 *   - One skin with 2 joints + identity inverseBindMatrices
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function makeMinimalGlb(withSecondPrimitive = false, patch?: (json: any) => void): Uint8Array {
  // Binary buffer layout (one BIN chunk):
  //   [0..47]:   4 vertex positions (4 * 3 * 4 = 48 bytes)
  //   [48..59]:  6 indices (6 * 2 = 12 bytes)
  //   [60..63]:  2 bytes pad to 4-align
  //   [64..191]: 2 mat4 inverse-bind matrices (2 * 64 = 128 bytes)
  //   [192..223]: 4 UV coordinates (4 * 2 * 4 = 32 bytes)
  //   [224..227]: embedded image bytes used to verify material extraction
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const ibm = new Float32Array([
    1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
    1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
  ]);
  const texcoords = new Float32Array([
    0, 0,
    1, 0,
    1, 1,
    0, 1,
  ]);

  // BIN buffer
  const bin = new Uint8Array(228);
  new Float32Array(bin.buffer, 0, 12).set(positions);
  new Uint16Array(bin.buffer, 48, 6).set(indices);
  // pad bytes 60..63 = 0
  new Float32Array(bin.buffer, 64, 32).set(ibm);
  new Float32Array(bin.buffer, 192, 8).set(texcoords);
  bin.set([137, 80, 78, 71], 224);

  const json = {
    asset: { version: "2.0", generator: "augengine-test" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [
      { name: "root", mesh: 0, skin: 0, children: [1, 2] },
      { name: "joint0", translation: [0, 0, 0] },
      { name: "joint1", translation: [0, 1, 0] },
    ],
    meshes: [{
      name: "quad",
      primitives: [{
        attributes: { POSITION: 0, TEXCOORD_0: 3 },
        indices: 1,
        material: 0,
      }, ...(withSecondPrimitive ? [{
        attributes: { POSITION: 0, TEXCOORD_0: 3 },
        indices: 1,
        material: 0,
      }] : [])],
    }],
    skins: [{
      joints: [1, 2],
      inverseBindMatrices: 2,
    }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorFactor: [0.5, 0.6, 0.7, 1],
        metallicFactor: 0.2,
        roughnessFactor: 0.4,
        baseColorTexture: { index: 0 },
        metallicRoughnessTexture: { index: 0 },
      },
      normalTexture: { index: 0 },
    }],
    textures: [{ source: 0 }],
    images: [{ bufferView: 4, mimeType: "image/png" }],
    accessors: [
      // 0: positions vec3 float32 × 4
      { bufferView: 0, componentType: 5126, count: 4, type: "VEC3" },
      // 1: indices scalar uint16 × 6
      { bufferView: 1, componentType: 5123, count: 6, type: "SCALAR" },
      // 2: ibm mat4 float32 × 2
      { bufferView: 2, componentType: 5126, count: 2, type: "MAT4" },
      // 3: texcoords vec2 float32 x 4
      { bufferView: 3, componentType: 5126, count: 4, type: "VEC2" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 48 },
      { buffer: 0, byteOffset: 48, byteLength: 12 },
      { buffer: 0, byteOffset: 64, byteLength: 128 },
      { buffer: 0, byteOffset: 192, byteLength: 32 },
      { buffer: 0, byteOffset: 224, byteLength: 4 },
    ],
    buffers: [{ byteLength: bin.length }],
  };
  patch?.(json);
  const jsonText = JSON.stringify(json);
  const jsonBytes = new TextEncoder().encode(jsonText);
  // glTF spec: each chunk's data length must be multiple of 4. Pad with spaces (JSON) / zeros (BIN).
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const jsonChunkLen = jsonBytes.length + jsonPad;
  const binPad = (4 - (bin.length % 4)) % 4;
  const binChunkLen = bin.length + binPad;

  const total = 12 + 8 + jsonChunkLen + 8 + binChunkLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  // Header
  dv.setUint32(0, 0x46546c67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  // JSON chunk
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  out.set(jsonBytes, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBytes.length + i] = 0x20; // space
  // BIN chunk
  const binChunkOff = 20 + jsonChunkLen;
  dv.setUint32(binChunkOff, binChunkLen, true);
  dv.setUint32(binChunkOff + 4, 0x004e4942, true); // 'BIN\0'
  out.set(bin, binChunkOff + 8);
  return out;
}

describe("augengine / gltf loader", () => {
  it("parses a minimal GLB", () => {
    const glb = makeMinimalGlb();
    const { root, bin } = parseGlb(glb);
    expect(root.asset.version).toBe("2.0");
    expect(root.meshes).toHaveLength(1);
    expect(bin).not.toBeNull();
    expect(bin!.byteLength).toBeGreaterThan(0);
  });

  it("rejects bad magic", () => {
    const bad = new Uint8Array(64);
    new DataView(bad.buffer).setUint32(0, 0xdeadbeef, true);
    expect(() => parseGlb(bad)).toThrow(/bad magic/);
  });

  it("rejects unsupported version", () => {
    const glb = makeMinimalGlb();
    new DataView(glb.buffer).setUint32(4, 99, true);
    expect(() => parseGlb(glb)).toThrow(/unsupported version/);
  });

  it("reads positions accessor as Float32Array", () => {
    const glb = makeMinimalGlb();
    const parsed = parseGlb(glb);
    const positions = readAccessor(parsed, 0);
    expect(positions).toBeInstanceOf(Float32Array);
    expect(positions.length).toBe(12); // 4 verts × 3 floats
    expect(positions[0]).toBe(0);
    expect(positions[3]).toBe(1);
    expect(positions[7]).toBe(1);
  });

  it("reads indices accessor as Uint16Array", () => {
    const glb = makeMinimalGlb();
    const parsed = parseGlb(glb);
    const indices = readAccessor(parsed, 1);
    expect(indices).toBeInstanceOf(Uint16Array);
    expect(indices.length).toBe(6);
    expect(indices[0]).toBe(0);
    expect(indices[5]).toBe(3);
  });

  it("extracts a primitive into AugEngine mesh shape", () => {
    const glb = makeMinimalGlb();
    const parsed = parseGlb(glb);
    const mesh = extractPrimitive(parsed, parsed.root.meshes![0]!.primitives[0]!);
    expect(mesh.vertexCount).toBe(4);
    expect(mesh.indexCount).toBe(6);
    expect(mesh.vertexStride).toBe(8);
    expect(mesh.vertices.length).toBe(32); // 4 verts x (pos+normal+uv)
    expect(mesh.vertices[6]).toBe(0);
    expect(mesh.vertices[14]).toBe(1);
    expect(mesh.indices).toBeInstanceOf(Uint32Array);
    // Indices upcast from u16 → u32
    expect(mesh.indices[5]).toBe(3);
  });

  it("extracts a skin with inverse bind matrices", () => {
    const glb = makeMinimalGlb();
    const parsed = parseGlb(glb);
    const skin = extractSkin(parsed, parsed.root.skins![0]!);
    expect(skin.joints).toEqual([1, 2]);
    expect(skin.inverseBindMatrices.length).toBe(32); // 2 joints × 16 floats
    expect(skin.inverseBindMatrices[0]).toBe(1); // identity diagonal
    expect(skin.inverseBindMatrices[5]).toBe(1);
  });

  it("extracts PBR base-color factors and an embedded image source", () => {
    const parsed = parseGlb(makeMinimalGlb());
    const mesh = extractPrimitive(parsed, parsed.root.meshes![0]!.primitives[0]!);
    expect(mesh.pbrMaterial).toEqual(expect.objectContaining({
      baseColor: [0.5, 0.6, 0.7],
      metallic: 0.2,
      roughness: 0.4,
    }));
    expect(mesh.pbrMaterial!.baseColorTexture!.mimeType).toBe("image/png");
    expect(Array.from(mesh.pbrMaterial!.baseColorTexture!.bytes!)).toEqual([137, 80, 78, 71]);
    expect(mesh.pbrMaterial!.metallicRoughnessTexture!.bytes!.byteLength).toBe(4);
    expect(mesh.pbrMaterial!.normalTexture!.bytes!.byteLength).toBe(4);
  });

  it("loadGlb top-level helper returns mesh + skin together", () => {
    const glb = makeMinimalGlb();
    const result = loadGlb(glb);
    expect(result.mesh).not.toBeNull();
    expect(result.primitives).toHaveLength(1);
    expect(result.skin).not.toBeNull();
    expect(result.animations).toEqual([]);
  });

  it("loadGlb exposes every primitive with stable flattened indices", () => {
    const result = loadGlb(makeMinimalGlb(true));
    expect(result.primitives).toHaveLength(2);
    expect(result.primitives.map((primitive) => primitive.primitiveIndex)).toEqual([0, 1]);
    expect(result.primitives.map((primitive) => primitive.meshIndex)).toEqual([0, 0]);
    expect(result.primitives.map((primitive) => primitive.meshPrimitiveIndex)).toEqual([0, 1]);
    expect(result.primitives[1]!.mesh.indexCount).toBe(6);
  });

  it("alphaMode MASK carries the cutoff (default 0.5), the factor alpha and doubleSided; OPAQUE carries none", () => {
    const opaque = loadGlb(makeMinimalGlb()).primitives[0]!.mesh.pbrMaterial!;
    expect(opaque.alphaCutoff).toBeUndefined();
    expect(opaque.baseColorAlpha).toBeUndefined();
    expect(opaque.doubleSided).toBeUndefined();
    const masked = loadGlb(makeMinimalGlb(false, (j) => {
      j.materials[0].alphaMode = "MASK";
      j.materials[0].alphaCutoff = 0.3;
      j.materials[0].doubleSided = true;
      j.materials[0].pbrMetallicRoughness.baseColorFactor = [1, 1, 1, 0.6];
    })).primitives[0]!.mesh.pbrMaterial!;
    expect(masked.alphaCutoff).toBeCloseTo(0.3, 9);
    expect(masked.baseColorAlpha).toBeCloseTo(0.6, 9);
    expect(masked.doubleSided).toBe(true);
    const defaultCutoff = loadGlb(makeMinimalGlb(false, (j) => { j.materials[0].alphaMode = "MASK"; })).primitives[0]!.mesh.pbrMaterial!;
    expect(defaultCutoff.alphaCutoff).toBe(0.5);
    // BLEND is drawn opaque (no cutoff).
    expect(loadGlb(makeMinimalGlb(false, (j) => { j.materials[0].alphaMode = "BLEND"; })).primitives[0]!.mesh.pbrMaterial!.alphaCutoff).toBeUndefined();
  });

  it("EXT_texture_webp: the extension's image wins over the fallback, and works with no core source", () => {
    // Required extension, no fallback (kael-varr.webp.glb's layout).
    const required = loadGlb(makeMinimalGlb(false, (j) => {
      j.extensionsUsed = ["EXT_texture_webp"];
      j.extensionsRequired = ["EXT_texture_webp"];
      j.textures = [{ extensions: { EXT_texture_webp: { source: 0 } } }];
      j.images = [{ bufferView: 4, mimeType: "image/webp" }];
    })).primitives[0]!.mesh.pbrMaterial!;
    expect(required.baseColorTexture!.mimeType).toBe("image/webp");
    expect(Array.from(required.baseColorTexture!.bytes!)).toEqual([137, 80, 78, 71]);
    // Optional extension with a PNG fallback: the WebP image is used; mimeType inferred when absent.
    const both = loadGlb(makeMinimalGlb(false, (j) => {
      j.extensionsUsed = ["EXT_texture_webp"];
      j.textures = [{ source: 0, extensions: { EXT_texture_webp: { source: 1 } } }];
      j.images = [{ bufferView: 4, mimeType: "image/png" }, { uri: "tex.webp" }];
    })).primitives[0]!.mesh.pbrMaterial!;
    expect(both.baseColorTexture!.uri).toBe("tex.webp");
    expect(both.baseColorTexture!.mimeType).toBe("image/webp");
    // A dangling extension index falls back to the core source.
    const dangling = loadGlb(makeMinimalGlb(false, (j) => {
      j.textures = [{ source: 0, extensions: { EXT_texture_webp: { source: 9 } } }];
    })).primitives[0]!.mesh.pbrMaterial!;
    expect(dangling.baseColorTexture!.mimeType).toBe("image/png");
  });
});

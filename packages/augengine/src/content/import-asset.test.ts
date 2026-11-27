import { describe, it, expect } from "vitest";
import { sniffAssetKind, parseKtx2Header, importAssetBytes } from "./import-asset.js";

describe("importAssetBytes", () => {
  it("sniffs KTX2 magic and GLB magic", () => {
    const ktx2 = new Uint8Array([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a, ...new Array(40).fill(0)]);
    expect(sniffAssetKind("floor.ktx2", ktx2)).toBe("ktx2");
    const glb = new Uint8Array(12);
    glb[0] = 0x67; glb[1] = 0x6c; glb[2] = 0x54; glb[3] = 0x46; // 'glTF' little-endian 0x46546c67
    expect(sniffAssetKind("hero.glb", glb)).toBe("glb");
    expect(sniffAssetKind("notes.txt", new Uint8Array([1, 2, 3]))).toBe("unknown");
  });

  it("rejects unknown files with a fail warning", () => {
    const result = importAssetBytes(new Uint8Array([1, 2, 3, 4]), "readme.md");
    expect(result.kind).toBe("unknown");
    expect(result.warnings[0]?.level).toBe("fail");
  });

  it("reads KTX2 width/height from the header", () => {
    const bytes = new Uint8Array(80);
    const magic = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.set(magic, 0);
    const view = new DataView(bytes.buffer);
    view.setUint32(20, 1024, true);
    view.setUint32(24, 512, true);
    const info = parseKtx2Header(bytes);
    expect(info.pixelWidth).toBe(1024);
    expect(info.pixelHeight).toBe(512);
  });

  it("rejects truncated and mislabeled KTX2 files", () => {
    expect(() => importAssetBytes(new Uint8Array(64), "broken.ktx2")).toThrow("header too short");
    expect(() => importAssetBytes(new Uint8Array(80), "fake.ktx2")).toThrow("invalid magic");
  });
});

import { describe, it, expect } from "vitest";
import { PLAYABLE_RENDERER, assertPlayableRenderer } from "./bootstrap.js";

describe("playable renderer contract", () => {
  it("locks the playable path to webgpu", () => {
    expect(PLAYABLE_RENDERER).toBe("webgpu");
    expect(() => assertPlayableRenderer("webgpu")).not.toThrow();
  });

  it("rejects Three.js as a playable path", () => {
    expect(() => assertPlayableRenderer("three")).toThrow(/WebGpuLiveRenderer/);
    expect(() => assertPlayableRenderer("webgl")).toThrow(/retired/);
  });
});

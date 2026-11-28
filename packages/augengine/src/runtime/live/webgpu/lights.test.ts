/**
 * CPU-side light collection and directional shadow fitting.
 * No WebGPU device: these pin the sign choices the shadow pass depends on.
 *
 * Identity-rotation lights shine from their position toward the origin
 * (authored key/fill/rim). A rotated light shines along local −Z, which is
 * the negation of column 2 of its column-major world matrix. The shadow
 * fit looks down the key's travel direction: a point closer to the light
 * gets a smaller clip depth in [0, 1].
 */
import { describe, expect, it } from "vitest";
import { Scene } from "../../../scene/scene.js";
import { Tag, type Entity, type EntityId } from "../../../scene/types.js";
import { CHARACTER_PROOF_ROOM, PROOF_ROOM_IBL } from "../../../scenes/character-proof-room.js";
import {
  FRAME_LAYOUT,
  FRAME_UNIFORM_FLOATS,
  FRAME_UNIFORM_SIZE,
  collectSceneLights,
  fitDirectionalShadow,
  lightDirection,
  packFrameLights,
  transformPoint,
} from "./lights.js";

function sceneOf(entities: Entity[]): Scene {
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "lights-test" },
    entities,
  });
}

function lightEntity(
  entityId: string,
  position: [number, number, number],
  kind: "directional" | "point" | "spot" | "ambient",
  color: [number, number, number],
  intensity: number,
  rotation: [number, number, number, number] = [0, 0, 0, 1],
): Entity {
  return {
    id: entityId as EntityId,
    name: entityId,
    components: {
      [Tag.Transform]: { position, rotation, scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind, color, intensity },
    },
  };
}

describe("scene lights", () => {
  it("sends an identity-rotation light at the proof-room key toward the origin, downward", () => {
    const world = new Float32Array(16);
    world[0] = 1;
    world[5] = 1;
    world[10] = 1;
    world[15] = 1;
    world[12] = -4.5;
    world[13] = 6.5;
    world[14] = 5;
    const dir = lightDirection(world, [0, 0, 0, 1]);
    const len = Math.hypot(4.5, 6.5, 5);
    expect(dir[0]).toBeCloseTo(4.5 / len);
    expect(dir[1]).toBeCloseTo(-6.5 / len);
    expect(dir[2]).toBeCloseTo(-5 / len);
    expect(dir[1]).toBeLessThan(0);
  });

  it("sends a rotated light along −column 2, not toward the origin", () => {
    // Column 2 (local +Z) is (0, 0, 1). Quaternion is 180° around Y so the
    // identity-rotation branch cannot be the one that runs.
    const world = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      4, 5, 0, 1,
    ]);
    const dir = lightDirection(world, [0, 1, 0, 0]);
    expect(dir[0]).toBeCloseTo(0);
    expect(dir[1]).toBeCloseTo(0);
    expect(dir[2]).toBeCloseTo(-1);
    const towardOrigin = [-4, -5, 0];
    const towardLen = Math.hypot(towardOrigin[0], towardOrigin[1], towardOrigin[2]);
    const dot =
      dir[0] * (towardOrigin[0] / towardLen) +
      dir[1] * (towardOrigin[1] / towardLen) +
      dir[2] * (towardOrigin[2] / towardLen);
    expect(dot).toBeLessThan(0.5);
  });

  it("makes the brightest proof-room directional the key and the IBL an ambient", () => {
    const scene = new Scene(structuredClone(CHARACTER_PROOF_ROOM));
    const lights = collectSceneLights(scene, [0, -1, 0], [0, 1.7, 5.6]);
    const len = Math.hypot(4.5, 6.5, 5);
    expect(lights.keyFromScene).toBe(true);
    expect(lights.key.dir[0]).toBeCloseTo(4.5 / len);
    expect(lights.key.dir[1]).toBeCloseTo(-6.5 / len);
    expect(lights.key.dir[2]).toBeCloseTo(-5 / len);
    expect(lights.key.radiance[0]).toBeCloseTo(3.4);
    expect(lights.key.radiance[1]).toBeCloseTo(0.9 * 3.4);
    expect(lights.key.radiance[2]).toBeCloseTo(0.74 * 3.4);
    expect(lights.extraDirectional).toHaveLength(2);
    expect(lights.local).toHaveLength(0);
    expect(lights.ambient?.[0]).toBeCloseTo(PROOF_ROOM_IBL.irradiance[0] * PROOF_ROOM_IBL.intensity);
    expect(lights.ambient?.[1]).toBeCloseTo(PROOF_ROOM_IBL.irradiance[1] * PROOF_ROOM_IBL.intensity);
    expect(lights.ambient?.[2]).toBeCloseTo(PROOF_ROOM_IBL.irradiance[2] * PROOF_ROOM_IBL.intensity);
  });

  it("keeps the fallback direction and white radiance when the scene has no lights", () => {
    const lights = collectSceneLights(sceneOf([]), [0.2, -0.4, 0.1], [0, 0, 0]);
    const len = Math.hypot(0.2, -0.4, 0.1);
    expect(lights.keyFromScene).toBe(false);
    expect(lights.key.dir[0]).toBeCloseTo(0.2 / len);
    expect(lights.key.dir[1]).toBeCloseTo(-0.4 / len);
    expect(lights.key.dir[2]).toBeCloseTo(0.1 / len);
    expect(lights.key.radiance).toEqual([1, 1, 1]);
    expect(lights.ambient).toBeNull();
    expect(lights.extraDirectional).toHaveLength(0);
    expect(lights.local).toHaveLength(0);
  });

  it("reads a rotated scene light from −column 2 of the world matrix", () => {
    const scene = sceneOf([
      lightEntity("yawed", [4, 5, 0], "directional", [1, 1, 1], 1, [0, Math.SQRT1_2, 0, Math.SQRT1_2]),
    ]);
    const lights = collectSceneLights(scene, [0, -1, 0], [0, 0, 0]);
    const wt = scene.worldTransforms();
    const i = wt.indexById.get("yawed" as EntityId)! * 16;
    const col2x = wt.worldMatrices[i + 8]!;
    const col2y = wt.worldMatrices[i + 9]!;
    const col2z = wt.worldMatrices[i + 10]!;
    const colLen = Math.hypot(col2x, col2y, col2z);
    expect(lights.key.dir[0]).toBeCloseTo(-col2x / colLen);
    expect(lights.key.dir[1]).toBeCloseTo(-col2y / colLen);
    expect(lights.key.dir[2]).toBeCloseTo(-col2z / colLen);
    // Toward-origin from (4, 5, 0) has a large −Y. A yaw keeps the beam level.
    expect(Math.abs(lights.key.dir[1]!)).toBeLessThan(0.05);
  });

  it("maps the shadow-sphere centre inside the clip volume, closer to the light than a point below it", () => {
    const dir: [number, number, number] = [0, -1, 0];
    const m = fitDirectionalShadow(dir, [0, 0, 0], 10, 1024);
    const center = transformPoint(m, [0, 0, 0]);
    const above = transformPoint(m, [0, 1, 0]);
    const below = transformPoint(m, [0, -5, 0]);
    expect(Math.abs(center[0])).toBeLessThan(1e-5);
    expect(Math.abs(center[1])).toBeLessThan(1e-5);
    expect(center[2]).toBeGreaterThan(0);
    expect(center[2]).toBeLessThan(1);
    // Light travels −Y, so a point above the centre is closer to the source
    // and must land at a smaller clip depth. The opposite inequality means
    // the light basis is flipped.
    expect(above[2]).toBeLessThan(center[2]);
    expect(below[2]).toBeGreaterThan(center[2]);
    expect(below[2]).toBeLessThan(1);
  });

  it("packs key radiance and the shadow flag on a 16-byte-aligned 1152-byte frame (natural-light env SH9 + exposure + PCSS + atmosphere added 2026-11-27)", () => {
    expect(FRAME_UNIFORM_FLOATS).toBe(288);
    expect(FRAME_LAYOUT.shadowSoft).toBe(FRAME_LAYOUT.exposure + 4);
    expect(FRAME_LAYOUT.shadowSoft2).toBe(FRAME_LAYOUT.shadowSoft + 4);
    expect(FRAME_LAYOUT.probeParams % 4).toBe(0);
    expect(FRAME_LAYOUT.ambientSH % 4).toBe(0);
    expect(FRAME_LAYOUT.envSH % 4).toBe(0);
    expect(FRAME_LAYOUT.envParams).toBe(FRAME_LAYOUT.envSH + 36);
    expect(FRAME_LAYOUT.exposure).toBe(FRAME_LAYOUT.envParams + 4);
    // 912 bytes as of 2026-11-23 + 144 (SH9) + 16 (env params) + 16 (exposure) + 32 (PCSS) + 32 (atmosphere).
    expect(FRAME_UNIFORM_SIZE).toBe(1152);
    expect(FRAME_UNIFORM_SIZE % 16).toBe(0);
    expect(FRAME_LAYOUT.localLights % 4).toBe(0);
    const lights = collectSceneLights(sceneOf([
      lightEntity("key", [-4.5, 6.5, 5], "directional", [1, 0.9, 0.74], 3.4),
    ]), [-0.4, -1, -0.2], [0, 0, 0]);
    const viewProj = fitDirectionalShadow(lights.key.dir, [0, 0, 0], 8, 1024);
    const out = new Float32Array(FRAME_UNIFORM_FLOATS);
    packFrameLights(out, lights, {
      enabled: true,
      viewProj,
      texelSize: 1 / 1024,
      depthBias: 0.0006,
      normalOffset: 0.02,
      pcfTaps: 1,
    });
    expect(out[FRAME_LAYOUT.lightColor]).toBeCloseTo(3.4);
    expect(out[FRAME_LAYOUT.lightColor + 1]).toBeCloseTo(0.9 * 3.4);
    expect(out[FRAME_LAYOUT.lightColor + 2]).toBeCloseTo(0.74 * 3.4);
    expect(out[FRAME_LAYOUT.lightColor + 3]).toBe(1);
    expect(out[FRAME_LAYOUT.lightDir + 1]).toBeLessThan(0);
    expect(out[FRAME_LAYOUT.shadowParams]).toBeCloseTo(1 / 1024);
    expect(out[FRAME_LAYOUT.shadowViewProj]).toBeCloseTo(viewProj[0]!);

    packFrameLights(out, lights, null);
    expect(out[FRAME_LAYOUT.lightColor]).toBeCloseTo(3.4);
    expect(out[FRAME_LAYOUT.lightColor + 3]).toBe(0);
  });

  it("encodes the shadow pass before the colour pass and skips contact discs", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(new URL("./renderer.ts", import.meta.url), "utf-8");
    const shadow = src.indexOf('label: "shadow-pass"');
    const color = src.indexOf("const pass = encoder.beginRenderPass(passDescriptor)");
    expect(shadow).toBeGreaterThan(0);
    expect(color).toBeGreaterThan(shadow);
    expect(src).toContain("fitDirectionalShadow(");
    expect(src).toContain("packFrameLights(");
    expect(src).toContain('e.id.endsWith(".shadow")');
    expect(src).toContain("lights.key.dir");
  });
});

describe("shadow cascades", () => {
  it("far cascade contains the near one and starts at the camera", async () => {
    const { cascadeSpheres } = await import("./lights.js");
    const [near, far] = cascadeSpheres(12, 4);
    expect(near).toEqual({ centerDist: 6, radius: 6 });
    expect(far).toEqual({ centerDist: 24, radius: 24 });
    // Both spheres start at the camera (centerDist − radius = 0), and the far
    // one reaches 4× as far, so the near sphere is inside it.
    expect(near!.centerDist - near!.radius).toBe(0);
    expect(far!.centerDist - far!.radius).toBe(0);
    expect(far!.centerDist + far!.radius).toBe(4 * (near!.centerDist + near!.radius));
  });

  it("packs the far cascade and counts.z = 2 only when present", async () => {
    const { packFrameLights, FRAME_LAYOUT, FRAME_UNIFORM_FLOATS } = await import("./lights.js");
    const lights = { key: { dir: [0, -1, 0] as [number, number, number], radiance: [1, 1, 1] as [number, number, number] }, extraDirectional: [], local: [], ambient: null, keyFromScene: false };
    const vp = new Float32Array(16).fill(2);
    const out = new Float32Array(FRAME_UNIFORM_FLOATS);
    packFrameLights(out, lights, { enabled: true, viewProj: vp, texelSize: 0.001, depthBias: 0.0006, normalOffset: 0.02, pcfTaps: 1 });
    expect(out[FRAME_LAYOUT.counts + 2]).toBe(1);
    expect(out[FRAME_LAYOUT.shadowViewProj2]).toBe(0);
    const vp2 = new Float32Array(16).fill(3);
    packFrameLights(out, lights, {
      enabled: true, viewProj: vp, texelSize: 0.001, depthBias: 0.0006, normalOffset: 0.02, pcfTaps: 1,
      far: { viewProj: vp2, texelSize: 0.001, depthBias: 0.0006, normalOffset: 0.08, blend: 0.15 },
    });
    expect(out[FRAME_LAYOUT.counts + 2]).toBe(2);
    expect(out[FRAME_LAYOUT.shadowViewProj2]).toBe(3);
    expect(out[FRAME_LAYOUT.shadowParams2 + 2]).toBeCloseTo(0.08, 6);
    expect(out[FRAME_LAYOUT.shadowParams2 + 3]).toBeCloseTo(0.15, 6);
    // WGSL alignment: the mat4 must start on a 16-byte boundary.
    expect((FRAME_LAYOUT.shadowViewProj2 * 4) % 16).toBe(0);
  });
});

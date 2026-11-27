/**
 * Tier 2 raycast — verify each primitive + scene-level dispatch.
 */
import { describe, it, expect } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId } from "../scene/types.js";
import { raycast, rayAabb, raySphere, rayCapsule } from "./raycast.js";
import type { AABB, Capsule } from "./colliders.js";

describe("ray ↔ AABB", () => {
  const box: AABB = { min: [-1, -1, -1], max: [1, 1, 1] };
  it("hits the front face from straight on", () => {
    const hit = rayAabb([0, 0, 5], [0, 0, -1], 100, box);
    expect(hit).not.toBeNull();
    expect(hit!.distance).toBeCloseTo(4, 4);
    expect(hit!.normal[2]).toBeCloseTo(1, 4);
  });
  it("misses when the ray points away", () => {
    expect(rayAabb([0, 0, 5], [0, 0, 1], 100, box)).toBeNull();
  });
  it("respects maxDistance", () => {
    expect(rayAabb([0, 0, 100], [0, 0, -1], 50, box)).toBeNull();
  });
});

describe("ray ↔ sphere", () => {
  it("hits a centered sphere", () => {
    const hit = raySphere([0, 0, 5], [0, 0, -1], 100, [0, 0, 0], 1);
    expect(hit).not.toBeNull();
    expect(hit!.distance).toBeCloseTo(4, 4);
    expect(hit!.normal[2]).toBeCloseTo(1, 4);
  });
  it("misses a tangent ray", () => {
    expect(raySphere([2, 0, 5], [0, 0, -1], 100, [0, 0, 0], 1)).toBeNull();
  });
});

describe("ray ↔ capsule", () => {
  it("hits the cylinder body", () => {
    const cap: Capsule = { kind: "capsule", p0: [0, -1, 0], p1: [0, 1, 0], radius: 0.5 };
    const hit = rayCapsule([2, 0, 0], [-1, 0, 0], 100, cap);
    expect(hit).not.toBeNull();
    expect(hit!.distance).toBeCloseTo(1.5, 4);
  });
  it("hits the upper hemisphere when ray passes above the segment", () => {
    const cap: Capsule = { kind: "capsule", p0: [0, -1, 0], p1: [0, 1, 0], radius: 0.5 };
    const hit = rayCapsule([2, 1.4, 0], [-1, 0, 0], 100, cap);
    expect(hit).not.toBeNull();
  });
  it("misses when far above", () => {
    const cap: Capsule = { kind: "capsule", p0: [0, -1, 0], p1: [0, 1, 0], radius: 0.5 };
    expect(rayCapsule([2, 5, 0], [-1, 0, 0], 100, cap)).toBeNull();
  });
});

describe("raycast — scene-level dispatch", () => {
  function makeScene(entities: Entity[]): Scene {
    return new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "raycast-test" },
      entities,
    });
  }

  it("returns the nearest hit across mixed colliders", () => {
    const wall: Entity = {
      id: "wall" as EntityId,
      name: "wall",
      components: {
        [Tag.Transform]: { position: [0, 0, -2], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "box", extents: [0.5, 1, 0.5] },
      },
    };
    const farball: Entity = {
      id: "farball" as EntityId,
      name: "farball",
      components: {
        [Tag.Transform]: { position: [0, 0, -10], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "sphere", radius: 1 },
      },
    };
    const scene = makeScene([wall, farball]);
    const hit = raycast(scene, [0, 0, 5], [0, 0, -1]);
    expect(hit).not.toBeNull();
    expect(hit!.entityId).toBe("wall");
    expect(hit!.distance).toBeCloseTo(6.5, 4);
  });

  it("respects ignore list", () => {
    const a: Entity = {
      id: "a" as EntityId,
      name: "a",
      components: {
        [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "sphere", radius: 1 },
      },
    };
    const b: Entity = {
      id: "b" as EntityId,
      name: "b",
      components: {
        [Tag.Transform]: { position: [0, 0, -5], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "sphere", radius: 1 },
      },
    };
    const scene = makeScene([a, b]);
    const hit = raycast(scene, [0, 0, 10], [0, 0, -1], { ignore: ["a" as EntityId] });
    expect(hit).not.toBeNull();
    expect(hit!.entityId).toBe("b");
  });

  it("returns null when nothing in maxDistance", () => {
    const a: Entity = {
      id: "a" as EntityId,
      name: "a",
      components: {
        [Tag.Transform]: { position: [0, 0, -50], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "sphere", radius: 1 },
      },
    };
    const scene = makeScene([a]);
    expect(raycast(scene, [0, 0, 0], [0, 0, -1], { maxDistance: 10 })).toBeNull();
  });
});

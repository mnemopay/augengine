/**
 * One collider convention for every consumer: box extents, radius and
 * halfHeight are world-space; transform scale is visual only.
 *
 * Before 2026-11-23 PhysicsSystem multiplied extents by scale while the
 * character controller and raycast did not, so the same container was
 * 2.5 x 3 x 6 m to the player and 6.25 x 9 x 36 m to Rapier.
 */
import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../scene/types.js";
import { PhysicsSystem } from "./system.js";
import { KinematicCharacterController } from "./character-controller.js";
import { raycast } from "./raycast.js";

const systems: PhysicsSystem[] = [];
afterEach(() => {
  for (const s of systems) s.dispose();
  systems.length = 0;
});

function scene(entities: Entity[]): Scene {
  return new Scene({ schemaVersion: 1, source: { kind: "authored", path: "convention" }, entities });
}

/** A visually scaled box whose collider is authored in world units. */
function scaledBox(id: string, position: Vec3, scale: Vec3): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale, parent: null },
      [Tag.Collider]: { shape: "box", extents: [scale[0] / 2, scale[1] / 2, scale[2] / 2] },
    },
  };
}

function ball(id: string, position: Vec3): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Collider]: { shape: "sphere", radius: 0.25 },
      [Tag.RigidBody]: { velocity: [0, 0, 0], angularVelocity: [0, 0, 0], mass: 1, isKinematic: false, restitution: 0, friction: 0.5 },
    },
  };
}

describe("collider convention (world-space sizes, scale is visual)", () => {
  it("Rapier rests a ball on the top face of a scaled box, not 3x higher", async () => {
    // 4 x 2 x 4 box centered at y=1: top face at y=2.
    const box = scaledBox("pedestal", [0, 1, 0], [4, 2, 4]);
    const b = ball("ball", [0, 4, 0]);
    const s = new PhysicsSystem();
    s.gravity = -9.81;
    systems.push(s);
    await s.init();
    const w = scene([box, b]);
    for (let i = 0; i < 240; i++) s.step(w, 1 / 60);
    const y = b.components[Tag.Transform]!.position[1];
    // Resting height = top face 2 + radius 0.25. The old scaled cuboid put
    // the top at y=1+2*1=3, so the ball would rest at 3.25.
    expect(y).toBeGreaterThan(2.15);
    expect(y).toBeLessThan(2.35);
  });

  it("the character controller stands on the same top face", () => {
    const box = scaledBox("pedestal", [0, 1, 0], [4, 2, 4]);
    const player: Entity = {
      id: "player" as EntityId,
      name: "player",
      components: {
        [Tag.Transform]: { position: [0, 4, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.9 },
      },
    };
    const w = scene([box, player]);
    const c = new KinematicCharacterController(w, { entityId: "player" as EntityId });
    for (let i = 0; i < 120; i++) c.update(1 / 60, [0, 0, 0]);
    expect(c.grounded).toBe(true);
    // Capsule centre = top face 2 + halfHeight 0.9 + radius 0.4 = 3.3.
    expect(w.get("player" as EntityId)!.components[Tag.Transform]!.position[1]).toBeCloseTo(3.3, 1);
  });

  it("raycast hits the scaled box at its world-space face", () => {
    const box = scaledBox("pedestal", [0, 1, 0], [4, 2, 4]);
    const w = scene([box]);
    const hit = raycast(w, [0, 10, 0], [0, -1, 0]);
    expect(hit?.entityId).toBe("pedestal");
    expect(hit!.point[1]).toBeCloseTo(2, 3);
  });

  it("centerOffset moves the controller's collider like it moves Rapier's", () => {
    const step: Entity = {
      id: "step" as EntityId,
      name: "step",
      components: {
        [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "box", extents: [2, 0.5, 2], centerOffset: [0, 0.5, 0] },
      },
    };
    const player: Entity = {
      id: "player" as EntityId,
      name: "player",
      components: {
        [Tag.Transform]: { position: [0, 3, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.9 },
      },
    };
    const w = scene([step, player]);
    const c = new KinematicCharacterController(w, { entityId: "player" as EntityId });
    for (let i = 0; i < 120; i++) c.update(1 / 60, [0, 0, 0]);
    // Offset box spans y 0..1, so the capsule centre rests at 1 + 1.3 = 2.3.
    expect(w.get("player" as EntityId)!.components[Tag.Transform]!.position[1]).toBeCloseTo(2.3, 1);
  });
});

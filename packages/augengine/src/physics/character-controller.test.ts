/**
 * Tier 2 game-readiness — kinematic FPS controller spec.
 *
 * Each test sets up a one-frame scene and walks the controller a known delta.
 * The expectations are framed in "from the player's perspective" — does W
 * carry them forward, does a wall stop them, does the floor catch them, does
 * the world come back together when they straddle two surfaces.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId } from "../scene/types.js";
import { KinematicCharacterController } from "./character-controller.js";

const PLAYER_ID = "player" as EntityId;

function makeScene(extras: Entity[] = []): Scene {
  const player: Entity = {
    id: PLAYER_ID,
    name: "Player",
    components: {
      [Tag.Transform]: {
        position: [0, 1.7, 5],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.9 },
    },
  };
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "controller-test" },
    entities: [player, ...extras],
    defaultCamera: PLAYER_ID,
  });
}

function floorEntity(): Entity {
  return {
    id: "floor" as EntityId,
    name: "Floor",
    components: {
      [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [36, 1, 36], parent: null },
      [Tag.Collider]: { shape: "box", extents: [18, 0.1, 18] },
    },
  };
}

function wallEntity(id: string, position: [number, number, number], extents: [number, number, number]): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [extents[0] * 2, extents[1] * 2, extents[2] * 2], parent: null },
      [Tag.Collider]: { shape: "box", extents },
    },
  };
}

describe("KinematicCharacterController", () => {
  it("moves forward in free space", () => {
    const scene = makeScene([floorEntity()]);
    const ctl = new KinematicCharacterController(scene, { entityId: PLAYER_ID, gravity: 0 });
    const before = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[2];
    ctl.update(0.1, [0, 0, -1]);
    const after = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[2];
    // moveSpeed 4 m/s × 0.1 s = 0.4 m forward (–Z).
    expect(after).toBeCloseTo(before - 0.4, 3);
  });

  it("stops when walking into a wall", () => {
    // Wall at x=1, extends to x=2. Player capsule radius 0.4. With the player
    // starting at x=0, walking +x at 4 m/s for 1 s would take them to x=4, but
    // the wall blocks them — final x should be ~0.6 (wall.min.x - radius).
    const wall = wallEntity("wall", [1.5, 1, 5], [0.5, 1, 0.5]);
    const scene = makeScene([floorEntity(), wall]);
    const ctl = new KinematicCharacterController(scene, { entityId: PLAYER_ID, gravity: 0 });
    for (let i = 0; i < 60; i++) ctl.update(1 / 60, [1, 0, 0]);
    const finalX = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[0];
    // Player.center.x + radius ≤ wall.min.x (with skin width).
    expect(finalX + 0.4).toBeLessThanOrEqual(1.0 + 0.02);
  });

  it("falls under gravity until grounded", () => {
    // Player at y=1.7 with floor top at y=0.1 (extents.y=0.1). Player feet =
    // center - halfHeight - radius = 1.7 - 0.9 - 0.4 = 0.4. After fall + grounding,
    // feet should be flush with the floor top.
    const scene = makeScene([floorEntity()]);
    const ctl = new KinematicCharacterController(scene, { entityId: PLAYER_ID });
    for (let i = 0; i < 60; i++) ctl.update(1 / 60, [0, 0, 0]);
    expect(ctl.grounded).toBe(true);
    const y = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[1];
    // Player's bottom = y - 0.9 - 0.4 should be at floor top (0.1).
    const playerBottom = y - 0.9 - 0.4;
    expect(playerBottom).toBeCloseTo(0.1, 1);
  });

  it("slides along a wall when input is diagonal", () => {
    // Diagonal push into a wall on +x; the capsule should still progress in -z.
    const wall = wallEntity("wall", [1.5, 1, 5], [0.5, 1, 5]);
    const scene = makeScene([floorEntity(), wall]);
    const ctl = new KinematicCharacterController(scene, { entityId: PLAYER_ID, gravity: 0 });
    const startZ = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[2];
    for (let i = 0; i < 30; i++) ctl.update(1 / 60, [1, 0, -1]);
    const endZ = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[2];
    expect(endZ).toBeLessThan(startZ - 0.2);
  });

  it("jumps while grounded, ignores jump while airborne", () => {
    const scene = makeScene([floorEntity()]);
    const ctl = new KinematicCharacterController(scene, { entityId: PLAYER_ID });
    for (let i = 0; i < 60; i++) ctl.update(1 / 60, [0, 0, 0]);
    expect(ctl.grounded).toBe(true);
    const beforeY = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[1];
    ctl.update(1 / 60, [0, 0, 0], true);
    expect(ctl.velocity[1]).toBeGreaterThan(0);
    // A second jump mid-air must not re-launch.
    const peakVel = ctl.velocity[1];
    ctl.update(1 / 60, [0, 0, 0], true);
    expect(ctl.velocity[1]).toBeLessThanOrEqual(peakVel);
    // After several frames the capsule must be higher than the rest pose.
    for (let i = 0; i < 5; i++) ctl.update(1 / 60, [0, 0, 0]);
    const afterY = scene.get(PLAYER_ID)!.components[Tag.Transform]!.position[1];
    expect(afterY).toBeGreaterThan(beforeY);
  });
});

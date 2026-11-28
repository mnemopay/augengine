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

function sensor(id: string, position: Vec3, radius = 0.3): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Collider]: { shape: "sphere", radius, sensor: true },
      [Tag.RigidBody]: { velocity: [0, 0, 0], angularVelocity: [0, 0, 0], mass: 1, isKinematic: true },
    },
  };
}
function box(id: string, position: Vec3, half: Vec3): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Collider]: { shape: "box", extents: half },
    },
  };
}
const scene = (entities: Entity[]) => new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities });

async function physics(): Promise<PhysicsSystem> {
  const p = new PhysicsSystem();
  p.gravity = 0;
  systems.push(p);
  await p.init();
  return p;
}

describe("PhysicsSystem.overlapShape", () => {
  it("finds sensors and solids at the query pose, sorted, honouring exclude and filter", async () => {
    const s = scene([sensor("hurt_b", [1, 1, 0]), sensor("hurt_a", [1.2, 1, 0]), box("wall", [5, 1, 0], [0.5, 1, 1])]);
    const p = await physics();
    p.stepFixed(s, 1 / 60);
    expect(p.overlapShape({ kind: "sphere", radius: 0.3 }, [1.1, 1, 0])).toEqual(["hurt_a", "hurt_b"]);
    expect(p.overlapShape({ kind: "sphere", radius: 0.3 }, [1.1, 1, 0], undefined, { exclude: new Set(["hurt_a" as EntityId]) })).toEqual(["hurt_b"]);
    expect(p.overlapShape({ kind: "capsule", radius: 0.2, halfHeight: 0.5 }, [4.4, 1, 0])).toEqual(["wall"]);
    expect(p.overlapShape({ kind: "box", halfExtents: [0.2, 0.2, 0.2] }, [0, 5, 0])).toEqual([]);
    expect(p.overlapShape({ kind: "sphere", radius: 2 }, [3, 1, 0], undefined, { filter: (id) => id.startsWith("hurt") })).toEqual(["hurt_a", "hurt_b"]);
  });

  it("tracks a kinematic sensor moved between steps (no one-step lag)", async () => {
    const e = sensor("hurt", [0, 1, 0]);
    const s = scene([e]);
    const p = await physics();
    p.stepFixed(s, 1 / 60);
    s.setLocalTransform("hurt" as EntityId, [3, 1, 0], [0, 0, 0, 1], [1, 1, 1]);
    p.stepFixed(s, 1 / 60);
    expect(p.overlapShape({ kind: "sphere", radius: 0.1 }, [3, 1, 0])).toEqual(["hurt"]);
    expect(p.overlapShape({ kind: "sphere", radius: 0.1 }, [0, 1, 0])).toEqual([]);
  });

  it("sensors do not block the character controller or raycasts", async () => {
    const s = scene([
      box("floor", [0, -0.5, 0], [10, 0.5, 10]),
      sensor("hurt", [1, 0.9, 0], 0.6),
      { id: "cap" as EntityId, name: "cap", components: { [Tag.Transform]: { position: [0, 0.9, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null }, [Tag.Collider]: { shape: "capsule", radius: 0.3, halfHeight: 0.6 } } },
    ]);
    const kcc = new KinematicCharacterController(s, { entityId: "cap" as EntityId, moveSpeed: 4 });
    for (let i = 0; i < 60; i++) kcc.update(1 / 60, [1, 0, 0]);
    expect(s.get("cap" as EntityId)!.components[Tag.Transform]!.position[0]).toBeGreaterThan(3);
    const hit = raycast(s, [-2, 0.9, 0], [1, 0, 0], { ignore: ["cap" as EntityId] });
    expect(hit?.entityId ?? null).toBeNull();
  });
});

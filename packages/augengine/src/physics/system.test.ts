import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../scene/types.js";
import { PhysicsSystem } from "./system.js";
import { FixedStepClock } from "../runtime/fixed-step.js";
const systems: PhysicsSystem[] = [];
afterEach(() => {
  for (const s of systems) s.dispose();
  systems.length = 0;
});
async function solver(gravity = 0): Promise<PhysicsSystem> {
  const s = new PhysicsSystem();
  s.gravity = gravity;
  systems.push(s);
  await s.init();
  return s;
}
function sphere(
  id: string,
  position: Vec3,
  velocity: Vec3 = [0, 0, 0],
  mass = 1,
): Entity {
  return {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: {
        position,
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Collider]: { shape: "sphere", radius: 0.5 },
      [Tag.RigidBody]: {
        velocity,
        angularVelocity: [0, 0, 0],
        mass,
        isKinematic: false,
        restitution: 1,
        friction: 0,
      },
    },
  };
}
function world(entities: Entity[]): Scene {
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities,
  });
}
function floor(): Entity {
  return {
    id: "floor" as EntityId,
    name: "floor",
    components: {
      [Tag.Transform]: {
        position: [0, -0.1, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Collider]: { shape: "box", extents: [20, 0.1, 20] },
    },
  };
}
describe("Rapier rigid-body integration", () => {
  it("conserves momentum and elastic kinetic energy for unequal-mass collisions", async () => {
    const a = sphere("a", [-1, 0, 0], [3, 0, 0], 2),
      b = sphere("b", [1, 0, 0], [-3, 0, 0]);
    const scene = world([a, b]),
      s = await solver();
    for (let i = 0; i < 120; i++) s.step(scene, 1 / 120);
    const va = a.components.rigid_body!.velocity[0],
      vb = b.components.rigid_body!.velocity[0];
    expect(2 * va + vb).toBeCloseTo(3, 3);
    expect(va).toBeCloseTo(-1, 2);
    expect(vb).toBeCloseTo(5, 2);
    expect(va * va + 0.5 * vb * vb).toBeCloseTo(13.5, 2);
  });
  it("rests on a box without sinking and reports contact", async () => {
    const ball = sphere("ball", [0, 2, 0]);
    ball.components.rigid_body!.restitution = 0;
    const scene = world([floor(), ball]),
      s = await solver(-9.81);
    let contacts = false;
    for (let i = 0; i < 1200; i++) {
      s.step(scene, 1 / 120);
      contacts ||= s.contacts.length > 0;
    }
    expect(ball.components.transform!.position[1]).toBeCloseTo(0.5, 2);
    expect(Math.abs(ball.components.rigid_body!.velocity[1])).toBeLessThan(
      0.02,
    );
    expect(contacts).toBe(true);
    expect(s.isSleeping(ball.id)).toBe(true);
  });
  it("CCD prevents a fast small body from passing through a thin wall", async () => {
    const wall = floor();
    wall.components.transform!.position = [0, 0, 0];
    wall.components.collider!.extents = [0.01, 5, 5];
    const ball = sphere("bullet", [-2, 0, 0], [1000, 0, 0]);
    ball.components.collider!.radius = 0.05;
    const scene = world([wall, ball]),
      s = await solver();
    s.step(scene, 1 / 120);
    expect(ball.components.transform!.position[0]).toBeLessThan(0);
  });
  it("supports rotation, friction and stable box stacks", async () => {
    const boxes = Array.from({ length: 4 }, (_, i) => {
      const b = sphere(`box.${i}`, [0, 0.5 + i * 1.02, 0]);
      b.components.collider = { shape: "box", extents: [0.5, 0.5, 0.5] };
      b.components.rigid_body!.restitution = 0;
      b.components.rigid_body!.friction = 0.8;
      return b;
    });
    const scene = world([floor(), ...boxes]),
      s = await solver(-9.81);
    for (let i = 0; i < 1200; i++) s.step(scene, 1 / 120);
    for (let i = 0; i < boxes.length; i++)
      expect(boxes[i]!.components.transform!.position[1]).toBeCloseTo(
        0.5 + i,
        1,
      );
    const b = boxes[3]!;
    b.components.rigid_body!.angularVelocity = [0, 2, 0];
    s.step(scene, 1 / 120);
    expect(Math.abs(b.components.transform!.rotation[1])).toBeGreaterThan(0);
  });
  it("preserves tangential velocity instead of zeroing every velocity axis", async () => {
    const ball = sphere("ball", [0, 0.5, 0], [2, -1, 0]);
    ball.components.rigid_body!.restitution = 0;
    const ground = floor();
    ground.components.rigid_body = {
      velocity: [0, 0, 0],
      angularVelocity: [0, 0, 0],
      mass: 1,
      isKinematic: true,
      friction: 0,
    };
    const scene = world([ground, ball]),
      s = await solver(-9.81);
    for (let i = 0; i < 120; i++) s.step(scene, 1 / 120);
    expect(ball.components.rigid_body!.velocity[0]).toBeCloseTo(2, 3);
  });
  it("matches at 30/60/120 Hz render rates and stable insertion order", async () => {
    const outcomes = [];
    for (const hz of [30, 60, 120]) {
      const a = sphere("a", [-1, 0, 0], [3, 0, 0]),
        b = sphere("b", [1, 0, 0], [-2, 0, 0]);
      const scene = world(hz === 60 ? [b, a] : [a, b]),
        s = await solver();
      for (let i = 0; i < hz * 2; i++) s.step(scene, 1 / hz);
      outcomes.push(
        JSON.stringify([
          a.components.transform,
          a.components.rigid_body,
          b.components.transform,
        ]),
      );
    }
    expect(new Set(outcomes).size).toBe(1);
  });
  it("updates/removes edited colliders and rejects invalid configurations before stepping", async () => {
    const ball = sphere("ball", [0, 2, 0]),
      scene = world([ball]),
      s = await solver();
    s.step(scene, 1 / 120);
    expect(s.bodyCount).toBe(1);
    ball.components.rigid_body!.mass = -1;
    expect(() => s.stepFixed(scene, 1 / 120)).toThrow("Invalid mass");
    ball.components.rigid_body!.mass = 1;
    scene.remove(ball.id);
    s.step(scene, 1 / 120);
    expect(s.bodyCount).toBe(0);
    expect(() => s.step(scene, NaN)).toThrow("finite");
  });
  it("does not leak a world if disposed while WASM initializes", async () => {
    const s = new PhysicsSystem();
    const pending = s.init();
    s.dispose();
    await pending;
    expect(s.ready).toBe(false);
  });
  it("friction slows lateral sliding on a resting surface", async () => {
    const box = sphere("box", [0, 0.5, 0], [3, 0, 0]);
    box.components.collider = { shape: "box", extents: [0.5, 0.5, 0.5] };
    box.components.rigid_body!.friction = 0.8;
    box.components.rigid_body!.restitution = 0;
    const scene = world([floor(), box]),
      s = await solver(-9.81);
    for (let i = 0; i < 240; i++) s.step(scene, 1 / 120);
    expect(Math.abs(box.components.rigid_body!.velocity[0])).toBeLessThan(0.05);
  });
  it("supports capsules on explicit static triangle meshes", async () => {
    const ground = floor();
    ground.components.collider = {
      shape: "mesh",
      vertices: [-10, 0, -10, 10, 0, -10, 10, 0, 10, -10, 0, 10],
      indices: [0, 2, 1, 0, 3, 2],
    };
    ground.components.transform!.position = [0, 0, 0];
    const capsule = sphere("capsule", [0, 3, 0]);
    capsule.components.collider = {
      shape: "capsule",
      radius: 0.25,
      halfHeight: 0.5,
    };
    capsule.components.rigid_body!.restitution = 0;
    const scene = world([ground, capsule]),
      s = await solver(-9.81);
    for (let i = 0; i < 600; i++) s.step(scene, 1 / 120);
    expect(capsule.components.transform!.position[1]).toBeCloseTo(0.75, 2);
  });
});
describe("fixed-step scheduling", () => {
  it("reports discarded wall time under overload and rejects backwards time", () => {
    const c = new FixedStepClock();
    let steps = 0;
    c.advance(10, () => steps++);
    expect(steps).toBe(32);
    expect(c.droppedSeconds).toBeCloseTo(10 - 32 / 120);
    expect(() => c.advance(-1, () => {})).toThrow();
    expect(() => c.advance(Infinity, () => {})).toThrow();
  });
});
it("rejects malformed vectors and values outside WASM float32 range", async () => {
  const ball = sphere("invalid", [0, 0, 0]), scene = world([ball]), s = await solver();
  ball.components.rigid_body!.mass = 1e300;
  expect(() => s.stepFixed(scene, 1 / 120)).toThrow("unrepresentable");
  ball.components.rigid_body!.mass = 1;
  ball.components.transform!.position = [0, 0] as unknown as Vec3;
  expect(() => s.stepFixed(scene, 1 / 120)).toThrow("dimensions");
});

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type VehicleComponent,
} from "./components.js";
import { VehicleSystem } from "./vehicle-system.js";
import { buildSystemContext } from "./system.js";
import { v3Length } from "../../scene/math.js";

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [
      {
        id: "car" as EntityId,
        name: "sports-car",
        components: {
          [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        },
      },
      {
        id: "floor" as EntityId,
        name: "floor",
        components: {
          [Tag.Transform]: { position: [0, -1, 0], rotation: [0, 0, 0, 1], scale: [100, 0.1, 100], parent: null },
          [Tag.Collider]: { shape: "box", extents: [50, 0.05, 50] },
        },
      },
    ],
  };
  const scene = new Scene(spec);
  setGameComponent<VehicleComponent>(scene.get("car" as EntityId)!, GameTag.Vehicle, {
    throttle: 0,
    steering: 0,
    handbrake: false,
    enginePower: 40.0,
    maxSpeed: 30.0,
    brakeForce: 20.0,
    turnSpeed: 2.0,
    aerodynamicDrag: 0.05,
    rollingResistance: 0.02,
    tireGrip: 0.8,
    suspensionStiffness: 15.0,
    suspensionDamping: 2.0,
    suspensionRestLength: 1.0,
    centerOfMassOffset: -0.1,
    mass: 1000.0,
    velocity: [0, 0, 0],
    yaw: 0,
    isGrounded: false,
    groundNormal: [0, 1, 0],
  });
  return scene;
}

describe("VehicleSystem", () => {
  it("accelerates forward when throttle is positive", () => {
    const scene = makeScene();
    const system = new VehicleSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    
    const car = scene.get("car" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(car, GameTag.Vehicle)!;
    vehicle.throttle = 1.0;

    system.update(scene, 0.1, ctx);
    
    expect(vehicle.velocity[2]).toBeLessThan(0); // in glTF, forward is -Z
    const tx = car.components[Tag.Transform]!;
    expect(tx.position[2]).toBeLessThan(0);
  });

  it("steers and changes yaw when steering is non-zero", () => {
    const scene = makeScene();
    const system = new VehicleSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    
    const car = scene.get("car" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(car, GameTag.Vehicle)!;
    vehicle.throttle = 1.0;
    vehicle.steering = 1.0; // turn right
    vehicle.velocity = [0, 0, -5]; // need velocity to steer

    system.update(scene, 0.1, ctx);
    
    expect(vehicle.yaw).toBeLessThan(0); // steering right decreases yaw in right-handed coords
  });

  it("decelerates when braking", () => {
    const scene = makeScene();
    const system = new VehicleSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    
    const car = scene.get("car" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(car, GameTag.Vehicle)!;
    vehicle.velocity = [0, 0, -10]; // going forward
    vehicle.throttle = -1.0; // brake

    system.update(scene, 0.1, ctx);
    
    // Braking reduces the speed magnitude (Z goes from -10 to closer to 0)
    expect(vehicle.velocity[2]).toBeGreaterThan(-10);
  });

  it("drifts with reduced grip when handbrake is engaged", () => {
    const scene = makeScene();
    const system = new VehicleSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    
    const car = scene.get("car" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(car, GameTag.Vehicle)!;
    vehicle.velocity = [5, 0, -5]; // sliding diagonally
    vehicle.handbrake = true;

    // Tick the update
    system.update(scene, 0.1, ctx);
    
    // Slide should persist longer because grip is reduced, so lateral velocity (X) is still high
    expect(Math.abs(vehicle.velocity[0])).toBeGreaterThan(4);
  });

  it("adjusts handling, applies water drag and sets surfaceType/waterSpray when on water_glide", () => {
    const scene = makeScene();
    const system = new VehicleSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);

    // Set the floor entity to be a water_glide race surface
    const floor = scene.get("floor" as EntityId)!;
    (floor.components as any)["game.race_surface"] = {
      surface: "water_glide",
      beat: "spray_gate",
      routeIndex: 4
    };

    const car = scene.get("car" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(car, GameTag.Vehicle)!;
    vehicle.velocity = [0, 0, -20]; // moving fast forward

    system.update(scene, 0.1, ctx);

    // Assert that the surface type is identified as "water"
    expect(vehicle.surfaceType).toBe("water");
    // Assert that waterSpray is calculated (proportional to speed) and is > 0
    expect(vehicle.waterSpray).toBeGreaterThan(0);
    expect(vehicle.waterSpray).toBeLessThanOrEqual(1.0);
    // Since drag multiplier is 3x on water, velocity decay should be faster than standard drag.
    expect(vehicle.velocity[2]).toBeGreaterThan(-20);
  });
});

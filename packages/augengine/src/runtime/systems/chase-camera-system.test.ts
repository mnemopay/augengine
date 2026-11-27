import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type VehicleComponent,
} from "./components.js";
import { ChaseCameraSystem } from "./chase-camera-system.js";
import { buildSystemContext } from "./system.js";

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [
      {
        id: "cam" as EntityId,
        name: "camera",
        components: {
          [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
          [Tag.Camera]: { fovY: Math.PI / 3, near: 0.1, far: 100, active: true },
        },
      },
      {
        id: "car" as EntityId,
        name: "sports-car",
        components: {
          [Tag.Transform]: { position: [0, 1, -5], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        },
      },
    ],
    defaultCamera: "cam" as EntityId,
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
    isGrounded: true,
    groundNormal: [0, 1, 0],
  });
  return scene;
}

describe("ChaseCameraSystem", () => {
  it("interpolates camera position to follow vehicle from behind", () => {
    const scene = makeScene();
    const system = new ChaseCameraSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    
    // Set parameters
    system.distance = 5.0;
    system.height = 2.0;
    system.lerpSpeed = 10.0; // High speed to snap quickly in test

    // Vehicle is at [0, 1, -5], Yaw is 0 (facing -Z).
    // Behind the vehicle (+Z) is [0, 1 + 2, -5 + 5] = [0, 3, 0].
    
    system.update(scene, 0.5, ctx);

    const cam = scene.get("cam" as EntityId)!;
    const camTx = cam.components[Tag.Transform]!;

    // Camera position should have interpolated from [0, 0, 0] towards [0, 3, 0]
    expect(camTx.position[1]).toBeGreaterThan(0);
    expect(camTx.position[2]).toBe(0); // since x and z are at 0
  });
});

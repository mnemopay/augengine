import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type RacerComponent,
  type RaceStateComponent,
  type VehicleComponent,
} from "./components.js";
import { RaceSystem } from "./race-system.js";
import { buildSystemContext } from "./system.js";

function makeRaceScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test-race" },
    entities: [
      {
        id: "player_vehicle" as EntityId,
        name: "Player Racer",
        components: {
          [Tag.Transform]: { position: [0, 0.5, 0], rotation: [0, 0, 0, 1], scale: [1.9, 0.7, 4.3], parent: null },
        },
      },
      {
        id: "racer_ai" as EntityId,
        name: "AI Racer",
        components: {
          [Tag.Transform]: { position: [4, 0.5, -4], rotation: [0, 0, 0, 1], scale: [1.9, 0.7, 4.3], parent: null },
        },
      },
      // Track nodes for spline path
      {
        id: "track_00" as EntityId,
        name: "Track node 0",
        components: {
          [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
          "game.race_surface": { surface: "road", beat: "start", routeIndex: 0 },
        },
      },
      {
        id: "track_01" as EntityId,
        name: "Track node 1",
        components: {
          [Tag.Transform]: { position: [0, 0, -20], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
          "game.race_surface": { surface: "road", beat: "mid", routeIndex: 1 },
        },
      },
      {
        id: "track_02" as EntityId,
        name: "Track node 2",
        components: {
          [Tag.Transform]: { position: [0, 0, -40], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
          "game.race_surface": { surface: "road", beat: "end", routeIndex: 2 },
        },
      },
      // Checkpoints
      {
        id: "checkpoint_start" as EntityId,
        name: "Start Checkpoint",
        components: {
          [Tag.Transform]: { position: [0, 2.0, 0], rotation: [0, 0, 0, 1], scale: [12.0, 4.0, 0.5], parent: null },
          "game.race_checkpoint": { id: "start", split: 0, routeIndex: 0, stageLaps: 3, stageId: "lagoon-crown" },
        },
      },
      {
        id: "checkpoint_mid" as EntityId,
        name: "Mid Checkpoint",
        components: {
          [Tag.Transform]: { position: [0, 2.0, -20], rotation: [0, 0, 0, 1], scale: [12.0, 4.0, 0.5], parent: null },
          "game.race_checkpoint": { id: "mid", split: 1, routeIndex: 1, stageLaps: 3, stageId: "lagoon-crown" },
        },
      },
    ],
  };

  const scene = new Scene(spec);

  // Set Player components
  setGameComponent<VehicleComponent>(scene.get("player_vehicle" as EntityId)!, GameTag.Vehicle, {
    throttle: 0,
    steering: 0,
    handbrake: false,
    enginePower: 47000,
    maxSpeed: 60,
    brakeForce: 32000,
    turnSpeed: 2.15,
    aerodynamicDrag: 0.24,
    rollingResistance: 0.035,
    tireGrip: 0.9,
    suspensionStiffness: 28000,
    suspensionDamping: 4200,
    suspensionRestLength: 1.0,
    centerOfMassOffset: -0.25,
    mass: 980,
    velocity: [0, 0, 0],
    yaw: 0,
    isGrounded: true,
    groundNormal: [0, 1, 0],
  });
  setGameComponent<RacerComponent>(scene.get("player_vehicle" as EntityId)!, GameTag.Racer, {
    racerId: "player",
    currentLap: 0,
    currentCheckpointIndex: -1,
    currentRank: 1,
  });

  // Set AI components
  setGameComponent<VehicleComponent>(scene.get("racer_ai" as EntityId)!, GameTag.Vehicle, {
    throttle: 0,
    steering: 0,
    handbrake: false,
    enginePower: 47000,
    maxSpeed: 60,
    brakeForce: 32000,
    turnSpeed: 2.15,
    aerodynamicDrag: 0.24,
    rollingResistance: 0.035,
    tireGrip: 0.9,
    suspensionStiffness: 28000,
    suspensionDamping: 4200,
    suspensionRestLength: 1.0,
    centerOfMassOffset: -0.25,
    mass: 980,
    velocity: [0, 0, 0],
    yaw: 0,
    isGrounded: true,
    groundNormal: [0, 1, 0],
  });
  setGameComponent<RacerComponent>(scene.get("racer_ai" as EntityId)!, GameTag.Racer, {
    racerId: "ai",
    currentLap: 0,
    currentCheckpointIndex: -1,
    currentRank: 2,
  });

  return scene;
}

describe("RaceSystem", () => {
  it("freezes vehicle throttles during countdown", () => {
    const scene = makeRaceScene();
    const system = new RaceSystem();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);

    const player = scene.get("player_vehicle" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(player, GameTag.Vehicle)!;
    vehicle.throttle = 1.0;

    system.update(scene, 0.1, ctx);

    // After system update during countdown, throttle is set to 0
    expect(vehicle.throttle).toBe(0);

    const stateEnt = Array.from(scene.entities.values()).find((e) => e.components[GameTag.RaceState])!;
    const state = stateEnt.components[GameTag.RaceState] as RaceStateComponent;
    expect(state.countdown).toBeCloseTo(2.9);
    expect(state.active).toBe(false);
  });

  it("activates race when countdown finishes", () => {
    const scene = makeRaceScene();
    const system = new RaceSystem();
    
    // First update to initialize state
    const ctx1 = buildSystemContext(0, [0, 0, 0], 1);
    system.update(scene, 0.1, ctx1);

    const stateEnt = Array.from(scene.entities.values()).find((e) => e.components[GameTag.RaceState])!;
    const state = stateEnt.components[GameTag.RaceState] as RaceStateComponent;
    state.countdown = 0.05; // Force near end of countdown

    const ctx2 = buildSystemContext(100, [0, 0, 0], 1);
    system.update(scene, 0.1, ctx2);

    expect(state.countdown).toBe(0);
    expect(state.active).toBe(true);
    expect(ctx2.events).toContainEqual(expect.objectContaining({ type: "race-started" }));
  });

  it("updates checkpoint crossings and lap loops", () => {
    const scene = makeRaceScene();
    const system = new RaceSystem();
    
    // Set state to active so we skip countdown logic and process checkpoints
    const ctx = buildSystemContext(3500, [0, 0, 0], 1);
    system.update(scene, 3.5, ctx); // countdown reaches 0

    const player = scene.get("player_vehicle" as EntityId)!;
    const racer = getGameComponent<RacerComponent>(player, GameTag.Racer)!;

    // Player position is at [0, 0.5, 0], which overlaps checkpoint_start ([0, 2, 0] scale [12, 4, 0.5])
    // The expected next checkpoint index is 0. So checkpoint_start (split 0) is matched!
    system.update(scene, 0.1, ctx);
    expect(racer.currentCheckpointIndex).toBe(0);

    // Move player to mid checkpoint [0, 0.5, -20]
    scene.setLocalTransform("player_vehicle" as EntityId, [0, 0.5, -20], [0, 0, 0, 1], [1.9, 0.7, 4.3]);
    system.update(scene, 0.1, ctx);
    expect(racer.currentCheckpointIndex).toBe(1);

    // Cross checkpoint_start (split 0) again to complete a lap
    scene.setLocalTransform("player_vehicle" as EntityId, [0, 0.5, 0], [0, 0, 0, 1], [1.9, 0.7, 4.3]);
    system.update(scene, 0.1, ctx);
    expect(racer.currentCheckpointIndex).toBe(0);
    expect(racer.currentLap).toBe(2);
    expect(ctx.events).toContainEqual(expect.objectContaining({ type: "race-lap-completed", lap: 1 }));
  });

  it("manages AI autopilot steering and throttle", () => {
    const scene = makeRaceScene();
    const system = new RaceSystem();

    // Fast-forward countdown
    const ctx = buildSystemContext(3500, [0, 0, 0], 1);
    system.update(scene, 3.5, ctx); // countdown reaches 0

    const ai = scene.get("racer_ai" as EntityId)!;
    const vehicle = getGameComponent<VehicleComponent>(ai, GameTag.Vehicle)!;
    
    // Initial velocity and transform
    system.update(scene, 0.1, ctx);

    // AI racer is at [4, 0.5, -4]
    // Nearest node is track_00 at [0, 0, 0]. Lookahead index will be (0 + 2) % 3 = 2, which is track_02 at [0, 0, -40].
    // Target position is [0, 0, -40].
    // Yaw steering is computed to align towards that target.
    // The throttle should be updated to drive towards the track spline.
    expect(vehicle.throttle).toBeGreaterThan(0);
    expect(vehicle.steering).not.toBe(0);
  });

  it("calculates rankings based on lap, checkpoints, and track proximity", () => {
    const scene = makeRaceScene();
    const system = new RaceSystem();

    // Fast-forward countdown
    const ctx = buildSystemContext(3500, [0, 0, 0], 1);
    system.update(scene, 3.5, ctx);

    const player = scene.get("player_vehicle" as EntityId)!;
    const playerRacer = getGameComponent<RacerComponent>(player, GameTag.Racer)!;

    const ai = scene.get("racer_ai" as EntityId)!;
    const aiRacer = getGameComponent<RacerComponent>(ai, GameTag.Racer)!;

    // Player lap = 1, AI lap = 0
    playerRacer.currentLap = 1;
    playerRacer.currentCheckpointIndex = 0;
    aiRacer.currentLap = 0;
    aiRacer.currentCheckpointIndex = 1;

    system.update(scene, 0.1, ctx);

    const stateEnt = Array.from(scene.entities.values()).find((e) => e.components[GameTag.RaceState])!;
    const state = stateEnt.components[GameTag.RaceState] as RaceStateComponent;

    // Player should be ranked 1st since lap is higher
    expect(state.rankings[0]).toBe("player");
    expect(playerRacer.currentRank).toBe(1);
    expect(aiRacer.currentRank).toBe(2);

    // Now set AI lap = 2
    aiRacer.currentLap = 2;
    system.update(scene, 0.1, ctx);

    // AI should now be ranked 1st
    expect(state.rankings[0]).toBe("ai");
    expect(aiRacer.currentRank).toBe(1);
    expect(playerRacer.currentRank).toBe(2);
  });
});

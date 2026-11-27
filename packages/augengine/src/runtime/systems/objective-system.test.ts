/**
 * Track 5 — ObjectiveSystem unit tests.
 *
 * Covers: secure-on-click event, capture accumulator + decay outside
 * ring, extract gated on hostiles-down predicate, score crediting via
 * MissionComponent.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type Entity, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  MISSION_ENTITY_ID,
  getGameComponent,
  setGameComponent,
  type MissionComponent,
  type ObjectiveComponent,
} from "./components.js";
import { ObjectiveSystem } from "./objective-system.js";
import { createMissionEntity } from "./mission-system.js";
import { buildSystemContext } from "./system.js";

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [],
  };
  const scene = new Scene(spec);
  createMissionEntity(scene, { mapId: "test", durationSec: 240 });
  return scene;
}

function addObjective(scene: Scene, id: string, obj: ObjectiveComponent): EntityId {
  const entity: Entity = {
    id: id as EntityId,
    name: id,
    components: {
      [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
    },
  };
  setGameComponent<ObjectiveComponent>(entity, GameTag.Objective, obj);
  scene.add(entity);
  return id as EntityId;
}

describe("ObjectiveSystem", () => {
  const sys = new ObjectiveSystem({
    allHostilesDown: () => true,
    missionEntityId: MISSION_ENTITY_ID,
  });

  it("secure() flips completed + credits score + emits event", () => {
    const scene = makeScene();
    const id = addObjective(scene, "crate_1", {
      kind: "secure", goalId: "crate_1", completed: false, scoreReward: 50,
    });
    const ctx = buildSystemContext(0, null, 1);
    expect(sys.secure(scene, id, ctx)).toBe(true);
    const obj = getGameComponent<ObjectiveComponent>(scene.get(id)!, GameTag.Objective)!;
    expect(obj.completed).toBe(true);
    const mission = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    expect(mission.score).toBe(50);
    expect(ctx.events.some((e) => e.type === "objective-completed")).toBe(true);
  });

  it("capture accumulates inside ring, decays outside, completes at threshold", () => {
    const scene = makeScene();
    const id = addObjective(scene, "cap_1", {
      kind: "capture", goalId: "cap_1", completed: false,
      position: [0, 0, 0], radius: 2, requiredSeconds: 1, progressSeconds: 0,
      scoreReward: 400,
    });
    // Inside the ring.
    let ctx = buildSystemContext(0, [0, 0, 0], 1);
    sys.update(scene, 0.5, ctx);
    const obj = getGameComponent<ObjectiveComponent>(scene.get(id)!, GameTag.Objective)!;
    expect(obj.progressSeconds).toBe(0.5);
    // Step out — decays at half speed.
    ctx = buildSystemContext(0, [10, 0, 0], 1);
    sys.update(scene, 0.4, ctx);
    expect(obj.progressSeconds).toBeCloseTo(0.5 - 0.4 * 0.5);
    // Step back in for the rest.
    ctx = buildSystemContext(0, [0, 0, 0], 1);
    sys.update(scene, 1.0, ctx);
    expect(obj.completed).toBe(true);
    expect(ctx.events.some((e) => e.type === "objective-completed")).toBe(true);
  });

  it("extract waits for both proximity AND hostiles-down predicate", () => {
    const scene = makeScene();
    const id = addObjective(scene, "ext_1", {
      kind: "extract", goalId: "ext_1", completed: false,
      position: [0, 0, 0], radius: 2, scoreReward: 200,
    });
    let hostilesDown = false;
    const sys = new ObjectiveSystem({
      allHostilesDown: () => hostilesDown,
      missionEntityId: MISSION_ENTITY_ID,
    });
    // In zone but hostiles up — no completion.
    let ctx = buildSystemContext(0, [0, 0, 0], 1);
    sys.update(scene, 0.1, ctx);
    expect(getGameComponent<ObjectiveComponent>(scene.get(id)!, GameTag.Objective)!.completed).toBe(false);
    // Hostiles down — completes.
    hostilesDown = true;
    ctx = buildSystemContext(0, [0, 0, 0], 1);
    sys.update(scene, 0.1, ctx);
    expect(getGameComponent<ObjectiveComponent>(scene.get(id)!, GameTag.Objective)!.completed).toBe(true);
  });
});

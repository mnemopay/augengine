import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  setGameComponent,
  type HealthComponent,
  type UIComponent,
  type MissionComponent,
  MISSION_ENTITY_ID,
} from "./components.js";
import { UISystem } from "./ui-system.js";
import { buildSystemContext } from "./system.js";

function makeBaseScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [
      {
        id: "p" as EntityId,
        name: "player",
        components: {
          [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        },
      },
    ],
  };
  return new Scene(spec);
}

describe("UISystem", () => {
  it("interpolates variables bound to the same entity's components", () => {
    const scene = makeBaseScene();
    const player = scene.get("p" as EntityId)!;

    // Attach Health component
    setGameComponent<HealthComponent>(player, GameTag.Health, {
      hp: 85, maxHp: 100, dead: false, lastDamageAt: -1,
    });

    // Attach UI component
    setGameComponent<UIComponent>(player, GameTag.UI, {
      template: "<div>HP: {{hp}} / {{maxHp}}</div>",
      bindings: {
        hp: { componentTag: GameTag.Health, fieldKey: "hp" },
        maxHp: { componentTag: GameTag.Health, fieldKey: "maxHp" },
      },
    });

    const uiSystem = new UISystem();
    const ctx = buildSystemContext(1000, [0, 0, 0], 0.85);

    uiSystem.update(scene, 0.016, ctx);

    // Verify rendered output in cached maps (graceful Node fallback)
    expect(uiSystem.renderedOutput.get("p" as EntityId)).toBe("<div>HP: 85 / 100</div>");
    
    const uiComp = scene.get("p" as EntityId)!.components[GameTag.UI] as UIComponent;
    expect(uiComp.lastRendered).toBe("<div>HP: 85 / 100</div>");
  });

  it("falls back to the mission singleton entity for bindings", () => {
    const scene = makeBaseScene();
    const player = scene.get("p" as EntityId)!;

    // Create mission singleton entity
    const missionEntity = {
      id: MISSION_ENTITY_ID,
      name: "Mission",
      components: {
        [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      },
    };
    scene.add(missionEntity);

    setGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission, {
      mapId: "desert_track",
      durationSec: 120,
      timeLeftSec: 105.5,
      score: 450,
      ended: false,
      won: false,
      endReason: "open",
      checkpoints: ["checkpoint_1"],
    });

    // Attach UI component to player
    setGameComponent<UIComponent>(player, GameTag.UI, {
      template: "<div>Score: {{score}} | Time: {{time}}</div>",
      bindings: {
        score: { componentTag: GameTag.Mission, fieldKey: "score" },
        time: { componentTag: GameTag.Mission, fieldKey: "timeLeftSec" },
      },
    });

    const uiSystem = new UISystem();
    const ctx = buildSystemContext(1000, [0, 0, 0], 1);

    uiSystem.update(scene, 0.016, ctx);

    expect(uiSystem.renderedOutput.get("p" as EntityId)).toBe("<div>Score: 450 | Time: 105.5</div>");
  });

  it("falls back to general scene entity search when component is not on current or mission entity", () => {
    const scene = makeBaseScene();
    
    // Add another entity with health
    const otherEntity = {
      id: "enemy" as EntityId,
      name: "enemy_recruit",
      components: {
        [Tag.Transform]: { position: [10, 0, 10], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      },
    };
    scene.add(otherEntity);
    setGameComponent<HealthComponent>(scene.get("enemy" as EntityId)!, GameTag.Health, {
      hp: 50, maxHp: 50, dead: false, lastDamageAt: -1,
    });

    // UI component on player referencing Health component (not on player, but on enemy)
    const player = scene.get("p" as EntityId)!;
    setGameComponent<UIComponent>(player, GameTag.UI, {
      template: "<div>Enemy HP: {{enemy_hp}}</div>",
      bindings: {
        enemy_hp: { componentTag: GameTag.Health, fieldKey: "hp" },
      },
    });

    const uiSystem = new UISystem();
    const ctx = buildSystemContext(1000, [0, 0, 0], 1);

    uiSystem.update(scene, 0.016, ctx);
    expect(uiSystem.renderedOutput.get("p" as EntityId)).toBe("<div>Enemy HP: 50</div>");
  });

  it("handles formatting arrays and object bindings cleanly", () => {
    const scene = makeBaseScene();
    const player = scene.get("p" as EntityId)!;

    const missionEntity = {
      id: MISSION_ENTITY_ID,
      name: "Mission",
      components: {
        [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      },
    };
    scene.add(missionEntity);
    setGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission, {
      mapId: "desert_track",
      durationSec: 120,
      timeLeftSec: 100,
      score: 100,
      ended: false,
      won: false,
      endReason: "open",
      checkpoints: ["start", "checkpoint_1", "checkpoint_2"],
    });

    setGameComponent<UIComponent>(player, GameTag.UI, {
      template: "<div>Passed: {{checkpoints}}</div>",
      bindings: {
        checkpoints: { componentTag: GameTag.Mission, fieldKey: "checkpoints" },
      },
    });

    const uiSystem = new UISystem();
    const ctx = buildSystemContext(1000, [0, 0, 0], 1);

    uiSystem.update(scene, 0.016, ctx);
    expect(uiSystem.renderedOutput.get("p" as EntityId)).toBe("<div>Passed: start, checkpoint_1, checkpoint_2</div>");
  });

  it("cleans up cache entries for removed entities or detached UI", () => {
    const scene = makeBaseScene();
    const player = scene.get("p" as EntityId)!;

    setGameComponent<UIComponent>(player, GameTag.UI, {
      template: "<div>Hello</div>",
      bindings: {},
    });

    const uiSystem = new UISystem();
    const ctx = buildSystemContext(1000, [0, 0, 0], 1);

    uiSystem.update(scene, 0.016, ctx);
    expect(uiSystem.renderedOutput.has("p" as EntityId)).toBe(true);

    // Remove UI component
    delete player.components[GameTag.UI];
    uiSystem.update(scene, 0.016, ctx);

    expect(uiSystem.renderedOutput.has("p" as EntityId)).toBe(false);
  });
});

/**
 * Track 5 — SaveSystem unit tests.
 *
 * Covers: round-trips an entire scene including game components via the
 * in-memory IO adapter. Verifies that mission + trust + health
 * components survive a save → load cycle, which is the load-bearing
 * guarantee for "saves persist via the engine's persistence layer" in
 * the Track 5 acceptance criterion.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag, MISSION_ENTITY_ID, getGameComponent, setGameComponent,
  type HealthComponent, type MissionComponent, type SquadTrustComponent,
} from "./components.js";
import { createMissionEntity } from "./mission-system.js";
import { SquadTrustSystem } from "./squad-trust-system.js";
import { SaveSystem, memorySaveIO } from "./save-system.js";
import { buildSystemContext } from "./system.js";

describe("SaveSystem", () => {
  it("round-trips mission + trust + health components through gzip + JSON", async () => {
    const spec: SceneSpec = {
      schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [
        { id: "p" as EntityId, name: "p", components: {
          [Tag.Transform]: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        }},
        { id: "maya" as EntityId, name: "Maya", components: {
          [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        }},
      ],
    };
    const scene = new Scene(spec);
    setGameComponent<HealthComponent>(scene.get("p" as EntityId)!, GameTag.Health, {
      hp: 75, maxHp: 100, dead: false, lastDamageAt: 1234,
    });
    createMissionEntity(scene, { mapId: "block13", durationSec: 240 });
    const m = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    m.score = 350;
    m.checkpoints.push("first-kill");
    SquadTrustSystem.register(scene, "maya" as EntityId, "Maya");
    const ctx = buildSystemContext(1000, null, 1);
    SquadTrustSystem.recordCoveringFire(scene, "maya" as EntityId, ctx);

    const io = memorySaveIO();
    const save = new SaveSystem("test-key", io);
    await save.persist(scene);
    const restored = await save.restore();
    expect(restored).not.toBeNull();
    const restoredScene = restored!.scene;

    const restoredHealth = getGameComponent<HealthComponent>(
      restoredScene.get("p" as EntityId)!, GameTag.Health,
    )!;
    expect(restoredHealth.hp).toBe(75);
    expect(restoredHealth.lastDamageAt).toBe(1234);

    const restoredMission = getGameComponent<MissionComponent>(
      restoredScene.get(MISSION_ENTITY_ID)!, GameTag.Mission,
    )!;
    expect(restoredMission.score).toBe(350);
    expect(restoredMission.checkpoints).toContain("first-kill");

    const restoredTrust = getGameComponent<SquadTrustComponent>(
      restoredScene.get("maya" as EntityId)!, GameTag.SquadTrust,
    )!;
    expect(restoredTrust.score).toBe(55);
  });

  it("restore returns null when no save exists", async () => {
    const save = new SaveSystem("missing", memorySaveIO());
    expect(await save.restore()).toBeNull();
  });
});

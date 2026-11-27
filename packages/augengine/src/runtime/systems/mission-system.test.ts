/**
 * Track 5 — MissionSystem unit tests.
 *
 * Covers: timer countdown, timeout end reason, win on configured goal,
 * death-of-player end reason, idempotent end (no double-events),
 * checkpoint one-shot firing.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import type { EntityId, SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  MISSION_ENTITY_ID,
  getGameComponent,
  type MissionComponent,
} from "./components.js";
import { MissionSystem, createMissionEntity } from "./mission-system.js";
import { buildSystemContext } from "./system.js";

const PLAYER = "p" as EntityId;

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [],
  };
  const scene = new Scene(spec);
  createMissionEntity(scene, { mapId: "block13", durationSec: 5 });
  return scene;
}

describe("MissionSystem", () => {
  it("decrements timer per frame", () => {
    const scene = makeScene();
    const sys = new MissionSystem({ mapId: "block13", durationSec: 5, playerEntityId: PLAYER });
    const ctx = buildSystemContext(0, null, 1);
    sys.update(scene, 1, ctx);
    sys.update(scene, 1, ctx);
    const m = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    expect(m.timeLeftSec).toBeCloseTo(3);
  });

  it("ends on timeout with reason=timeout", () => {
    const scene = makeScene();
    const sys = new MissionSystem({ mapId: "block13", durationSec: 1, playerEntityId: PLAYER });
    const m0 = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    m0.timeLeftSec = 1; // override
    const ctx = buildSystemContext(0, null, 1);
    sys.update(scene, 2, ctx);
    expect(m0.ended).toBe(true);
    expect(m0.endReason).toBe("timeout");
    expect(m0.won).toBe(false);
    expect(ctx.events.some((e) => e.type === "mission-ended")).toBe(true);
  });

  it("ends on win when objective-completed matches winOnGoalId", () => {
    const scene = makeScene();
    const sys = new MissionSystem({
      mapId: "block13", durationSec: 5, playerEntityId: PLAYER,
      winOnGoalId: "block13.extraction",
    });
    const ctx = buildSystemContext(0, null, 1);
    ctx.events.push({ type: "objective-completed", goalId: "block13.extraction", kind: "extract", score: 200 });
    sys.update(scene, 0.016, ctx);
    const m = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    expect(m.ended).toBe(true);
    expect(m.won).toBe(true);
    expect(m.endReason).toBe("won");
  });

  it("ends on player death", () => {
    const scene = makeScene();
    const sys = new MissionSystem({ mapId: "block13", durationSec: 5, playerEntityId: PLAYER });
    const ctx = buildSystemContext(0, null, 1);
    ctx.events.push({ type: "death", target: PLAYER, source: "recruit" });
    sys.update(scene, 0.016, ctx);
    const m = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    expect(m.endReason).toBe("death");
    expect(m.won).toBe(false);
  });

  it("end is idempotent — second tick after end emits no new event", () => {
    const scene = makeScene();
    const sys = new MissionSystem({ mapId: "block13", durationSec: 5, playerEntityId: PLAYER });
    const m0 = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    m0.timeLeftSec = 0;
    const ctx = buildSystemContext(0, null, 1);
    sys.update(scene, 1, ctx);
    const eventsAfterFirst = ctx.events.length;
    sys.update(scene, 1, ctx);
    expect(ctx.events.length).toBe(eventsAfterFirst);
  });

  it("checkpoints fire once and persist into MissionComponent.checkpoints", () => {
    const scene = makeScene();
    let fired = false;
    const sys = new MissionSystem({
      mapId: "block13", durationSec: 5, playerEntityId: PLAYER,
      checkpoints: [{ id: "first-kill", predicate: () => (fired = !fired, !fired ? false : true) }],
    });
    // First update: predicate returns true → checkpoint records.
    fired = false; // ensure predicate returns true on first call (true → !true=false on first; flip logic):
    // simplify by using a stable predicate:
    const sys2 = new MissionSystem({
      mapId: "block13", durationSec: 5, playerEntityId: PLAYER,
      checkpoints: [{ id: "always", predicate: () => true }],
    });
    const ctx = buildSystemContext(0, null, 1);
    sys2.update(scene, 0.016, ctx);
    const m = getGameComponent<MissionComponent>(scene.get(MISSION_ENTITY_ID)!, GameTag.Mission)!;
    expect(m.checkpoints).toContain("always");
    const eventsAfter = ctx.events.length;
    sys2.update(scene, 0.016, ctx);
    // Predicate still true, but checkpoint id already recorded — no new event.
    const checkpointEventsAfter = ctx.events.filter((e) => e.type === "mission-checkpoint").length;
    expect(checkpointEventsAfter).toBe(1);
    expect(ctx.events.length - eventsAfter).toBe(0);
    // First sys (unused) just here to demonstrate the API surface:
    void sys;
  });
});

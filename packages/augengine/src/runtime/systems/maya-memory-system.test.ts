/**
 * Track 5 — MayaMemorySystem unit tests.
 *
 * Covers: friendly fire bumps count + flips betrayed at 2nd hit,
 * objective + kill consequences, onConsequence sink invoked.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import { GameTag, getGameComponent, type MayaMemoryComponent } from "./components.js";
import { MayaMemorySystem, ensureMayaMemoryComponent } from "./maya-memory-system.js";
import { buildSystemContext } from "./system.js";

const MAYA = "maya" as EntityId;

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1, source: { kind: "authored", path: "t" },
    entities: [{
      id: MAYA, name: "Maya",
      components: { [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null } },
    }],
  };
  const scene = new Scene(spec);
  ensureMayaMemoryComponent(scene, MAYA);
  return scene;
}

describe("MayaMemorySystem", () => {
  it("friendly fire on Maya bumps count + flips betrayed at 2nd hit", () => {
    const scene = makeScene();
    const sink: string[] = [];
    const sys = new MayaMemorySystem({ mayaEntityId: MAYA, onConsequence: (l) => sink.push(l) });
    const ctx = buildSystemContext(1000, null, 1);
    ctx.events.push({ type: "damage", target: MAYA, amount: 10, remainingHp: 90, source: "friendly-fire" });
    sys.update(scene, 0.016, ctx);
    const mem1 = getGameComponent<MayaMemoryComponent>(scene.get(MAYA)!, GameTag.MayaMemory)!;
    expect(mem1.friendlyFireCount).toBe(1);
    expect(mem1.betrayed).toBe(false);

    const ctx2 = buildSystemContext(2000, null, 1);
    ctx2.events.push({ type: "damage", target: MAYA, amount: 10, remainingHp: 80, source: "friendly-fire" });
    sys.update(scene, 0.016, ctx2);
    const mem2 = getGameComponent<MayaMemoryComponent>(scene.get(MAYA)!, GameTag.MayaMemory)!;
    expect(mem2.friendlyFireCount).toBe(2);
    expect(mem2.betrayed).toBe(true);
    expect(sink.length).toBe(2);
  });

  it("objective completion increments witnessed counter", () => {
    const scene = makeScene();
    const sys = new MayaMemorySystem({ mayaEntityId: MAYA });
    const ctx = buildSystemContext(0, null, 1);
    ctx.events.push({ type: "objective-completed", goalId: "crate_1", kind: "secure", score: 50 });
    sys.update(scene, 0.016, ctx);
    const mem = getGameComponent<MayaMemoryComponent>(scene.get(MAYA)!, GameTag.MayaMemory)!;
    expect(mem.objectivesWitnessed).toBe(1);
    expect(ctx.events.some((e) => e.type === "maya-consequence")).toBe(true);
  });

  it("first kill emits one consequence; subsequent kills do not duplicate", () => {
    const scene = makeScene();
    const sys = new MayaMemorySystem({ mayaEntityId: MAYA });
    const ctx = buildSystemContext(0, null, 1);
    ctx.events.push({ type: "shot-hit", weapon: "w" as EntityId, target: "x" as EntityId, damage: 25, killed: true });
    sys.update(scene, 0.016, ctx);
    const mem = getGameComponent<MayaMemoryComponent>(scene.get(MAYA)!, GameTag.MayaMemory)!;
    expect(mem.killsWitnessed).toBe(1);
    const consequences = ctx.events.filter((e) => e.type === "maya-consequence").length;
    expect(consequences).toBe(1);

    const ctx2 = buildSystemContext(0, null, 1);
    ctx2.events.push({ type: "shot-hit", weapon: "w" as EntityId, target: "y" as EntityId, damage: 25, killed: true });
    sys.update(scene, 0.016, ctx2);
    const mem2 = getGameComponent<MayaMemoryComponent>(scene.get(MAYA)!, GameTag.MayaMemory)!;
    expect(mem2.killsWitnessed).toBe(2);
    const consequences2 = ctx2.events.filter((e) => e.type === "maya-consequence").length;
    expect(consequences2).toBe(0);
  });
});

/**
 * Track 5 — HealthSystem unit tests.
 *
 * Covers: damage with armor pre-absorb, damage without armor, killing
 * blow + death event, applying damage to a missing entity, healing.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type ArmorComponent,
  type HealthComponent,
} from "./components.js";
import { HealthSystem } from "./health-system.js";
import { buildSystemContext } from "./system.js";

function makeScene(): Scene {
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
  const scene = new Scene(spec);
  setGameComponent<HealthComponent>(scene.get("p" as EntityId)!, GameTag.Health, {
    hp: 100, maxHp: 100, dead: false, lastDamageAt: -1,
  });
  return scene;
}

describe("HealthSystem", () => {
  it("deducts hp and pushes damage event", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(1000, [0, 0, 0], 1);
    const hp = HealthSystem.applyDamage(scene, "p" as EntityId, 25, ctx, { source: "recruit" });
    expect(hp).toBe(75);
    expect(ctx.events).toHaveLength(1);
    expect(ctx.events[0]).toMatchObject({ type: "damage", target: "p", amount: 25, remainingHp: 75 });
  });

  it("armor absorbs configured fraction first", () => {
    const scene = makeScene();
    setGameComponent<ArmorComponent>(scene.get("p" as EntityId)!, GameTag.Armor, {
      value: 50, max: 50, absorptionFraction: 0.5,
    });
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    HealthSystem.applyDamage(scene, "p" as EntityId, 40, ctx, { source: "recruit" });
    // 40 damage × 0.5 fraction = 20 absorbed; armor 50→30; hp 100→80
    const armor = getGameComponent<ArmorComponent>(scene.get("p" as EntityId)!, GameTag.Armor)!;
    const health = getGameComponent<HealthComponent>(scene.get("p" as EntityId)!, GameTag.Health)!;
    expect(armor.value).toBe(30);
    expect(health.hp).toBe(80);
  });

  it("emits death event on killing blow", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    HealthSystem.applyDamage(scene, "p" as EntityId, 200, ctx, { source: "recruit" });
    const health = getGameComponent<HealthComponent>(scene.get("p" as EntityId)!, GameTag.Health)!;
    expect(health.hp).toBe(0);
    expect(health.dead).toBe(true);
    expect(ctx.events.map((e) => e.type)).toEqual(["damage", "death"]);
  });

  it("returns null for missing entity instead of throwing", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    const r = HealthSystem.applyDamage(scene, "ghost" as EntityId, 10, ctx, { source: "recruit" });
    expect(r).toBeNull();
  });

  it("ignores damage on already-dead entity", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    HealthSystem.applyDamage(scene, "p" as EntityId, 200, ctx, { source: "recruit" });
    const before = ctx.events.length;
    HealthSystem.applyDamage(scene, "p" as EntityId, 50, ctx, { source: "recruit" });
    expect(ctx.events.length).toBe(before);
  });

  it("heal restores hp up to max", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    HealthSystem.applyDamage(scene, "p" as EntityId, 40, ctx, { source: "recruit" });
    HealthSystem.heal(scene, "p" as EntityId, 80);
    const health = getGameComponent<HealthComponent>(scene.get("p" as EntityId)!, GameTag.Health)!;
    expect(health.hp).toBe(100);
  });
});

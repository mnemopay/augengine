/**
 * Track 5 — WeaponSystem unit tests.
 *
 * Covers: fire cadence (cooldown enforces semi-auto), reload lockout +
 * draw, dry click, hit pipeline applies damage via HealthSystem,
 * friendly-fire vs player source tagging.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type HealthComponent,
  type WeaponComponent,
} from "./components.js";
import { WeaponSystem } from "./weapon-system.js";
import { buildSystemContext } from "./system.js";

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [
      {
        id: "weapon" as EntityId,
        name: "ARM",
        components: {
          [Tag.Transform]: { position: [0, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        },
      },
      {
        id: "target" as EntityId,
        name: "target",
        components: {
          [Tag.Transform]: { position: [3, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        },
      },
    ],
  };
  const scene = new Scene(spec);
  setGameComponent<WeaponComponent>(scene.get("weapon" as EntityId)!, GameTag.Weapon, {
    name: "ARM", ammo: 2, magSize: 12, reserve: 96, reserveMax: 96,
    shotDamage: 25, fireCooldown: 0.16, reloadLockout: 1.5,
    cooldownTimer: 0, reloadTimer: 0, recoilKick: 0,
  });
  setGameComponent<HealthComponent>(scene.get("target" as EntityId)!, GameTag.Health, {
    hp: 100, maxHp: 100, dead: false, lastDamageAt: -1,
  });
  return scene;
}

describe("WeaponSystem", () => {
  const sys = new WeaponSystem();

  it("fires when ammo + cooldown allow", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    const r = WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    expect(r.fired).toBe(true);
    if (r.fired) expect(r.remainingAmmo).toBe(1);
    expect(ctx.events.some((e) => e.type === "shot-fired")).toBe(true);
  });

  it("blocks consecutive fires until cooldown drains", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    const second = WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    expect(second.fired).toBe(false);
    if (!second.fired) expect(second.reason).toBe("cooldown");
    sys.update(scene, 0.2, ctx); // > fireCooldown (0.16) → cleared
    const third = WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    expect(third.fired).toBe(true);
  });

  it("dry click reports empty without spending ammo", () => {
    const scene = makeScene();
    const w = getGameComponent<WeaponComponent>(scene.get("weapon" as EntityId)!, GameTag.Weapon)!;
    w.ammo = 0;
    const ctx = buildSystemContext(0, null, 1);
    const r = WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    expect(r.fired).toBe(false);
    if (!r.fired) expect(r.reason).toBe("empty");
    expect(w.ammo).toBe(0);
  });

  it("reload locks out then draws from reserve", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    const w = getGameComponent<WeaponComponent>(scene.get("weapon" as EntityId)!, GameTag.Weapon)!;
    expect(WeaponSystem.reload(scene, "weapon" as EntityId, ctx)).toBe(true);
    expect(w.reloadTimer).toBe(1.5);
    // While reloading, firing is blocked.
    const blocked = WeaponSystem.fire(scene, "weapon" as EntityId, ctx);
    expect(blocked.fired).toBe(false);
    sys.update(scene, 1.6, ctx);
    expect(w.reloadTimer).toBe(0);
    expect(w.ammo).toBe(12);
    expect(w.reserve).toBe(96 - 10);
    expect(ctx.events.some((e) => e.type === "reload-complete")).toBe(true);
  });

  it("hit pipeline deducts damage via HealthSystem", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    WeaponSystem.fire(scene, "weapon" as EntityId, ctx, { hit: { target: "target" as EntityId } });
    const hpHealth = getGameComponent<HealthComponent>(scene.get("target" as EntityId)!, GameTag.Health)!;
    expect(hpHealth.hp).toBe(75);
    expect(ctx.events.some((e) => e.type === "shot-hit")).toBe(true);
  });

  it("friendly-fire source tags damage event correctly", () => {
    const scene = makeScene();
    const ctx = buildSystemContext(0, null, 1);
    WeaponSystem.fire(scene, "weapon" as EntityId, ctx, {
      hit: { target: "target" as EntityId, isFriendly: true },
    });
    const damageEvt = ctx.events.find((e) => e.type === "damage");
    expect(damageEvt && "source" in damageEvt && damageEvt.source).toBe("friendly-fire");
  });
});

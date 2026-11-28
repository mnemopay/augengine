import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../scene/types.js";
import { PhysicsSystem } from "../physics/system.js";
import {
  applyDamage,
  CameraShake,
  CombatWorld,
  createHealth,
  cycleLockTarget,
  Hitbox,
  HitStop,
  knockbackVelocity,
  pickLockTarget,
  shouldBreakLock,
  signedAngle,
  tickHealth,
  type CombatantRig,
} from "./combat.js";

const systems: PhysicsSystem[] = [];
afterEach(() => {
  for (const s of systems) s.dispose();
  systems.length = 0;
});

/** A tiny skeleton: root → nodes[i] with parent indices (−1 = root). */
function skeleton(prefix: string, root: Vec3, nodes: Array<{ name: string; parent: number; t: Vec3 }>): { entities: Entity[]; rig: CombatantRig; rootId: EntityId } {
  const rootId = `${prefix}_root` as EntityId;
  const entities: Entity[] = [{ id: rootId, name: rootId, components: { [Tag.Transform]: { position: root, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null } } }];
  const nodeToEntity = new Map<number, EntityId>();
  nodes.forEach((n, i) => {
    const id = `${prefix}_node_${i}` as EntityId;
    nodeToEntity.set(i, id);
    entities.push({ id, name: n.name, components: { [Tag.Transform]: { position: n.t, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: n.parent < 0 ? rootId : (`${prefix}_node_${n.parent}` as EntityId) } } });
  });
  return { entities, rig: { nodeNames: nodes.map((n) => n.name), nodeToEntity }, rootId };
}

async function arena(attackerRoot: Vec3 = [-1.2, 0, 0], bladeLen = 1.1) {
  const def = skeleton("def", [0, 0, 0], [{ name: "Hips", parent: -1, t: [0, 1, 0] }, { name: "Head", parent: 0, t: [0, 0.6, 0] }]);
  const att = skeleton("att", attackerRoot, [{ name: "Hand.R", parent: -1, t: [0, 1.2, 0] }, { name: "Weapon_end", parent: 0, t: [bladeLen, 0, 0] }]);
  const ally = skeleton("ally", [0, 0, 4], [{ name: "Hips", parent: -1, t: [0, 1, 0] }, { name: "Head", parent: 0, t: [0, 0.6, 0] }]);
  const scene = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [...def.entities, ...att.entities, ...ally.entities] });
  const physics = new PhysicsSystem();
  physics.gravity = 0;
  systems.push(physics);
  await physics.init();
  const world = new CombatWorld(scene, physics);
  const hurt = [{ bone: "hips", toBone: "head", radius: 0.25, zone: "body" }, { bone: "Head", radius: 0.15, zone: "head", damageScale: 1.5 }];
  world.register("def", "enemy", def.rig, hurt);
  world.register("ally", "player", ally.rig, hurt);
  const sword = new Hitbox(world, "att", att.rig, { bone: "Hand.R", toBone: "Weapon_end", radius: 0.06 });
  world.register("att", "player", att.rig, []);
  const step = () => { world.syncHurtboxes(); physics.stepFixed(scene, 1 / 60); };
  return { scene, physics, world, sword, step, att };
}

describe("hitboxes vs Rapier hurtboxes", () => {
  it("a blade reaching the torso hits it once per activation, with zone and direction", async () => {
    const { sword, step } = await arena();
    step();
    sword.begin();
    step();
    const hits = sword.query();
    expect(hits.map((h) => [h.target, h.zone])).toEqual([["def", "body"]]);
    expect(hits[0]!.direction[0]).toBeCloseTo(1, 5);
    expect(sword.query()).toEqual([]); // same activation: no repeat
    sword.end();
    sword.begin();
    step();
    expect(sword.query()).toHaveLength(1); // a new swing can hit again
  });

  it("misses when the blade stops short, and never queries while inactive", async () => {
    const { sword, step } = await arena([-1.2, 0, 0], 0.6);
    step();
    expect(sword.query()).toEqual([]);
    sword.begin();
    step();
    expect(sword.query()).toEqual([]);
  });

  it("sweeps between steps so a fast blade cannot tunnel through a body", async () => {
    const { sword, step, scene, att } = await arena([-0.6, 0, -3], 0.9);
    step();
    sword.begin(); // previous pose: 3 m in front of the target
    scene.setLocalTransform(att.rootId, [-0.6, 0, 3], [0, 0, 0, 1], [1, 1, 1]); // 6 m in one step
    step();
    expect(sword.query().map((h) => h.target)).toEqual(["def"]);
  });

  it("skips teammates and disabled hurtboxes", async () => {
    const { sword, step, world, scene, att } = await arena();
    world.setEnabled("def", false);
    step();
    sword.begin();
    step();
    expect(sword.query()).toEqual([]);
    // Ally ("player" team like the attacker) right in the blade's path.
    world.setEnabled("def", true);
    scene.setLocalTransform(att.rootId, [-1.2, 0, 4], [0, 0, 0, 1], [1, 1, 1]);
    sword.end();
    sword.begin();
    step();
    expect(sword.query()).toEqual([]);
    sword.friendlyFire = true;
    sword.end();
    sword.begin();
    step();
    expect(sword.query().map((h) => h.target)).toEqual(["ally"]);
  });

  it("hurtbox sensors follow their bones", async () => {
    const { world, step, scene, physics } = await arena();
    scene.setLocalTransform("def_root" as EntityId, [5, 0, 5], [0, 0, 0, 1], [1, 1, 1]);
    step();
    const found = physics.overlapShape({ kind: "sphere", radius: 0.05 }, [5, 1.6, 5], undefined, { filter: (id) => world.hurtbox(id)?.owner === "def" });
    expect(found.length).toBeGreaterThan(0);
    expect(found.map((id) => world.hurtbox(id)!.zone).sort()).toEqual(["body", "head"]);
  });
});

describe("health, poise, knockback", () => {
  it("drains HP, staggers when poise breaks, refills poise, and dies once", () => {
    const h = createHealth(100, { poise: 30, poiseRegen: 10, poiseDelay: 1 });
    let r = applyDamage(h, 20);
    expect([h.hp, h.poise, r.staggered]).toEqual([80, 10, false]);
    r = applyDamage(h, 15);
    expect(r.staggered).toBe(true);
    expect(h.poise).toBe(30); // refilled on break
    applyDamage(h, 5, { poiseDamage: 25 });
    expect(h.poise).toBe(5);
    tickHealth(h, 0.5);
    expect(h.poise).toBe(5); // still inside the regen delay
    tickHealth(h, 1);
    expect(h.poise).toBeCloseTo(10, 5); // 0.5 s of regen at 10/s
    r = applyDamage(h, 500);
    expect([r.applied, r.killed, r.staggered, h.dead, h.hp]).toEqual([60, true, false, true, 0]);
    expect(applyDamage(h, 10).ignored).toBe(true);
  });

  it("invulnerability frames ignore hits until they run out", () => {
    const h = createHealth(50);
    applyDamage(h, 10, { iFrames: 0.3 });
    expect(applyDamage(h, 10).ignored).toBe(true);
    tickHealth(h, 0.31);
    expect(applyDamage(h, 10).applied).toBe(10);
  });

  it("knockback follows the hit direction", () => {
    expect(knockbackVelocity([0.6, 0, 0.8], 5, 1)).toEqual([3, 1, 4]);
  });
});

describe("hit-stop and camera shake", () => {
  it("freezes participants for the duration, keeping the longest request", () => {
    const hs = new HitStop();
    hs.add(["a", "b"], 0.08);
    hs.add(["a"], 0.05);
    expect([hs.scale("a"), hs.scale("b"), hs.scale("c")]).toEqual([0, 0, 1]);
    hs.step(0.06);
    expect(hs.frozen("a")).toBe(true);
    hs.step(0.03);
    expect([hs.scale("a"), hs.scale("b")]).toEqual([1, 1]);
  });

  it("shake scales with trauma squared, decays to zero and is deterministic per seed", () => {
    const a = new CameraShake(7), b = new CameraShake(7);
    a.add(0.5); b.add(0.5);
    const oa = a.update(1 / 60), ob = b.update(1 / 60);
    expect(oa).toEqual(ob);
    const mag = Math.hypot(...oa);
    expect(mag).toBeLessThanOrEqual(a.maxOffset * 0.5 * 0.5 * 1.5 + 1e-9);
    for (let i = 0; i < 60; i++) a.update(1 / 60);
    expect(a.trauma).toBe(0);
    expect(a.update(1 / 60)).toEqual([0, 0, 0]);
    const big = new CameraShake(3), small = new CameraShake(3);
    big.add(1); small.add(0.3);
    let sb = 0, ss = 0;
    for (let i = 0; i < 10; i++) { sb += Math.hypot(...big.update(1 / 60)); ss += Math.hypot(...small.update(1 / 60)); }
    expect(sb).toBeGreaterThan(ss * 4);
  });
});

describe("lock-on", () => {
  const origin: Vec3 = [0, 0, 0];
  const forward: Vec3 = [0, 0, 1];
  const cands = [
    { id: "front-far", position: [0, 0, 12] as Vec3 },
    { id: "front-near-right", position: [2, 0, 4] as Vec3 },
    { id: "left", position: [-4, 0, 3] as Vec3 },
    { id: "behind", position: [0, 0, -3] as Vec3 },
    { id: "too-far", position: [0, 0, 40] as Vec3 },
  ];

  it("signed angle is positive to the right of forward", () => {
    expect(signedAngle(origin, forward, [1, 0, 1])).toBeCloseTo(-Math.PI / 4, 6);
    expect(signedAngle(origin, forward, [-1, 0, 1])).toBeCloseTo(Math.PI / 4, 6);
  });

  it("picks the best-scored target in the cone and ignores what is behind or out of range", () => {
    expect(pickLockTarget(origin, forward, cands)).toBe("front-near-right");
    expect(pickLockTarget(origin, forward, cands, { visible: (c) => c.id !== "front-near-right" })).toBe("front-far");
    expect(pickLockTarget(origin, [0, 0, -1], cands.filter((c) => c.id !== "behind"))).toBeNull();
  });

  it("cycles left/right through targets in range, wrapping", () => {
    const order = ["front-near-right", "front-far", "left", "behind"];
    // Sorted by signed angle: right (negative) → left (positive), behind wraps at ±π.
    const right = cycleLockTarget(origin, forward, cands, "front-far", -1);
    expect(order).toContain(right);
    expect(right).toBe("front-near-right");
    expect(cycleLockTarget(origin, forward, cands, "front-far", 1)).toBe("left");
    expect(cycleLockTarget(origin, forward, cands, null, 1)).toBe("front-near-right");
  });

  it("breaks the lock beyond the break distance", () => {
    expect(shouldBreakLock(origin, [0, 0, 19])).toBe(false);
    expect(shouldBreakLock(origin, [0, 0, 21])).toBe(true);
    expect(shouldBreakLock(origin, null)).toBe(true);
  });
});

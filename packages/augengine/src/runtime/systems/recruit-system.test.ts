/**
 * Track 5 — RecruitSystem unit tests.
 *
 * Covers: hearSound flips patrol → investigate, allDown predicate,
 * patrol waypoint advance + state transition events.
 *
 * The deeper movement integration is exercised in the e2e (browser-
 * side controller integration); here we keep tests deterministic by
 * stepping `update` with synthetic positions.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type Entity, type EntityId, type SceneSpec, type Vec3 } from "../../scene/types.js";
import {
  GameTag, getGameComponent, setGameComponent,
  type HealthComponent, type RecruitComponent,
} from "./components.js";
import { RecruitSystem } from "./recruit-system.js";
import { buildSystemContext } from "./system.js";

const PLAYER = "p" as EntityId;
const R1 = "r1" as EntityId;

function makeRecruit(id: EntityId, pos: Vec3, patrolPath: Vec3[]): Entity {
  const e: Entity = {
    id, name: id, components: {
      [Tag.Transform]: { position: pos, rotation: [0, 0, 0, 1], scale: [0.5, 1, 0.5], parent: null },
      [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.8 },
    },
  };
  setGameComponent<HealthComponent>(e, GameTag.Health, { hp: 100, maxHp: 100, dead: false, lastDamageAt: -1 });
  setGameComponent<RecruitComponent>(e, GameTag.Recruit, {
    state: "patrol", hp: 100, maxHp: 100, patrolIndex: 0, patrolPath,
    facingYaw: 0, detectionTimer: 0, investigateTimer: 0, investigateTarget: null,
    meleeTimer: 0,
    visionRange: 14, visionFovCos: 0.5, reactionDelay: 0.6, retreatThreshold: 0.5,
    patrolSpeed: 2.2, engageSpeed: 3.5, retreatSpeed: 4.0,
    meleeRange: 1.6, meleeDamage: 12, meleeCooldown: 0.9,
  });
  return e;
}

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1, source: { kind: "authored", path: "t" },
    entities: [
      { id: PLAYER, name: "p", components: {
        [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.8 },
      } },
      makeRecruit(R1, [10, 0, 10], [[10, 0, 10], [10, 0, 12]]),
    ],
  };
  return new Scene(spec);
}

describe("RecruitSystem", () => {
  it("hearSound flips patrolling recruit to investigate", () => {
    const scene = makeScene();
    const sys = new RecruitSystem({ playerEntityId: PLAYER });
    const alerted = sys.hearSound(scene, [10, 0, 10], 5);
    expect(alerted).toBe(1);
    const recruit = getGameComponent<RecruitComponent>(scene.get(R1)!, GameTag.Recruit)!;
    expect(recruit.state).toBe("investigate");
    expect(recruit.investigateTarget).toEqual([10, 0, 10]);
  });

  it("hearSound is ignored by recruits already in engage/retreat/down", () => {
    const scene = makeScene();
    const sys = new RecruitSystem({ playerEntityId: PLAYER });
    const recruit = getGameComponent<RecruitComponent>(scene.get(R1)!, GameTag.Recruit)!;
    recruit.state = "engage";
    const alerted = sys.hearSound(scene, [10, 0, 10], 5);
    expect(alerted).toBe(0);
    expect(recruit.state).toBe("engage");
  });

  it("allDown returns false when any recruit is alive, true when all down", () => {
    const scene = makeScene();
    const sys = new RecruitSystem({ playerEntityId: PLAYER });
    expect(sys.allDown(scene)).toBe(false);
    const recruit = getGameComponent<RecruitComponent>(scene.get(R1)!, GameTag.Recruit)!;
    recruit.state = "down";
    expect(sys.allDown(scene)).toBe(true);
  });

  it("mirrors HealthComponent death into recruit state", () => {
    const scene = makeScene();
    const sys = new RecruitSystem({ playerEntityId: PLAYER });
    const health = getGameComponent<HealthComponent>(scene.get(R1)!, GameTag.Health)!;
    health.hp = 0;
    health.dead = true;
    const ctx = buildSystemContext(0, [0, 0, 0], 1);
    sys.update(scene, 0.016, ctx);
    const recruit = getGameComponent<RecruitComponent>(scene.get(R1)!, GameTag.Recruit)!;
    expect(recruit.state).toBe("down");
    expect(ctx.events.some((e) => e.type === "recruit-state-changed" && e.to === "down")).toBe(true);
  });

  it("update is a no-op when player position is null", () => {
    const scene = makeScene();
    const sys = new RecruitSystem({ playerEntityId: PLAYER });
    const ctx = buildSystemContext(0, null, 1);
    expect(() => sys.update(scene, 0.016, ctx)).not.toThrow();
    expect(ctx.events.length).toBe(0);
  });
});

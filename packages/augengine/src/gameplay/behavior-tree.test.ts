import { describe, expect, it } from "vitest";
import type { Vec3 } from "../scene/types.js";
import { action, BehaviorTree, condition, inverter, MeleeBrain, selector, sequence, type MeleeInput } from "./behavior-tree.js";

describe("behaviour tree nodes", () => {
  it("selector takes the first non-failing child; sequence stops at the first non-success", () => {
    const log: string[] = [];
    const bb = { hungry: false, tired: true };
    const tree = new BehaviorTree(selector<typeof bb>("root",
      sequence("eat", condition("hungry", (b) => b.hungry), action("eat", () => { log.push("eat"); return "success"; })),
      sequence("sleep", condition("tired", (b) => b.tired), action("sleep", () => { log.push("sleep"); return "running"; })),
      action("wander", () => { log.push("wander"); return "running"; }),
    ));
    expect(tree.tick(bb, 0.1)).toBe("running");
    expect(tree.trace).toEqual(["sleep"]);
    bb.hungry = true;
    expect(tree.tick(bb, 0.1)).toBe("success");
    expect(tree.trace).toEqual(["eat"]);
    bb.hungry = false; bb.tired = false;
    tree.tick(bb, 0.1);
    expect(log).toEqual(["sleep", "eat", "wander"]);
  });

  it("inverter flips success and failure, passes running through", () => {
    const t = (s: "success" | "failure" | "running") => new BehaviorTree(inverter(action<object>("x", () => s))).tick({}, 0);
    expect([t("success"), t("failure"), t("running")]).toEqual(["failure", "success", "running"]);
  });
});

describe("melee brain", () => {
  const input = (self: Vec3, target: Vec3 | null, extra: Partial<MeleeInput> = {}): MeleeInput => ({
    self, target, targetAlive: true, staggered: false, dead: false, attacking: false, ...extra,
  });

  it("idles out of aggro range, chases inside it, and keeps chasing until the lose range", () => {
    const b = new MeleeBrain({ aggroRange: 10, loseRange: 15 });
    expect(b.tick(input([0, 0, 0], [12, 0, 0]), 0.1).state).toBe("idle");
    const c = b.tick(input([0, 0, 0], [8, 0, 0]), 0.1);
    expect(c.state).toBe("chase");
    expect(c.move[0]).toBeCloseTo(1, 5);
    expect(c.run).toBe(true);
    expect(b.tick(input([0, 0, 0], [13, 0, 0]), 0.1).state).toBe("chase"); // hysteresis
    expect(b.tick(input([0, 0, 0], [16, 0, 0]), 0.1).state).toBe("idle");
  });

  it("follows pathfinder waypoints and re-plans on its interval", () => {
    const calls: Array<[Vec3, Vec3]> = [];
    const b = new MeleeBrain({ repathInterval: 0.5, pathfind: (from, to) => { calls.push([from, to]); return [[0, 0, 5], [5, 0, 5]]; } });
    const c = b.tick(input([0, 0, 0], [6, 0, 6]), 0.1);
    expect(c.move[2]).toBeCloseTo(1, 5); // heads to the first waypoint (0,0,5), not straight at the target
    b.tick(input([0, 0, 1], [6, 0, 6]), 0.1);
    expect(b.pathCalls).toBe(1);
    // Waypoint 1 reached → steer to waypoint 2.
    const c2 = b.tick(input([0, 0, 4.8], [6, 0, 6]), 0.1);
    expect(c2.move[0]).toBeGreaterThan(0.9);
    for (let i = 0; i < 4; i++) b.tick(input([1, 0, 5], [6, 0, 6]), 0.1);
    expect(b.pathCalls).toBe(2); // 0.5 s interval elapsed
    b.tick(input([1, 0, 5], [6, 0, 9]), 0.01); // target jumped 3 m → immediate re-plan
    expect(b.pathCalls).toBe(3);
  });

  it("winds up, attacks on cooldown, and waits facing the target in between", () => {
    const b = new MeleeBrain({ attackRange: 1.7, attackCooldown: 1, windup: 0.3 });
    const states: string[] = [];
    let attacks = 0;
    for (let i = 0; i < 30; i++) {
      const c = b.tick(input([0, 0, 0], [1.2, 0, 0]), 0.1);
      states.push(c.state);
      if (c.attack) attacks++;
      expect(c.move).toEqual([0, 0, 0]);
      expect(c.face).toEqual([1.2, 0, 0]);
    }
    expect(states.slice(0, 3)).toEqual(["recover", "recover", "attack"]); // 0.3 s windup
    expect(attacks).toBe(3); // at 0.3 s, 1.3 s, 2.3 s
  });

  it("stagger and death pre-empt everything; a dead target is ignored", () => {
    const b = new MeleeBrain();
    expect(b.tick(input([0, 0, 0], [1, 0, 0], { staggered: true }), 0.1).state).toBe("stagger");
    expect(b.tick(input([0, 0, 0], [1, 0, 0], { dead: true, staggered: true }), 0.1).state).toBe("dead");
    expect(b.tick(input([0, 0, 0], [1, 0, 0], { targetAlive: false }), 0.1).state).toBe("idle");
    expect(b.tick(input([0, 0, 0], [1, 0, 0], { attacking: true }), 0.1).state).toBe("attack");
  });
});

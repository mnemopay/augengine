/**
 * A tiny behaviour tree and a melee-enemy brain built on it.
 *
 * Nodes are reactive: every tick re-evaluates from the root, so a
 * higher-priority branch (stagger, death) pre-empts a running one without
 * abort plumbing. That is the right trade for small trees; it is not a
 * replacement for an event-driven BT with memory nodes.
 *
 * The melee brain does not import a pathfinder: pass `pathfind(from, to)`
 * (the demo wires `@mnemopay/augengine-nav`'s A*), which keeps the engine
 * free of a nav dependency.
 */

import type { Vec3 } from "../scene/types.js";

export type BTStatus = "success" | "failure" | "running";

export interface BTNode<B> {
  readonly name: string;
  tick(bb: B, dt: number, trace: string[]): BTStatus;
}

/** Leaf that runs `fn`; it may return success / failure / running. */
export function action<B>(name: string, fn: (bb: B, dt: number) => BTStatus): BTNode<B> {
  return { name, tick: (bb, dt, trace) => { const s = fn(bb, dt); if (s !== "failure") trace.push(name); return s; } };
}

/** Leaf that succeeds when `fn` is true. */
export function condition<B>(name: string, fn: (bb: B) => boolean): BTNode<B> {
  return { name, tick: (bb) => (fn(bb) ? "success" : "failure") };
}

/** Children in order until one fails (failure) or runs (running); all succeed → success. */
export function sequence<B>(name: string, ...children: BTNode<B>[]): BTNode<B> {
  return {
    name,
    tick(bb, dt, trace) {
      for (const c of children) {
        const s = c.tick(bb, dt, trace);
        if (s !== "success") return s;
      }
      return "success";
    },
  };
}

/** Children in order until one succeeds or runs; all fail → failure. */
export function selector<B>(name: string, ...children: BTNode<B>[]): BTNode<B> {
  return {
    name,
    tick(bb, dt, trace) {
      for (const c of children) {
        const s = c.tick(bb, dt, trace);
        if (s !== "failure") return s;
      }
      return "failure";
    },
  };
}

export function inverter<B>(child: BTNode<B>): BTNode<B> {
  return {
    name: `not(${child.name})`,
    tick(bb, dt, trace) {
      const s = child.tick(bb, dt, trace);
      return s === "success" ? "failure" : s === "failure" ? "success" : s;
    },
  };
}

export class BehaviorTree<B> {
  /** Names of the leaves that succeeded or ran on the last tick, root first. */
  trace: string[] = [];
  constructor(readonly root: BTNode<B>) {}
  tick(bb: B, dt: number): BTStatus {
    this.trace = [];
    return this.root.tick(bb, dt, this.trace);
  }
}

// ──────────────────────────────────────────────────────────────────────
// Melee enemy brain
// ──────────────────────────────────────────────────────────────────────

export type MeleeState = "idle" | "chase" | "attack" | "recover" | "stagger" | "dead";

export interface MeleeInput {
  /** Feet position. */
  self: Vec3;
  /** Target feet position, or null when there is none. */
  target: Vec3 | null;
  targetAlive: boolean;
  /** True while the body is in a stagger/hit reaction. */
  staggered: boolean;
  dead: boolean;
  /** True while the body is still playing its attack (the brain waits). */
  attacking: boolean;
}

export interface MeleeCommand {
  /** World XZ direction, magnitude 0..1. */
  move: Vec3;
  run: boolean;
  /** Start an attack this tick. */
  attack: boolean;
  /** Point to face, or null to face travel. */
  face: Vec3 | null;
  state: MeleeState;
}

export interface MeleeBrainOptions {
  /** Start chasing inside this range. Default 14 m. */
  aggroRange?: number;
  /** Give up beyond this range. Default 22 m. */
  loseRange?: number;
  /** Attack inside this range. Default 1.7 m. */
  attackRange?: number;
  /** Seconds between attacks. Default 1.6. */
  attackCooldown?: number;
  /** Re-plan the path this often (s), or sooner when the target moves 1.5 m. Default 0.6. */
  repathInterval?: number;
  /** Waypoint reached radius. Default 0.6 m. */
  waypointRadius?: number;
  /** Path from → to (feet points). Null = no path; the brain then steers straight. */
  pathfind?: (from: Vec3, to: Vec3) => Vec3[] | null;
  /** Delay before the first attack after closing in, so attacks are readable. Default 0.35 s. */
  windup?: number;
}

interface MeleeBB {
  in: MeleeInput;
  out: MeleeCommand;
  dist: number;
  chasing: boolean;
  cooldown: number;
  inRangeFor: number;
  path: Vec3[];
  pathIndex: number;
  repathIn: number;
  pathGoal: Vec3 | null;
  pathCalls: number;
}

const flatDist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[2] - b[2]);

export class MeleeBrain {
  readonly tree: BehaviorTree<MeleeBB>;
  private readonly bb: MeleeBB;
  private readonly o: Required<Omit<MeleeBrainOptions, "pathfind">> & { pathfind: MeleeBrainOptions["pathfind"] | undefined };

  constructor(opts: MeleeBrainOptions = {}) {
    this.o = {
      aggroRange: opts.aggroRange ?? 14,
      loseRange: opts.loseRange ?? 22,
      attackRange: opts.attackRange ?? 1.7,
      attackCooldown: opts.attackCooldown ?? 1.6,
      repathInterval: opts.repathInterval ?? 0.6,
      waypointRadius: opts.waypointRadius ?? 0.6,
      windup: opts.windup ?? 0.35,
      pathfind: opts.pathfind,
    };
    this.bb = {
      in: { self: [0, 0, 0], target: null, targetAlive: false, staggered: false, dead: false, attacking: false },
      out: { move: [0, 0, 0], run: false, attack: false, face: null, state: "idle" },
      dist: Infinity, chasing: false, cooldown: 0, inRangeFor: 0,
      path: [], pathIndex: 0, repathIn: 0, pathGoal: null, pathCalls: 0,
    };
    const o = this.o;
    const stop = (bb: MeleeBB, state: MeleeState) => { bb.out.move = [0, 0, 0]; bb.out.state = state; };
    this.tree = new BehaviorTree<MeleeBB>(selector("root",
      sequence("dead", condition("isDead", (bb) => bb.in.dead), action("die", (bb) => { stop(bb, "dead"); bb.chasing = false; return "running"; })),
      sequence("stagger", condition("isStaggered", (bb) => bb.in.staggered), action("reel", (bb) => { stop(bb, "stagger"); bb.inRangeFor = 0; return "running"; })),
      sequence("busy", condition("isAttacking", (bb) => bb.in.attacking), action("followThrough", (bb) => { stop(bb, "attack"); bb.out.face = bb.in.target; return "running"; })),
      sequence("fight",
        condition("hasTarget", (bb) => !!bb.in.target && bb.in.targetAlive),
        condition("inAttackRange", (bb) => bb.dist <= o.attackRange),
        selector("attackOrWait",
          sequence("strike",
            condition("ready", (bb) => bb.cooldown <= 0 && bb.inRangeFor >= o.windup),
            action("attack", (bb) => { stop(bb, "attack"); bb.out.attack = true; bb.out.face = bb.in.target; bb.cooldown = o.attackCooldown; return "success"; })),
          action("recover", (bb) => { stop(bb, "recover"); bb.out.face = bb.in.target; return "running"; }))),
      sequence("chase",
        condition("hasTarget", (bb) => !!bb.in.target && bb.in.targetAlive),
        condition("inAggro", (bb) => bb.dist <= (bb.chasing ? o.loseRange : o.aggroRange)),
        action("follow", (bb, dt) => this.follow(bb, dt))),
      action("idle", (bb) => { stop(bb, "idle"); bb.chasing = false; bb.path = []; return "running"; }),
    ));
  }

  /** Number of pathfinder calls so far (for tests and perf HUDs). */
  get pathCalls(): number {
    return this.bb.pathCalls;
  }

  /** Current path (feet points), for debug drawing. */
  get path(): readonly Vec3[] {
    return this.bb.path;
  }

  tick(input: MeleeInput, dt: number): MeleeCommand {
    const bb = this.bb;
    bb.in = input;
    bb.out = { move: [0, 0, 0], run: false, attack: false, face: null, state: bb.out.state };
    bb.dist = input.target ? flatDist(input.self, input.target) : Infinity;
    bb.cooldown = Math.max(0, bb.cooldown - dt);
    bb.inRangeFor = bb.dist <= this.o.attackRange ? bb.inRangeFor + dt : 0;
    this.tree.tick(bb, dt);
    return bb.out;
  }

  private follow(bb: MeleeBB, dt: number): BTStatus {
    const target = bb.in.target!;
    bb.chasing = true;
    bb.repathIn -= dt;
    const moved = bb.pathGoal ? flatDist(bb.pathGoal, target) : Infinity;
    if (this.o.pathfind && (bb.repathIn <= 0 || moved > 1.5 || bb.pathIndex >= bb.path.length)) {
      bb.pathCalls++;
      bb.path = this.o.pathfind(bb.in.self, target) ?? [];
      bb.pathIndex = 0;
      bb.pathGoal = target;
      bb.repathIn = this.o.repathInterval;
    }
    // Skip waypoints already reached.
    while (bb.pathIndex < bb.path.length && flatDist(bb.in.self, bb.path[bb.pathIndex]!) < this.o.waypointRadius) bb.pathIndex++;
    const goal = bb.pathIndex < bb.path.length ? bb.path[bb.pathIndex]! : target;
    const dx = goal[0] - bb.in.self[0], dz = goal[2] - bb.in.self[2];
    const l = Math.hypot(dx, dz);
    bb.out.move = l > 1e-4 ? [dx / l, 0, dz / l] : [0, 0, 0];
    bb.out.run = bb.dist > this.o.attackRange * 2.5;
    bb.out.state = "chase";
    return "running";
  }
}

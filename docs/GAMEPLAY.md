# Gameplay kit (alpha)

`@mnemopay/augengine/gameplay` is a small set of runtime building blocks for
action games: an animation graph, combat primitives, behaviour trees, prefabs
and trigger volumes, and procedural terrain. It is **alpha**: the APIs are
covered by unit tests and one playable vertical slice, nothing more. The
section at the end lists what is missing compared with Unity, Godot and Bevy.

The demo is `examples/arena` in the development monorepo (it also uses the
private `augengine-nav` and `augengine-net` packages, so it is not in the
public mirror). Run it with:

```bash
npm ci && npm run build        # repo root
cd examples/arena && npx vite --port 5195
# http://localhost:5195          play (WASD, Shift walk, Space jump, click attack)
# http://localhost:5195/?auto=1  autopilot
# http://localhost:5195/?bench=1 autopilot + frame-time capture (window.__arenaBench)
# http://localhost:5195/?inspect=attack   hold one animation state, close side camera
```

Measured numbers below come from an Intel UHD laptop in headed Chrome.

## 1. Animation graph

glTF clips in, a runtime state machine out. Pure data and math, no GPU or DOM,
deterministic for a given sequence of `update(dt)` calls.

```ts
import { loadGlb, buildGltfSceneInstance } from "@mnemopay/augengine/scene";
import { AnimGraph, createAnimRig } from "@mnemopay/augengine/gameplay";

const loaded = loadGlb(bytes);
const instance = buildGltfSceneInstance(loaded, { idPrefix: "hero", url });
const graph = new AnimGraph(createAnimRig(loaded.parsed.root.nodes ?? []), loaded.animations, {
  params: { speed: 0 },
  states: [
    { name: "idle", motion: { clip: "Idle_Sword" } },
    { name: "move", motion: { blend1d: { param: "speed", children: [{ clip: "Walk", at: 1.56 }, { clip: "Run", at: 3.19 }] }, speedParam: "rate" } },
    { name: "attack", motion: { clip: "Sword_Slash", loop: false } },
    { name: "hit", motion: { clip: "HitRecieve", loop: false } },
    { name: "death", motion: { clip: "Death", loop: false } },
  ],
  transitions: [
    { from: "*", to: "death", when: [{ flag: "dead" }], duration: 0.12 },
    { from: "*", to: "hit", when: [{ trigger: "hit" }], allowSelf: true, except: ["death"], duration: 0.06 },
    { from: "*", to: "attack", when: [{ trigger: "attack" }], except: ["death", "hit", "attack"], duration: 0.08 },
    { from: "idle", to: "move", when: [{ param: "speed", op: ">", value: 0.2 }], duration: 0.18 },
    { from: "move", to: "idle", when: [{ param: "speed", op: "<=", value: 0.2 }], duration: 0.22 },
    { from: "attack", to: "idle", exitTime: 0.93, duration: 0.16 },
    { from: "hit", to: "idle", exitTime: 0.95, duration: 0.18 },
  ],
  layers: [{ name: "flinch", mode: "additive", mask: ["spine"] }],
  events: { Sword_Slash: [{ time: 0.19, name: "swingPeak" }], Run: [{ time: 0.4, name: "footL" }, { time: 0.79, name: "footR" }] },
});

// every fixed step
graph.setParam("speed", horizontalSpeed);
for (const e of graph.update(dt)) if (e.name === "swingPeak") playWhoosh();
graph.applyToScene(scene, instance.nodeToEntity); // joint local TRS → compute skinning
```

What it does:

- **Rig** (`createAnimRig`): one slot per glTF node with its rest TRS; finds the
  hips by humanoid name for root motion. Works on any skinned file whose clips
  target its nodes: Quaternius, Mixamo, VRoid, and the Studio auto-rig after
  `retargetClip` (tested in `examples/arena/tests/graph-spec.test.ts`).
- **States** play a clip (loop or one-shot, speed, a held start time) or a
  **1D blend tree** with phase sync: all children share one normalised time, so
  walk and run keep their foot timing while blending. `speedParam` scales
  playback (the demo speeds the run cycle up past its authored pace instead of
  letting the feet slide).
- **Transitions** are data: parameter comparisons, flags, triggers (consumed when
  used), `exitTime` (normalised), any-state (`from: "*"`) with `allowSelf` and
  `except`. Any-state transitions are checked first. One transition per update.
- **Cross-fades** are smoothstep-weighted. Interrupting a cross-fade freezes the
  blended pose on screen and fades from that, so there is no pop.
- **Events** fire on clip time, once per crossing, including several loop wraps
  in one long step, and at time 0 on a state's first update. Blend states fire
  from the dominant child.
- **Root motion** (`rootMotion: true` per state): the hips' horizontal travel is
  removed from the pose and returned as `graph.rootMotion` in model space,
  scaled through the armature node (Blender's 100x / -90 degree X export is
  handled). Vertical travel stays in the pose.
- **Layers**: `override` (replace masked bones by weight) or `additive`
  (base * w * (clip - first frame)). Masks are bone subtrees, by node name or
  canonical humanoid id. `playLayer` fades in, and one-shots fade out to reach
  zero exactly at their end.
- Rotation blending is normalised lerp with hemisphere correction.

Tests: `packages/augengine/src/gameplay/anim-graph.test.ts` (state machine
transitions and timing, 1D weights, cross-fade weights, nlerp and additive math,
event crossings across wraps, root-motion deltas across wraps, layer fades and
masks) and `examples/arena/tests/graph-spec.test.ts` (the real character: every
clip binds, foot events within 50 ms of measured touchdowns, the sword swing
peaks inside the hit window, the graph driving an auto-rig skeleton through
retargeted clips).

Demo specifics: the bundled character is the CC0 Quaternius robot also used as
the Studio clip library. Its clips are in-place (no hips travel), so the demo
moves the character with the kinematic controller and leaves root motion off.
It has **no jump clip**: "jump" holds one Run frame with both knees raised.
Clip timings in `examples/arena/src/graph-spec.ts` were measured from the clip
data (foot heights, sword tip speed), not eyeballed.

## 2. Combat primitives

```ts
import { PhysicsSystem } from "@mnemopay/augengine/physics";
import { CombatWorld, Hitbox, applyDamage, createHealth, knockbackVelocity, HitStop, CameraShake, pickLockTarget } from "@mnemopay/augengine/gameplay";

const physics = new PhysicsSystem();
await physics.init();
const combat = new CombatWorld(scene, physics);
const rig = { nodeNames: graph.rig.names, nodeToEntity: instance.nodeToEntity };
combat.register("hero", "player", rig, [
  { bone: "Hips", toBone: "Neck", radius: 0.24, zone: "body" },
  { bone: "Head", toBone: "Head_end", radius: 0.19, zone: "head", damageScale: 1.5 },
]);
const sword = new Hitbox(combat, "hero", rig, { bone: "Weapon", toBone: "Weapon_end", radius: 0.09, extend: 0.9 });

// each fixed step, after animation is applied:
for (const e of graph.update(dt)) {
  if (e.name === "swingStart") sword.begin();
  if (e.name === "swingEnd") sword.end();
}
combat.syncHurtboxes();          // bone poses → sensor colliders
physics.stepFixed(scene, dt);    // Rapier picks the poses up
for (const hit of sword.query()) {
  const r = applyDamage(healthOf(hit.target), 24 * hit.damageScale, { iFrames: 0.05 });
  // r.killed / r.staggered → graph flags/triggers; knockbackVelocity(hit.direction, 4)
}
```

- **Hurtboxes** are kinematic **sensor** colliders (new `ColliderComponent.sensor`: no contact
  forces, ignored by the character controller and `raycast`) placed on bones every step: a
  sphere on one bone, or a capsule between two bones. Bones are named by glTF node name or
  canonical humanoid id. Sensor poses are read from one world-transform pass, then written;
  capsule length stays at its registration value (rebuilding Rapier colliders when an elbow
  bends cost 6 ms a step in the first version of the demo).
- **Hitboxes** query `PhysicsSystem.overlapShape` (new: sphere/capsule/box overlap through
  Rapier's `intersectionsWithShape`, sorted, with `exclude`/`filter`). Each query is swept
  from the previous step's pose in sub-steps no longer than the radius, so a fast blade
  cannot tunnel through a body between two steps. One hit per target per activation; same
  team skipped unless `friendlyFire`; disabled owners (dead, dodging) skipped.
- **Health** (`createHealth`, `applyDamage`, `tickHealth`): HP, **poise** (damage drains it;
  breaking it staggers and refills it; regenerates after a delay), invulnerability frames.
  `knockbackVelocity(direction, strength, up)`.
- **Hit-stop** (`HitStop`): per-entity freeze timers; multiply each participant's dt by
  `scale(id)`. **Camera shake** (`CameraShake`): trauma model, offset ∝ trauma², seeded smooth
  noise, linear decay.
- **Lock-on**: `pickLockTarget` (cone + distance score, optional line-of-sight callback),
  `cycleLockTarget` (left/right by signed angle, wrapping), `shouldBreakLock`.
- **Behaviour tree** (`BehaviorTree`, `selector`, `sequence`, `condition`, `action`,
  `inverter`): reactive, re-evaluated from the root every tick, with a trace of the leaves
  that ran. **`MeleeBrain`**: idle → chase (with hysteresis) → wind-up → attack on cooldown
  → recover, pre-empted by stagger and death. Pathing is injected (`pathfind(from, to)`), so
  the engine has no nav dependency; the demo passes `@mnemopay/augengine-nav` A* on a 2D
  occupancy grid (obstacles inflated by the capsule radius, line-of-sight simplified,
  iteration-capped).

Demo: two melee enemies per wave (a tinted copy of the player's model), sword slash driven
by the clip's swing events, stagger on poise break, additive flinch otherwise, hit-stop
(75 ms, 130 ms on a kill), sparks at the contact point, shake when the player is involved,
Tab lock-on with an over-the-shoulder camera, new wave two seconds after the last corpse
despawns. `?debug=1` draws the sword capsule and one enemy's hurtboxes; that capture is how
the sword's `extend: 0.9` was matched to the blade mesh (the Weapon bone itself is 0.14 m).

Tests: `combat.test.ts` (blade vs Rapier sensors: hit once per activation, miss when short,
sweep through a 6 m jump in one step, team/disabled filters, sensors follow bones; health,
poise regen timing, i-frames; hit-stop; shake decay and determinism; lock-on picking and
cycling), `behavior-tree.test.ts` (node semantics, chase hysteresis, waypoint following and
re-planning, wind-up and cooldown timing, pre-emption), `physics/overlap.test.ts` (overlap
queries, kinematic sensor tracking with no one-step lag, sensors not blocking the
controller or raycasts) and, on the real rig, `examples/arena/tests/combat.test.ts` (a slash
at 1.3 m lands inside the swing window and pushes away from the attacker; at 2.6 m it
misses; an enemy paths around a wall with augengine-nav and attacks).

Engine fix found while building this: compute skinning wrote one output buffer per mesh, so
two characters from the same .glb drew as one. Each skinned entity now has its own buffer
(e2e "two instances of one skinned mesh", which fails on the old renderer).

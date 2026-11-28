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

## 3. World tooling (runtime)

```ts
import { PrefabLibrary, TriggerSystem, Spawner, CheckpointTracker, generateTerrain, scatter, grassTuftMesh } from "@mnemopay/augengine/gameplay";

const lib = new PrefabLibrary();
for (const p of levelJson.prefabs) lib.register(p);          // validated; throws PrefabError listing every problem
const layout = lib.instantiate(scene, "arena", { position: [0, 0, 0], yaw: 0 }, "level");
const triggers = new TriggerSystem();
layout.triggers.forEach((t) => triggers.add(t));
const spawners = layout.spawners.map((s) => new Spawner(s));
const checkpoints = new CheckpointTracker(layout.checkpoints, startPoint);

const terrain = generateTerrain({ seed: 4, size: 90, cells: 72, amplitude: 3.6, flatRadius: 12, rim: 6 });
scene.add({ id: "terrain", name: "terrain", components: { transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
  mesh: { geometry: { kind: "procedural", seed: 4, generator: "terrain" }, material: { kind: "pbr", baseColor: [1, 1, 1], metallic: 0, roughness: 0.95 }, visible: true },
  collider: terrain.collider } });                             // Rapier heightfield; the controller and raycast read it too
renderer.uploadMesh("proc:terrain:4", terrain.mesh(), 8);
renderer.uploadMesh("proc:grass:0", grassTuftMesh(11), 8);
for (const g of scatter(terrain, { seed: 31, count: 700, minSpacing: 0.9, minNormalY: 0.82, alignToNormal: 0.5 }))
  scene.add({ /* mesh: { geometry: { kind: "procedural", seed: 0, generator: "grass" }, … } at g.position/rotation/scale */ });
```

- **Prefabs** (`PrefabLibrary`, `parsePrefab`): an entity tree with local ids, transforms
  (`position`, `rotation` or `yaw` in degrees, `scale`), scene components copied as-is,
  nested prefabs (cycles rejected with the chain in the message), per-instance `overrides`
  by local path ("entrance/base") that shallow-patch components without touching the
  registered spec. Colliders and rigid bodies are flattened to world-space roots on
  instantiation (physics requires roots); visual children keep their hierarchy. Validation
  reports every problem with its JSON path (unknown component, non-unit quaternion,
  duplicate id, unknown parent, parent cycle, bad trigger/spawner fields).
- **Gameplay entries** ride on prefab entities under `gameplay` and come back placed in
  world space: `trigger` (box/sphere, `event`, `once`), `spawner` (`kind`, `points`,
  `interval`, `maxAlive`, `count`, `activateOn`, `seed`), `checkpoint` (`order`, `radius`).
- **`TriggerSystem`**: enter/exit events per subject; `once` triggers fire one enter and go
  quiet. **`Spawner`**: spawns on its interval, never above `maxAlive` alive or `count` in
  total, points from a seeded shuffle bag, at most one interval of credit while capped (no
  burst when a slot frees). **`CheckpointTracker`**: only forward progress counts;
  `respawnPoint()`. All pure and deterministic.
- **`generateTerrain`**: seeded fBm heights with an optional flat disc and raised rim; the
  collider, the render mesh (vertex-coloured grass/dirt/rock by slope and height) and
  `heightAt`/`normalAt` all use Rapier's cell split, so they agree (tested against Rapier
  raycasts to < 2 mm). **`scatter`**: seeded dart throwing with a spatial hash; min spacing,
  slope and height filters, avoid zones, scale range, tilt-to-normal, sink.
  **`grassTuftMesh`/`rockMesh`**: small procedural meshes. Entities that share a mesh key are
  instanced by the renderer automatically (groups of 4+): in the demo, 750 scatter entities
  draw in 6 instanced draws.

Demo: 90 m terrain (72x72 cells) with a flat 12 m combat disc and a raised rim, 46 rocks (21
large ones get sphere colliders and block nav), 700 grass tufts, four standing stones from a
nested prefab, a sphere trigger at the centre that activates two spawners (one enemy alive
each), two checkpoints (respawn after death). The nav grid marks stones, big rocks and
terrain steeper than normal.y 0.72. All of it is in `examples/arena/src/level.json` and
`level.ts`.

Engine changes found while building this (each with tests):

- **Scene transform cache**: moving a childless root (sensors, sparks, cameras, props) now
  patches its world matrix in place instead of dirtying the whole scene, and moving joints
  refreshes matrices along the cached parent-first order instead of re-sorting every entity
  (a re-parent behind the cache's back is detected and falls back to a full pass). In the
  arena this took the hurtbox sync from 6.4 ms to 0.5–0.8 ms per step and hit queries from
  0.9 to 0.1 ms.
- **PhysicsSystem** validated every heightfield sample on every fixed step; it now checks
  each heights array once.
- **augengine-nav A\***: binary-heap open set with numeric keys, allocation-free neighbour
  loop (was a linear scan of a Map plus a string key per neighbour). Same API, optimal
  paths (tested against Dijkstra). Measured on Node 25 (warm): a 12 m detour round a
  stone 150 → 7 ms, a blocked goal at a 4 000-node cap 234 → 57 ms, a 180x180 grid detour
  900 → 480 ms. The arena also snaps start/goal to the nearest free cell, because a player
  standing against a stone put the goal inside the stone's margin and every re-plan ran to
  the cap: per-call cost went from 42 ms mean / 244 ms max to about 1.5 ms mean / 17 ms max.

## 4. Multiplayer proof (two clients, authoritative server)

Not part of `./gameplay`: the transport lives in the private `augengine-net` package and the
server in the demo. What exists:

- **`augengine-net` WebSocket transport** (`createWsServer`, `WsNetServerTransport`,
  `WsNetClientTransport`): the engine's typed protocol (`join` / `input` / `ping` →
  `welcome` / `snapshot` / `pong`) over a minimal RFC 6455 server on `node:http` and
  `node:crypto` (no dependency: fragmentation, ping/pong, close, 16/64-bit lengths, mask
  enforcement, size cap) and the global `WebSocket` on the client (browsers, Node 22+).
  Optional injected latency/jitter per direction, order preserved. WebSocket rather than
  the existing WebTransport adapter because WebTransport on the server needs Node's
  experimental QUIC and TLS certificates even on localhost.
- **`ClockSync`** (server-time estimate from the lowest-RTT pings) and **`SnapshotBuffer`**
  (linear position, short-arc yaw, discrete fields from the earlier snapshot, capped
  extrapolation, underrun/extrapolation counters).
- **Arena server** (`examples/arena/server/arena-server.ts`, `npx tsx server/arena-server.ts
  [--port 5296] [--lag ms] [--jitter ms]`): runs the same `ArenaSim` as single-player in Node
  (terrain, Rapier, players, enemies, AI, combat, spawners) at 60 Hz and broadcasts a
  snapshot (one compact state per fighter + hit events) at 20 Hz. Inputs are validated and
  clamped; an input carries one step's intent; a client that stops sending is held for at
  most 100 ms and then stops. `GET /trace` returns the authoritative trajectory, `GET /stats`
  tick timing and traffic.
- **Client** (`?net=ws://127.0.0.1:5296`): the same level in mirror mode. The local player
  is **predicted** through the same controller and reconciled against the acked input
  (15 % per step, snap above 1.5 m; corrections shift every not-yet-acked prediction so they
  are not applied twice). Everyone else is **interpolated** at server time minus an
  RTT-adaptive delay (one-way latency + 1.5 snapshot intervals + jitter: ~80 ms on
  localhost, ~150 ms at 130 ms RTT).

Measured on this laptop (Intel UHD, one Chrome with two 640x360 clients, both on
autopilot, 20 s, server in Node on the same machine; `e2e/net.spec.ts`,
`results/arena/net-lag*/metrics.json`):

| | localhost | +50 ms one-way, 10 ms jitter |
| --- | --- | --- |
| round trip, p50 / p95 | 4–5 / 32 ms | 133–135 / 236–264 ms |
| other player as drawn vs server at the same server time, p95 | 3 cm | 2–4 cm |
| all remote entities (player + enemies), p95 / max | 2 cm / 0.1–0.2 m | 2–3 cm / 0.39 m |
| samples extrapolated (buffer ran dry) | 1–2 % | 5 % |
| local prediction error at ack, mean / p95 / max | 9–12 cm / 0.35–0.55 m / 1.2–1.6 m | 9–10 cm / 0.44 m / 1.3–1.6 m |
| traffic per client, down / up (bytes over a ~22–25 s connection) | ~9–11 / ~10–12 KB/s | ~10–13 / ~12–15 KB/s |

The 1.6 m maximum is the second client's spawn offset before its first snapshot. Most of
the p95 prediction error comes from things the client does not predict: server-side hit
knockback and hit-stop, and pushing against enemy capsules that the client sees ~80 ms in
the past. Headless (`tests/net.test.ts`, two in-process clients over real sockets, one
stalling 1.5 s mid-run) the median is 1–2.5 cm at both latencies.

Limits, honestly:

- **Two players on localhost, not a netcode library.** TCP (head-of-line blocking on loss),
  JSON snapshots of every fighter every 50 ms (~0.5 KB each), no delta compression, no
  interest management, no lag compensation for hits (the server resolves swings in its own
  present, so at 130 ms RTT you hit where the enemy was when the server stepped, not where
  you saw it), no authority handover or reconnection, no anti-cheat beyond input clamping.
  Not measured over a real network or with packet loss.
- Remote animation is approximate: discrete states (attack, hit, death, jump) are
  cross-faded when the snapshot's state name changes; locomotion follows the speed. Clip
  time is not synced.
- The server shares the arena's code, so a Node server needs the character .glb on disk.
  Spawning an enemy costs ~20 ms on the server (and a visible hitch in the browser); the
  first physics step builds the heightfield (~230 ms). Neither is pooled or amortised yet.

## The demo, end to end

`examples/arena`: a third-person character on procedural terrain walks into the arena; the
gate trigger starts two spawners; two melee enemies (behaviour tree + A* on the nav grid)
come at you; sword slashes resolve through bone hitboxes against Rapier hurtboxes, with
stagger, flinch, knockback, hit-stop, sparks and camera shake; lock-on frames the fight;
dying respawns you at the last checkpoint. With `?net=` a second player joins through the
authoritative server and both fight the same enemies.

Tests: engine `src/gameplay/*.test.ts`, `src/physics/overlap.test.ts`,
`src/scene/scene-fastpath.test.ts`; nav `src/astar.test.ts`; net `tests/ws-transport.spec.ts`,
`tests/interpolation.spec.ts`; arena `tests/*.test.ts` (real rig, combat, nav pathing,
headless multiplayer) and `e2e/*.spec.ts` (animation states and events; 20 s of game time
in which enemies take damage and die; terrain/scatter/trigger/spawner/checkpoint; two
instances of one skinned mesh; two browser clients against the server).

Performance (`results/arena/SCORECARD.md`, Intel UHD gen-11, headed Chrome, 1280x720, `?bench=1`, three runs on a quiet machine): **59.9 fps median, 59.8–59.9 mean, frame p95 16.8–16.9 ms**, GPU ~7 ms, 750 scatter entities in 6 instanced draws. Spawning an enemy still causes a 33–50 ms frame. Runs made while other agents' WebGPU test suites were loading the machine dropped to 22–52 fps mean.

## What is missing compared with Unity, Godot and Bevy

Being plain about the size of the gap:

- **Animation**: no 2D blend spaces or IK/look-at nodes in the graph, no sync groups across
  layers, no animation retargeting UI at runtime, no visual graph editor or debugger, no
  animation compression, no ragdoll blending. Unity's Mecanim, Godot's AnimationTree and
  Bevy's animation graph all have editors; this has JSON.
- **Combat**: no block/parry/guard-break, no combo or input-buffer system, no
  animation-driven root-motion attacks in the demo (the bundled clips are in-place), no
  projectile or ranged primitives, behaviour trees are reactive only (no memory nodes,
  blackboard tooling or editor).
- **World**: no prefab editor or prefab variants; triggers test points, not shapes; terrain
  is one heightfield (no holes, caves, splat maps, LOD or streaming — the streamed 1 km level
  in `runtime/world` is a separate, level-specific system); scatter has no wind, GPU culling
  or per-instance LOD.
- **Networking**: a two-player proof over WebSocket, not a netcode framework (see section 4).
- **Shipped titles**: none. This is one vertical slice, played by an autopilot in tests and
  by hand on one laptop.

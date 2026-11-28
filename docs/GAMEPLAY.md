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

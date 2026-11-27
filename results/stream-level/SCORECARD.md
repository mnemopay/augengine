# Gate 2 — "one level, not a room": measured results

Measured 2026-11-23 on the same LOW-END baseline as the proof room: Intel
Core i5-1035G1, 7.8 GB RAM (about 2.2 GB free during the run), Intel UHD
Graphics (Ice Lake gen-11), Chrome via `--project=chromium-webgpu-headed`,
smoke-game dev server (unbundled). Other jobs were running on the laptop.
Raw data: `2026-11-23-walk-capture.json`, `2026-11-23-route-walk.json`.
Stills: `2026-11-23-west-gate.png`, `2026-11-23-signal-spire.png`,
`2026-11-23-east-beacon.png`.

**Gate 1 is not claimed.** 16.7 ms p95 on a shipping GPU still cannot be
measured on this machine (see `../proof-room/SCORECARD.md`).

## What the level is

The Loading Range (`#/stream-level`): 1024 x 1024 m in 16 x 16 tiles of
64 m. A road runs west to east past five landmarks: West Gate, Water Tower,
Signal Spire, Monolith Ring, East Beacon. Every floor, building, pillar,
dome and landmark belongs to a tile and streams with it
(`runtime/world/level-content.ts`, deterministic per tile). The only
permanent entities are the player, lights, four map-edge walls and a far
ground plane with no collider. Content is primitives, not art.

## Results

| Criterion | Measured | Verdict |
|---|---|---|
| Tiles carry real content (meshes, colliders, nav) and unload | Floors, props, colliders and a per-tile nav grid spawn on load and are removed on evict (unit + browser) | PASS |
| Memory / entities bounded on a long walk | ~2 km walk in Node: peak 449 streamed entities of 6,280 in the whole level; peak 18 resident tiles; peak bytes 299,520 vs 299,808 budget; 101 tile evictions. Browser 1 km route: peak 448 entities, 18 tiles | PASS |
| No hitch on load at play speed | 600 frames sprinting (8.5 m/s) on streamed floors: frame p50 16.7 / p95 17.6 / p99 33.5 / max 36.2 ms, 57.7 FPS, 10 of 599 frames over 33 ms, 0 over 50 ms | PASS at play speed; p95 is 0.9 ms over a 60 FPS budget |
| No hitch on load at extreme speed | Player moved 4 m per frame (~240 m/s) down the whole road: p50 16.7 / p95 49.9 / max 67.2 ms | FAIL: tile spawn work lands in one frame; needs time-slicing before fast vehicles |
| LOD: 2–3 mesh levels + impostor, with hysteresis | Pillars 128/48/24 tris, domes ~1,200/~170/~40 tris, box impostor 12 tris; building detail parts drop out after 28 m; hysteresis band 12% (unit-tested, no flicker on a boundary) | PASS |
| LOD + culling reduce work | From the spawn, facing east: 42 draws / 2,532 tris drawn vs 153 draws / 16,148 tris if every resident mesh drew at LOD0. Facing the map edge: 4 draws / 48 tris. Browser walk: 19–58 draws | PASS |
| Player can't fall through unloaded edges | No permanent floor. `guardMove` blocks steps onto non-resident tiles and off the map; 30 s controller walk in Node and the 600-frame browser walk both keep the capsule at y 1.305 | PASS |
| Navigation regenerates per tile | One occupancy grid per resident tile, built on load and dropped on evict; A* paths only cross resident ground and route around colliders | PASS |
| One collider convention | PhysicsSystem scaled box extents by transform scale; the controller and raycast did not; smoke-game nav multiplied again (a 2.5 x 3 x 6 container blocked 6 x 9 x 36 m of nav). Now all read world-space sizes; tested across Rapier, controller and raycast | FIXED |
| A playable ~1 km level with landmarks | Walkable end to end on the road with keyboard controls; five landmarks | PASS (greybox) |

## Follow-up 2026-11-23: time-sliced streaming + renderer instancing

Same laptop and setup (headed Chrome, dev server, Intel UHD, ~1.7–2.1 GB
free RAM, other jobs running). Raw data: `2026-11-23-after-*.json`,
`2026-11-23-after-run2-*.json` (everything on), `2026-11-23-noinst-*.json`
(time-slicing on, `?instancing=0`). Stills:
`2026-11-23-instanced-signal-spire.png`, `2026-11-23-instanced-east-beacon.png`
(visually the same as the pre-instancing stills; tiles ahead now arrive from
further out).

What changed:

- **Time-sliced spawning** (`streamed-level.ts`). A tile becoming resident or
  evicted only queues work; each tick runs steps (floor + nav, one prop, a
  batch of 24 removals) until a 3 ms budget is spent, nearest/ahead tiles
  first, every pending floor before any prop. The floor the camera is on or
  within 24 m of is built immediately whatever the budget. Evicted entities
  are hidden at once and removed a batch per tick. Tiles dead ahead are
  wanted from one extra tile out (`forwardRadiusBonusTiles`, tile-manager).
- **Instancing** (`renderer.ts`, `pipeline-factory.ts`, `instanced-draws.ts`).
  Visible entities sharing a mesh + pipeline variant (rigid, untextured,
  no vertex colours, unlit/pbr/cel) and numbering >= 4 draw with one
  `drawIndexed(n, instances)` from a storage buffer of the same 112-byte draw
  records; smaller groups draw per entity as before. Culling is untouched
  (a hidden entity is simply not in that frame's buffer); the shadow pass
  has an instanced depth pipeline too.

| Capture | Frame p50 / p95 / p99 / max (ms) | > 33 ms | > 50 ms | Draw calls (renderer) |
|---|---|---|---|---|
| Sprint, 600 frames — before (gate-2 PR) | 16.7 / 17.6 / 33.5 / 36.2 | 10 | 0 | one per visible entity |
| Sprint — time-slicing, no instancing | 16.7 / 17.6 / 33.5 / 33.8 | 10 | 0 | 70–87 |
| Sprint — both, run 1 | 16.7 / 17.0 / 19.3 / 50.2 | 4 | 1 | 6–12 |
| Sprint — both, run 2 | 16.7 / 17.1 / 33.3 / 50.1 | 3 | 1 | 5–12 |
| 1 km at ~240 m/s — before | 16.7 / 49.9 / — / 67.2 | — | — | one per entity |
| 1 km fast travel — time-slicing, no instancing | 16.7 / 33.5 / 49.9 / 51.2 | 19 | 2 | 29–104 |
| 1 km fast travel — both, run 1 | 16.7 / 17.1 / 33.3 / 33.5 | 3 | 0 | 5–14 |
| 1 km fast travel — both, run 2 | 16.7 / 33.3 / 33.7 / 49.5 | 8 | 0 | 5–14 |

Read honestly:

- **Fast travel: the goal is met.** p95 is 17.1–33.3 ms (was 49.9), max
  33.5–49.5 ms (was 67.2), and no frame over 50 ms in either run. The page
  runs on a 60 Hz vsync, so frame times sit on the 16.7 / 33.3 ms steps and
  p95 lands on one or the other run to run.
- **Draw calls dropped ~10x** (5–14 instead of 64–104 for the same visible
  entities) with identical images.
- **Machine load moves these numbers a lot.** After rebasing on master, one
  sprint run with everything on came out p95 33.2 ms (20 frames > 33 ms).
  Run back to back right after it, under the same load: instancing off
  p95 33.3 ms (18 > 33 ms), instancing on p95 17.0 ms (4 > 33 ms)
  (`rebased-noinst-walk-capture.json`, `rebased-inst-walk-capture.json`).
  Compare runs taken together, not across sessions.
- **Sprint p95 is unchanged within noise (17.0–17.6 ms)**, still about 0.3–0.9
  ms over a 60 FPS budget, and each "both" run had one ~50 ms frame that the
  capture does not localise (start-up pipeline compile is the suspect, not
  verified).
- **A wrong turn, kept for the record:** the first instanced shader read the
  draw record from the storage buffer in the fragment stage too. On this GPU,
  with floors covering the screen, that cost ~the whole budget: sprint p95
  went to 33.3 ms with 22 frames over 33 ms. Passing colour/params/emissive
  to the fragment stage as flat varyings (one storage load per vertex) fixed
  it.
- The 3 ms budget is a soft cap: a step that starts under budget finishes,
  and forced floors are outside it. Measured spawn work per tick peaked at
  3.5–5.3 ms in the instanced runs.
- Props (and their colliders) are not forced like floors: on a teleport the
  floor is there at once and the props follow within a few ticks. Nav blocks
  prop footprints from the moment the floor exists.

## Art kit with baked LODs (2026-11-23)

The Loading Range now builds from an original procedural kit by default
(`@mnemopay/augengine-lod`; `?kit=0` keeps the primitive level for A/B):
6 buildings (4 facade palettes, lit/dark window grids, ledges, cornice, roof
plant, water tanks), street lamps both sides of the road every 16 m, benches,
jersey barriers, 3 low-poly trees and a kit water tower for that landmark. No
downloaded or third-party assets. Each asset is one vertex-coloured mesh baked
to LOD0-3 with meshoptimizer (1/4, 1/16 with pruning) plus an impostor (bounds
box for solid shapes, crossed cards for trees and lamps). Example chain:
5-floor building 1036 / 258 / 64 / 10 triangles. Stills:
`2026-11-23-kit-west-gate.png`, `-kit-water-tower.png`, `-kit-signal-spire.png`,
`-kit-overview.png` (LOD0 of every asset).

The renderer now instances vertex-coloured meshes too (they were excluded for
no structural reason), so kit props of the same asset + LOD share a draw.

600-frame sprint, headed Chrome, Intel UHD, kit vs primitives run alternately
in one session after a warm-up load, CPU below ~55% at the start of each run
(Windows Search/Defender were indexing; an earlier pair taken at 91% CPU was
discarded: primitives measured 21.6 FPS with a 733 ms stall):

| run | FPS | p50 | p95 | p99 | max | >33 ms | >50 ms | draw calls | max tris | peak entities |
|---|---|---|---|---|---|---|---|---|---|---|
| kit a | 57.1 | 16.7 | 33.0 | 33.5 | 35.4 | 12 | 0 | 31-46 | 4907 | 188 |
| primitives a | 51.8 | 16.7 | 32.9 | 33.6 | 1086.3 | 9 | 1 | 6-12 | 2896 | 347 |
| kit b | 58.5 | 16.7 | 17.1 | 33.4 | 50.6 | 6 | 1 | 30-46 | 4907 | 188 |
| primitives b | 58.5 | 16.7 | 17.2 | 33.4 | 49.7 | 8 | 0 | 7-12 | 2896 | 347 |

(`2026-11-23-{kit,nokit}-{a,b}-walk-capture.json`.) The 1086 ms frame in
primitives a is a background stall, not the scene.

- **Frame time is unchanged within noise** on this GPU: the run-b pair is
  identical (58.5 FPS, p95 17.1 vs 17.2 ms). Both are pinned to the 60 Hz
  display here, so this shows the kit costs nothing measurable on Intel UHD,
  not that it is free on every GPU.
- **Draw calls went up (30-46 vs 6-12)**: 13 assets x 4 LODs means many
  groups below `INSTANCE_MIN_GROUP` (4) draw singly. Entities went down
  (188 vs 347) because a kit building is one mesh, not a box plus detail
  boxes. Triangles roughly doubled at peak (4.9k vs 2.9k).
- 1 km route walk with the kit (`2026-11-23-kit-route-walk.json`): p95 33.5 ms,
  max 35.7 ms, 0 frames over 50 ms, peak 16 resident tiles / 249 entities,
  spawn work peaked at 2.4 ms per tick. All landmarks streamed in and out.
- The first kit run after a dev-server restart was slow (47.7 FPS, 2 frames
  over 50 ms); the warm reruns above were not. Cold start (shader/pipeline
  compile + module load) is the suspect, not verified.

## Terrain, occlusion, NPCs and ambient audio (2026-11-23, evening)

`StreamExtras` (`runtime/world/stream-extras.ts`) is on by default in the
Loading Range; `?extras=0` keeps the flat slabs, `?occlusion=0` turns off just
the occlusion pass.

- **Terrain.** One global height function (4-octave value noise, 22 m peaks),
  flattened to y = 0 on the road and the five landmark plazas. Each tile is a
  33 x 33 sample grid (2 m) that drives the mesh (3 LODs with skirts, no
  cracks), a Rapier heightfield collider and nav slope blockers. Props stand
  on the lowest ground under their footprint.
- **Occlusion.** Conservative CPU software occlusion (160-px depth buffer;
  inner-coverage, far-depth and outer-bounds rules). Occluders: the nearest 48
  solid buildings within 180 m (primitive and kit buildings only, via
  `PropSpec.occluder`: domes, the kit water tower's open legs, trees and
  street props never occlude) and hills within 180 m (lowered LOD2 surface).
- **NPCs + audio.** Persona NPCs spawn with their tile, save their state on
  evict and restore it on reload; ambient emitters fade in/out with an 8-voice
  budget. In the game the audio sink waits for the first click to unlock audio.

### What restoring it took (2026-11-23)

The work was parked when a three.js renderer looked likely; backend A stayed
default (`../renderer-benchmark/SCORECARD.md`), so it was restored on top of
the art kit and the renderer adapter. Merging it surfaced and fixed:

- **Kit levels had no occluders at all**: the old rule only used
  `shape === "box"` props, and kit buildings are `shape: "kit"`. Unit test
  walk: 0 props occluded without the fix, up to 54 with it.
- **A per-tile terrain build cost ~13 ms of CPU** (4.7 ms heights, 8.1 ms for
  three LOD meshes, each vertex re-evaluating the noise 4 times for its
  normal), done inside the floor step, which forced floors run outside the
  3 ms budget. First browser capture after the merge
  (`prefix-terrain-sync-*.json`): 1 km fast travel p95 166 ms, max 316 ms,
  85 frames over 50 ms, spawn work up to 79.6 ms per tick.
  Fix: a tile's terrain (heights, shared normal grid, 3 meshes, slope
  blockers, hill occluder) is prepared while the tile loads, one part per
  idle task; the floor step only places the entity (p50 0.02 ms). Node
  fast-travel profile: spawn work max 3.2 ms per tick (was ~13 ms per floor).
- `releaseMesh` is on the `LiveRenderer` adapter; the three backend also drops
  objects/instance groups built from a released mesh.

### Measured (Intel UHD, headed Chrome, dev server, same session)

`?bench=1` harness, fixed camera path, 600 frames after 180 warm-up
(`../renderer-benchmark/2026-11-23-webgpu-{extras,flat}-stream-level.json`):

| | Extras on | `?extras=0` (flat) |
|---|---|---|
| GPU frame p50 / p95 | **8.33 / 11.25 ms** | 8.68 / 9.29 ms |
| CPU render p50 / p95 | 5.7 / 10.1 ms | 3.8 / 9.7 ms |
| rAF interval p50 / p95 | 18.7 / 31.3 ms | 17.2 / 23.6 ms |
| Triangles (p50) | 24,954 | 2,971 |
| Draw calls (p50) | 47 | 38 |
| Frames > 33 / > 50 ms | 23 / 5 | 11 / 1 |

600-frame sprint and 1 km route, run twice each, alternating
(`ab-{on,occ0,flat}-r{1,2}-*.json`):

| run | Sprint FPS | Sprint > 50 ms | Route p95 / max (ms) | Route > 50 ms | Route spawn work max |
|---|---|---|---|---|---|
| extras on r1 | 52.9 | 1 | 33.8 / 166.6 | 5 | 4.3 ms |
| extras on r2 | 55.7 | 2 | 66.6 / 233.3 | 27 | 5.7 ms |
| occlusion off r1 | 52.9 | 1 | 66.6 / 282.3 | 29 | 13.8 ms |
| occlusion off r2 | 56.0 | 1 | 79.8 / 281.8 | 63 | 9.1 ms |
| flat r1 | 55.9 | 0 | 33.5 / 34.3 | 0 | 3.0 ms |
| flat r2 | 56.4 | 0 | 61.6 / 249.9 | 28 | 9.7 ms |

Read honestly:

- **GPU cost is flat** despite ~8x the triangles: terrain LODs + occlusion
  keep the typical frame at the flat level's cost (p50 8.3 vs 8.7 ms). The p95
  is ~2 ms higher.
- **The cost is CPU**: ~+1.9 ms render CPU at p50, mostly the occlusion pass
  (Node phase profile: 1.8 ms p50, 7 ms max per frame) and more draws (terrain
  tiles do not instance: each tile is its own mesh).
- **Occlusion on vs off is a wash in the browser** on this level (sprint FPS
  52.9/55.7 vs 52.9/56.0): it hides few draws from the road (2 of 59 tested at
  one spot in the e2e) and those draws are cheap after instancing. It stays
  on by default because it is conservative and pays off among buildings, but
  it is the first thing to revisit (lower resolution, fewer occluders, or
  only when the camera is inside the city).
- **Fast travel is not solved with extras on.** Spawn work stays near budget
  (4.3–5.7 ms, was 79.6), but frame spikes of 166–233 ms remain. The flat
  level also spiked to 250 ms in one run, so machine load is part of it; the
  rest is not localised (GPU buffer uploads for new terrain meshes and idle
  tasks forced by their 60 ms timeout are the suspects, not verified).
- Browser e2e (`e2e/stream-level-extras.spec.ts`, all pass): the player
  crossed the x = 128 seam over 5.4 m of ground relief without sinking (min
  clearance 1.305 m); NPC `stream.4.8.npc0` restored where it was saved after
  a forced evict/reload; over a 2 km walk at most 2 ambient voices played at
  once (budget 8), 7 started and 7 released, audio context running.

## Fast-travel spikes found and fixed (2026-11-23, night)

Measured first, then fixed. New harness: `e2e/stream-level-spikes.spec.ts`
drives the 1 km route at 4 m/frame (~240 m/s) and records per frame the rAF
interval, `StreamedLevel.stats.phases` (new: where each tick's time went),
terrain idle-task runs (`extras.prepLog`), long-animation-frame attribution,
and optionally a Chrome trace (`STREAM_TRACE=1`, main-thread tasks + GC) and a
CPU sampling profile (`STREAM_PROFILE=1`) attributed per spike frame.
Laptop Intel UHD, headed Chrome, 1440x900; other agents' Chrome/Node
processes were loading the machine, so runs alternate and CPU load is noted.

**Cause of the 107–183 ms spikes (evidence, `spikes-base-on-r2-spikes.json`):**
the CPU profile of every large spike is ambient-audio buffer synthesis on the
main thread — `bufferFor → synth` in the smoke game's ambient sink, called the
first time a tile's emitter starts (one 4 s buffer per sound, a per-sample
`forEach` closure calling `Math.sin`), 80–143 ms plus ~25 ms of GC from it.
`stats.phases.actors` on the previous tick was 183 / 107 / 93 ms for the three
spikes. Streaming itself (floors, props, despawn, terrain prep) stayed under
~10 ms in those frames. Not terrain, colliders, nav or GPU uploads.

Fixes:

- **Ambient sounds are synthesised in a Web Worker** as soon as the sink
  exists (`ambient-synth.worker.ts`), with a main-thread fallback that does one
  sound per task. A voice whose buffer is not ready is a silent stand-in that
  starts when it lands; starting a voice never computes samples. The synth
  (`ambient-synth.ts`) uses phasor recurrences and one small function per sound
  (one function holding all three sample loops got re-tiered by V8, leaving
  `hum-high` at ~300 ms): 1–12 ms per sound, output equal to the old synth
  within 1.5e-8.
- **Occlusion raster ~3x faster** (Node bench, 48 box occluders + 9 hill tiles:
  7.0 → 2.4 ms per frame, same culls within the epsilon): the per-pixel
  four-corner edge test became one incremental value per edge
  (E(x,y) + min(0,A) + min(0,B) is the minimum over the pixel's corners), with
  an epsilon that can only reject borderline pixels. New randomised test checks
  it never covers a pixel the four-corner rule does not, never nearer, and a
  mutation (loosened epsilon) fails it.
- **Occlusion runs only while it pays**: every frame while the last pass hid at
  least 4 draws and 5% of the visible ones, otherwise a probe every 8 frames
  (`STREAM_LEVEL.occlusion`). Skipping hides nothing, so it can never hide
  something visible. On the road it mostly probes: browser p50 cost 2.7 → 1.7 ms
  with the faster raster, then ~0 on non-probe frames.
- Per-frame garbage cut in occlusion setup (items built once per prop at spawn,
  no per-slab spreads, scratch arrays reused).
- Terrain idle tasks wait for at least 3 ms of idle time (up to 4 re-queues),
  and the forced-run timeout is 250 ms instead of 60 ms.

**After** (route at ~240 m/s, terrain on vs flat floors, alternating;
`final-route-{on,flat}-r{1..5}-spikes.json`):

| run | CPU | on: p50 / p95 / max (ms) | on: > 50 ms | flat: p95 / max | flat: > 50 ms |
|---|---|---|---|---|---|
| r1 | 24–17% | 35.4 / 54.9 / 77.1 | 23 | 37.5 / 40.2 | 0 |
| r2 | 39–29% | 18.1 / 19.2 / 51.2 | 1 | 19.1 / 20.3 | 0 |
| r3 | 9–8% | 18.2 / 19.2 / 36.2 | 0 | 19.1 / 21.1 | 0 |
| r4 | 13–35% | 18.2 / 19.3 / 21.6 | 0 | 19.6 / 38.1 | 0 |
| r5 | 9–25% | 18.3 / 35.5 / 38.2 | 0 | 19.5 / 37.6 | 0 |

Before on the same harness (`spikes-base-on-r{1,2}`): max 180.4 / 178.9 ms,
5 / 4 frames over 50 ms. Streaming per tick after: p95 5–6 ms, max 8.5–10.3 ms;
terrain prep tasks max 4–5 ms. Round 1 ran while the machine was busy (the
paired sprint run that round, flat floors, hit 30 fps and a 260 ms frame), so
it says nothing about the code; the r2 51.2 ms frame had only ~9 ms of
streaming work. Traced runs (`final-route-on-trace-r{3,4}`) are not frame-time
evidence (tracing overhead put the browser at a 30 fps cadence) but show that
the remaining long frames carry 10–30 ms of main-thread work in total, with
GC scavenges of 3–9 ms and the occasional V8 incremental-marking step.

Sprint (600 frames), on vs flat, alternating (`final-sprint-*`):

| run | on: fps / p95 / max | on: > 50 | flat: fps / p95 / max | flat: > 50 |
|---|---|---|---|---|
| r2 | 55.0 / 19.1 / 38.5 | 0 | 55.4 / 19.1 / 36.7 | 0 |
| r3 | 54.5 / 19.2 / 36.0 | 0 | 55.2 / 19.0 / 38.3 | 0 |
| r4 (CPU 46% on) | 54.6 / 19.6 / 71.9 | 2 | 54.8 / 19.4 / 36.9 | 0 |

Read honestly:

- **Goal "no frame > 50 ms at 240 m/s with terrain on": met in the three
  quiet rounds (r3–r5: 0), missed by 1.2 ms once (r2), missed badly in r1 under
  machine load.** Terrain on is now in the flat-floor range.
- **Goal "sprint p95 ≈ 16.7–17 ms": not met.** p95 is ~19.1 ms with terrain on
  *and* with flat floors; rAF p50 is ~18.1 ms either way (the PR #13 capture
  had the same 17–18.7 ms p50). That gap is the frame's base cost on this
  laptop (main thread ~15–17 ms busy per frame in traces: render CPU, game
  loop, streaming, GC), not streaming spikes.
- Occlusion still does not pay on the road; it now costs ~nothing there.
- Browser e2e flakes seen once each under load and passed on two reruns each:
  "NPCs keep their state…" (`liveState(id)` undefined after the 2 s wait; the
  test picks any live NPC, which the memory budget can evict) and the proof-head
  line test. Smoke-game: the same 3 pre-existing failures.

## What gate 2 still needs

- Hand-made or scanned assets through the bake (`augengine-lod bake model.glb`
  takes Blender/Maya glTF exports); the procedural kit proves the pipeline,
  it is not final art. Impostors are untextured (no offscreen bake yet).
- Fewer draw calls for varied kits: merge per-tile static props or lower the
  instancing threshold for small meshes. Terrain tiles add one draw each.
- Base frame cost: the main thread is ~15–17 ms busy per frame on this laptop
  (render CPU, game loop, GC), which keeps sprint p95 at ~19 ms with or
  without terrain. Cutting per-frame garbage outside the world module (renderer,
  scene) and render CPU is the next lever.
- GPU-driven culling/indirect draws if entity counts grow far past today's
  ~450 resident.
- A shipping-GPU capture, and a production (bundled) build capture.

---

# 2026-11-24 — per-tile irradiance probes

Intel UHD gen-11, headed Chrome, 1440×900, interleaved off/on runs. The
machine was shared with other agents' browsers during part of the runs, so
single numbers drift; compare the paired runs.

**What it is.** Each 64 m tile bakes a 4 × 4 × 2 grid of irradiance probes
(16 m apart; 1.5 m and 13.5 m above the ground under each column) when it
loads, as L0 + L1 SH relative to the flat ambient (open ground averages 1).
Bake: 64 fixed directions per probe; the tile's and its 8 neighbours' props as
boxes plus the terrain (8 m height grid, 12-step march), one bounce, one
shadow ray to the sun per hit, guessed albedos (see the header of
`runtime/world/irradiance-probes.ts`). StreamExtras runs it in idle tasks (one
for the box list + height grid, then one per row of four probes), hands the
tile to the renderer's 128 KB probe ring with the floor, and clears it on
evict. The PBR / cel / toon shaders read the ring per **vertex** (bilinear
ground base, trilinear over 8 probes, renormalised over valid ones, so tiles
blend across seams and a missing neighbour fades), and dot the interpolated
coefficients with the per-pixel normal. Opt-in: `?probes=1`.

| Measure | Probes off | Probes on |
|---|---|---|
| Main pass GPU p50, 4 interleaved rounds (`2026-11-24-probes-ab-bench.json`) | 2.58, 2.82, 2.59, 2.57 ms | 2.72, 3.20, 2.99, 2.94 ms → **+0.38 ms** median |
| gpu_sum p50 (same runs) | 11.04 ms (median) | 11.20 ms (median) |
| First try: per-PIXEL sampling (`2026-11-24-probes-ab-perpixel.json`) | main 1.75 ms | main 4.7 ms → **+3.0 ms**, rejected; moved to the vertex stage |
| 1 km fast travel at ~240 m/s, frames > 50 ms (`2026-11-24-probes-ab.json`, 3 rounds) | 0 / 86 / 54 | 0 / 32 / 0 |
| Bake idle tasks per run (route) | — | 185–254 tasks, p50 2.2–2.7 ms, max 9.8–17 ms |

- Fast travel: in the quiet rounds both sides have 0 frames over 50 ms; the
  noisy middle round hit both (off was worse). Probes add **no new** frames
  over 50 ms that we could measure. The biggest bake tasks (10–17 ms) are the
  first ones after load (JIT warm-up) and the per-tile setup.
- Unit tests: L1 projection (constant and linear radiance), open-ground
  normalisation, wall occlusion + colour bleed, probes lifted out of
  buildings, ring aliasing / clearing, seam continuity (max step < 0.08 per
  0.5 m across a tile seam), layer selection over a hill, StreamExtras
  residency (baked in prepare, set with the floor, cleared on evict, reused
  when a tile streams back).

**Honest visual verdict** (`2026-11-24-probes-<spot>-<off|on>.png`): the probes
work and the seams are invisible, but on this level **the difference is barely
visible**: mean absolute difference 1.2–1.6 / 255 at the West Gate and the
Water Tower (an 8× difference image shows building faces shifting a few
levels toward their surroundings' colour and shaded faces gaining a little
contrast; the road and ground barely move). The key light here is ~10× the
ambient, so ambient — the only thing probes change — is a small part of every
pixel, and the kit's colours are in vertex colours the bake does not read.
(The Signal Spire pair differs mostly because a tile was still streaming in
the "on" still, not because of the probes.)

**Decision: off by default.** It is within the ~0.5 ms GPU bar (+0.4 ms) and
adds no fast-travel spikes, but +0.4 ms (~4% of the frame) for a change you
have to diff to see is not worth it on Intel UHD today. Worth revisiting with
a dimmer key light / interiors / dusk lighting, where ambient carries more of
the image, or once the bake reads real albedo.

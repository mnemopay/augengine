# Changelog

## 0.4.1-alpha.0 — 2026-11-27

Fix release. **The root import works under Vite.**

- **Root import works under Vite (issue #26).** `import … from "@mnemopay/augengine"` failed in
  0.4.0-alpha.0 at Vite's dependency pre-bundle and in `vite build` (`failed to resolve import
  "@xenova/transformers"`), because `LocalWebGpuProvider` contained a literal
  `import("@xenova/transformers")`, and bundlers resolve literal dynamic imports at build time.
  No optional package is now referenced as a literal import anywhere the published entry points
  reach: `@xenova/transformers` (local LLM), `@mnemopay/sdk/recall` (in-process brain) and
  `@mnemopay/augengine-onnx` (Kokoro) are loaded through computed specifiers only when the
  feature is used, and a missing package throws an `Install … to use …` error. Verified with a
  Vite 8 project outside the monorepo: `vite build`, the dev server and `vite preview` all work
  with only `@mnemopay/augengine` installed, and the page renders in headed Chrome.
- **New loader options for optional packages (bundled apps).** A computed import is invisible to
  your bundler, so in a Vite/webpack/esbuild app pass your own import:
  `createInProcessBrain({ loadRecall: () => import("@mnemopay/sdk/recall") })`,
  `new LocalWebGpuProvider({ loadTransformers: () => import("@xenova/transformers") })`,
  `new KokoroTtsProvider(url, { loadOnnx: () => import("@mnemopay/augengine-onnx") })`.
  Without the option the default runtime import still works in Node, Bun and with import maps.
  **Behaviour change:** a bundled browser app that used `createInProcessBrain` with
  `@mnemopay/sdk` installed and no option now gets the `Install @mnemopay/sdk …` error on the
  first memory call; add `loadRecall` as above.
- `@xenova/transformers` is declared as an optional peer dependency.
- **`loadGltfIntoRenderer(renderer, scene, url, opts)`** (root and `./runtime/live`): fetches a
  .glb, uploads every primitive and its PBR textures under the `gltf:<url>#<n>` keys the renderer
  looks up, and adds the model's entities to the scene (pass `null` to upload only). Replaces the
  per-primitive upload loop in the getting-started guide.
- **`InProcessBrain.forget(namespace)`.** The method was published as `augenginet()`, a
  find-and-replace accident; `augenginet()` stays as a deprecated alias until 0.5. Maya's dialogue
  line and the memory-question pattern also said "augenginet" instead of "forget" (the pattern
  now matches "forget" again).
- `AUGENGINE_VERSION` reported `"0.0.1"`. It is now generated from `package.json` on every build
  (`scripts/write-version.mjs` → `src/version.ts`) and a test checks they match.
- Repo hygiene: tracked `tsconfig.tsbuildinfo` files (already gitignored) are removed from git so
  builds no longer dirty the tree.
- Tests: a dist scan (`src/optional-deps-dist.test.ts`) lexes every built file with
  es-module-lexer, the lexer Vite uses, and fails if any optional package is a literal import, or if
  any published entry point reaches a package that is not a declared dependency (`three` is allowed
  only under `./runtime/live/three`). Run `npm run build` before `npm test`.

## 0.4.0-alpha.0 — 2026-11-24

Everything merged since 0.3.0-alpha.0 (PRs #18–#23). Only the engine changes ship in this
package; the Studio, the detective game and the LOD tool live elsewhere in the monorepo and
are listed for context.

In this package:

- **Characters (`./characters`):** humanoid bone mapping (`mapHumanoidBones`,
  `resolveHumanoidRig`, `missingHumanoidBones`) that recognises Mixamo, VRoid, Unreal,
  Rigify and Quaternius bone names, plus an ARKit-52 face-morph coverage check
  (`arkitCoverage`); clip retargeting between humanoid rigs (`retargetClip`) with facing
  correction and limb-only rest-pose correction (T-pose clip on an A-pose rig); exact
  runtime two-bone IK and look-at (`IK_TWO_BONE` / `LOOK_AT` components, `applyIkComponents`),
  which refuses disconnected chains instead of producing a wrong pose.
- **Per-tile irradiance probes (`./runtime/world`, `./runtime/live`):** L0+L1 probes baked
  when a tile loads, streamed into a fixed 128 KB GPU ring and read per vertex. **Off by
  default**: on Intel UHD they cost about +0.38 ms of main pass and were barely visible in
  the test range. The smoke game turns them on with `?probes=1`.

Elsewhere in the monorepo (not part of this package):

- **Studio:** skinned (skeletal) model preview, "Use a clip from another model"
  retargeting, IK components in the inspector, FBX imports converted from centimetres to
  metres, and an **Add character** wizard (text/image → FLUX image → TRELLIS mesh →
  in-browser auto-rig → Idle/Walk/Wave clips → persona). Paid steps run only on a click,
  show their estimate, and the dev-server proxy refuses calls past a spend cap (about
  $0.03 per generated character). See `docs/CHARACTERS.md`.
- **Detective game (gate 4 slice):** *The Wrenfield Observatory* with painted-noir suspect
  portraits, TRELLIS busts and click-to-interact, deployed at augengine.com/detective with a
  key-wide spend guard on the shared model key (scripted "storyteller is resting" fallback).
- **`@mnemopay/augengine-lod`:** 0.1.0-alpha.0 is published separately. `simplifySmooth`
  for organic meshes was added in the repo afterwards and is not in the published
  0.1.0-alpha.0 tarball.

Known gaps:

- The auto-rig is a heuristic 22-joint humanoid: no fingers, face, hair or loose-clothing
  bones; good for stylised NPCs, not hero characters.
- Generated (TRELLIS / auto-rigged) characters have no face morphs, so no lip-sync on them.
- Irradiance probes are off by default (see above).
- Sprint p95 is still ≈ 19 ms on Intel UHD; gate 1 (16.7 ms p95 on a shipping GPU) is not
  claimed.
- `@mnemopay/augengine-onnx` (needed for Kokoro TTS) is a private workspace package and is not
  on npm, so for anyone installing from the registry Kokoro throws its "Install …" error.

## 0.3.0-alpha.0 — 2026-11-23

Everything merged since 0.2.0-alpha.0 (PRs #6–#16). Measured on an Intel UHD laptop
(headed Chrome, WebGPU); numbers are in the repo's `results/*/SCORECARD.md`.

- **World streaming (gate 2):** a 1 km streamed level — tiles carry floors, buildings,
  props, colliders and a per-tile nav grid; LOD with hysteresis + frustum culling;
  time-sliced loading under a per-frame budget (floors first, preload ahead); instanced
  draws for repeated meshes (draw calls ~150 → 5–46).
- **Terrain, occlusion, NPCs, audio:** per-tile heightfield terrain with seamless borders
  and Rapier heightfield colliders; conservative CPU occlusion culling (gated so it only runs
  while it pays); NPCs that persist their state across tile unload/reload; streamed
  positional ambient audio (synthesised off the main thread).
- **Physics:** one box-collider convention (world-space half extents) across physics, the
  character controller, raycasts and nav.
- **Renderer:** a `LiveRenderer` adapter boundary; per-pass GPU timestamp profiler and a
  deterministic `?bench=1` benchmark; an optional three.js WebGPU backend
  (`./runtime/live/three`, `three` is an optional peer) — benchmarked and kept opt-in
  because it was 15–65 % slower on UHD; render scale + TAA/TAAU with a single fused
  TAA/bloom/tonemap pass and a velocity buffer for skinned/morph motion; half-res SSAO
  (high preset); second shadow cascade (high preset); tonemapper choices (ACES, Khronos PBR
  Neutral, AgX), colour grade and height fog; sky diffuse ambient and opt-in sky specular.
- **Content:** procedural proof head now sits on a bust; art-kit meshes (`proc:kit-*`)
  render through the same instancing path.

Known gaps: no irradiance probe grid; sprint p95 ≈ 19 ms on Intel UHD (60 Hz-capped
display, base frame cost — not streaming); gate 1 (16.7 ms p95 on a shipping GPU) is not
claimed; the LOD bake tool (`@mnemopay/augengine-lod`) is not published yet.

## 0.2.0-alpha.0 — 2026-11-23

First update since 0.1.0-alpha.0 (2026-05-25). Highlights from 42 commits:

- **Rendering:** unified WebGPU playable path; native glTF PBR textures and morph-target draws;
  scene lights with one shadow map and a single ACES pass; GPU particle pass (fire, smoke,
  embers); GPU shallow water; WebGPU timestamp queries + HUD.
- **World:** tile streaming + LOD + culling + instancing (Track 7), and `StreamedLevel` /
  `attachStreamedLevel` in `runtime/world` — tiles become scene entities and leave memory
  when evicted.
- **Characters:** skeletal animation playback, motion matching, character Proof Room, and a
  procedural proof head with a lip-synced line (`content`: `buildProofHeadMesh`,
  `proofHeadLineWeights`, …).
- **Physics:** Rapier rigid bodies, surface-aware physics, vehicle system + chase camera,
  OBB raycast fix for sinking vehicles; conserving chemistry simulation.
- **Gameplay:** game-native ECS components + systems (Track 5); `fighter` module (moves,
  combos, meter, CPU opponent, rig, roster, sim); DOM-overlay UI system.
- **Assets & export:** KTX2 + meshopt packer with per-asset budget report; WebCodecs
  recorder + render-scene CLI.
- **AI:** client-side AI providers, incl. Kokoro TTS. `@mnemopay/augengine-onnx` (Kokoro) and
  `@mnemopay/sdk` (in-process brain) are optional peer dependencies, loaded only when used:
  `npm install @mnemopay/augengine` works on its own, and Kokoro / the brain throw an
  "Install …" error naming the missing package.
- Rename Forge → AugEngine completed across packages.

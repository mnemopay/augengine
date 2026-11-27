# `@mnemopay/augengine`

AugEngine is a web-native runtime for persistent, memory-bearing, guard-railed AI
characters inside deterministic game state, drivable by agents over MCP. One scene
graph feeds a native WebGPU renderer (live play) and a frame-deterministic WebGPU +
WebCodecs path (export). Gameplay runs on a fixed 60 Hz tick with Rapier physics, so
the same inputs give the same world. Any entity can carry a persona; the AI layer
gives it memory, goals and an action grammar that refuses verbs it was not granted.
The same scene can be driven by a person in the browser or by an agent through the
AugEngine MCP server.

**Status: alpha (`0.4.1-alpha.0`).** Good for evaluation, prototypes and vertical
slices. APIs can change between `0.x` releases. It is not yet a production-ready
general game engine.

- Live game built on it: **[augengine.com/detective](https://augengine.com/detective)**
  (*The Wrenfield Observatory*, a noir interrogation game)
- Site: [augengine.com](https://augengine.com/)
- Source, issues and the MCP server:
  [github.com/mnemopay/augengine](https://github.com/mnemopay/augengine)
- Getting-started guide: [`docs/GETTING-STARTED.md`](https://github.com/mnemopay/augengine/blob/main/docs/GETTING-STARTED.md)
  in the repo

## Quick start (60 seconds)

You need a WebGPU browser: Chrome or Edge 113+ on desktop.

```bash
mkdir my-scene && cd my-scene
npm init -y
npm install @mnemopay/augengine vite
```

`index.html`:

```html
<!doctype html>
<html>
  <body style="margin:0;background:#000">
    <canvas id="game" style="width:100vw;height:100vh;display:block"></canvas>
    <script type="module" src="/main.ts"></script>
  </body>
</html>
```

`main.ts`:

```ts
import { Scene } from "@mnemopay/augengine";
import { LiveRuntime, lookAtTransform } from "@mnemopay/augengine/runtime/live";

const at = (x: number, y: number, z: number) => ({
  position: [x, y, z], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null,
});

const scene = Scene.fromJson({
  schemaVersion: 1,
  source: { kind: "authored", path: "quickstart" },
  defaultCamera: "cam",
  entities: [
    { id: "cam", name: "Camera", components: {
        transform: { ...lookAtTransform([0, 2, 6], [0, 1, 0], [0, 1, 0]), parent: null },
        camera: { fovY: 0.9, near: 0.1, far: 100, active: true } } },
    { id: "sun", name: "Sun", components: {
        transform: { ...lookAtTransform([4, 8, 4], [0, 0, 0], [0, 1, 0]), parent: null },
        light: { kind: "directional", color: [1, 0.95, 0.9], intensity: 3 } } },
    { id: "floor", name: "Floor", components: {
        transform: { ...at(0, 0, 0), scale: [10, 1, 10] },
        mesh: { geometry: { kind: "primitive", shape: "plane" },
                material: { kind: "pbr", baseColor: [0.35, 0.35, 0.4], metallic: 0, roughness: 0.9 }, visible: true } } },
    { id: "maya", name: "Maya", components: {
        transform: at(0, 1, 0),
        mesh: { geometry: { kind: "primitive", shape: "sphere" },
                material: { kind: "pbr", baseColor: [0.9, 0.4, 0.2], metallic: 0, roughness: 0.4 }, visible: true },
        // Any entity can carry a persona; the AI layer reads it.
        persona: { traits: ["curious", "remembers you"], voice: null, memoryId: "maya",
                   goals: [], mood: 0.2 } } },
  ],
});

const canvas = document.querySelector<HTMLCanvasElement>("#game")!;
const runtime = new LiveRuntime(canvas, scene, {
  onSimTick: (_dt, t) => {
    // Fixed 60 Hz sim tick: deterministic game state lives here.
    scene.setLocalTransform("maya" as never, [0, 1 + Math.sin(t * 2) * 0.25, 0], [0, 0, 0, 1], [1, 1, 1]);
  },
});
// Needs a WebGPU browser (Chrome/Edge 113+).
runtime.init().then(() => runtime.start());
```

```bash
npx vite
```

Open the printed URL: a lit, bobbing sphere on a floor. This exact project was built with
`vite build` and run in headed Chrome against the `0.4.1-alpha.0` tarball before this
README shipped.

**Root or sub-path imports.** Since 0.4.1 the root import (`import … from
"@mnemopay/augengine"`) works under Vite and other bundlers with no config: no optional
package is referenced at build time. The browser runtimes stay on their sub-paths
(`./runtime/live`, `./runtime/render`). Sub-paths are still the way to keep bundles small.

**Optional packages.** Features that need an optional package load it only when used, and
throw an `Install …` error if it is missing. In a bundled app, hand the feature your own
import so your bundler resolves it:

```ts
createInProcessBrain({ loadRecall: () => import("@mnemopay/sdk/recall") });           // npm i @mnemopay/sdk
new LocalWebGpuProvider({ loadTransformers: () => import("@xenova/transformers") });  // npm i @xenova/transformers
```

**Load a glTF in one call:**

```ts
import { loadGltfIntoRenderer } from "@mnemopay/augengine";
const { instance } = await loadGltfIntoRenderer(renderer, scene, "/character.glb", { position: [0, 0, 0] });
```

It uploads every primitive and its PBR textures under the keys the renderer looks up and
adds the model's entities (with skin and animations) to the scene.

## What works today

"Stable" here means shipped, covered by unit tests and exercised by the live detective
game; the API can still change before 1.0. "Alpha" means it works with known gaps.

| Area | Import | Status | Notes |
| --- | --- | --- | --- |
| Scene graph, components, transforms | `./scene` | stable | JSON `SceneSpec`, validated by `Scene.fromJson` |
| glTF / GLB loading (mesh, skin, morph, animation) | `./scene`, root | stable | `loadGltfIntoRenderer` loads a .glb into the live renderer in one call; meshopt + KTX2 decoders in the root export |
| Native WebGPU live renderer | `./runtime/live` | stable | PBR textures, shadows (2 cascades on high), TAA, SSAO, bloom, tonemappers, GPU particles, shallow water |
| Fixed-step simulation + Rapier physics | `./physics`, root | stable | one box-collider convention across physics, controller, raycasts, nav |
| World streaming (1 km tiles, LOD, culling, instancing) | `./runtime/world` | alpha | per-tile terrain, nav and NPC state; irradiance probes off by default |
| Personas, observation bus, action grammar | `./ai` | alpha | unregistered verbs and unresolvable preconditions are refused |
| Persistent NPC memory | `./ai` + optional `@mnemopay/sdk` | alpha | memory lives in the optional SDK peer; bundled apps pass `loadRecall` |
| Characters: humanoid mapping, retargeting, two-bone IK | `./characters` | alpha | Mixamo, VRoid, Unreal, Rigify, Quaternius names; IK refuses disconnected chains |
| Auto-rig for generated characters | Studio | alpha | heuristic 22-joint humanoid: **no fingers, face, hair or cloth bones** |
| Lip-sync on hand-authored rigs (ARKit-52 morphs) | `./audio` | alpha | needs face morph targets in the GLB |
| Lip-sync on generated (TRELLIS / auto-rigged) characters | | not yet | generated meshes have no face morphs |
| Kokoro voices (TTS) | `./ai` | not yet | needs `@mnemopay/augengine-onnx`, which is not public; without it `load()` throws an "Install …" error |
| Deterministic render + WebCodecs export | `./runtime/render` | alpha | |
| Optional three.js backend | `./runtime/live/three` | alpha | opt-in; measured 15–65 % slower on Intel UHD |
| Performance on a shipping GPU (16.7 ms p95) | | not measured | all published numbers are from an Intel UHD laptop; sprint p95 ≈ 19 ms there |
| AugEngine Studio (browser editor) | not public | alpha | not on npm and not in the public repo yet |
| AugEngine Studio for Windows | not public | alpha | Electron build is **unsigned** |
| MCP server (`augengine-mcp`) | public repo | alpha | not on npm; see tool list below |

## Drive it from an agent (MCP)

[`packages/augengine-mcp`](https://github.com/mnemopay/augengine/tree/main/packages/augengine-mcp) in the repo is a stdio MCP server (JSON-RPC 2.0, full
`initialize` / `tools/list` / `tools/call` handshake). It is not published to npm yet;
build it from a clone and register `node packages/augengine-mcp/dist/cli.js serve --stdio`.

| Tool | What it does |
| --- | --- |
| `augengine_create_scene` | prompt → `SceneSpec`, deterministic per (prompt, seed) |
| `augengine_add_entity` | add an entity to a scene |
| `augengine_attach_persona` | attach a persona component |
| `augengine_list_entities` | query by component tag or name, with world positions |
| `augengine_dispatch_action` | run an action through the grammar (refuses unknown verbs) |
| `augengine_subscribe_observation` | open a buffered observation channel |
| `augengine_poll_observation` | drain queued observation events |
| `augengine_describe_scene` | component census, verbs, predicates, effects |
| `augengine_export_godot` | `SceneSpec` → Godot 4.4 `project.godot` + `Main.tscn` (not yet opened in a real Godot editor) |
| `augengine_render_clip` | returns an explicit error over stdio (needs a browser host) |
| `detective_new_session`, `detective_status`, `detective_act` | play the detective game as an agent |

## Links

- Live game: [augengine.com/detective](https://augengine.com/detective)
- Source: [github.com/mnemopay/augengine](https://github.com/mnemopay/augengine)
- MCP server: [`packages/augengine-mcp`](https://github.com/mnemopay/augengine/tree/main/packages/augengine-mcp)
- Detective game source: [`examples/detective`](https://github.com/mnemopay/augengine/tree/main/examples/detective)
- Benchmark scorecards (measured on Intel UHD, headed Chrome):
  [renderer](https://github.com/mnemopay/augengine/blob/main/results/renderer-benchmark/SCORECARD.md),
  [streamed level](https://github.com/mnemopay/augengine/blob/main/results/stream-level/SCORECARD.md),
  [proof room](https://github.com/mnemopay/augengine/blob/main/results/proof-room/SCORECARD.md)
- [CHANGELOG](https://github.com/mnemopay/augengine/blob/main/packages/augengine/CHANGELOG.md)
- LOD baking tool: [`@mnemopay/augengine-lod`](https://www.npmjs.com/package/@mnemopay/augengine-lod)

## Build from source

```bash
git clone https://github.com/mnemopay/augengine.git
cd augengine
npm ci
npm run build
npm test --workspace=@mnemopay/augengine
```

## License

Apache-2.0. See [LICENSE](LICENSE).

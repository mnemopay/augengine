# `@mnemopay/augengine` — Engine Architecture

**Date:** 2026-05-08 (original architecture decision record), updated through 2026-11.
**Status:** Public alpha. The design decisions below are the original ADR; they are not a
production-readiness claim. Current status lives in the
[engine README](https://github.com/mnemopay/augengine/blob/main/packages/augengine/README.md) and the
[CHANGELOG](https://github.com/mnemopay/augengine/blob/main/packages/augengine/CHANGELOG.md).

> This is the public edition of the architecture record. Some paths it mentions as
> measurement harnesses (`examples/smoke-game`, `e2e/`) live in the private development
> monorepo and are not part of this repository; the scorecards they produce are in `results/`.

---

## What this is

A single browser-native runtime for **games + UGC + films**, sharing one scene-graph and feeding two renderers (real-time WebGPU for interactive play, frame-deterministic WebGPU+WebCodecs for video export). AI is a first-class subsystem, not a bolt-on. ML-model generation stays as external API calls because the models themselves are not replicatable; everything else is owned native code on browser primitives.

The canonical one-liner:

> AugEngine is one scene graph, two renderers, one charter authoring grammar — running every game, UGC short, and film through the same pipeline.

## Why this shape

Three findings forced the architecture:

1. **WebGPU has crossed critical mass (~83% as of March 2026).** Real-world perf is 5-150× WebGL2 on draw-calls + compute; 2026 is the right year to commit.
2. **WebCodecs is mature.** `VideoEncoder` runs ~5-15× faster than ffmpeg.wasm and ~200 fps for sustained 1080p H.264 on M-series hardware. Headless-browser render paths are a slower indirection over this exact API. A native engine that owns the canvas → encoder pipeline is faster and skips the headless bottleneck entirely.
3. **For character animation, the math is shared between game-time and render-time.** Skinning, retargeting, IK, VRM expressions, spring physics — all work the same way at 60 fps real-time as they do at frame-stepped render. Splitting these between a "game engine" and a "video engine" is wasted code. One implementation, two consumers.

---

## Package layout

```
packages/
  augengine/            @mnemopay/augengine        scene graph + dual renderer + physics + characters + AI subsystem
    src/
      scene/            entity/component types, transform hierarchy, scene loader
      runtime/
        live/           WebGPU real-time renderer + game loop (optional three.js backend at runtime/live/three)
        render/         WebGPU + WebCodecs frame-deterministic renderer + encoder
        world/          1 km tile streaming, LOD, culling, instancing
      authoring/        YAML charter → scene-spec compiler
      characters/       skinning, retargeting, IK, humanoid mapping
      audio/            Web Audio mixer (live) + OfflineAudioContext (render) + lip-sync
      ai/               inference scheduler, personas, prompt templates, action grammar, memory
      physics/          fixed-step simulation on Rapier
  augengine-lod/        @mnemopay/augengine-lod    LOD baking CLI + procedural city kit
  augengine-detective/  detective game core (case state, rules, guardrails, memory)
  augengine-mcp/        stdio MCP server over the engine and the detective game
examples/
  detective/            The Wrenfield Observatory, the finished game
```

---

## The seven AI-native design decisions (locked)

These are commitments, not suggestions. Every line of code respects them.

### 1. Inference is async, frame-budgeted, fallback-always

Every AI call goes through one abstraction:

```ts
type AIRequest<T> = {
  task: string;            // "dialogue" | "ambient" | "action_plan" | …
  payload: unknown;
  timeoutMs: number;       // hard deadline; 800 for dialogue, 50 for ambient
  priority: 1 | 2 | 3;     // 1 = critical (frame-blocking allowed off-thread), 3 = background
  fallback: () => T;       // deterministic, instant, always defined
};
```

Dispatched to an NVIGI-style scheduler that interleaves with the render queue. Game logic never blocks. If inference misses the deadline, fallback fires and the AI result becomes a future frame's update.

### 2. Persona is an ECS component, not a class

Every entity *can* carry a `Persona` component. Most don't. The component is a thin handle:

```ts
type Persona = {
  traits: string[];        // serialized for LLM prompts
  voice: VoiceRef;         // TTS profile
  memoryId: string;        // memory recall key — portable across titles
  goals: Goal[];
  mood: number;            // -1..1, drives animation/voice modulation
};
```

A door can have a persona. A weapon can have a persona. Most don't.

### 3. All authored text is a `.prompt` template

There is no `dialogue.csv`. There are `dialogue/<scene>/<beat>.prompt` files with slot-fillers, few-shot examples, and an `expected_outputs` test block CI runs against a frozen LLM. Static lines are zero-slot templates. The shipped runtime never distinguishes between "scripted" and "AI" dialogue — both go through one renderer.

### 4. Determinism core, AI-artifact manifest

The deterministic engine state (physics, transforms, RNG seeds, inventory) is what saves and replicates. AI-generated artifacts (text, images, voice clips) are content-addressed by `(persona_id, prompt_hash, world_state_hash)` and stored in a side-cache. **Save = core state + artifact manifest.** Multiplayer = server runs AI, broadcasts artifact hashes, clients fetch from cache or stream from server. SQLite for the artifact manifest, content-addressable blob store for media.

### 5. Unified action grammar

One IR for every gameplay action:

```ts
type Action = {
  verb: ActionVerb;        // "MoveTo" | "Talk" | "Attack" | "Give" | …
  actor: EntityId;
  target?: EntityId | Vec3;
  params: Record<string, unknown>;
  preconditions: Predicate[];
  effects: Effect[];
};
```

Player input maps to it. Behavior trees emit it. LLM agents emit it. **Every system is symmetric** — humans and bots are the same kind of entity from the engine's point of view.

### 6. One scene loader for authored / procgen / AI-gen

A scene is a JSON document conforming to a JSON-Schema. The loader doesn't care which path produced the document. `SceneSource` is a tagged union — `Authored | Procgen(seed) | Generated(prompt)` — but downstream is identical. AI-gen scenes are cached by prompt-hash so reload is deterministic.

### 7. Observation bus = engine's public agent API

Typed pub/sub of all game events: `Visual`, `Audible`, `Social`, `Inventory`, `Combat`, `Dialogue`. Cameras render `Visual`. Speakers render `Audible`. UI consumes `Inventory`. **Agents subscribe to the same bus** with policy-controlled view filters. A bot is a `Player` with no input device and an `Agent` component. Replay/telemetry/training corpus are the same event log, serialized.

---

## Dual renderer split

### Live runtime (`runtime/live/`)

- **Backend interface** — `LiveRenderer` (`runtime/live/renderer-api.ts`) with `WebGpuLiveRenderer` (primary, default) and `ThreeLiveRenderer` (opt-in A/B backend on three.js WebGPURenderer, whose WebGL2 fallback replaces the planned `WebGL2LiveRenderer`). See "Renderer adapter" below.
- **Frame loop** — `requestAnimationFrame`. Fixed-step physics tick at 60 Hz. Render at display refresh.
- **Skinning** — WebGPU compute shader writes joint matrices to a storage buffer; vertex shader reads `JOINTS_0`/`WEIGHTS_0`.
- **Audio** — live `AudioContext`. AudioWorklet for analysis (raw samples for beat detection + formant tracking). `AnalyserNode` is UI-debug only.
- **Input** — pointer + keyboard + gamepad + touch. Maps to `Action` IR.

### Render runtime (`runtime/render/`)

- **Backend** — same `LiveRenderer` trait, but driven by frame index instead of wall clock. Time advances exactly `1 / fps` per frame.
- **Encoder pipeline** — Canvas → `GPUTexture` → `VideoFrame` (zero-copy where supported) → `VideoEncoder` (H.264 default, HEVC opt-in, AV1 stretch). Muxer writes MP4 in-browser.
- **Audio** — `OfflineAudioContext` with same graph as live mode; result interleaved into the MP4 audio track.
- **Determinism** — every RNG call respects a clip-level seed. Same charter + same seed = byte-identical output.
- **No headless intermediate.** Skip Remotion's chrome-for-testing detour entirely. The browser-native path is faster *and* unlocks GPU acceleration that Remotion's headless mode cannot.

### Why this isn't Three.js + Remotion

Three.js and Remotion are convenient but they're abstractions over what the browser already gives you. Three.js bundle is ~155 KB gzipped after tree-shake (much more in practice); a focused engine ships ~50-100 KB. Remotion's render path goes Canvas → headless Chrome → ffmpeg, which collapses GPU acceleration and adds a 5-15× speed penalty vs WebCodecs direct. Native means: AugEngine owns the canvas, owns the encoder, ships nothing it doesn't render with.

### Renderer adapter (2026-11-23)

The 2026-11 research pass ("Borrow the renderer, own the characters") proposed moving pixels onto three.js WebGPURenderer if it held up on the floor GPU. That was tested rather than assumed:

- **Boundary.** `LiveRenderer` is everything a shell asks of a renderer: `init/render/resize/destroy`, `uploadMesh`, `uploadPbrTextures`, `setBlendshapeWeights`, sky / IBL / shadows / post chain / `setLook`, instancing, frame stats, and GPU timing (`enableGpuTimestamps`, per-pass `enablePassTimings`). `EngineRunner` and the smoke-game take the interface. Backends read `Scene.entities` + `Scene.worldTransforms()` each frame and own no game state, so the scene format, ECS, physics, streaming and personas are untouched by a backend swap.
- **Backend B** lives at `@mnemopay/augengine/runtime/live/three` with `three` (0.187.0) as an **optional peer**. It is a separate subpath so apps that do not opt in never bundle three.js. This is the architecture review required by contributor rule 3: no runtime payload is added to the default path.
- **Measurement.** `?bench=1` drives a fixed camera path by rendered frame and records per-pass GPU timestamps, CPU time inside `render()`, and rAF intervals. A/B runs are interleaved to cancel drifting machine load (`e2e/renderer-benchmark.spec.ts`, `BENCH_BACKEND=ab`).
- **Decision (Intel UHD, 2026-11-23): backend A stays the default.** B was 15% (proof room) and 49–65% (Loading Range) slower on GPU, and 2.9–3.5× slower on CPU per frame, at matching image quality, against an adoption bar of ≤10%. Details and raw captures: `results/renderer-benchmark/SCORECARD.md`. B stays in tree as the opt-in path for experiments that need three's node library (TRAA/TAAU, GTAO, SSGI, splats via Spark); re-run the benchmark before revisiting.
- **Step 2 on A.** Tonemapper choice (ACES default / Khronos PBR Neutral / AgX), a display grade (contrast, vibrance, lift), and exponential height fog (`fogHeightFalloff`). The smoke-game's `?quality=high` bundles them with high shadows; the default stays the UHD baseline.
- **Temporal AA / upscaling on A (2026-11-23).** `setImageQuality()` (optional on `LiveRenderer`; B does not implement it) adds TAAU: the main pass renders at `renderScale` (0.7 default when on) with a Halton(2,3) projection jitter, and one full-resolution pass (`runtime/live/webgpu/temporal.ts`) reprojects the history by depth for static geometry, clips it to the current 5-tap YCoCg neighbourhood, blends with luma weighting, and in the same pass composites bloom, tonemaps, grades and writes the canvas (MRT: history + display). Fusing the tonemap is what makes it pay on Intel UHD, where every full-resolution pass costs ~2 ms. Optional half-resolution depth-only SSAO (4×4 depth-aware blur, accumulated in the history's alpha) and an opt-in sky-shaped ambient (`ibl.ts`, zonal SH of the sky gradient normalised to the flat ambient) ride on the same call. TAA needs 1× MSAA and a sampleable depth buffer. Moving, skinned and morphing meshes get per-object motion vectors from a small extra pass (`motion.ts`, ~0.2 ms; only moving draws are redrawn, against the main depth with `@invariant` positions), so TAA tracks them instead of clipping them; the proof room now runs TAA at full render scale. Also on A: an optional second key-light shadow cascade (`setShadowCascades(2)`, far map 4× the near range in a 2:1 atlas, high preset), split-sum sky specular on PBR (`skySpecular`, analytic: zonal-SH prefiltered sky + Karis env-BRDF fit, opt-in), and distance-faded SSAO. Streamed irradiance probes (2026-11-24, opt-in): each Loading Range tile bakes a 4×4×2 L0+L1 probe grid on load (`runtime/world/irradiance-probes.ts`, idle tasks), streamed into a fixed storage ring (`probe-ring.ts`) the PBR/cel/toon shaders read per vertex; off by default (+0.4 ms on UHD, barely visible on that level). Not yet: reflection probes, lightmaps, motion vectors for instanced draws. Measurements: `results/renderer-benchmark/SCORECARD.md`.

---

## Character system (the hard piece)

Mocap retargeting is the single piece that makes characters work in both games and film. The math is shared.

### Skinning

Standard glTF 2.0: flat joint array + `inverseBindMatrices` accessor + `JOINTS_0`/`WEIGHTS_0` per vertex. Per joint per frame:

```
skinMatrix = Σ_j  weight[j] · (globalJointTransform[j] · inverseBindMatrix[j])
worldPos   = modelMatrix · skinMatrix · vertexPos
```

WebGPU compute shader computes joint matrices once, vertex shader reads from storage buffer. ~50 LOC of shader, ~150 LOC of CPU.

### Retargeting (~500 LOC MVP)

Three phases:

**Phase 0 — Bind alignment (one-time per pair):** Force both rigs to canonical T-pose by computing per-joint delta quaternions; cache `bindSrcWorld`, `bindTrgWorld`, parent transforms. Compute uniform leg-length scale ratio.

**Phase 1 — Per-frame retarget** (the formula):

```
trgLocalQ = invBindTrgParentQ_i · bindSrcParentQ_i · srcLocalQ · invBindSrcQ_i · bindTrgQ_i
```

For each mapped joint, parent-first traversal. Quaternion-only, ignore per-joint translation except root.

**Phase 2 — Root motion:** scale source hip displacement by leg-length ratio, apply to target hip world position.

**Phase 3 — IK foot-lock fixup:** when source foot is planted (height < ε), pin target foot to last planted position, run two-bone analytic IK on (hip, knee, ankle) using cosine-rule.

Edge cases — A-pose vs T-pose, mirrored skeletons, mismatched joint counts, mismatched proportions, root yaw drift — are handled in Phase 0 with documented heuristics. See `references/retargeting-threejs/Algorithm.md` for the precedent we're replicating.

### VRM 1.0 expressions + spring bones

- Humanoid mapping: 54 named bones from `VRMC_vrm`.
- Expressions: 4 categories (Emotion / LipSync / Blink / LookAt), each a named morph-target preset.
- Spring bones: Verlet-integrated chain with stiffness/drag/gravity/hitRadius. ~200 LOC. Sphere/capsule/plane colliders.
- Constraints: roll/aim/rotation from `VRMC_node_constraint`.

### AI lip-sync — two paths

**Path A (default, native, ~5 ms per frame):** FFT in AudioWorklet → mel filterbank → MFCC → Mahalanobis distance against viseme prototypes → Oculus 15-viseme blend-shape values. Quality ceiling: "great for VTubing, not film." Implementation reference: `met4citizen/HeadAudio`.

**Path B (premium, optional, port of NVIDIA Audio2Face-3D):** Audio2Face-3D was open-sourced September 2025 (MIT code + NVIDIA Open Model License for weights). ONNX export ~80 MB; ports to ONNX Runtime Web with WebGPU at ~30 fps regression of 52 ARKit blendshapes. **This is the 2026 native opportunity.** Phase 4 stretch.

### What stays as API call

- **Text-to-motion** (MDM, MotionGPT3, Lodge) — diffusion is too slow for runtime; asset-pipeline batch only via Replicate/Fal. License-encumbered (HumanML3D research-only) so we treat outputs as user-owned per-call.
- **Music-to-dance** (EDGE) — same, plus stricter dataset license (Motorica research-only). API-only.
- **Talking-head video** (Wav2Lip / MuseTalk / SCAIL) — server-side only. UGC mode wires Replicate/Fal endpoints.
- **Character video gen from one image** (Animate-X, StableAnimator++) — same.
- **Premium TTS** (Inworld, ElevenLabs, Azure) — remote. Native fallback uses WebGPU TTS (HeadTTS / Kokoro ~80 MB) when latency budget allows.
- **LLM brain** — Anthropic / OpenAI / WebLLM in-browser. Routed through MnemoPay's `recall` for memory persistence.

---

## AI integration surface

The seven AI-native design decisions above describe the *philosophy*. The concrete API surface:

```ts
// Engine-side
class AIScheduler {
  dispatch<T>(req: AIRequest<T>): Promise<T>;
  // Frame-budgeted, deadline-driven, with deterministic fallback.
}

// Per entity
type PersonaComponent = Persona; // see decision #2

// Authoring
class PromptTemplate {
  static load(path: string): PromptTemplate;
  render(slots: Record<string, string>, persona?: Persona, world?: WorldSnapshot): string;
}

// Action IR
class ActionDispatcher {
  dispatch(action: Action): ActionResult;
  validate(action: Action): ValidationResult;
}

// Observation bus
class ObservationBus {
  publish<T>(channel: ChannelTag, event: T): void;
  subscribe<T>(channel: ChannelTag, filter: ViewFilter, cb: (e: T) => void): Unsubscribe;
}

// Memory (delegated to MnemoPay)
import { recall, remember } from '@mnemopay/sdk';
// Persona memory key = portable persona memory across titles
```

External services used:

- **Anthropic / OpenAI / WebLLM** — LLM brain (per `Persona`)
- **Inworld / ElevenLabs / Azure / Kokoro WebGPU** — TTS (per `Persona.voice`)
- **Whisper API or Whisper-tiny WebGPU** — STT
- **MnemoPay SDK** — persistent persona memory (the Mantella pattern done right: SQLite-backed character facts injected per prompt, not LLM context-window memory)
- **NVIDIA Audio2Face-3D ONNX** (phase 4) — high-quality blendshape lip-sync

---

## Roadmap

The original phase plan (MVP runtime, character system, AI integration, networking, UI,
visual fidelity, audio, physics tier 2, i18n/telemetry/bundler) is tracked release by
release in the [CHANGELOG](https://github.com/mnemopay/augengine/blob/main/packages/augengine/CHANGELOG.md), with measured status in
[`results/`](https://github.com/mnemopay/augengine/tree/main/results). Gates that are still open are listed in the
[README](https://github.com/mnemopay/augengine/blob/main/README.md#where-things-stand-the-gates).

---

## AugEngine as an MCP server (`packages/augengine-mcp`)

AugEngine exposes itself over the Model Context Protocol so any MCP-aware agent (Claude, Cursor, ChatGPT desktop, custom) can author and inspect scenes. Decision #7 (observation bus) is already the right shape for an MCP surface: exposing its typed channels as MCP tools is mostly a routing layer, not new architecture.

Tools (names exactly as the server lists them): `augengine_create_scene`, `augengine_add_entity`, `augengine_attach_persona`, `augengine_list_entities`, `augengine_describe_scene`, `augengine_dispatch_action`, `augengine_subscribe_observation`, `augengine_poll_observation`, `augengine_export_godot`, `augengine_render_clip` (returns an explicit error over stdio; it needs a browser host), and `detective_new_session` / `detective_status` / `detective_act` for playing the detective game as an agent.

Run it after `npm run build`: `node packages/augengine-mcp/dist/cli.js serve --stdio`.

### What about Fortnite + Roblox?

Out of scope as ship targets. Both are walled gardens (UEFN-only and Roblox-Studio-only). AugEngine is **upstream** of those platforms — generate assets in AugEngine (GLB / animations), import them elsewhere. The MCP server's Godot export (`augengine_export_godot`) is the pattern for "export this scene to a different engine".

---

## Non-goals (deliberately)

- Not Unreal/Unity scope as a whole — no native console targets (export to Godot instead, see `augengine_export_godot` in the MCP server). The editor (above) is deliberately small: scene layout, import, animation preview — not a full DCC tool.
- Not a competitor to Roblox/Bevy — web-first, charter-authored, single-player and small-multiplayer scope.
- Not generative AI as window dressing — see decision #3 (every line is a template) and #4 (artifacts are content-addressed). If we can't ship determinism + multiplayer parity, we don't ship the AI feature.
- Not "the next Three.js" — we're not making a general-purpose graphics library. AugEngine does specifically what's needed for games + UGC + films and nothing else.

---

## Citations + reference reads

Research findings that drove the design:

- WebGPU adoption + perf — `caniuse.com/webgpu`, `volumeshader.dev/en/blog/browser-gpu-benchmark-report-2026`, `kaelan.fyi/research/webgpu-future-roadmap/` (accessed 2026-05-08)
- WebCodecs benchmarks — `github.com/remotion-dev/webcodecs-benchmark`, `webcodecsfundamentals.org/datasets/codec-analysis-2026/` (accessed 2026-05-08)
- Web Audio determinism — `padenot.github.io/web-audio-perf/`, `w3.org/TR/webaudio-1.1/` (accessed 2026-05-08)
- Mocap retargeting — `github.com/upf-gti/retargeting-threejs/blob/main/docs/Algorithm.md`, Unreal IK Rig docs, `arxiv.org/html/2512.10881v2` (accessed 2026-05-08)
- VRM + glTF — `github.com/vrm-c/vrm-specification`, `github.com/pixiv/three-vrm`, `github.khronos.org/glTF-Tutorials` (accessed 2026-05-08)
- Audio2Face-3D — `github.com/NVIDIA/Audio2Face-3D-SDK`, `huggingface.co/nvidia/Audio2Face-3D-v3.0` (accessed 2026-05-08)
- AI engine patterns — `inworld.ai/blog/introducing-unreal-ai-runtime-sdk`, NVIDIA ACE In-Game Inference SDK, SIMA 2 paper `arxiv.org/abs/2512.04797`, Mantella pattern `github.com/art-from-the-machine/Mantella` (accessed 2026-05-08)

---

## What's expected of every AugEngine contributor

1. Read this doc end-to-end before opening a PR.
2. Respect the seven AI-native design decisions. They are not negotiable without an architecture-doc revision.
3. No third-party runtime dependency without an architecture review. (Build-time deps are fine; runtime payload bytes are sacred.)
4. Determinism is opt-in, AI is async, the action grammar is symmetric. If you find yourself writing a special case for "AI vs human" or "game vs film," you're working against the grain — stop and ask.


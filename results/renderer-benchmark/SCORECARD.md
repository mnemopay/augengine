# Renderer A/B benchmark — 2026-11-23

**Decision: keep backend A (AugEngine's WebGPU renderer) as the default.** Backend B (three.js 0.187.0 WebGPURenderer behind the same `LiveRenderer` adapter) failed the adoption bar (GPU frame time within ~10% of A at equal or better quality) in both scenes: +15% (proof room) and +49–65% (Loading Range) GPU, 2.9–3.5× CPU per frame, at matching image quality. B stays in tree, opt-in via `?renderer=three`.

## Setup

- Hardware: Intel UHD Graphics gen-11 (device 0x8a56, real adapter, not a fallback), Chrome 148 (headed, system channel), 1440×900 canvas, 60 Hz panel.
- MSAA: A resolves `sampleCount: "auto"` to 1× on this GPU. B never uses MSAA (`antialias: false`). Both use FXAA in the "full" chain.
- Scenes:
  - **Proof room**: lit characters, Maya glTF skinned with morph targets, proof-head bust, key/fill/rim + ambient, 1024² shadows, "tonemap" post chain.
  - **Loading Range**: streamed 1 km level, instancing on, 1024² shadows, "full" post chain (bloom + FXAA).
- Camera path: `?bench=1` (examples/smoke-game/src/bench.ts) moves the player camera by rendered-frame index, so both backends draw the same views. It runs 180 warm-up frames, then measures 600. Overlays are hidden during the run, because the HUD's CSS backdrop blur is a compositor cost that would land on whichever backend is measured.
- Metrics:
  - `gpu_sum_ms`: sum of per-pass GPU timestamp durations. A times skin, shadow, main, water and post separately. three resolves its own per-pass queries and sums them for the frame.
  - `cpu_render_ms`: wall time inside `render()` (scene sync, encode, submit).
  - `raf_ms`: interval between frames. It is vsync-quantised (16.7 / 33.3 ms) on this panel, so the GPU numbers are the vsync-independent comparison.
- **Machine load:** the laptop was shared with other agents running their own Playwright and Vite jobs. Single runs drifted by up to 40%. The decision therefore rests on **interleaved** runs (`BENCH_BACKEND=ab`: A,B / B,A / A,B / B,A), reported as the mean and median of the per-run p50s.

## Result (final, interleaved, current master incl. the art kit #11)

Clean set: 4 independent runs per backend per scene for the proof room and the Loading Range, plus a 3-round set for the Loading Range after adding vertex-coloured instancing to B. Every run has a distinct timestamp in the JSON.

| Scene | Metric | A (webgpu) | B (three) | B vs A |
|---|---|---|---|---|
| Proof room | GPU sum p50 (mean / median of runs) | **7.12 / 6.98 ms** | 8.19 / 8.39 ms | **+15% / +20%** |
| | GPU sum p95 (mean) | 7.87 ms | 10.33 ms | +31% |
| | CPU in render() p50 (mean) | 2.15 ms | 6.22 ms | 2.9× |
| | rAF p95 (mean) | 18.1 ms | 20.3 ms | |
| Loading Range | GPU sum p50 (mean) | **11.66 ms** | 17.40 ms | **+49%** |
| | GPU sum p95 (mean) | 13.24 ms | 19.34 ms | +46% |
| | CPU in render() p50 (mean) | 3.8 ms | 13.3 ms | 3.5× |
| | rAF p95 (mean) | 29.6 ms | 36.2 ms | |
| Loading Range, B instancing vertex-coloured kit meshes too (3 rounds) | GPU sum p50 (mean) | **9.98 ms** | 16.50 ms | **+65%** |
| | CPU p50 (mean) | 3.7 ms | 10.9 ms | 2.9× |

Per-run GPU sum p50 (ms):
- Proof room: A 6.92, 6.89, 7.03, 7.63; B 6.52, 9.46, 7.32, 9.47. B is also more variable run to run.
- Loading Range: A 11.14, 11.98, 11.64, 11.90; B 16.59, 18.75, 16.01, 18.25.
- Loading Range with vertex-coloured instancing in B: A 8.27, 10.97, 10.70; B 16.28, 16.72, 16.50.

Raw captures:
- `2026-11-23-ab-final-proof-room.json`
- `2026-11-23-ab-final-stream-level.json`
- `2026-11-23-ab-final-vcinst-stream-level.json`

### Superseded sets (kept, do not use)

These files had a **harness bug**:
- `2026-11-23-ab-proof-room.json`, `-ab-stream-level.json`
- `-ab-tonemap-*`, `-ab-rebased-*`, `-ab-rebased2-*`

**What went wrong:** two back-to-back runs with the same URL are a same-document hash navigation, which does not reload the page. The second run read the first run's results again, so those sets contain duplicated runs (identical numbers to three decimals).
- **Effect on the decision:** they still showed B slower in every set, by +2% to +53% depending on load. The duplication skewed their medians, so the spread is not trustworthy.
- **Fixed:** the spec now navigates through about:blank with a unique `run=` query per run.

### Attribution

- **B's core path is slower, not just its bloom.** Loading Range with `post=tonemap` on both backends (clean 3-round set, `2026-11-23-ab-final-tonemap-stream-level.json`):
  - GPU sum p50 mean: A 6.71 ms, B 12.12 ms (+81%).
  - CPU p50 mean: A 3.7 ms, B 7.3 ms.
  - three's BloomNode costs more than A's half-res blur, but the gap is already there without it.
- **Draw calls:** after vertex-coloured instancing, B submits 65 draw calls per frame across all passes, against A's 38 main-pass draws. These counts are not directly comparable (B counts shadow, background and post).
- **Not profiled:** the CPU side of B (scene mirror vs three's render lists) and B's per-pass GPU split, because three sums its passes without labels. Both are the obvious next steps if B is ever revisited.
- **Shadows-off pair** (single runs, noisy; `*-noshadow-proof-room.json`): GPU sum p50 A 5.56 ms, B 6.93 ms.

### Image parity

Stills at the same camera spots are near-identical in layout, lighting and fog. They cover the primitive level (`2026-11-23-webgpu-*.png` vs `2026-11-23-three-*.png`), the art-kit level after #11 (`*-kit-stream-level.png`, including B's instanced vertex-coloured kit meshes) and the proof-room bust (`bust-webgpu.png` / `bust-three.png`). Remaining differences:
- B's sky is slightly darker.
- A's ACES "saturation" knob has no three equivalent, so B's colour is a little more saturated.
- Local-light falloff differs within ~1 m of a light: A uses window²/(d²+1), three uses window²/d².
- B does not implement GPU particles, shallow water, or the outline material (drawn unlit). Those scenes were not benchmarked.

**Equal quality, worse time, so B is not adopted.**

## Step 2 on A (the adoption bar failed, so the quality work went into A)

| Scene | Preset | GPU sum p50 / p95 | Per-pass p50 (ms) |
|---|---|---|---|
| Proof room | default (UHD baseline) | 6.94 / 9.95 | skin 0.20, shadow 0.10, main 4.86, post 1.85 |
| Proof room | `quality=high` | 9.90 / 10.87 | skin 0.25, shadow 0.43, main 6.89, post 2.35 |
| Loading Range | default | 11.70 / 13.12 | shadow 0.85, main 5.47, post 5.37 |
| Loading Range | `quality=high` | 14.70 / 19.43 | shadow 0.76, main 7.67, post 5.96 |

These are single runs, not interleaved, captured before the rebase onto #11 (primitive Loading Range), so read the deltas as ±1–2 ms.

`quality=high` combines:
- AgX tonemap;
- a display grade (contrast 1.08, vibrance 1.12, lift 0.01);
- 2048² shadows with 3×3 PCF;
- exponential height fog.

The main-pass increase is the 3×3 PCF shadow taps in the lit shader.

**Honest look assessment** (stills `2026-11-23-webgpu-low-*.png` vs `2026-11-23-webgpu-high-*.png`):
- AgX mutes the saturated cyan and pink, and shadow edges are a little crisper.
- The Loading Range is nearly indistinguishable.
- It is a different look, not a clearly better one, and it costs about +3 ms GPU.
- It stays opt-in. The tonemapper is also selectable on its own (`?look=neutral|agx`).

## What was not done (from the step-2 list)

- **TRAA/TAAU, 0.7 render scale + upscale + sharpen.** A has no velocity buffer or jitter/history path. three ships `TRAANode`, `TAAUNode`, `FSR1Node` and `SharpenNode`, but B lost the A/B. Building TAA into A is the next real image-quality lever, and the largest remaining piece.
- **Baked/probe lighting, 2-cascade PCF, half-res GTAO.**
  - A has a hemisphere IBL proxy and one fitted shadow map.
  - None of baked/probe lighting, a second cascade or GTAO was built here.
  - GTAO needs a normal/depth prepass A does not have yet.
- **SSGI/VXGI high tier.** Not built. They exist in three r184+ (WebGPU only), so they would be a B-only feature.
- **A gate-1 claim.** This capture is the floor GPU only.

## Reproduce

```
# dev server
cd examples/smoke-game && npx vite --port 5191 --strictPort
# from repo root, headed Chrome on the real GPU
BASE_URL=http://localhost:5191 BENCH_BACKEND=ab BENCH_ROUNDS=3 npx playwright test e2e/renderer-benchmark.spec.ts --project=chromium-webgpu-headed
BASE_URL=http://localhost:5191 BENCH_BACKEND=webgpu BENCH_TAG=high BENCH_QUERY=quality=high npx playwright test e2e/renderer-benchmark.spec.ts --project=chromium-webgpu-headed
```


---

# Image-quality pass on backend A — 2026-11-23 (later)

Built on A (the renderer that stays default): render scale + TAAU + sharpen, half-res SSAO, and a sky-shaped ambient. Cascaded shadows, a probe grid and specular IBL were **not built** in this pass (see the end).

## What shipped and the defaults

| Feature | Default | How to try it |
|---|---|---|
| TAAU: 0.7 render scale, Halton jitter, depth reprojection, 5-tap YCoCg variance clip, fused composite + tonemap | **On** for every map except the proof room | `?aa=taa`, `?aa=fxaa`, `?scale=0.5-1`, `?sharpen=0-1` |
| Half-res SSAO, 4×4 depth-aware blur, accumulated in TAA history | Off, on in `quality=high` (12 samples) | `?ao=1` (needs TAA) |
| Sky-shaped ambient (diffuse IBL, zonal SH L0–L2 of the sky gradient, normalised to the flat ambient) | **Off** | `?ibl=sky` |

## Measurements (Intel UHD gen-11, headed Chrome, 1440×900, GPU sum p50 in ms)

The machine was shared with other agents and drifted (the same baseline read 5.5–6.9 ms on the proof room across the session), so decisions rest on **interleaved** runs, base / TAA / TAA+AO alternated, 2 rounds (`*-iqab-*`):

| Scene | Baseline (full res, FXAA) | TAA 0.7 | TAA 0.7 + AO (6 spp) |
|---|---|---|---|
| Proof room (per round) | 6.30, 5.88 | 7.55, 7.42 (**+1.4**) | 9.40, 8.61 (+2.9) |
| Loading Range (per round) | 8.82, 8.59 | 8.13, 7.88 (**−0.7**) | 9.59, 9.52 (+0.85) |

Final defaults, interleaved FXAA vs default on the Loading Range (`*-iqfinal-*`): FXAA 8.12 / 9.46, default (TAA) 7.63 / 8.13, about **0.9 ms cheaper**.

`quality=high` (TAA + 12-spp AO + the earlier high look), single runs: proof room 9.95 (earlier high preset 9.90), Loading Range 11.35 (earlier high preset 14.70).

Per-pass p50 on the Loading Range, TAA default: main 1.8–1.9 (was 3.6), bloom 0.8 (was 3.9 for bloom + composite + FXAA at full res), TAA resolve incl. tonemap 4.6–4.8. AO costs 1.0–1.2 for the pass + blur, plus ~0.5 inside the resolve.

### What made it affordable

- A first, unfused TAA resolve cost **4.0 ms** on its own, plus a separate full-res composite (~2 ms): proof room +2.8 ms over budget. Neither a 5-tap instead of 9-tap neighbourhood nor a single depth tap changed it (4.29 vs 4.07, 4.63 vs 4.44, inside noise).
- On this GPU every full-resolution pass costs ~2 ms regardless of taps. The fix was structural: **one pass** writes the new history and the tonemapped canvas (MRT), and bloom is built from the 0.7-scale frame. That removed a full-res pass and is the whole saving.

## Honest look assessment

- **Loading Range, TAA:** edges and window frames are smoother and the poles stay intact; slightly softer overall. Stills cannot show TAA's main benefit (less crawling/shimmer in motion). Cheaper than before, so it is the default.
- **Proof room, TAA:** **worse for the character.** At 3× zoom the face is blurrier (0.7 scale + a morphing mouth that has no motion vectors, so the history is clipped rather than tracked), and the first sharpen put a dark halo round the head. The halo is fixed (overshoot limited to ¼σ), the softness is not. The proof room keeps FXAA by default.
- **SSAO:** the raw 6-sample AO showed visible mottled grain; the 4×4 depth-aware blur removed it. Contact shading under the cubes, the dais edge and the characters' feet reads correctly. A faint dark band where the far road meets the buildings at the horizon is the main artefact. At +2 ms it is a high-preset feature, not the UHD default.
- **Sky-shaped ambient:** costs nothing measurable (main pass within noise), but because it keeps the average ambient level and the key light dominates both scenes, the difference is **barely visible**. Shipped opt-in, not default.
- AO multiplies all lighting, not just ambient (applied in the resolve, outside the material shaders). A deliberate compromise.

Stills: `*-iqab-{base,taa,taaao}-r2-*.png`, `*-iq-aoblur-*.png` (AO after blur), `*-iq-fused-proof-room.png` (AO before blur), `*-iq-sharp06-proof-room.png` (halo), `*-iq-sky{b,i}-*.png` (sky ambient off/on), `*-iqfinal-high-*.png`.

After rebasing onto master with terrain/occlusion/NPCs (#13): Loading Range default GPU p50 7.70 ms (main 1.67, bloom 0.83, TAA 4.55), no page errors; `stream-level-extras.spec.ts` 4/4 and `stream-level-and-proof-head.spec.ts` 3/3 pass with TAA on (`*-iq-rebased-default-stream-level.*`).

## Not done (from the step 2–3 list)

- **Second shadow cascade + PCF.** Not built. The frame layout, shadow pass and lit shaders need a second map and matrix; it did not fit after the TAA work.
- **Prefiltered specular IBL, light probes or baked lightmaps.** Not built. Only the diffuse sky-shaped ambient exists.
- **Velocity buffer** for skinned/morph/moving objects. Without it TAA softens animated characters; that is the next step if TAA is to cover the proof room.
- **Gate 1** (16.7 ms on a shipping GPU) is still not claimed; all numbers are the floor GPU.

## Reproduce

```
cd examples/smoke-game && npx vite --port 5191 --strictPort
BASE_URL=http://localhost:5191 BENCH_BACKEND=webgpu BENCH_TAG=x BENCH_QUERY="aa=taa&scale=0.7" npx playwright test e2e/renderer-benchmark.spec.ts --project=chromium-webgpu-headed
```

---

# 2026-11-23 (iq2) — finishing the step 2–3 list on backend A

Intel UHD gen-11, headed Chrome 148, 1440×900, `?bench=1` (180 warm-up + 600 measured frames), per-pass GPU timestamps, interleaved runs (r1/r2) because machine load drifts. All numbers are GPU p50 in ms. Files: `2026-11-23-webgpu-iq2-*`, `face-ab/`, `iq2-*.png`.

| Item | Result | Kept? |
|---|---|---|
| 1. Per-object velocity (`motion.ts`) for skinned / morph / moved meshes; TAA reprojects them along their own motion | Motion pass **0.16–0.22 ms** (proof room). Static pixels keep exact depth reprojection. | **On** with TAA (`?mv=0` for A/B) |
| 1b. TAA in the proof room | TAA at **full render scale** (0.7 still softens the eyes): gpu_sum 9.7–10.2 vs FXAA 6.2–8.0 → ~**+2 ms** | **Default on** in the proof room |
| 2. Second shadow cascade (far map 4× the near range, 2:1 atlas, 15% blend, texel-snapped) | Shadow pass 0.59–0.70 → **1.08–1.55 ms** (+0.5–0.85, Loading Range) | `quality=high` only (`?cascades=2`) — over the ~0.7 ms low bar |
| 3a. Split-sum sky specular on PBR (`ibl.ts`): zonal-SH prefiltered sky + Karis analytic env-BRDF | Main pass +0.4 ms proof room, +0.1–0.2 ms Loading Range | **Off** (opt-in `?spec=1`) |
| 3b. Irradiance probe grid per tile | **Built 2026-11-24** (see below): +0.38 ms main pass, barely visible here | Off (opt-in `?probes=1`) |
| 4. SSAO far-band fix (distance-scaled bias + fade 40→70 m) | Band where the far road meets buildings gone (A/B diff map); AO pass 0.88 ms (unchanged cost class) | **On** whenever AO is on |

## Honest visual verdicts

- **Talking face (face-ab/face-grid.png, 3× crops):** TAA (with or without motion vectors) gives clean silhouettes where FXAA leaves stair-steps on the head and plinth; eyes and mouth read about the same. **Motion vectors made no measurable difference to face sharpness at this camera distance** (edge variance 916 vs 917): the mouth moves < 1 px/frame, so the old softness was TAA resampling, not trailing. A 5-tap Catmull-Rom history fetch was tried (face-ab/catmull-rom-trial/) and also made no visible or measured difference — reverted (kept as `sampleHistoryCR` for A/B). The edge-variance metric counts FXAA's jaggies as "detail", so it cannot separate sharpness from aliasing; the judgement above is from looking.
- **Second cascade (iq2-cascades-c1-vs-c2.png):** distant trees and lamp posts get ground shadows that the single map leaves out. Real but subtle.
- **Sky specular (iq2-spec-*.png):** on these matte primitive materials every dielectric gets a uniform sheen — floor and cubes wash out and lose contrast. Not an improvement for the current art; useful only once there are glossy materials. Off by default.
- **SSAO band (iq2-ssao-band-ab.png):** top = old constants, middle = fixed, bottom = 8× difference. The bright line on the difference map is exactly the old dark band at the road/building junction; AO now also fades out on distant buildings by design.

## Cost notes

- The fused TAA resolve is ~5 ms on UHD at any render scale (it is full-resolution and also tonemaps); TAA at 0.7 costs about the same as the FXAA chain overall because the main pass gets cheaper.
- Frame uniform grew 768 → 896 bytes (far cascade matrix/params + sky-specular SH).

## Still not done

- Irradiance/reflection probe grid baked per tile; baked lightmaps.
- Motion vectors for instanced draws (assumed static) and for a mesh that is both skinned and morphed (uses current skinned positions + previous weights).
- Gate 1 (16.7 ms on a shipping GPU) is not claimed.

---

# 2026-11-24 — irradiance probe grid (the item iq2 left)

Built on backend A: per-tile L0+L1 irradiance probes baked on load, streamed
into a fixed 128 KB storage ring (`probe-ring.ts`, group(0) binding 3), read
per vertex in the PBR / cel / toon pipelines. Full write-up and numbers in
`results/stream-level/SCORECARD.md` (2026-11-24). Summary: main pass **+0.38 ms**
GPU p50 on UHD (per-pixel sampling was +3.0 ms and was rejected), no new
fast-travel frames over 50 ms, visual change barely visible on the Loading
Range → **off by default**, opt-in `?probes=1`. Frame uniform 896 → 912 bytes
(probe grid params).

Still not done: reflection probes, baked lightmaps, motion vectors for
instanced draws.

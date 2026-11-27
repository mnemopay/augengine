# Character Proof Room — measured results and Unreal Engine 5 scorecard

Measured 2026-11-20 on a LOW-END baseline: Intel Core i5-1035G1, 7.8 GB RAM,
Intel UHD Graphics (Ice Lake gen-11, integrated). Chrome 148, WebGPU, real
(non-software) adapter. Raw data: `2026-11-20-intel-uhd-gen11.json`. Stills:
`2026-11-20-frame-3.png`, `2026-11-20-frame-6.png`.

**AugEngine does not beat Unreal Engine 5.** It wins on reach (a link that
runs in a browser on a laptop like this) and on persistent NPC memory. It is
behind on every rendering, tooling and content axis.

## Measured on this machine

Capture: `npx playwright test --project=chromium-webgpu-headed proof-room-capture`
with the smoke-game dev server on :5173. One 600-frame window after DEPLOY,
player camera facing the dais.

| Metric | Value |
|---|---|
| GPU main pass (timestamp-query) | 5.43 ms, p95 6.35 ms |
| CPU frame p50 / p95 / p99 / max | 16.1 / 81.7 / 333 / 1580 ms |
| Delivered FPS over the window | 29.4 (288 of 600 frames over 16.7 ms) |
| Page load (DOMContentLoaded / load) | 2.82 s / 2.85 s |
| Dev-server transfer | 18.0 MB over 237 requests (unbundled dev mode) |
| WebGPU validation errors / console errors | 0 / 0 |

Reading: the GPU has headroom (5.4 ms of a 16.7 ms budget). The frame tail is
CPU-side: p95 82 ms misses both 60 and 30 FPS. Other agents' test runs were
active on this laptop during the capture, so the tail is not an idle number;
it still has to be profiled before any 60 FPS claim. The production bundle is
224 kB JS+CSS+HTML (PERFORMANCE.md, 2026-05-16); the 18 MB above is dev mode
plus assets.

### After branch `render/shadows-lights-perf` (re-measured 2026-11-20 20:36 UTC)

Same laptop, same dev server. Scene lights, one key-light shadow map ("low":
1024², 2×2 PCF), textured skinned Maya, one ACES pass (proof room only), HUD
writes only on change, proof-room physics tick off, unchanged uniforms not
re-uploaded. Window: `e2e/proof-room-profile.spec.ts` (599 frames, no
screenshots). Raw: `2026-11-20-shadows-lights-profile.json`. Still:
`2026-11-20-shadows-lights-frame-3.png`. CPU load ~27% from other jobs.

| Metric | Before | After |
|---|---|---|
| Delivered FPS | 29.4 | 57.5 |
| CPU frame p50 / p95 / p99 / max | 16.1 / 81.7 / 333 / 1580 ms | 16.7 / 17.0 / 17.3 / 433 ms |
| Frames over 33 ms | 288 of 600 over 16.7 ms | 1 of 599 |
| GPU pass mean / p95 (timestamp-query) | 5.43 / 6.35 ms | 7.25 / 11.84 ms (shadow pass added) |

p95 17.0 ms is still 0.3 ms over the 16.7 ms (60 FPS) gate, with one 433 ms
stall in the window. Grok's run of the same spec on 2302a18 reported 58.5 fps
and p95 17.4 ms. The 6 s screenshot capture (`proof-room-capture.spec.ts`)
reads lower (33.1 fps, p95 50 ms) because it includes screenshot stalls.

Visual: cast shadows now land on the floor and the characters are lit from a
direction, but both are still low-poly stand-ins; Kael is boxes.

## Proof Room criteria (from AUGENGINE-P0-OWNER-PROPOSAL.md and the public site)

| Criterion | Status |
|---|---|
| Runs on real WebGPU hardware, no validation errors | PASS (after fixing the timestamp readback bug) |
| Measured frame timing on real GPU hardware | PASS (numbers above, real run) |
| Textured PBR render path (base colour, metal-rough, normal) | PASS in the renderer; NOT exercised by the room's characters |
| Morph-target facial path | PASS in the renderer (52-slot ARKit layout); the Maya asset has only 3 morphs |
| A textured, rigged, morph-capable character in the room | FAIL: Kael is box primitives; Maya is a 3-joint, 3-morph, untextured GLB |
| 60 FPS on the floor GPU | FAIL: p95 82 ms |

The remaining gap is an asset, not engine code: one CC-BY character with
base-colour/metal-rough/normal textures, a skeleton and ARKit blendshapes.

## Scorecard vs Unreal Engine 5

Epic sources fetched 2026-11-20: Nanite and Lumen docs, hardware specs page.
The licence/royalty page returned 403 to automated fetches, so that row is
from prior knowledge and unverified.

| Area | Unreal Engine 5 | AugEngine today | Verdict |
|---|---|---|---|
| Geometry detail | Nanite virtualized geometry, pixel-scale detail, up to 16M instances (Epic docs) | Conventional meshes; Gaussian splats in WebGPU, native splat drawing is a scaffold | Unreal ahead |
| Global illumination | Lumen dynamic GI and reflections (Epic docs) | Constant ambient colour labelled "IBL proxy" | Unreal ahead |
| Direct lights and shadows | Many dynamic lights, shadow maps | One directional light; scene light entities are not read; contact-shadow discs only | Unreal ahead |
| Materials | Node-based material editor | GGX PBR with base-colour/metal-rough/normal textures, cel, MToon | Unreal ahead |
| Characters and faces | MetaHuman pipeline | 52-slot morph path, skinning; demo asset 3 joints / 3 morphs | Unreal ahead |
| Post-processing | Full cinematic stack | Bloom, tone mapping, blur, edge AA | Unreal ahead |
| VFX | Niagara | GPU compute particles (fire, smoke, embers), combustion flame, water spray | Unreal ahead |
| Water and fluids | Water system | Shallow-water compute simulation + water surface | Unreal ahead |
| Physics | Chaos (rigid, destruction, cloth) | Rapier rigid bodies | Unreal ahead |
| Audio | MetaSounds, spatial audio | Web Audio HRTF spatial mixer (tested, not yet wired into the room) | Unreal ahead |
| Editor and tooling | Unreal Editor | Browser Studio, early | Unreal ahead |
| Ecosystem | Fab/Marketplace, large talent pool | None | Unreal ahead |
| Hardware floor | Editor recommends 32 GB RAM, 8 GB VRAM, DirectX 12 (Epic specs page) | This room's GPU pass is 5.4 ms on a 7.8 GB laptop with integrated graphics | AugEngine ahead |
| Time to play | Download and install | Open a link; ~2.8 s to load here | AugEngine ahead |
| Platform reach | PC, consoles, mobile, native | Any WebGPU browser; no consoles | Split |
| Persistent NPC memory | No built-in persistent LLM memory | MnemoPay remember/recall persona memory that survives reloads | AugEngine ahead |
| Licensing cost | Royalty above a revenue threshold (unverified this session) | Own code | AugEngine ahead (unverified) |

Count: Unreal ahead 12, AugEngine ahead 4, split 1, parity 0.

Biggest gaps, in order: (1) lighting — no real GI, one light, no shadow
maps; (2) content — no production character asset; (3) tooling — no
editor at Unreal's level. Where AugEngine genuinely wins: zero-install
browser play on low-end hardware, and NPCs whose memory persists.

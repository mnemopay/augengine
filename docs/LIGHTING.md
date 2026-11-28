# Natural lighting

AugEngine's WebGPU renderer has an opt-in, physically based natural-light tier. A real sun and sky (or an HDR photo of one) lights the scene in photometric units. It drives the key light, the visible sky, diffuse and specular ambient, bounced light, shadows and haze, and a physical camera exposes the result.

The tier is **off by default**. A scene that never calls `setNaturalLight` renders exactly as before.

Everything here was measured on an **Intel UHD (gen-11) laptop, Chrome 148, 1440×900** in November 2026. Other agents shared that machine, so single benchmark runs drift by up to ±30% in GPU time. Every number below names its run.

## Use it

```ts
import { createLiveRenderer, loadRadianceHdr } from "@mnemopay/augengine/runtime/live";

const r = createLiveRenderer(canvas);
await r.init();
r.setLightingPreset("medium");            // low | medium | high | ultra
r.setNaturalLight({
  sun: { latitudeDeg: 40, dayOfYear: 172, solarHour: 15 },  // or { elevationDeg, azimuthDeg } / { direction }
  turbidity: 2.5,                           // 1.7 very clear … 10 hazy
  cloudCover: 0,                            // 1 = CIE overcast, sun hidden
  exposure: { mode: "auto", compensation: -0.5 },  // or { mode: "manual", ev100: 14 }
  whiteBalanceK: 6504,
});
// A time-of-day slider only needs to pass the sun:
r.setNaturalLight({ sun: { latitudeDeg: 40, dayOfYear: 172, solarHour: 18.9 } });

// HDR photo instead of the analytic sky:
r.setEnvironmentHdr(await loadRadianceHdr("/hdr/sky.hdr"), { horizontalLux: 60000 });
r.setNaturalLight({ environment: "hdr" });

r.getNaturalLightState(); // { ev100, preExposure, sunDir, sunIlluminanceLux, skyHorizontalLux, … }
r.setNaturalLight(false); // back to the authored lights
```

In the smoke game, the same tier is available through URL parameters:

- `?light=natural&preset=medium&tod=15`
- `&lat= &day= &cloud= &turbidity= &ev= &evcomp= &wb=`
- `&hdr=/hdr/kloofendal_48d_partly_cloudy_puresky_1k.hdr`

The lighting lab is at `examples/smoke-game/lighting-lab.html?scene=parlour|macbeth|furnace`.

**What natural light replaces:**

- The sun becomes the key light, measured in lux.
- The sky replaces the scene's fill lights and ambient lights.
- Scene point and spot lights are display-referred, so they are multiplied by `localLightScale` (default 50; intensity 1 gives about 10 lux at 2 m).
- Emissive and unlit colours stay display-referred (`emissiveScale` 1), so markers and UI-like props keep their look at any exposure.
- Water and particles are not exposed. Their look is still authored for the legacy path (known gap).

## The pieces

| Piece | File | Method | Source |
|---|---|---|---|
| Sun + sky | `sky-model.ts`, `physical-sky.ts` | Preetham clear sky (Perez distribution, turbidity coefficients, absolute zenith luminance) blended with a CIE overcast sky. Sun position from declination + hour angle. Sun illuminance from Rayleigh + Ångström aerosol transmittance along the Kasten–Young air mass. Limb-darkened disc. | Preetham, Shirley & Smits 1999; Moon & Spencer 1942; Krochmann overcast zenith luminance; Kasten & Young 1989 |
| Sky as environment | `env-lighting.ts`, `sh9.ts` | Sky (without the sun disc) is rendered to a cube, then box-downsampled and GGX-prefiltered (importance sampling + PDF-based source mip). Diffuse is order-2 SH (irradiance/π) computed on the CPU from the same model. Re-baked only when the sun or atmosphere changes. | Karis 2013; Colbert & Křivánek 2007; Ramamoorthi & Hanrahan 2001 |
| Specular BRDF | `ibl-brdf.ts`, `pipeline-factory.ts` | Split-sum with a GPU-integrated 64² DFG LUT, plus multiple-scattering compensation. | Fdez-Agüera 2019 |
| Exposure | `exposure.ts`, `auto-exposure.ts` | EV100 (camera or metered). Saturation-based exposure H = 1/(1.2·2^EV). Lighting is pre-exposed. A centre-weighted 64-bin GPU log-luminance histogram (40–90th percentile) is read back asynchronously and adapted at 3 EV/s brightening and 1.5 EV/s darkening. | Lagarde & de Rousiers 2014 |
| White balance | `exposure.ts` | Von Kries gains in linear sRGB from the Planckian locus. | Kim et al. 2002 |
| HDR IBL | `hdr-loader.ts` | Radiance RGBE decoder (adaptive RLE, flat and old RLE, ±Y, EXPOSURE). The sun is extracted and becomes a shadow-casting key light. Same cube/prefilter/SH pipeline as the sky. **No OpenEXR**: convert .exr to .hdr. | Ward 1991 |
| Sun shadows | `lights.ts`, `pipeline-factory.ts` | PCSS over the existing near/far cascades (with blend and normal offset): blocker search, then penumbra = 2·d·tan(sun radius), then PCF. Vogel disc, rotated by per-frame IGN, early-outs. | Fernando 2005 |
| Indirect light | `gi-probes.ts` | DDGI-lite: a camera-centred probe window, ray-traced in compute against scene AABB proxies plus a ground plane, with infinite bounce through history. Stored as L1 irradiance/π in the existing probe ring. Gives specular occlusion. | Majercik et al. 2019 (simplified) |
| Atmosphere | `pipeline-factory.ts`, `sky-model.ts`, `volumetric.ts` | Per-channel aerial extinction from the sky model's own optical depths. In-scatter = horizon sky + sun × Henyey–Greenstein. Optional half-resolution shadow-map ray march for sun shafts. | — |
| Camera post | `bloom-chain.ts`, `post-processing.ts`, `temporal.ts` | Energy-conserving mip-chain bloom (13-tap Karis down, tent up, `mix(scene, bloom, 4%)`). Lens dirt and film grain are included, **off by default**. ACES / AgX / Khronos Neutral tonemappers already existed. | Jimenez 2014 |

### Why GI probes and not SSGI

This was measured, not assumed. On the Loading Range, the existing half-res SSAO alone costs **0.88 ms (6 samples) to 1.26 ms (12)**, plus about 0.5 ms in the TAA resolve. SSGI needs all of that plus colour gathering, ray marching and a denoiser. It also cannot see off-screen sunlit surfaces, which is exactly what breaks when time of day moves the sun.

The probe trace costs **0.10–0.2 ms at medium** (12×12×2 probes, 32 rays, half of them per frame), independent of resolution.

## Presets

| Preset | Env cube | Shadows | GI probes | Shafts | Loading Range fps (median) | GPU sum p50 |
|---|---|---|---|---|---|---|
| low | 32² | 1024², hardware PCF | off | off | **54.1** | 9.9 ms |
| **medium** (default) | 64² | 1024², PCSS 8 taps | 12×12×2, 32 rays, ½ per frame | off | **52.1 / 52.9 / 51.3** (3 runs) | 15.8 / 11.8 / 14.8 ms |
| high | 128² | 2048² × 2 cascades, PCSS 16 | 16×16×2, 32 rays | 12 steps | 28.1 | 34.9 ms |
| ultra | 256² | 2048² × 2 cascades, PCSS 16 | 16×16×2, 64 rays | 24 steps | 27.6 | 33.5 ms |

How these were measured:

- Benchmark: `?bench=1&light=natural&preset=…&tod=15`, 600 frames walking the road after 180 warm-up frames (`e2e/renderer-benchmark.spec.ts`, results in `results/renderer-benchmark/2026-11-28-webgpu-natural-*.json`).
- **Medium kept the ≥ 50 fps median in 3 of 3 runs**, with a 1–3 fps margin.
- For comparison, the legacy look on the same map runs at 54.9 fps before this work and 55.2 after.
- High and ultra are meant for discrete GPUs. On the UHD they run at about 28 fps; the 2048² two-cascade shadow pass alone costs 7–8 ms there.

Per-pass costs at medium (stills and benchmark runs):

| Pass | Cost |
|---|---|
| Exposure histogram | 0.2–0.3 ms |
| GI trace | 0.10–0.24 ms |
| PCSS | +0.6–1.3 ms over fixed PCF in stills (the first version, with a private Poisson table, cost +2.3 ms) |
| Physical bloom | +0.45 ms over threshold bloom |
| Sun shafts (half resolution) | 3.6 ms (7.3 ms at full resolution) |
| Environment bake | only on frames where the sun or atmosphere changed |

## Correctness checks

| Check | Result | Where |
|---|---|---|
| Sky irradiance vs reference | Raw Preetham integrates to **1.75–2.0×** the IES clear-sky horizontal sky illuminance (0.8 + 15.5·√sin α klux). One constant (0.532), fitted at a 45° sun, gives **1.06 / 1.03 / 1.00 / 0.95 / 0.93** for suns at 15/30/45/60/75°. | `natural-light.test.ts` |
| Direct sun | Within ±15% of the IES direct-normal equation (127.5·e^(−0.21 m) klux) for suns ≥ 15° (0.86 at 15°). | `natural-light.test.ts` |
| Sun position | Equinox noon at 40° N gives 50° elevation due south. The Tropic of Cancer at solstice gives overhead. The equator at 06:00 gives the east horizon. | `natural-light.test.ts` |
| DFG | Karis's analytic fit is up to **0.16** off the integrated split-sum, which is why the tier uses a LUT. The LUT's sample count is within 0.01 of a 4× denser integral. | `ibl-brdf.test.ts` |
| White furnace (CPU) | A white metal reflects 1.000 at every roughness with multiple scattering, and 0.41 at roughness 1 without it. A white dielectric reflects between 0.95 and 1.005. | `ibl-brdf.test.ts` |
| White furnace (GPU) | White diffuse, rough-metal and smooth-metal spheres in a uniform environment read **209 = background (209)** after tonemapping. Controls prove the spheres are drawn: at albedo 0.5 they read 166–171, and with multiple scattering off the rough metal reads 203–204. | `lab-furnace.png`, `lab-furnace-control-*.png` |
| SH9 | A uniform environment evaluates exactly. A hemisphere sky gives 1 / 0.5 / ~0 (up / side / down). Equirect projection matches sphere projection. | `natural-light.test.ts` |
| RGBE decoder | Pixel-exact against an encoder for RLE and flat files. A real HDRI (Poly Haven *kloofendal_48d*) yields a sun at **47.9°**; the file is named for a 48° sun. | `hdr-loader.test.ts` |
| ColorChecker neutrals (noon) | Display values white 204, N8 180, N6.5 147, N5 109, N3.5 70, black 43; monotonic, B/R 0.95–0.97. That is slightly warm (a ~5800 K sun against a 6504 K white balance) and about 0.3 EV dark on a front-lit chart, because of the −0.5 EV default tuned for backlit scenes. | `lab-macbeth-noon-final.png` |
| Exposure math | Sunny-16 gives EV100 14.64. Average luminance and EV round-trip with K = 12.5. Histogram percentiles ignore a 1% sun. Adaptation is rate-limited without overshoot. | `natural-light.test.ts` |

## Visual verdict (honest)

Captures are in `results/renderer-benchmark/natural-light/`:

- `stream-level-{dawn,noon,dusk,overcast,hdr}-final.png`
- `lab-parlour-{dawn,noon,dusk,overcast,hdr}-final.png`
- `lab-macbeth-noon-final.png`
- `lab-parlour-noon-nogi-final.png` (GI off, for comparison)

**What reads as real light:**

- The direction and colour of light follow the sun.
- Golden-hour warmth on façades at dawn and dusk.
- Long shadows that are crisp at their base and softer with distance.
- Haze that glows toward a low sun and blues with distance.
- A soft, shadowless overcast.
- A chrome ball that reflects sky, horizon, ground and a small sun glint.
- With GI, the parlour interior seen through the door is dark. Without GI it is lit like open sky.
- The HDR shots look the most natural, because the sky is a photograph and the extracted sun casts the shadows.

**What does not read as a photograph yet:**

- The Loading Range's authored fog (0.012/m, there to hide tile streaming) washes out contrast at every time of day.
- The flat-colour, untextured low-poly assets have no surface detail.
- There is no contact darkening under the tree in overcast, because the 4 m probe spacing is too coarse and SSAO is off at medium.
- The noon parlour shows jagged overhang-shadow streaks under the eave (1024² over 32 m at grazing angles).
- A thin dark line appears on the Loading Range horizon at dawn: the unlit far impostor stays display-referred.

Against a reference street photograph, the *lighting* is plausible. The *image* still looks like a clean CG massing render. More light won't close that gap; assets, materials and fog tuning will.

## Known gaps

- OpenEXR is not decoded.
- Sun shafts are a shadow-map ray march, not a froxel grid. They only use the near cascade.
- GI probes have no per-probe visibility, so thin walls without a probe inside can leak.
- Scene geometry is approximated by up to 256 AABBs. Skinned meshes are not GI occluders.
- Water, particles, the unlit far impostor and emissive markers are not physically exposed.
- Hosek–Wilkie would model low suns better than Preetham. It is not implemented, for its table size.
- **Possible legacy-path cost (unconfirmed):** in the cleanest proof-room run, the main pass with natural light *off* measured 5.17 ms against 4.59 ms before (+12%). The likely cause is the larger shader. The machine load prevented isolating it. The fix is to compile the natural-light code as a pipeline variant.

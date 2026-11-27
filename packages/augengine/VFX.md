# GPU particles — fire, smoke, embers

Implemented in source on 2026-10-24. This is the first particle feature in
this engine that actually puts pixels on screen. It is not a fluid solver and
it is not a complete VFX authoring system.

## What was missing before

The repo already held three particle implementations, none of which rendered:

- `src/physics/particles.ts` — a CPU ring buffer plus WGSL source, never drawn.
- `@mnemopay/augengine-vfx` — a compute dispatch with no draw pass.
- `src/runtime/systems/vfx-system.ts` — spawns one **scene entity** per
  particle, which costs a draw call each and does not scale.

`src/runtime/live/webgpu/particle-pass.ts` closes that gap. The earlier three
are untouched and still do what they did; prefer `ParticlePass` for anything
that needs to be visible.

## Using it

Emitters are owned by the renderer, not by the scene graph — a 4096-particle
flame as scene entities would exhaust the entity and draw-call budget.

```ts
import { FIRE_PRESET, SMOKE_PRESET, EMBER_PRESET } from "@mnemopay/augengine/runtime/live";

renderer.addParticleEmitter("torch", { origin: [0, 0.4, -2] }, FIRE_PRESET);
renderer.addParticleEmitter("torch.smoke", { origin: [0, 0.9, -2] }, SMOKE_PRESET);

// Particles only advance when dt is supplied. Without it they draw frozen.
renderer.render(scene, { dt: deltaSeconds });
```

`renderer.getParticleStats()` returns emitter count, pool bytes and
`estimatedAlive`. **`estimatedAlive` is derived** from rate × lifetime, not
read back from the GPU; do not present it as a measured particle count.

## How it works

Particles live only in a GPU storage buffer, 48 bytes each. A compute pass
steps them; the draw is encoded into the live renderer's existing main render
pass, after opaque geometry:

```
skin compute -> particle compute -> [sky -> opaque -> PARTICLES] -> post
```

Particles are depth-tested against the world but write no depth, so geometry
occludes them and they never occlude each other. Output is premultiplied HDR
into the renderer's `rgba16float` target, so a flame with `intensity > 1` is
picked up by the existing bloom-extract pass — that is what makes fire read as
hot rather than as orange paint.

Emission is GPU-side and allocation-free: particle `i` first spawns at
`i / emissionRate` and respawns on a `capacity / emissionRate` period. No
atomics, no CPU re-seed loop, no `Math.random()` on the frame path. When
`lifetime > capacity / emissionRate` the pool saturates and the effective rate
clamps to `capacity / lifetime` — pool-limited, by design.

Turbulence uses an ABC flow (`sin z + cos y`, `sin x + cos z`, `sin y + cos x`),
which is exactly divergence-free, so particles swirl without clumping or
tearing voids. Temperature and size growth are **derived** from remaining life
each frame rather than integrated, so neither can drift over a long run.

Determinism: the RNG is a hash of (particle index, frame counter, emitter
seed), so replaying the same dt sequence reproduces the same simulation **on
the same device**. Float results are not bit-identical across GPU vendors, so
this is *not* cross-device deterministic and must not be used as lockstep
multiplayer input.

Memory: `capacity * 48` bytes per emitter. `PLAYABLE_PARTICLE_CAPACITY`
(16384 → 786 KB) is ~2.5 % of the 32 MB GPU ceiling in `content/budget.ts`;
`MAX_PARTICLE_CAPACITY` (65536 → 3.1 MB) is the hard cap.

## Driving it from combustion

`flameFromCombustion()` maps a burning body's temperature and fuel consumption
rate onto emitter knobs, so the flame grows, brightens and dies with the
chemistry in `src/chemistry`. It takes plain numbers, not a
`ReactiveComponent`, so the renderer never imports the chemistry module.

The mapping is **authored art direction, not physics**. Real flame luminosity
comes from soot radiation and a temperature⁴ term this does not evaluate, and
flame height comes from a buoyant plume this does not solve. Nothing here is
calibrated against measured flame data. Its only claim is that it responds
monotonically to the simulated state, and that a stalled reaction (for example
exhausted oxygen) puts the flame out even while the body is still hot.

## Not implemented

Do not describe any of these as working:

- **Depth-buffer soft particles.** Fading against scene depth needs the depth
  texture as a sampled input, which it cannot be while it is the pass's depth
  attachment. "Soft" here means a radial alpha falloff only.
- **Back-to-front sorting.** Additive blending is order-independent, so the
  fire and ember presets are correct; the alpha-blended smoke preset shows
  intra-emitter ordering artifacts.
- **Lighting.** Particles are emissive only. They neither receive light nor
  cast any — a torch does not illuminate the wall behind it.
- **Collision.** Particles pass straight through scene geometry and have no
  interaction with the physics solver.
- **Water.** There is no fluid simulation and no water surface renderer in this
  engine. The same pass is the intended foundation for spray and foam later,
  but no water feature exists today.
- **Texture atlases / flipbooks.** Particles are procedural radial sprites;
  there is no sprite-sheet animation.

## Validation

Node cannot execute WGSL, so verification is split in two.

`src/runtime/live/webgpu/particle-pass.test.ts` (66 tests) and
`combustion-flame.test.ts` (14 tests) drive the real classes through a
recording fake `GPUDevice`, covering uniform packing at documented byte
offsets, spec clamping, pool seeding and stagger, blend state, depth
configuration, sample-count matching, dispatch and instance counts, buffer
lifetime, draw ordering, and the camera basis against a real `m4LookAt`
matrix.

`tests/integration/fire-particles.spec.ts` is where the shaders actually run.
It loads `examples/smoke-game/fire-probe.html` in installed Chrome, compiles
both pipelines, steps the simulation, reads the particle pool back off the GPU,
then renders into an HDR target and reads the pixels back. Measured on this
machine (Intel gen-11, `isFallbackAdapter: false`, 2048-slot pool, 90 frames at
dt = 1/60):

| Check | Result |
| --- | --- |
| Alive particles after 90 frames | 896 |
| Mean / max height above emitter | 0.588 m / 1.760 m |
| Temperature range | 0.0034 – 1.0 |
| Non-finite fields in the pool | 0 |
| Particles with life > maxLife | 0 |
| Alive after deactivating the emitter | 0 |
| Non-zero pixels in the HDR target | 1531 / 4096 |
| Peak HDR channel | 185.1 (bloom threshold is 0.9) |

The real-adapter run is what caught `active` being a reserved WGSL keyword;
no Node test could have. The reserved-word list is now pinned in the unit test.

Two honest caveats on that table. The peak-channel figure comes from a 64×64
probe target where each pixel covers a large solid angle, so additive overlap
is far heavier than at playable resolution — read it as "comfortably above the
bloom threshold", not as the shipping look. And none of this is a frame-rate
measurement: there is **no measured p50/p99 for the particle pass**, on this or
any machine. `PERFORMANCE.md` stays pending headed capture.

```sh
npm run test --workspace=@mnemopay/augengine
npm run verify:integration
```

## Not yet wired

The pass is registered on `WebGpuLiveRenderer` and verified, but no shipping
scene or Studio panel creates an emitter yet. The Chemistry Lab still renders
its glowing diagnostic solids, not this flame. Wiring the lab's combustion
body to `flameFromCombustion` is the next step and is not done.

# Physics and chemistry foundation

Implemented in source on 2026-10-24. This is a first working subsystem, not
completion of the full research roadmap or a laboratory chemistry simulator.

## Try the Studio lab

Build the engine, then start Studio:

```sh
npm run build --workspace=@mnemopay/augengine
npm run dev --workspace=@mnemopay/augengine-studio
```

Choose **Chemistry Lab**, then **Play**. The reactor has finite carbon and
oxygen; the separate water sample has a switchable heater. The panel shows
temperature, pressure, inventories, externally supplied heat, simulation
timing, and overload counters. Four rigid blocks settle under gravity.
Pause/resume preserves the simulation and its accounting. **Save simulation
checkpoint** exports the scene, native physics state and accounting; restore
it through the API below. The ordinary scene export alone does not contain
the complete simulation checkpoint.

## Runtime API

```ts
import { CoupledSimulation, Scene } from "@mnemopay/augengine";
import { createChemistryLab } from "@mnemopay/augengine/scenes";

const scene = new Scene(createChemistryLab());
const simulation = new CoupledSimulation();
await simulation.init();
simulation.advance(scene, 1 / 60); // seconds of wall time

const checkpoint = simulation.snapshot(scene);
const restored = await CoupledSimulation.restore(
  JSON.parse(JSON.stringify(checkpoint)),
);
// Use restored.scene and restored.simulation together.
restored.simulation.dispose();
simulation.dispose();
```

`EngineRunner.start()` is now asynchronous: await it. `PhysicsSystem` users
must also await `init()` before stepping. The WASM backend loads lazily.

Physics runs at 120 Hz, object chemistry at 30 Hz, and local fields at 15 Hz
by default. For full-rate reference comparisons, construct
`new CoupledSimulation({ chemistryEveryTicks: 1, fieldsEveryTicks: 1 })`.
Update strides must be integers from 1 to 12. Inputs to slower systems take
effect at their next scheduled update. Contact exposure is accumulated
between chemistry updates, avoiding continuous-heating assumptions for
brief contacts. Work discarded after a large wall-time jump is reported by
`simulation.clock.droppedSeconds`; it is not silently passed into a huge dt.

## Implemented physics

The deterministic-compatible Rapier 3D WASM build is pinned to 0.21.0. The
adapter supports dynamic, fixed, and position-driven kinematic bodies;
sphere, box and capsule collision; explicitly supplied static triangle
meshes; friction, restitution, angular motion, damping, sleeping and CCD.
Rapier handles broad/narrow phase and contact impulses instead of the old
all-pairs sphere loop that zeroed every velocity component.

Units are metres, kilograms and seconds. `Collider.extents` are half-extents
before transform scale. Physics entities must be roots; bake parent world
transforms before adding them. Sphere/capsule scales must be uniform.
Dynamic triangle meshes are rejected. The adapter checks dimensions,
finite float32-representable input, positive mass/sizes and unit rotations.
Other engine character, vehicle and ragdoll controllers retain their own
existing models; replacing the rigid-body solver does not certify every
controller or expose all of Rapier's joint/vehicle APIs.

## Thermodynamics and reaction assumptions

All quantities use mol, J, K, m³, Pa and W. Formation-enthalpy references are
at 298.15 K. Heat capacities are constant engineering approximations.

- A body with `volumeM3` stores internal energy `U = H - n_gas R T`, using
  gas `Cv = Cp - R`. A body without a fixed gas volume uses enthalpy and Cp.
  Condensed-phase pressure/volume work is neglected.
- The bundled carbon reaction is `C + O2 -> CO2`. Stoichiometric definitions
  are checked against element and mass balance. Reaction composition changes
  leave total stored energy unchanged: formation energy produces the
  temperature change. Reaction heat is not added a second time.
- Its activation threshold and first-order rate are authored gameplay
  parameters, not measured carbon-combustion kinetics. This is not a
  flame-speed, pyrolysis, pollution or arbitrary-reaction predictor.
- Water/ice/vapour equilibrium uses fixed 1-atm melting/boiling temperatures
  and latent heat. Partial phase changes pin the corresponding temperature.
  This mode is rejected in sealed-volume bodies, where a pressure-dependent
  equilibrium model would be required. The 1 K lower bound is a numerical
  guard, not a claim of cryogenic material accuracy.
- Heat links use an exact two-lump constant-capacity exchange step, bounded
  by the supported energy floors. Contact conductance is explicitly authored
  in W/K; contact area is not automatically inferred.
- Relief valves use a well-mixed adiabatic blowdown model with constant
  capacities. The rigid vessel stores U and discharged gas carries enthalpy.
  A monotone solve finds the retained gas fraction at the pressure limit.
  Discharged species and boundary-energy transfer remain in the ledger.
  Valve flow duration, exhaust momentum and structural rupture are not modeled.
- Chemistry ticks stage their inventory/energy changes, and roll back both
  the inventory and accounting if validation or a numerical overflow fails.

Reference anchors include NIST's [CO2 formation enthalpy](https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1)
and [liquid-water formation enthalpy](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=2).
Ice/vapour formation offsets in this reduced model are derived to make the
chosen constant capacities and latent heats internally consistent; they
are not an independent reproduction of a full NIST/IAPWS property model.
Cantera documents the distinction between [control-volume internal energy](https://www.cantera.org/3.2/reference/reactors/ideal-gas-reactor.html)
and [constant-pressure enthalpy](https://www.cantera.org/3.2/reference/reactors/ideal-gas-constant-pressure-reactor.html).

## Sparse local fields

`SparseReactiveField` supplies explicit local cells, thermal conduction,
gas concentration diffusion, a work/activity set and snapshots. Only
allocated neighbours exchange matter; unallocated boundaries are sealed.
Equilibrium cells sleep and changed cells wake relevant interfaces,
independently of camera visibility. Object/field interfaces can exchange
heat and tracked gas through `Reactive.fieldExchange`.

These are conservative scalar diffusion fields, not fluid dynamics.
Transport carries internal energy; pressure-driven bulk flow, advection,
buoyancy and momentum/pressure-work coupling are not simulated. Default
capacity is 128 cells. Increasing `maxCells` is an explicit workload choice,
not a promise of real-time performance. Setting/replacing cells is an
authoring operation; runtime sources must be accounted for explicitly.

## Validation and performance

Tests cover element/mass/energy conservation, oxygen exhaustion, Cp/Cv
heating, latent heat, adiabatic vent cooling, conduction, gas diffusion,
field sleeping, timestep scheduling, native checkpoint continuation,
collision momentum/energy, sliding friction, resting stacks, capsule/mesh
collision and high-speed CCD. The browser integration test runs the real
WASM backend, heats water through boiling, preserves accounting across
pause, and downloads a checkpoint.

```sh
npm run test --workspace=@mnemopay/augengine
npm run verify:integration
node scripts/benchmark-reactive-physics.mjs report.json
```

The CPU benchmark records the actual machine, Node version and workload.
It is not a rendering/FPS measurement. The 512-cell stress case still has
latency spikes above an 8.33 ms physics-tick budget on the tested integrated-
GPU laptop's CPU. See the [benchmark artifact](../../docs/research/chemistry-physics-cpu-benchmark-2026-10-24.json).
No production GPU p50/p99 or universal speedup is claimed.

Further work: temperature-dependent property tables and calibrated reaction
mechanisms; a general validated material authoring library; pressure-dependent
phase equilibrium; advection and buoyancy; GPU field/VFX detail; chemical
weakening/fracture and friction-to-heat coupling; broader device and large-world
validation. The current lab's glowing/coloured diagnostic solids are not
production fire or water-fluid graphics.

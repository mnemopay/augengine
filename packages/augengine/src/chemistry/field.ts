import { Scene } from "../scene/scene.js";
import { Tag, type EntityId } from "../scene/types.js";
import type { ReactiveComponent } from "./types.js";
import {
  ChemistrySystem,
  validateBody,
  minimumEnergy,
  equilibrateWater,
} from "./system.js";
import { substance, temperature, heatCapacity, energyAt } from "./materials.js";

export interface ReactiveFieldOptions {
  cellSizeM?: number;
  gasDiffusivityM2PerSecond?: number;
  thermalConductanceWPerK?: number;
  maxCells?: number;
}
/** Sparse local finite-volume field. Missing cells are sealed boundaries.
 * Scalar gas diffusion carries internal energy; pressure-driven bulk flow,
 * momentum/pressure work and Navier–Stokes advection are not modeled here.
 * Inactive cells sleep; changed cells wake adjacent interfaces, independent of visibility.
 */
export class SparseReactiveField {
  readonly scene = new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "reactive-field" },
    entities: [],
  });
  readonly chemistry = new ChemistrySystem();
  readonly cellSizeM: number;
  readonly diffusivity: number;
  readonly conductance: number;
  readonly maxCells: number;
  injectedEnergyJ = 0;
  lastActiveCells = 0;
  lastProcessedInterfaces = 0;
  private active = new Set<string>();
  constructor(options: ReactiveFieldOptions = {}) {
    this.cellSizeM = options.cellSizeM ?? 0.25;
    this.diffusivity = options.gasDiffusivityM2PerSecond ?? 0.001;
    this.conductance = options.thermalConductanceWPerK ?? 0.1;
    this.maxCells = options.maxCells ?? 128;
    if (
      !Number.isFinite(this.cellSizeM) ||
      this.cellSizeM <= 0 ||
      !Number.isFinite(this.diffusivity) ||
      this.diffusivity < 0 ||
      !Number.isFinite(this.conductance) ||
      this.conductance < 0 ||
      !Number.isSafeInteger(this.maxCells) ||
      this.maxCells < 1
    )
      throw new Error("Invalid reactive-field options");
  }
  setCell(x: number, y: number, z: number, body: ReactiveComponent): void {
    const key = this.key(x, y, z);
    if (!this.scene.has(key as EntityId) && this.scene.size() >= this.maxCells)
      throw new Error("Reactive field cell budget exceeded");
    const b = structuredClone(body);
    const initialTemperature = temperature(b);
    b.volumeM3 = this.cellSizeM ** 3;
    if (body.volumeM3 === undefined)
      b.energyJ = energyAt(b, initialTemperature);
    if (b.thermalLinks?.length)
      throw new Error(
        "Reactive fields use neighbour transport, not explicit thermal links",
      );
    validateBody(b, key);
    this.scene.add({
      id: key as EntityId,
      name: key,
      components: { [Tag.Reactive]: b },
    });
    this.wake(x, y, z);
  }
  getCell(
    x: number,
    y: number,
    z: number,
  ): Readonly<ReactiveComponent> | undefined {
    return this.scene.get(this.key(x, y, z) as EntityId)?.components[
      Tag.Reactive
    ];
  }
  wake(x: number, y: number, z: number): void {
    const key = this.key(x, y, z);
    this.active.add(key);
    for (const [dx, dy, dz] of DIRECTIONS)
      this.active.add(this.key(x + dx, y + dy, z + dz));
  }
  addHeat(x: number, y: number, z: number, joules: number): void {
    const b = this.scene.get(this.key(x, y, z) as EntityId)?.components[
      Tag.Reactive
    ];
    if (
      !b ||
      !Number.isFinite(joules) ||
      !Number.isFinite(b.energyJ + joules) ||
      b.energyJ + joules < energyAt(b, 1)
    )
      throw new Error("Invalid field heat source");
    b.energyJ += joules;
    this.injectedEnergyJ += joules;
    this.wake(x, y, z);
  }
  exchangeWithBody(
    coordinate: readonly [number, number, number],
    body: ReactiveComponent,
    heatConductance: number,
    gasConductance: number,
    dt: number,
  ): void {
    const cell = this.scene.get(this.key(...coordinate) as EntityId)
      ?.components[Tag.Reactive];
    if (
      !cell ||
      ![heatConductance, gasConductance, dt].every(Number.isFinite) ||
      heatConductance < 0 ||
      gasConductance < 0 ||
      dt <= 0 ||
      dt > 0.1
    )
      throw new Error("Invalid field/body interface");
    validateBody(body);
    validateBody(cell);
    if (gasConductance && !body.volumeM3)
      throw new Error("Gas exchange needs a defined body gas volume");
    const inverseC = 1 / heatCapacity(body) + 1 / heatCapacity(cell);
    const requested =
      ((temperature(body) - temperature(cell)) *
        -Math.expm1(-heatConductance * inverseC * dt)) /
      inverseC;
    const heat = Math.max(
      minimumEnergy(cell) - cell.energyJ,
      Math.min(requested, body.energyJ - minimumEnergy(body)),
    );
    body.energyJ -= heat;
    cell.energyJ += heat;
    if (gasConductance) {
      const va = body.volumeM3!,
        vb = cell.volumeM3!,
        inverseV = 1 / va + 1 / vb;
      const fraction = -Math.expm1(-gasConductance * inverseV * dt) / inverseV;
      for (const species of [
        ...new Set([
          ...Object.keys(body.amountsMol),
          ...Object.keys(cell.amountsMol),
        ]),
      ].sort()) {
        if (substance(species).phase !== "gas") continue;
        const delta =
          ((body.amountsMol[species] ?? 0) / va -
            (cell.amountsMol[species] ?? 0) / vb) *
          fraction;
        const donor = delta > 0 ? body : cell,
          receiver = delta > 0 ? cell : body;
        const amount = Math.abs(delta);
        if (!amount) continue;
        const energy = energyAt(
          {
            amountsMol: { [species]: amount },
            energyJ: 0,
            volumeM3: donor.volumeM3!,
          },
          temperature(donor),
        );
        donor.amountsMol[species] = Math.max(
          0,
          (donor.amountsMol[species] ?? 0) - amount,
        );
        receiver.amountsMol[species] =
          (receiver.amountsMol[species] ?? 0) + amount;
        donor.energyJ -= energy;
        receiver.energyJ += energy;
      }
    }
    if (body.waterPhaseEquilibrium)
      equilibrateWater(body, "field-interface", this.chemistry.events);
    this.wake(...coordinate);
  }
  stepFixed(dt: number): void {
    if (!Number.isFinite(dt) || dt <= 0 || dt > 0.1)
      throw new Error("Invalid field timestep");
    const keys = [...this.active]
      .filter((id) => this.scene.has(id as EntityId))
      .sort();
    this.active.clear();
    this.lastActiveCells = keys.length;
    this.lastProcessedInterfaces = 0;
    if (!keys.length) return;
    const reacting = keys
      .map((id) => this.scene.get(id as EntityId)!)
      .filter((e) => {
        const b = e.components[Tag.Reactive]!;
        return (
          b.heaterWatts ||
          b.reactionIds?.length ||
          b.waterPhaseEquilibrium ||
          b.maxPressurePa
        );
      });
    const activeScene = new Scene({
      ...this.scene.spec,
      entities: reacting,
    });
    if (reacting.length) this.chemistry.stepFixed(activeScene, dt);
    else this.chemistry.events = [];
    for (const event of this.chemistry.events)
      if (event.moles > 1e-12) this.active.add(event.body);
    for (const id of keys)
      if (this.scene.get(id as EntityId)!.components[Tag.Reactive]!.heaterWatts)
        this.active.add(id);
    const pairs = new Map<string, [string, string]>();
    for (const id of keys) {
      const [x, y, z] = id.split(",").map(Number) as [number, number, number];
      for (const [dx, dy, dz] of DIRECTIONS) {
        const other = this.key(x + dx, y + dy, z + dz);
        if (!this.scene.has(other as EntityId)) continue;
        const p: [string, string] = id < other ? [id, other] : [other, id];
        pairs.set(JSON.stringify(p), p);
      }
    }
    const alpha =
      -Math.expm1((-2 * this.diffusivity * dt) / this.cellSizeM ** 2) / 2;
    for (const [, [left, right]] of [...pairs].sort(([a], [b]) =>
      a < b ? -1 : a > b ? 1 : 0,
    )) {
      this.lastProcessedInterfaces++;
      const a = this.scene.get(left as EntityId)!.components[Tag.Reactive]!,
        b = this.scene.get(right as EntityId)!.components[Tag.Reactive]!;
      const inverseC = 1 / heatCapacity(a) + 1 / heatCapacity(b);
      const heat =
        ((temperature(a) - temperature(b)) *
          -Math.expm1(-this.conductance * inverseC * dt)) /
        inverseC;
      a.energyJ -= heat;
      b.energyJ += heat;
      let changed = Math.abs(heat) > 1e-9;
      for (const species of [
        ...new Set([
          ...Object.keys(a.amountsMol),
          ...Object.keys(b.amountsMol),
        ]),
      ].sort()) {
        if (substance(species).phase !== "gas") continue;
        const delta =
          ((a.amountsMol[species] ?? 0) - (b.amountsMol[species] ?? 0)) * alpha;
        const donor = delta > 0 ? a : b,
          receiver = delta > 0 ? b : a;
        const amount = Math.abs(delta);
        if (!amount) continue;
        const energy = energyAt(
          {
            amountsMol: { [species]: amount },
            energyJ: 0,
            volumeM3: donor.volumeM3!,
          },
          temperature(donor),
        );
        donor.amountsMol[species] = Math.max(
          0,
          (donor.amountsMol[species] ?? 0) - amount,
        );
        receiver.amountsMol[species] =
          (receiver.amountsMol[species] ?? 0) + amount;
        donor.energyJ -= energy;
        receiver.energyJ += energy;
        changed ||= amount > 1e-12;
      }
      if (changed) {
        this.active.add(left);
        this.active.add(right);
      }
    }
  }
  private key(x: number, y: number, z: number): string {
    if (
      ![x, y, z].every((n) => Number.isSafeInteger(n) && Math.abs(n) < 1000000)
    )
      throw new Error("Invalid reactive-cell coordinates");
    return `${x},${y},${z}`;
  }
  snapshot() {
    return {
      version: 1 as const,
      options: {
        cellSizeM: this.cellSizeM,
        gasDiffusivityM2PerSecond: this.diffusivity,
        thermalConductanceWPerK: this.conductance,
        maxCells: this.maxCells,
      },
      cells: [...this.scene.entities]
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([id, e]) => ({
          coordinate: id.split(",").map(Number) as [number, number, number],
          body: structuredClone(e.components[Tag.Reactive]!),
        })),
      active: [...this.active].sort(),
      chemistry: this.chemistry.snapshot(),
      injectedEnergyJ: this.injectedEnergyJ,
    };
  }
  static restore(
    state: ReturnType<SparseReactiveField["snapshot"]>,
  ): SparseReactiveField {
    if (state.version !== 1 || !Number.isFinite(state.injectedEnergyJ))
      throw new Error("Invalid reactive-field snapshot");
    const field = new SparseReactiveField(state.options);
    for (const cell of state.cells)
      field.setCell(...cell.coordinate, cell.body);
    field.active.clear();
    for (const key of state.active) {
      const xyz = key.split(",").map(Number);
      if (xyz.length !== 3 || field.key(xyz[0]!, xyz[1]!, xyz[2]!) !== key)
        throw new Error("Invalid active-cell snapshot");
      field.active.add(key);
    }
    field.chemistry.restore(state.chemistry);
    field.injectedEnergyJ = state.injectedEnergyJ;
    return field;
  }
}
const DIRECTIONS: readonly (readonly [number, number, number])[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

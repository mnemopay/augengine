import { Tag } from "../scene/types.js";
import { Scene } from "../scene/scene.js";
import type { PhysicsContact } from "../physics/system.js";
import type { ReactiveComponent, ChemistryEvent, Reaction } from "./types.js";
import {
  REACTIONS,
  WATER_BOILING_K,
  WATER_LATENT_J_PER_MOL,
  WATER_FREEZING_K,
  WATER_FUSION_J_PER_MOL,
  GAS_CONSTANT,
  substance,
  heatCapacity,
  temperature,
  pressure,
  enthalpyAt,
  energyAt,
  inventory,
} from "./materials.js";

/** Lumped constant-Cp/Cv thermochemistry, heat links, and adiabatic ideal-gas vents. */
export class ChemistrySystem {
  events: ChemistryEvent[] = [];
  externalEnergyJ = 0;
  escapedEnergyJ = 0;
  escapedAmountsMol: Record<string, number> = {};
  ticks = 0;
  constructor() {
    for (const r of Object.values(REACTIONS)) validateReaction(r);
  }

  stepFixed(
    scene: Scene,
    dt: number,
    contacts: readonly PhysicsContact[] = [],
  ): void {
    if (!Number.isFinite(dt) || dt <= 0 || dt > 0.1)
      throw new Error("Chemistry fixed dt must be in (0, 0.1]");
    // Stage reactive state only. Invalid/numerically overflowing ticks cannot
    // partly consume fuel or alter the accounting ledger.
    const entities = [...scene.entities.values()].filter(
      (e) => e.components[Tag.Reactive],
    );
    if (!entities.length) {
      this.events = [];
      this.ticks++;
      return;
    }
    const staged = new Scene({
      ...scene.spec,
      entities: entities.map((e) => ({
        ...e,
        components: {
          ...e.components,
          [Tag.Reactive]: {
            ...e.components[Tag.Reactive]!,
            amountsMol: { ...e.components[Tag.Reactive]!.amountsMol },
          },
        },
      })),
    });
    const before = this.snapshot();
    try {
      this.integrate(staged, dt, contacts);
      for (const e of staged.entities.values())
        validateBody(e.components[Tag.Reactive]!, e.id);
      if (![this.externalEnergyJ, this.escapedEnergyJ].every(Number.isFinite))
        throw new Error("Chemistry ledger overflow");
      for (const entity of entities)
        Object.assign(
          entity.components[Tag.Reactive]!,
          staged.get(entity.id)!.components[Tag.Reactive]!,
        );
    } catch (error) {
      this.restore(before);
      this.events = [];
      throw error;
    }
  }
  private integrate(
    scene: Scene,
    dt: number,
    contacts: readonly PhysicsContact[],
  ): void {
    if (!Number.isFinite(dt) || dt <= 0 || dt > 0.1)
      throw new Error("Chemistry fixed dt must be in (0, 0.1]");
    const bodies = new Map(
      [...scene.entities.values()]
        .filter((e) => e.components[Tag.Reactive])
        .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
        .map((e) => [e.id as string, e.components[Tag.Reactive]!]),
    );
    for (const [id, b] of bodies) validateBody(b, id);
    for (const [id, b] of bodies)
      for (const link of b.thermalLinks ?? []) {
        if (
          !bodies.has(link.to) ||
          link.to === id ||
          !Number.isFinite(link.conductanceWPerK) ||
          link.conductanceWPerK < 0
        )
          throw new Error(`Invalid thermal link on ${id}`);
      }
    // Links are undirected. Conflicting duplicate definitions are rejected.
    const links = new Map<string, { a: string; b: string; g: number }>();
    const addLink = (a: string, b: string, g: number) => {
      if (a > b) [a, b] = [b, a];
      const key = JSON.stringify([a, b]);
      const existing = links.get(key);
      if (existing && existing.g !== g)
        throw new Error("Conflicting thermal-link conductances");
      links.set(key, { a, b, g });
    };
    for (const [id, b] of bodies)
      for (const link of b.thermalLinks ?? [])
        addLink(id, link.to, link.conductanceWPerK);
    for (const c of contacts) {
      const a = bodies.get(c.a),
        b = bodies.get(c.b);
      if (a && b && c.separation <= 0.002) {
        const exposure = c.durationSeconds ?? dt;
        if (!Number.isFinite(exposure) || exposure < 0 || exposure > dt + 1e-9)
          throw new Error("Invalid contact exposure");
        const g =
          Math.min(
            a.contactConductanceWPerK ?? 0,
            b.contactConductanceWPerK ?? 0,
          ) * Math.min(1, exposure / dt);
        if (g > 0 && !links.has(JSON.stringify([c.a, c.b].sort())))
          addLink(c.a, c.b, g);
      }
    }
    this.events = [];
    // Formation energy is in the state: changing composition heats/cools
    // automatically. Adding reaction heat separately would double-count it.
    for (const [id, body] of bodies) {
      const requestedHeat = (body.heaterWatts ?? 0) * dt;
      const applied = Math.max(
        requestedHeat,
        minimumEnergy(body) - body.energyJ,
      );
      body.energyJ += applied;
      this.externalEnergyJ += applied;
      for (const reactionId of body.reactionIds ?? []) {
        const r = REACTIONS[reactionId]!;
        if (temperature(body) < r.activationTemperatureK) continue;
        let extent = Infinity;
        for (const [species, n] of Object.entries(r.reactants))
          extent = Math.min(extent, (body.amountsMol[species] ?? 0) / n);
        extent *= -Math.expm1(-r.ratePerSecond * dt);
        if (extent <= 0) continue;
        let requiredAtMin = 0;
        for (const [species, n] of Object.entries(r.products))
          requiredAtMin +=
            n * enthalpyAt({ amountsMol: { [species]: 1 }, energyJ: 0 }, 1);
        for (const [species, n] of Object.entries(r.reactants))
          requiredAtMin -=
            n * enthalpyAt({ amountsMol: { [species]: 1 }, energyJ: 0 }, 1);
        if (requiredAtMin > 0)
          extent = Math.min(
            extent,
            Math.max(0, body.energyJ - enthalpyAt(body, 1)) / requiredAtMin,
          );
        for (const [species, n] of Object.entries(r.reactants))
          body.amountsMol[species] = Math.max(
            0,
            (body.amountsMol[species] ?? 0) - n * extent,
          );
        for (const [species, n] of Object.entries(r.products))
          body.amountsMol[species] =
            (body.amountsMol[species] ?? 0) + n * extent;
        this.events.push({
          type: "reaction",
          body: id,
          process: r.id,
          moles: extent,
        });
      }
    }
    for (const [, { a, b, g }] of [...links].sort(([a], [b]) =>
      a < b ? -1 : a > b ? 1 : 0,
    )) {
      const left = bodies.get(a)!,
        right = bodies.get(b)!;
      const inverseC = 1 / heatCapacity(left) + 1 / heatCapacity(right);
      const requested =
        ((temperature(left) - temperature(right)) *
          -Math.expm1(-g * inverseC * dt)) /
        inverseC;
      const q = Math.max(
        minimumEnergy(right) - right.energyJ,
        Math.min(requested, left.energyJ - minimumEnergy(left)),
      );
      left.energyJ -= q;
      right.energyJ += q;
    }
    for (const [id, b] of bodies) {
      if (b.waterPhaseEquilibrium) equilibrateWater(b, id, this.events);
      if (b.maxPressurePa && pressure(b) > b.maxPressurePa) {
        const gasMoles = Object.entries(b.amountsMol).reduce(
          (sum, [id, n]) => sum + (substance(id).phase === "gas" ? n : 0),
          0,
        );
        let gasCv = 0,
          solidCp = 0;
        for (const [species, n] of Object.entries(b.amountsMol)) {
          const material = substance(species);
          if (material.phase === "gas")
            gasCv += n * (material.heatCapacityJPerMolK - GAS_CONSTANT);
          else solidCp += n * material.heatCapacityJPerMolK;
        }
        const t0 = temperature(b),
          initialEnergy = b.energyJ;
        const exponent = (GAS_CONSTANT * gasMoles) / gasCv;
        const finalTemperature = (fraction: number) =>
          t0 * ((solidCp + gasCv * fraction) / (solidCp + gasCv)) ** exponent;
        // Adiabatic, well-mixed blowdown: outflow carries H; the rigid
        // vessel stores U. Solve n(f) R T(f) / V = P_limit.
        let lo = 0,
          hi = 1;
        for (let i = 0; i < 52; i++) {
          const mid = (lo + hi) / 2;
          if (
            (gasMoles * mid * GAS_CONSTANT * finalTemperature(mid)) /
              b.volumeM3! >
            b.maxPressurePa
          )
            hi = mid;
          else lo = mid;
        }
        const retained = (lo + hi) / 2;
        let removed = 0;
        for (const [species, n] of Object.entries(b.amountsMol))
          if (substance(species).phase === "gas") {
            const amount = n * (1 - retained);
            b.amountsMol[species] = n * retained;
            this.escapedAmountsMol[species] =
              (this.escapedAmountsMol[species] ?? 0) + amount;
            removed += amount;
          }
        b.energyJ = energyAt(b, finalTemperature(retained));
        this.escapedEnergyJ += initialEnergy - b.energyJ;
        this.events.push({
          type: "vent",
          body: id,
          process: "pressure-relief",
          moles: removed,
        });
      }
    }
    this.ticks++;
  }
  /** Ledger includes discharged gas so open-system conservation is inspectable. */
  totals(scene: Scene): {
    energyJ: number;
    massKg: number;
    elementsMol: Record<string, number>;
  } {
    let energyJ = this.escapedEnergyJ,
      massKg = 0;
    const elementsMol: Record<string, number> = {};
    const bodies = [...scene.entities.values()].flatMap((e) =>
      e.components[Tag.Reactive] ? [e.components[Tag.Reactive]!] : [],
    );
    bodies.push({
      amountsMol: this.escapedAmountsMol,
      energyJ: this.escapedEnergyJ,
    });
    for (const b of bodies) {
      if (b !== bodies[bodies.length - 1]) energyJ += b.energyJ;
      const inv = inventory(b);
      massKg += inv.massKg;
      for (const [id, n] of Object.entries(inv.elementsMol))
        elementsMol[id] = (elementsMol[id] ?? 0) + n;
    }
    return { energyJ, massKg, elementsMol };
  }
  snapshot(): {
    ticks: number;
    externalEnergyJ: number;
    escapedEnergyJ: number;
    escapedAmountsMol: Record<string, number>;
  } {
    return {
      ticks: this.ticks,
      externalEnergyJ: this.externalEnergyJ,
      escapedEnergyJ: this.escapedEnergyJ,
      escapedAmountsMol: { ...this.escapedAmountsMol },
    };
  }
  restore(state: ReturnType<ChemistrySystem["snapshot"]>): void {
    if (
      !Number.isSafeInteger(state.ticks) ||
      state.ticks < 0 ||
      !Number.isFinite(state.externalEnergyJ) ||
      !Number.isFinite(state.escapedEnergyJ)
    )
      throw new Error("Invalid chemistry ledger snapshot");
    for (const [id, n] of Object.entries(state.escapedAmountsMol)) {
      substance(id);
      if (!Number.isFinite(n) || n < 0)
        throw new Error("Invalid escaped inventory");
    }
    this.ticks = state.ticks;
    this.externalEnergyJ = state.externalEnergyJ;
    this.escapedEnergyJ = state.escapedEnergyJ;
    this.escapedAmountsMol = { ...state.escapedAmountsMol };
    this.events = [];
  }
}
export function validateBody(b: ReactiveComponent, id = "body"): void {
  if (!b || !b.amountsMol || !Number.isFinite(b.energyJ))
    throw new Error(`Invalid reactive body ${id}`);
  for (const [name, amount] of Object.entries(b.amountsMol)) {
    substance(name);
    if (!Number.isFinite(amount) || amount < 0)
      throw new Error(`Invalid amount on ${id}`);
  }
  const t = temperature(b);
  if (!Number.isFinite(t) || t < 1 - 1e-8)
    throw new Error(`Invalid temperature on ${id}`);
  if (b.energyJ < minimumEnergy(b) - 1e-7)
    throw new Error(`Energy below the phase-equilibrium floor on ${id}`);
  if (
    b.volumeM3 !== undefined &&
    (!Number.isFinite(b.volumeM3) || b.volumeM3 <= 0)
  )
    throw new Error("Gas volume must be positive");
  if (
    b.maxPressurePa !== undefined &&
    (!b.volumeM3 || !Number.isFinite(b.maxPressurePa) || b.maxPressurePa <= 0)
  )
    throw new Error("Pressure relief needs a positive gas volume and pressure");
  if (b.waterPhaseEquilibrium && b.volumeM3)
    throw new Error(
      "Fixed-1-atm water phase model cannot be used in a sealed gas volume",
    );
  if (b.heaterWatts !== undefined && !Number.isFinite(b.heaterWatts))
    throw new Error("Heater power must be finite");
  if (
    b.contactConductanceWPerK !== undefined &&
    (!Number.isFinite(b.contactConductanceWPerK) ||
      b.contactConductanceWPerK < 0)
  )
    throw new Error("Invalid contact conductance");
  for (const r of b.reactionIds ?? [])
    if (!Object.hasOwn(REACTIONS, r)) throw new Error(`Unknown reaction ${r}`);
  if (
    b.reactionIds &&
    b.reactionIds.length > 1 &&
    new Set(b.reactionIds).size !== b.reactionIds.length
  )
    throw new Error("Duplicate reaction IDs");
  const link = b.fieldExchange;
  if (
    link &&
    (!Number.isSafeInteger(link.fieldIndex) ||
      link.fieldIndex < 0 ||
      !Array.isArray(link.cell) ||
      link.cell.length !== 3 ||
      !link.cell.every(
        (n) => Number.isSafeInteger(n) && Math.abs(n) < 1000000,
      ) ||
      !Number.isFinite(link.conductanceWPerK) ||
      link.conductanceWPerK < 0 ||
      !Number.isFinite(link.gasConductanceM3PerSecond ?? 0) ||
      (link.gasConductanceM3PerSecond ?? 0) < 0 ||
      (link.gasConductanceM3PerSecond && !b.volumeM3))
  )
    throw new Error("Invalid reactive-field interface");
}
export function validateReaction(r: Reaction): void {
  if (
    !Number.isFinite(r.ratePerSecond) ||
    r.ratePerSecond < 0 ||
    !Number.isFinite(r.activationTemperatureK) ||
    r.activationTemperatureK < 1
  )
    throw new Error("Invalid reaction rate/activation");
  const balance: Record<string, number> = {};
  let mass = 0;
  for (const [side, sign] of [
    [r.reactants, -1],
    [r.products, 1],
  ] as const) {
    if (!Object.keys(side).length)
      throw new Error("Reaction must have reactants and products");
    for (const [id, n] of Object.entries(side)) {
      if (!Number.isFinite(n) || n <= 0)
        throw new Error("Stoichiometric coefficients must be positive");
      const s = substance(id);
      mass += sign * n * s.molarMassKg;
      for (const [el, count] of Object.entries(s.elements))
        balance[el] = (balance[el] ?? 0) + sign * n * count;
    }
  }
  if (
    Object.values(balance).some((n) => Math.abs(n) > 1e-12) ||
    Math.abs(mass) > 1e-12
  )
    throw new Error(`Unbalanced reaction ${r.id}`);
}
export function minimumEnergy(b: ReactiveComponent): number {
  if (!b.waterPhaseEquilibrium) return energyAt(b, 1);
  const n =
    (b.amountsMol.ice ?? 0) +
    (b.amountsMol.water ?? 0) +
    (b.amountsMol.steam ?? 0);
  return enthalpyAt(
    { amountsMol: { ...b.amountsMol, ice: n, water: 0, steam: 0 }, energyJ: 0 },
    1,
  );
}
export function equilibrateWater(
  b: ReactiveComponent,
  id: string,
  events: ChemistryEvent[],
): void {
  const n =
    (b.amountsMol.ice ?? 0) +
    (b.amountsMol.water ?? 0) +
    (b.amountsMol.steam ?? 0);
  if (!n) return;
  const oldIce = b.amountsMol.ice ?? 0,
    oldSteam = b.amountsMol.steam ?? 0;
  const probe: ReactiveComponent = {
    amountsMol: { ...b.amountsMol, ice: n, water: 0, steam: 0 },
    energyJ: b.energyJ,
  };
  const frozenAtMelt = enthalpyAt(probe, WATER_FREEZING_K);
  const liquidAtMelt = frozenAtMelt + n * WATER_FUSION_J_PER_MOL;
  probe.amountsMol.ice = 0;
  probe.amountsMol.water = n;
  const liquidAtBoil = enthalpyAt(probe, WATER_BOILING_K);
  const vaporAtBoil = liquidAtBoil + n * WATER_LATENT_J_PER_MOL;
  let ice = 0,
    water = 0,
    steam = 0;
  if (b.energyJ <= frozenAtMelt) ice = n;
  else if (b.energyJ < liquidAtMelt) {
    water = (b.energyJ - frozenAtMelt) / WATER_FUSION_J_PER_MOL;
    ice = n - water;
  } else if (b.energyJ <= liquidAtBoil) water = n;
  else if (b.energyJ < vaporAtBoil) {
    steam = (b.energyJ - liquidAtBoil) / WATER_LATENT_J_PER_MOL;
    water = n - steam;
  } else steam = n;
  Object.assign(b.amountsMol, { ice, water, steam });
  const melted = oldIce - ice,
    vaporized = steam - oldSteam;
  if (Math.abs(melted) > 1e-14)
    events.push({
      type: melted > 0 ? "melting" : "freezing",
      body: id,
      process: "water-1atm",
      moles: Math.abs(melted),
    });
  if (Math.abs(vaporized) > 1e-14)
    events.push({
      type: vaporized > 0 ? "evaporation" : "condensation",
      body: id,
      process: "water-1atm",
      moles: Math.abs(vaporized),
    });
}

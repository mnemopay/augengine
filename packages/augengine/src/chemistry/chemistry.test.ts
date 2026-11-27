import { describe, it, expect } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type EntityId } from "../scene/types.js";
import {
  createReactiveBody,
  temperature,
  pressure,
  enthalpyAt,
  WATER_LATENT_J_PER_MOL,
  WATER_BOILING_K,
  inventory,
  GAS_CONSTANT,
} from "./materials.js";
import { ChemistrySystem, validateBody, validateReaction } from "./system.js";
import type { ReactiveComponent } from "./types.js";
import { FixedStepClock } from "../runtime/fixed-step.js";
function scene(...bodies: ReactiveComponent[]): Scene {
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: bodies.map((b, i) => ({
      id: `body.${i}` as EntityId,
      name: `body ${i}`,
      components: { [Tag.Reactive]: b },
    })),
  });
}
describe("conservative thermochemistry", () => {
  it("conserves elements, mass and total energy while oxidation raises temperature", () => {
    const b = createReactiveBody(
      { carbon: 1, oxygen: 0.2, nitrogen: 10 },
      800,
      { reactionIds: ["carbonOxidation"] },
    );
    const s = scene(b),
      chemistry = new ChemistrySystem(),
      before = chemistry.totals(s);
    for (let i = 0; i < 12000; i++) chemistry.stepFixed(s, 1 / 120);
    const after = chemistry.totals(s);
    expect(after.energyJ).toBeCloseTo(before.energyJ, 8);
    expect(after.massKg).toBeCloseTo(before.massKg, 10);
    for (const el of Object.keys(before.elementsMol))
      expect(after.elementsMol[el]).toBeCloseTo(before.elementsMol[el]!, 10);
    expect(b.amountsMol.carbonDioxide).toBeCloseTo(0.2, 4);
    expect(b.amountsMol.oxygen).toBeGreaterThanOrEqual(0);
    expect(b.amountsMol.carbon).toBeCloseTo(0.8, 4);
    expect(temperature(b)).toBeGreaterThan(800);
  });
  it("does not burn without oxygen or below the authored activation threshold", () => {
    for (const b of [
      createReactiveBody({ carbon: 1 }, 900, {
        reactionIds: ["carbonOxidation"],
      }),
      createReactiveBody({ carbon: 1, oxygen: 1 }, 300, {
        reactionIds: ["carbonOxidation"],
      }),
    ]) {
      const before = structuredClone(b);
      new ChemistrySystem().stepFixed(scene(b), 1 / 120);
      expect(b).toEqual(before);
    }
  });
  it("uses latent heat and pins partially boiling water at 373.15 K", () => {
    const b = createReactiveBody({ water: 1 }, WATER_BOILING_K, {
      waterPhaseEquilibrium: true,
    });
    b.energyJ += WATER_LATENT_J_PER_MOL / 2;
    const initial = b.energyJ;
    new ChemistrySystem().stepFixed(scene(b), 1 / 120);
    expect(b.amountsMol.steam).toBeCloseTo(0.5, 10);
    expect(b.amountsMol.water).toBeCloseTo(0.5, 10);
    expect(temperature(b)).toBeCloseTo(WATER_BOILING_K, 10);
    expect(b.energyJ).toBe(initial);
  });
  it("condenses and returns latent heat without losing water mass", () => {
    const b = createReactiveBody({ steam: 1 }, WATER_BOILING_K, {
      waterPhaseEquilibrium: true,
    });
    b.energyJ -= WATER_LATENT_J_PER_MOL / 4;
    const initial = inventory(b);
    new ChemistrySystem().stepFixed(scene(b), 1 / 120);
    expect(b.amountsMol.water).toBeCloseTo(0.25, 10);
    expect(temperature(b)).toBeCloseTo(WATER_BOILING_K, 10);
    expect(inventory(b)).toEqual(initial);
  });
  it("exchanges heat conservatively without overshoot even at very high conductance", () => {
    const a = createReactiveBody({ nitrogen: 1 }, 1000, {
      thermalLinks: [{ to: "body.1", conductanceWPerK: 1e9 }],
    });
    const b = createReactiveBody({ nitrogen: 3 }, 200);
    const s = scene(a, b),
      c = new ChemistrySystem(),
      before = c.totals(s);
    c.stepFixed(s, 0.1);
    expect(temperature(a)).toBeCloseTo(400, 10);
    expect(temperature(b)).toBeCloseTo(400, 10);
    expect(c.totals(s).energyJ).toBeCloseTo(before.energyJ, 9);
  });
  it("counts each thermal edge once and rejects conflicting duplicates before mutation", () => {
    const a = createReactiveBody({ nitrogen: 1 }, 800, {
      heaterWatts: 100,
      thermalLinks: [{ to: "body.1", conductanceWPerK: 1 }],
    });
    const b = createReactiveBody({ nitrogen: 1 }, 300, {
      thermalLinks: [{ to: "body.0", conductanceWPerK: 2 }],
    });
    const before = structuredClone(a);
    expect(() => new ChemistrySystem().stepFixed(scene(a, b), 0.01)).toThrow(
      "Conflicting",
    );
    expect(a).toEqual(before);
  });
  it("accounts for applied heating and cooling and never crosses absolute zero", () => {
    const b = createReactiveBody({ water: 1 }, 300, { heaterWatts: -1e9 });
    const s = scene(b),
      c = new ChemistrySystem(),
      before = b.energyJ;
    c.stepFixed(s, 0.01);
    expect(temperature(b)).toBeCloseTo(1, 8);
    expect(b.energyJ - before).toBeCloseTo(c.externalEnergyJ, 8);
    b.heaterWatts = 500;
    c.stepFixed(s, 0.1);
    expect(b.energyJ).toBeCloseTo(enthalpyAt(b, 1) + 50, 7);
  });
  it("adiabatic pressure relief conserves mass/energy and matches ideal-gas blowdown", () => {
    const b = createReactiveBody({ oxygen: 1, nitrogen: 3 }, 500, {
      volumeM3: 0.01,
      maxPressurePa: 100000,
    });
    const s = scene(b),
      c = new ChemistrySystem(),
      before = c.totals(s);
    const initialPressure = pressure(b);
    c.stepFixed(s, 0.01);
    expect(pressure(b)).toBeCloseTo(100000, 6);
    const cp = 29.4 + 3 * 29.1,
      cv = cp - 4 * GAS_CONSTANT,
      gamma = cp / cv;
    expect(temperature(b)).toBeCloseTo(
      500 * (100000 / initialPressure) ** ((gamma - 1) / gamma),
      7,
    );
    expect(c.totals(s).energyJ).toBeCloseTo(before.energyJ, 8);
    expect(c.totals(s).massKg).toBeCloseTo(before.massKg, 10);
    expect(c.escapedAmountsMol.oxygen).toBeGreaterThan(0);
  });
  it("matches chemistry across 30/60/120 Hz rendering", () => {
    const states = [];
    for (const hz of [30, 60, 120]) {
      const b = createReactiveBody(
        { carbon: 1, oxygen: 0.1, nitrogen: 10 },
        800,
        { reactionIds: ["carbonOxidation"] },
      );
      const s = scene(b),
        c = new ChemistrySystem(),
        clock = new FixedStepClock();
      for (let i = 0; i < 2 * hz; i++)
        clock.advance(1 / hz, (dt) => c.stepFixed(s, dt));
      states.push(JSON.stringify([b, c.snapshot()]));
    }
    expect(new Set(states).size).toBe(1);
  });
  it("rejects invalid amounts, missing links, sealed-water misuse and unbalanced reactions", () => {
    expect(() => createReactiveBody({ oxygen: -1 }, 300)).toThrow();
    expect(() => createReactiveBody({ nonexistent: 1 }, 300)).toThrow();
    expect(() => createReactiveBody({}, 300)).toThrow();
    const b = createReactiveBody({ water: 1 }, 300, {
      waterPhaseEquilibrium: true,
      volumeM3: 1,
    });
    expect(() => validateBody(b)).toThrow("sealed");
    expect(() =>
      validateReaction({
        id: "bad",
        reactants: { carbon: 1 },
        products: { carbonDioxide: 1 },
        ratePerSecond: 1,
        activationTemperatureK: 700,
      }),
    ).toThrow("Unbalanced");
    const invalid = createReactiveBody({ oxygen: 1 }, 300, {
      heaterWatts: 10,
      thermalLinks: [{ to: "missing", conductanceWPerK: 1 }],
    });
    const before = structuredClone(invalid);
    expect(() =>
      new ChemistrySystem().stepFixed(scene(invalid), 0.01),
    ).toThrow();
    expect(invalid).toEqual(before);
  });
  it("freezes and melts water with the fusion latent heat and conserves energy", () => {
    for (const from of ["ice", "water"]) {
      const b = createReactiveBody({ [from]: 1 }, 273.15, {
        waterPhaseEquilibrium: true,
      });
      b.energyJ += from === "ice" ? 6010 * 0.25 : -6010 * 0.75;
      const energy = b.energyJ;
      new ChemistrySystem().stepFixed(scene(b), 0.01);
      expect(b.amountsMol.water).toBeCloseTo(0.25, 10);
      expect(b.amountsMol.ice).toBeCloseTo(0.75, 10);
      expect(temperature(b)).toBeCloseTo(273.15, 10);
      expect(b.energyJ).toBe(energy);
    }
  });
  it("strong cooling respects the ice-phase absolute-temperature floor", () => {
    const b = createReactiveBody({ water: 1 }, 300, {
      waterPhaseEquilibrium: true,
      heaterWatts: -1e9,
    });
    new ChemistrySystem().stepFixed(scene(b), 0.01);
    expect(temperature(b)).toBeCloseTo(1, 8);
    expect(b.amountsMol.ice).toBeCloseTo(1, 10);
  });
  it("rolls back both inventory and ledger if a numerical tick overflows", () => {
    const b = createReactiveBody({ nitrogen: 1 }, 300);
    b.energyJ = 1.7e308;
    b.heaterWatts = 1.7e308;
    const c = new ChemistrySystem(),
      before = structuredClone(b),
      ledger = c.snapshot();
    expect(() => c.stepFixed(scene(b), 0.1)).toThrow();
    expect(b).toEqual(before);
    expect(c.snapshot()).toEqual(ledger);
  });
  it("restores ledger snapshots without sharing mutable inventories", () => {
    const b = createReactiveBody({ oxygen: 1 }, 300, {
      volumeM3: 0.001,
      maxPressurePa: 100000,
    });
    const c = new ChemistrySystem();
    c.stepFixed(scene(b), 0.01);
    const snapshot = c.snapshot(),
      restored = new ChemistrySystem();
    restored.restore(snapshot);
    expect(restored.snapshot()).toEqual(snapshot);
    snapshot.escapedAmountsMol.oxygen = 0;
    expect(restored.escapedAmountsMol.oxygen).toBeGreaterThan(0);
  });
  it("uses actual contact exposure when chemistry runs slower than physics", () => {
    const a = createReactiveBody({ nitrogen: 1 }, 1000, {
      contactConductanceWPerK: 10,
    });
    const b = createReactiveBody({ nitrogen: 1 }, 300, {
      contactConductanceWPerK: 10,
    });
    const before = a.energyJ;
    const expected = (700 * -Math.expm1((-10 * (2 / 29.1)) / 120)) / (2 / 29.1);
    new ChemistrySystem().stepFixed(scene(a, b), 1 / 30, [
      {
        a: "body.0" as EntityId,
        b: "body.1" as EntityId,
        normal: [0, 1, 0],
        point: [0, 0, 0],
        separation: 0,
        durationSeconds: 1 / 120,
      },
    ]);
    expect(before - a.energyJ).toBeCloseTo(expected, 8);
  });
  it("distinguishes Cp heating from fixed-volume Cv heating", () => {
    const open = createReactiveBody({ nitrogen: 1 }, 300, {
      heaterWatts: 1000,
    });
    const closed = createReactiveBody({ nitrogen: 1 }, 300, {
      heaterWatts: 1000,
      volumeM3: 1,
    });
    expect(temperature(closed)).toBeCloseTo(300, 10);
    new ChemistrySystem().stepFixed(scene(open, closed), 0.1);
    expect(temperature(open)).toBeCloseTo(300 + 100 / 29.1, 9);
    expect(temperature(closed)).toBeCloseTo(
      300 + 100 / (29.1 - GAS_CONSTANT),
      9,
    );
  });
});

import { describe, it, expect } from "vitest";
import { SparseReactiveField } from "./field.js";
import { createReactiveBody, temperature } from "./materials.js";
describe("sparse reactive fields", () => {
  it("diffuses gases and heat while conserving total inventory and enthalpy", () => {
    const f = new SparseReactiveField({
      cellSizeM: 1,
      gasDiffusivityM2PerSecond: 0.2,
    });
    f.setCell(0, 0, 0, createReactiveBody({ oxygen: 1, nitrogen: 1 }, 600));
    f.setCell(1, 0, 0, createReactiveBody({ nitrogen: 2 }, 300));
    const before = f.chemistry.totals(f.scene);
    for (let i = 0; i < 2000; i++) f.stepFixed(1 / 120);
    const after = f.chemistry.totals(f.scene);
    expect(after.massKg).toBeCloseTo(before.massKg, 10);
    expect(after.energyJ).toBeCloseTo(before.energyJ, 7);
    expect(f.getCell(1, 0, 0)!.amountsMol.oxygen).toBeCloseTo(0.5, 2);
    expect(temperature(f.getCell(0, 0, 0)!)).toBeGreaterThan(300);
    expect(after.elementsMol.O).toBeCloseTo(2, 10);
  });
  it("does not invent oxygen at unallocated boundaries", () => {
    const f = new SparseReactiveField();
    f.setCell(0, 0, 0, createReactiveBody({ nitrogen: 1 }, 300));
    f.stepFixed(0.01);
    expect(f.chemistry.totals(f.scene).elementsMol.O).toBeUndefined();
    expect(f.scene.size()).toBe(1);
  });
  it("sleeps equilibrium cells and wakes an adjacent interface after heating", () => {
    const f = new SparseReactiveField();
    f.setCell(0, 0, 0, createReactiveBody({ nitrogen: 1 }, 300));
    f.setCell(1, 0, 0, createReactiveBody({ nitrogen: 1 }, 300));
    f.stepFixed(0.01);
    f.stepFixed(0.01);
    expect(f.lastActiveCells).toBe(0);
    f.addHeat(0, 0, 0, 100);
    f.stepFixed(0.01);
    expect(f.lastActiveCells).toBe(2);
    expect(temperature(f.getCell(1, 0, 0)!)).toBeGreaterThan(300);
    expect(f.injectedEnergyJ).toBe(100);
  });
  it("enforces the cell budget, valid coordinates and source quantities", () => {
    const f = new SparseReactiveField({ maxCells: 1 });
    f.setCell(0, 0, 0, createReactiveBody({ nitrogen: 1 }, 300));
    expect(() =>
      f.setCell(1, 0, 0, createReactiveBody({ nitrogen: 1 }, 300)),
    ).toThrow("budget");
    expect(() =>
      f.setCell(0.5, 0, 0, createReactiveBody({ nitrogen: 1 }, 300)),
    ).toThrow("coordinates");
    expect(() => f.addHeat(0, 0, 0, -1e10)).toThrow();
    expect(() => f.stepFixed(NaN)).toThrow();
  });
  it("couples an object to a finite gas cell without inventing matter or energy", () => {
    const f = new SparseReactiveField({ cellSizeM: 1 });
    f.setCell(0, 0, 0, createReactiveBody({ oxygen: 1 }, 400));
    const body = createReactiveBody({ nitrogen: 1 }, 300, { volumeM3: 2 });
    const energy = f.getCell(0, 0, 0)!.energyJ + body.energyJ;
    for (let i = 0; i < 1000; i++)
      f.exchangeWithBody([0, 0, 0], body, 1, 2, 1 / 120);
    expect(body.amountsMol.oxygen).toBeCloseTo(2 / 3, 5);
    expect(
      body.amountsMol.oxygen! + f.getCell(0, 0, 0)!.amountsMol.oxygen!,
    ).toBeCloseTo(1, 10);
    expect(f.getCell(0, 0, 0)!.energyJ + body.energyJ).toBeCloseTo(energy, 7);
  });
});

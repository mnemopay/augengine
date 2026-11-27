import { describe, it, expect } from "vitest";
import { SUBSTANCES } from "../../../chemistry/materials.js";
import { ShallowWaterField } from "./shallow-water.js";
import { OCEAN_WATER_LOOK } from "./water-surface.js";
import { WATER_MOLAR_MASS_KG, waterFromSubstance, waterMolesFromVolumeM3 } from "./water-substance.js";

describe("chemistry water presentation", () => {
  const grid = { width: 8, height: 8, cellSize: 0.25 };
  it("round-trips chemical liquid moles through a flat water field volume", () => {
    expect(WATER_MOLAR_MASS_KG).toBe(SUBSTANCES.water.molarMassKg);
    const moles = 100000;
    const state = waterFromSubstance({ grid, temperatureK: 300, liquidMoles: moles });
    const field = new ShallowWaterField({ ...grid, bedLevel: 0, surfaceLevel: state.surfaceLevel });
    let volume = 0;
    for (let j = 0; j < grid.height; j++) for (let i = 0; i < grid.width; i++)
      volume += field.depthAt(i, j) * grid.cellSize ** 2;
    expect(waterMolesFromVolumeM3(volume)).toBeCloseTo(moles, 1);
  });
  it("restores the liquid look after freezing and thawing", () => {
    const frozen = waterFromSubstance({ grid, temperatureK: 260, liquidMoles: 10, iceMoles: 100 });
    const thawed = waterFromSubstance({ grid, temperatureK: 300, liquidMoles: 110 });
    const applied = { ...frozen.options, ...thawed.options };
    expect(frozen.options.running).toBe(false);
    expect(applied.running).toBe(true);
    expect(applied.fresnelF0).toBe(OCEAN_WATER_LOOK.fresnelF0);
    expect(applied.shallowTint).toEqual(OCEAN_WATER_LOOK.shallowTint);
    expect(applied.extinctionPerM).toEqual(OCEAN_WATER_LOOK.extinctionPerM);
  });
  it("does not turn vapor inventory into liquid or negative friction", () => {
    const state = waterFromSubstance({ grid, temperatureK: 400, liquidMoles: 0, vaporMoles: 10, referenceDamping: -1 });
    expect(state.phase).toBe("boiling");
    expect(state.volumeM3).toBe(0);
    expect(state.liquidFraction).toBe(0);
    expect(state.options.damping).toBe(0);
  });
});

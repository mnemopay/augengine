import { it, expect } from "vitest";
import { Scene } from "../scene/scene.js";
import { createChemistryLab } from "../scenes/chemistry-lab.js";
import { CoupledSimulation } from "./simulation.js";
import { SparseReactiveField } from "../chemistry/field.js";
import { createReactiveBody } from "../chemistry/materials.js";
import { Tag, type EntityId } from "../scene/types.js";

it("restores the full native physics state, chemical ledger and partial tick from JSON", async () => {
  const scene = new Scene(createChemistryLab()),
    simulation = new CoupledSimulation();
  await simulation.init();
  let restored:
    | Awaited<ReturnType<typeof CoupledSimulation.restore>>
    | undefined;
  try {
    const field = new SparseReactiveField();
    field.setCell(0, 0, 0, createReactiveBody({ oxygen: 1 }, 400));
    field.setCell(1, 0, 0, createReactiveBody({ nitrogen: 1 }, 300));
    simulation.fields.push(field);
    for (let i = 0; i < 57; i++) simulation.advance(scene, 1 / 144);
    const snapshot = JSON.parse(JSON.stringify(simulation.snapshot(scene)));
    restored = await CoupledSimulation.restore(snapshot);
    for (let i = 0; i < 100; i++) {
      simulation.advance(scene, 1 / 144);
      restored.simulation.advance(restored.scene, 1 / 144);
    }
    expect([...restored.scene.entities.values()]).toEqual([
      ...scene.entities.values(),
    ]);
    expect(restored.simulation.chemistry.snapshot()).toEqual(
      simulation.chemistry.snapshot(),
    );
    expect(restored.simulation.clock.snapshot()).toEqual(
      simulation.clock.snapshot(),
    );
    expect(restored.simulation.fields[0]!.snapshot()).toEqual(field.snapshot());
  } finally {
    simulation.dispose();
    restored?.simulation.dispose();
  }
});
it("rejects snapshots from unsupported versions without starting a simulation", async () => {
  await expect(
    CoupledSimulation.restore({ version: 2 } as never),
  ).rejects.toThrow("Unsupported");
});
it("integrates heater energy exactly over the lower-rate chemistry schedule", async () => {
  const b = createReactiveBody({ nitrogen: 1 }, 300, { heaterWatts: 100 });
  const scene = new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "heater" },
    entities: [
      {
        id: "heated" as EntityId,
        name: "heated",
        components: { [Tag.Reactive]: b },
      },
    ],
  });
  const s = new CoupledSimulation();
  await s.init();
  const before = b.energyJ;
  try {
    for (let i = 0; i < 60; i++) s.advance(scene, 1 / 30);
    expect(s.chemistry.ticks).toBe(60);
    expect(s.chemistry.externalEnergyJ).toBeCloseTo(200, 10);
    expect(b.energyJ - before).toBeCloseTo(200, 8);
  } finally {
    s.dispose();
  }
});

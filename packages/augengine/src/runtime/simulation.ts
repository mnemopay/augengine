import { Scene } from "../scene/scene.js";
import { PhysicsSystem, type PhysicsContact } from "../physics/system.js";
import { ChemistrySystem } from "../chemistry/system.js";
import { FixedStepClock } from "./fixed-step.js";
import { SparseReactiveField } from "../chemistry/field.js";
import { Tag } from "../scene/types.js";
import { validateBody } from "../chemistry/system.js";

/** One clock for rigid bodies, contact heat transfer and chemical reactions. */
export class CoupledSimulation {
  readonly clock = new FixedStepClock();
  readonly physics = new PhysicsSystem();
  readonly chemistry = new ChemistrySystem();
  readonly fields: SparseReactiveField[] = [];
  readonly chemistryEveryTicks: number;
  readonly fieldsEveryTicks: number;
  private tickIndex = 0;
  private pendingContacts = new Map<string, PhysicsContact>();
  readonly metrics = { lastTickCpuMs: 0, overBudgetTicks: 0 };
  constructor(
    options: { chemistryEveryTicks?: number; fieldsEveryTicks?: number } = {},
  ) {
    this.chemistryEveryTicks = options.chemistryEveryTicks ?? 4;
    this.fieldsEveryTicks = options.fieldsEveryTicks ?? 8;
    if (
      ![this.chemistryEveryTicks, this.fieldsEveryTicks].every(
        (n) => Number.isSafeInteger(n) && n >= 1 && n <= 12,
      )
    )
      throw new Error(
        "Simulation update strides must be integers from 1 to 12",
      );
  }
  async init(): Promise<void> {
    await this.physics.init();
  }
  advance(scene: Scene, seconds: number): number {
    for (const e of scene.entities.values()) {
      const body = e.components[Tag.Reactive];
      if (!body) continue;
      const link = body.fieldExchange;
      if (link) validateBody(body, e.id);
      if (link && !this.fields[link.fieldIndex]?.getCell(...link.cell))
        throw new Error("Reactive-field interface refers to a missing cell");
    }
    return this.clock.advance(seconds, (dt) => {
      const started = performance.now();
      this.physics.stepFixed(scene, dt);
      this.tickIndex++;
      const seen = new Set<string>();
      for (const contact of this.physics.contacts) {
        const key = JSON.stringify([contact.a, contact.b]);
        if (seen.has(key)) continue;
        seen.add(key);
        const previous = this.pendingContacts.get(key);
        this.pendingContacts.set(key, {
          ...contact,
          durationSeconds: (previous?.durationSeconds ?? 0) + dt,
        });
      }
      if (this.tickIndex % this.chemistryEveryTicks === 0) {
        this.chemistry.stepFixed(scene, dt * this.chemistryEveryTicks, [
          ...this.pendingContacts.values(),
        ]);
        this.pendingContacts.clear();
      }
      if (this.tickIndex % this.fieldsEveryTicks === 0)
        for (const field of this.fields)
          field.stepFixed(dt * this.fieldsEveryTicks);
      for (const e of scene.entities.values()) {
        const body = e.components[Tag.Reactive],
          link = body?.fieldExchange;
        if (body && link) {
          const field = this.fields[link.fieldIndex];
          if (!field)
            throw new Error(`Unknown reactive field ${link.fieldIndex}`);
          field.exchangeWithBody(
            link.cell,
            body,
            link.conductanceWPerK,
            link.gasConductanceM3PerSecond ?? 0,
            dt,
          );
        }
      }
      this.metrics.lastTickCpuMs = performance.now() - started;
      if (this.metrics.lastTickCpuMs > dt * 1000)
        this.metrics.overBudgetTicks++;
    });
  }
  dispose(): void {
    this.physics.dispose();
  }
  snapshot(scene: Scene) {
    return {
      version: 1 as const,
      options: {
        chemistryEveryTicks: this.chemistryEveryTicks,
        fieldsEveryTicks: this.fieldsEveryTicks,
      },
      tickIndex: this.tickIndex,
      pendingContacts: [...this.pendingContacts.values()].map((c) =>
        structuredClone(c),
      ),
      scene: structuredClone({
        ...scene.spec,
        entities: [...scene.entities.values()],
      }),
      clock: this.clock.snapshot(),
      physics: this.physics.snapshot(),
      chemistry: this.chemistry.snapshot(),
      fields: this.fields.map((field) => field.snapshot()),
    };
  }
  static async restore(
    state: ReturnType<CoupledSimulation["snapshot"]>,
  ): Promise<{ scene: Scene; simulation: CoupledSimulation }> {
    if (state.version !== 1) throw new Error("Unsupported simulation snapshot");
    const scene = Scene.fromJson(structuredClone(state.scene));
    for (const e of scene.entities.values())
      if (e.components[Tag.Reactive])
        validateBody(e.components[Tag.Reactive]!, e.id);
    const simulation = new CoupledSimulation(state.options);
    try {
      if (!Number.isSafeInteger(state.tickIndex) || state.tickIndex < 0)
        throw new Error("Invalid simulation tick snapshot");
      simulation.tickIndex = state.tickIndex;
      for (const contact of state.pendingContacts) {
        if (
          !scene.has(contact.a) ||
          !scene.has(contact.b) ||
          !Number.isFinite(contact.durationSeconds) ||
          contact.durationSeconds! < 0 ||
          contact.durationSeconds! >
            simulation.clock.dt * simulation.chemistryEveryTicks + 1e-9
        )
          throw new Error("Invalid contact snapshot");
        simulation.pendingContacts.set(
          JSON.stringify([contact.a, contact.b]),
          structuredClone(contact),
        );
      }
      simulation.clock.restore(state.clock);
      simulation.chemistry.restore(state.chemistry);
      simulation.fields.push(
        ...state.fields.map((field) => SparseReactiveField.restore(field)),
      );
      await simulation.init();
      simulation.physics.restore(scene, state.physics);
      return { scene, simulation };
    } catch (error) {
      simulation.dispose();
      throw error;
    }
  }
}

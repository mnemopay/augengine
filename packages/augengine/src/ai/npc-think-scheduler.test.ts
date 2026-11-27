import { describe, it, expect } from "vitest";
import {
  NpcThinkScheduler,
  type NpcCombatState,
} from "./npc-think-scheduler.js";
import { Scene, Tag, type Entity, type EntityId, type PersonaComponent } from "../scene/index.js";

const PLAYER: EntityId = "player" as EntityId;

function buildScene(entities: Entity[]): Scene {
  return new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test://scheduler" },
    entities: [
      {
        id: PLAYER,
        name: "Player",
        components: {
          [Tag.Transform]: {
            position: [0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1],
            parent: null,
          },
        },
      },
      ...entities,
    ],
  });
}

function npc(opts: {
  id: string;
  position: [number, number, number];
  persona?: Partial<PersonaComponent>;
}): Entity {
  const persona: PersonaComponent = {
    traits: [],
    voice: null,
    memoryId: null,
    goals: [],
    mood: 0,
    ...opts.persona,
  };
  return {
    id: opts.id as EntityId,
    name: opts.id,
    components: {
      [Tag.Transform]: {
        position: opts.position,
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Persona]: persona,
    },
  };
}

describe("NpcThinkScheduler", () => {
  it("ignores entities without a Persona component", () => {
    const scene = new Scene({
      schemaVersion: 1,
      source: { kind: "authored", path: "test://no-persona" },
      entities: [
        {
          id: PLAYER,
          name: "Player",
          components: {
            [Tag.Transform]: {
              position: [0, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [1, 1, 1],
              parent: null,
            },
          },
        },
        {
          id: "wall" as EntityId,
          name: "Wall",
          components: {
            [Tag.Transform]: {
              position: [1, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [1, 1, 1],
              parent: null,
            },
          },
        },
      ],
    });
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, { playerEntityId: PLAYER });
    expect(work).toHaveLength(0);
  });

  it("never schedules the player itself", () => {
    const scene = buildScene([npc({ id: "guard", position: [5, 0, 0] })]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, { playerEntityId: PLAYER });
    expect(work.find((w) => w.entity.id === PLAYER)).toBeUndefined();
    expect(work).toHaveLength(1);
  });

  it("sorts by priority descending", () => {
    const scene = buildScene([
      npc({ id: "far", position: [100, 0, 0] }),
      npc({ id: "near", position: [4, 0, 0] }),
      npc({ id: "mid", position: [30, 0, 0] }),
    ]);
    const states: Record<string, NpcCombatState> = {
      far: "idle",
      near: "engage",
      mid: "patrol",
    };
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: (id) => states[id as string] ?? "idle",
    });
    expect(work[0]!.entity.id).toBe("near");
    expect(work[work.length - 1]!.entity.id).toBe("far");
  });

  it("bumps story-critical NPCs to DeepReasoning regardless of distance", () => {
    const scene = buildScene([npc({ id: "vip", position: [200, 0, 200] })]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      storyCriticalIds: new Set(["vip" as EntityId]),
    });
    expect(work[0]!.tier).toBe("DeepReasoning");
  });

  it("bumps role='boss' NPCs to DeepReasoning", () => {
    const scene = buildScene([
      npc({ id: "boss", position: [200, 0, 200], persona: { role: "boss" } }),
    ]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, { playerEntityId: PLAYER });
    expect(work[0]!.tier).toBe("DeepReasoning");
  });

  it("melee-range NPCs are at least Tactical regardless of state", () => {
    const scene = buildScene([npc({ id: "melee", position: [2, 0, 2] })]);
    const scheduler = new NpcThinkScheduler({ tacticalDistance: 18 });
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "idle",
    });
    expect(["Tactical", "DeepReasoning"]).toContain(work[0]!.tier);
  });

  it("dialogue state forces DeepReasoning", () => {
    const scene = buildScene([npc({ id: "maya", position: [60, 0, 60] })]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "dialogue",
    });
    expect(work[0]!.tier).toBe("DeepReasoning");
  });

  it("down NPCs always Dormant", () => {
    const scene = buildScene([npc({ id: "corpse", position: [1, 0, 1] })]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "down",
    });
    expect(work[0]!.tier).toBe("Dormant");
  });

  it("far + idle NPCs drop to Dormant", () => {
    const scene = buildScene([npc({ id: "wanderer", position: [200, 0, 0] })]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "idle",
    });
    expect(work[0]!.tier).toBe("Dormant");
  });

  it("aggression axis biases combat-state priority up", () => {
    const scene = buildScene([
      npc({
        id: "calm",
        position: [25, 0, 0],
        persona: { axes: { aggression: 0.2, fear: 0.5, loyalty: 0.5, curiosity: 0.5 } },
      }),
      npc({
        id: "fierce",
        position: [25, 0, 0],
        persona: { axes: { aggression: 0.95, fear: 0.5, loyalty: 0.5, curiosity: 0.5 } },
      }),
    ]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "engage",
    });
    const calm = work.find((w) => w.entity.id === "calm")!;
    const fierce = work.find((w) => w.entity.id === "fierce")!;
    expect(fierce.priority).toBeGreaterThan(calm.priority);
  });

  it("memoryId presence adds a small priority bonus", () => {
    const scene = buildScene([
      npc({ id: "amnesiac", position: [25, 0, 0] }),
      npc({ id: "with-memory", position: [25, 0, 0], persona: { memoryId: "npc/m" } }),
    ]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, {
      playerEntityId: PLAYER,
      resolveState: () => "patrol",
    });
    const amnesiac = work.find((w) => w.entity.id === "amnesiac")!;
    const withMemory = work.find((w) => w.entity.id === "with-memory")!;
    expect(withMemory.priority).toBeGreaterThan(amnesiac.priority);
  });

  it("reason strings are populated for every NPC", () => {
    const scene = buildScene([
      npc({ id: "a", position: [5, 0, 0] }),
      npc({ id: "b", position: [70, 0, 0] }),
    ]);
    const scheduler = new NpcThinkScheduler();
    const work = scheduler.schedule(scene, { playerEntityId: PLAYER });
    for (const w of work) {
      expect(w.reason).toBeTruthy();
      expect(w.reason.length).toBeGreaterThan(0);
    }
  });
});

import { describe, it, expect } from "vitest";
import { Scene, Tag, type EntityId } from "../scene/index.js";
import {
  MAYA_DEFAULT_ENTITY_ID,
  MAYA_DIALOGUE,
  MAYA_MEMORY_NAMESPACE,
  MAYA_SEED_MEMORIES,
  createMaya,
  type MayaBrain,
} from "./maya.js";

function emptyScene(): Scene {
  return new Scene({
    spec_version: 1,
    id: "test",
    defaultCamera: null,
    entities: [],
  } as never);
}

function makeMockBrain(): MayaBrain & { records: Array<{ namespace?: string; content: string; tags?: string[] }> } {
  const records: Array<{ namespace?: string; content: string; tags?: string[] }> = [];
  return {
    records,
    async remember(input) {
      const r: { namespace?: string; content: string; tags?: string[] } = { content: input.content };
      if (input.namespace !== undefined) r.namespace = input.namespace;
      if (input.tags !== undefined) r.tags = input.tags;
      records.push(r);
      return { ok: true };
    },
    async recall({ namespace, query, limit }) {
      const ns = namespace ?? "";
      const qWords = (query ?? "").toLowerCase().split(/\W+/).filter((w) => w.length > 3);
      const hits = records
        .filter((r) => (r.namespace ?? "") === ns)
        .filter((r) => !qWords.length || qWords.some((w) => r.content.toLowerCase().includes(w)))
        .slice(0, limit ?? 5);
      return hits.map((r) => {
        const out: { content: string; tags?: string[] } = { content: r.content };
        if (r.tags !== undefined) out.tags = r.tags;
        return out;
      });
    },
    snapshot() {
      return records.map((r) => ({ namespace: r.namespace ?? "", content: r.content }));
    },
  };
}

describe("createMaya", () => {
  it("spawns Maya with a canonical persona attached", async () => {
    const scene = emptyScene();
    const maya = await createMaya(scene);
    expect(maya.entityId).toBe(MAYA_DEFAULT_ENTITY_ID);
    const entity = scene.get(MAYA_DEFAULT_ENTITY_ID);
    expect(entity).toBeTruthy();
    expect(entity!.name).toBe("Maya");
    const persona = entity!.components[Tag.Persona]!;
    expect(persona.memoryId).toBe(MAYA_MEMORY_NAMESPACE);
    expect(persona.role).toBe("ally");
    expect(persona.faction).toBe("inmates");
    expect(persona.axes!.loyalty).toBeCloseTo(0.85);
    expect(persona.voice).toBe("claude-haiku-4-5");
    expect(persona.traits).toContain("wry");
    expect(persona.biasTags).toContain("betrayed-by-corp");
  });

  it("idempotent — calling twice on the same scene reuses the entity", async () => {
    const scene = emptyScene();
    const m1 = await createMaya(scene);
    const m2 = await createMaya(scene);
    expect(m1.entityId).toBe(m2.entityId);
    expect(scene.size()).toBe(1);
  });

  it("seeds background memories into the brain on first apply", async () => {
    const scene = emptyScene();
    const brain = makeMockBrain();
    const maya = await createMaya(scene, { brain });
    expect(maya.seedCount()).toBe(MAYA_SEED_MEMORIES.length);
    expect(brain.records.filter((r) => r.namespace === MAYA_MEMORY_NAMESPACE)).toHaveLength(MAYA_SEED_MEMORIES.length);
  });

  it("seed is idempotent — same brain across two spawns doesn't double-insert", async () => {
    const brain = makeMockBrain();
    const sceneA = emptyScene();
    const sceneB = emptyScene();
    await createMaya(sceneA, { brain });
    await createMaya(sceneB, { brain });
    expect(brain.records.filter((r) => r.namespace === MAYA_MEMORY_NAMESPACE)).toHaveLength(MAYA_SEED_MEMORIES.length);
  });

  it("respond() routes via keyword beat first", async () => {
    const scene = emptyScene();
    const maya = await createMaya(scene);
    const reply = await maya.respond("tell me about the warden");
    expect(reply.source).toBe("beat");
    expect(reply.raises).toBe("maya/named-warden");
    expect(reply.line).toContain("Drev");
  });

  it("respond() falls through to brain recall when no keyword matches", async () => {
    const scene = emptyScene();
    const brain = makeMockBrain();
    const maya = await createMaya(scene, { brain });
    // Inject a memory that matches the query but no keyword in MAYA_DIALOGUE.
    await brain.remember({
      namespace: MAYA_MEMORY_NAMESPACE,
      content: "The player told you their favourite colour was rust orange.",
      tags: ["personal"],
    });
    const reply = await maya.respond("what's my favourite colour?");
    expect(reply.source).toBe("recall");
    expect(reply.line).toContain("rust orange");
  });

  it("respond() falls back to seeded round-robin when no keyword + no recall", async () => {
    const scene = emptyScene();
    const maya = await createMaya(scene);
    const replies = await Promise.all([
      maya.respond("xyz nothing matches"),
      maya.respond("xyz nothing matches"),
    ]);
    expect(replies[0]!.source).toBe("seeded");
    expect(replies[1]!.source).toBe("seeded");
    expect(replies[0]!.line).not.toBe(replies[1]!.line);
  });

  it("introLine() round-robins through the intro array", async () => {
    const scene = emptyScene();
    const maya = await createMaya(scene);
    const a = maya.introLine();
    const b = maya.introLine();
    const c = maya.introLine();
    const d = maya.introLine();
    expect(a).not.toBe(b);
    expect(d).toBe(a); // wraps after 3 intros
  });

  it("phaseLine() returns the right reaction per phase", async () => {
    const scene = emptyScene();
    const maya = await createMaya(scene);
    expect(maya.phaseLine("ally")).toContain("Stay on my left");
    expect(maya.phaseLine("tribunal")).toContain("up there");
    expect(maya.phaseLine("nonsense")).toBeNull();
  });

  it("react() shifts mood and emits a memory for grudge events", async () => {
    const scene = emptyScene();
    const brain = makeMockBrain();
    const maya = await createMaya(scene, { brain });
    const before = maya.mood();
    await maya.react({ kind: "player_shot_at_her" });
    expect(maya.mood()).toBeLessThan(before);
    expect(brain.records.some((r) => /aimed a weapon/i.test(r.content))).toBe(true);
  });

  it("react() ally_killed mood-drops + writes a loss memory", async () => {
    const scene = emptyScene();
    const brain = makeMockBrain();
    const maya = await createMaya(scene, { brain });
    const before = maya.mood();
    await maya.react({ kind: "ally_killed", allyId: "kael" as EntityId });
    expect(maya.mood()).toBeLessThan(before);
    expect(brain.records.some((r) => /kael did not survive/i.test(r.content))).toBe(true);
  });

  it("MAYA_DIALOGUE has the expected shape (no zero-length arrays)", () => {
    expect(MAYA_DIALOGUE.intro.length).toBeGreaterThan(0);
    expect(MAYA_DIALOGUE.beats.length).toBeGreaterThan(0);
    expect(MAYA_DIALOGUE.seeded.length).toBeGreaterThan(0);
  });
});

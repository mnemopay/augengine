/**
 * End-to-end brain wire-up: ObservationBus → recorder → in-process brain →
 * adapter → PersonaRunner. Proves the cross-session memory property: an NPC
 * spoken to in session 1 still remembers in session 2 after rehydrating
 * brain state from a snapshot (the same shape a save-file or localStorage
 * blob would carry).
 */

import { describe, it, expect } from "vitest";
import {
  AIScheduler,
  PersonaRunner,
  ObservationBus,
  createInProcessBrain,
  BRAIN_MISSING_SDK,
  createMnemoPayMemoryAdapter,
  startPersonaMemoryRecorder,
  ensurePersona,
  type BrainMemoryRecord,
  type LLMProvider,
  type LLMRequest,
  type LLMResponse,
} from "../index.js";
import { Tag, type Entity, type EntityId } from "../../scene/types.js";

class RecordingProvider implements LLMProvider {
  readonly id = "rec";
  readonly defaultModel = "rec-1";
  systems: string[] = [];
  reply = "ok";
  async complete(req: LLMRequest): Promise<LLMResponse> {
    if (req.system) this.systems.push(req.system);
    return { text: this.reply, finishReason: "end_turn" };
  }
}

function buildNpc(memoryId: string): Entity {
  const npc: Entity = {
    id: "npc-maya" as EntityId,
    name: "Maya",
    components: {},
  };
  ensurePersona(npc, { traits: ["curious", "guarded"], memoryId });
  return npc;
}

describe("augengine / ai / memory — in-process brain (localEmbed + cosineSimilarity)", () => {
  it("remembers + recalls memories within a single session", async () => {
    const brain = createInProcessBrain();
    await brain.remember({ namespace: "npc/maya", content: "the player brought the blue key" });
    await brain.remember({ namespace: "npc/maya", content: "the player likes hot tea" });
    await brain.remember({ namespace: "other", content: "irrelevant" });

    const result = await brain.recall({ namespace: "npc/maya", query: "blue key", limit: 2 });
    const rows = (result as { memories: Array<{ content?: string; score?: number }> }).memories;
    expect(rows).toHaveLength(2);
    // Namespace isolation: nothing from "other" leaks in
    expect(rows.every((r) => !r.content?.includes("irrelevant"))).toBe(true);
    // Top hit on lexical "blue key" query should be the blue-key memory.
    // localEmbed is bag-of-tokens hashing — strong lexical overlap wins.
    expect(rows[0]?.content).toContain("blue key");
  });

  it("snapshot + initial rehydrates state — cross-session persistence", async () => {
    const sessionOne = createInProcessBrain();
    await sessionOne.remember({ namespace: "npc/maya", content: "player saved Maya from the burn pit" });
    const snapshot = sessionOne.snapshot();
    expect(snapshot).toHaveLength(1);

    const sessionTwo = createInProcessBrain({ initial: snapshot });
    const result = await sessionTwo.recall({ namespace: "npc/maya", query: "what does Maya owe the player?" });
    const rows = (result as { memories: Array<{ content?: string }> }).memories;
    expect(rows[0]?.content).toContain("burn pit");
  });

  it("forget() wipes a namespace without touching others", async () => {
    const brain = createInProcessBrain();
    await brain.remember({ namespace: "npc/maya", content: "a" });
    await brain.remember({ namespace: "npc/maya", content: "b" });
    await brain.remember({ namespace: "npc/jorge", content: "c" });
    expect(brain.forget("npc/maya")).toBe(2);
    expect(brain.snapshot()).toHaveLength(1);
    expect(brain.snapshot()[0]?.namespace).toBe("npc/jorge");
  });

  it("deprecated augenginet() still works as an alias of forget()", async () => {
    const brain = createInProcessBrain();
    await brain.remember({ namespace: "npc/maya", content: "a" });
    await brain.remember({ namespace: "npc/jorge", content: "c" });
    expect(brain.augenginet("npc/maya")).toBe(1);
    expect(brain.snapshot().map((r) => r.namespace)).toEqual(["npc/jorge"]);
  });

  it("uses an injected loadRecall (the bundler path) instead of the runtime import", async () => {
    let calls = 0;
    const brain = createInProcessBrain({
      loadRecall: async () => {
        calls++;
        return {
          localEmbed: (_t: string, d: number) => new Float32Array(d).fill(1),
          cosineSimilarity: () => 0.5,
        } as never;
      },
    });
    await brain.remember({ namespace: "n", content: "x" });
    const res = (await brain.recall({ namespace: "n", query: "y" })) as { memories: Array<{ score: number }> };
    expect(res.memories[0]?.score).toBe(0.5);
    expect(calls).toBe(1);
  });

  it("a failing loadRecall surfaces the install hint", async () => {
    const brain = createInProcessBrain({ loadRecall: async () => { throw new Error("nope"); } });
    await expect(brain.remember({ namespace: "n", content: "x" })).rejects.toThrow(BRAIN_MISSING_SDK);
  });
});

describe("augengine / ai / memory — ObservationBus → recorder → brain", () => {
  it("auto-captures Dialogue events into the persona namespace", async () => {
    const brain = createInProcessBrain();
    const adapter = createMnemoPayMemoryAdapter({ client: brain });
    const bus = new ObservationBus();
    const npc = buildNpc("npc/maya");
    const entities = new Map<EntityId, Entity>([[npc.id, npc]]);

    const rec = startPersonaMemoryRecorder({
      bus,
      memory: adapter,
      resolveEntity: (id) => entities.get(id),
    });

    // Player says something Maya hears. The recorder routes to Maya by default
    // because she's the source — but we override to route to whichever persona
    // is listening.
    bus.publish({
      channel: "Dialogue",
      ts: new Date().toISOString(),
      payload: { text: "I brought you the blue key, Maya." },
      source: npc.id,
    });

    await rec.flush();
    expect(rec.count()).toBe(1);
    const snap = brain.snapshot();
    expect(snap).toHaveLength(1);
    expect(snap[0]?.namespace).toBe("npc/maya");
    expect(snap[0]?.content).toBe("I brought you the blue key, Maya.");
    expect(snap[0]?.metadata.channel).toBe("Dialogue");
    rec.stop();
  });

  it("skips events for entities without a memoryId", async () => {
    const brain = createInProcessBrain();
    const adapter = createMnemoPayMemoryAdapter({ client: brain });
    const bus = new ObservationBus();
    const npc: Entity = { id: "npc-noid" as EntityId, name: "Bystander", components: {} };
    ensurePersona(npc, { traits: ["mute"] }); // memoryId stays null
    const entities = new Map<EntityId, Entity>([[npc.id, npc]]);

    const rec = startPersonaMemoryRecorder({
      bus,
      memory: adapter,
      resolveEntity: (id) => entities.get(id),
    });

    bus.publish({
      channel: "Dialogue",
      ts: new Date().toISOString(),
      payload: { text: "hello?" },
      source: npc.id,
    });
    await rec.flush();
    expect(rec.count()).toBe(0);
    expect(brain.snapshot()).toHaveLength(0);
    rec.stop();
  });

  it("onPersist fires once per remembered observation — save-file hook", async () => {
    const persisted: BrainMemoryRecord[] = [];
    const brain = createInProcessBrain({ onPersist: (r) => persisted.push(r) });
    const adapter = createMnemoPayMemoryAdapter({ client: brain });
    const bus = new ObservationBus();
    const npc = buildNpc("npc/maya");
    const entities = new Map<EntityId, Entity>([[npc.id, npc]]);

    const rec = startPersonaMemoryRecorder({
      bus,
      memory: adapter,
      resolveEntity: (id) => entities.get(id),
    });

    bus.publish({
      channel: "Social",
      ts: new Date().toISOString(),
      payload: { text: "Maya nods" },
      source: npc.id,
    });
    bus.publish({
      channel: "Dialogue",
      ts: new Date().toISOString(),
      payload: { text: "thanks" },
      source: npc.id,
    });
    await rec.flush();

    expect(persisted).toHaveLength(2);
    rec.stop();
  });
});

describe("augengine / ai / memory — full stack: NPC remembers across sessions", () => {
  it("session-1 dialogue persists, session-2 PersonaRunner sees it in the prompt", async () => {
    // -------- Session 1: NPC hears something. --------
    const brainOne = createInProcessBrain();
    const adapterOne = createMnemoPayMemoryAdapter({
      client: brainOne,
      // Force a summarized recall path even with only one memory so the
      // adapter exercises the brain.summarize fallback.
      summaryBudget: 256,
    });
    const busOne = new ObservationBus();
    const npc = buildNpc("npc/maya");
    const entities = new Map<EntityId, Entity>([[npc.id, npc]]);

    const rec = startPersonaMemoryRecorder({
      bus: busOne,
      memory: adapterOne,
      resolveEntity: (id) => entities.get(id),
    });

    busOne.publish({
      channel: "Dialogue",
      ts: "2026-05-11T08:00:00.000Z",
      payload: { text: "the player saved Maya from the burn pit" },
      source: npc.id,
    });
    busOne.publish({
      channel: "Dialogue",
      ts: "2026-05-11T08:01:00.000Z",
      payload: { text: "Maya promised to repay the debt" },
      source: npc.id,
    });
    await rec.flush();
    rec.stop();

    const snapshot = brainOne.snapshot();
    expect(snapshot.length).toBeGreaterThanOrEqual(2);

    // -------- Session 2: fresh process, rehydrate from snapshot. --------
    const brainTwo = createInProcessBrain({ initial: snapshot });
    const adapterTwo = createMnemoPayMemoryAdapter({ client: brainTwo, summaryBudget: 256 });
    const scheduler = new AIScheduler();
    const provider = new RecordingProvider();
    const runner = new PersonaRunner({ scheduler, provider, memory: adapterTwo });

    const r = await runner.dialogue({
      entity: npc,
      user: { message: "do you remember the burn pit?", fromName: "player" },
      timeoutMs: 1000,
      fallback: "...",
    });

    expect(r).toBe("ok");
    const sys = provider.systems[0] ?? "";
    expect(sys).toContain("Maya");
    expect(sys).toContain("burn pit");
  });
});

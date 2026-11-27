/**
 * Phase 4 tests — provider request shape, template parsing/rendering,
 * persona runner end-to-end with a mock provider.
 *
 * Pure data + deterministic mocks. No real network calls.
 */

import { describe, it, expect } from "vitest";
import { Scene, Tag, type EntityId, type SceneSpec } from "../scene/index.js";
import {
  AIScheduler,
  AnthropicProvider,
  OpenAIProvider,
  OpenRouterProvider,
  PersonaRunner,
  createMnemoPayMemoryAdapter,
  parsePromptFile,
  renderTemplate,
  renderFewShot,
  ensurePersona,
  type LLMProvider,
  type LLMRequest,
  type LLMResponse,
} from "./index.js";

class MockProvider implements LLMProvider {
  readonly id = "mock";
  readonly defaultModel = "mock-1";
  lastReq: LLMRequest | null = null;
  reply = "mock reply";
  shouldThrow = false;

  async complete(req: LLMRequest): Promise<LLMResponse> {
    this.lastReq = req;
    if (this.shouldThrow) throw new Error("mock provider error");
    return { text: this.reply, finishReason: "end_turn" };
  }
}

describe("augengine / ai phase 4 — providers", () => {
  it("AnthropicProvider has the expected id + default model", () => {
    const p = new AnthropicProvider({ apiKey: "test" });
    expect(p.id).toBe("anthropic");
    expect(p.defaultModel).toContain("claude");
  });

  it("OpenAIProvider has the expected id", () => {
    const p = new OpenAIProvider({ apiKey: "test" });
    expect(p.id).toBe("openai");
  });

  it("OpenRouterProvider has the expected id + free-tier default", () => {
    const p = new OpenRouterProvider({ apiKey: "test" });
    expect(p.id).toBe("openrouter");
    expect(p.defaultModel).toContain(":free");
  });
});

describe("augengine / ai phase 4 — prompt template format", () => {
  it("parses frontmatter + body, auto-detects slots", () => {
    const raw = `---
slots: ["persona_name", "mood"]
examples:
  - input: {"persona_name": "Maya", "mood": "tense"}
    output: "Quiet."
---
You are {{persona_name}}. Mood is {{mood}}.`;
    const parsed = parsePromptFile("test.prompt", raw);
    expect(parsed.template.slots).toContain("persona_name");
    expect(parsed.template.slots).toContain("mood");
    expect(parsed.template.examples).toHaveLength(1);
    expect(parsed.template.body).toContain("{{persona_name}}");
  });

  it("auto-detects slots when frontmatter omits them", () => {
    const raw = `Hello {{name}}, you said: {{message}}`;
    const parsed = parsePromptFile("auto.prompt", raw);
    expect(parsed.template.slots.sort()).toEqual(["message", "name"]);
  });

  it("renderTemplate fills slots; missing slots leave literal by default", () => {
    const raw = `Hello {{name}}, age {{age}}`;
    const parsed = parsePromptFile("t.prompt", raw);
    const out = renderTemplate(parsed.template, { name: "Jorge" });
    expect(out).toBe("Hello Jorge, age {{age}}");
  });

  it("renderTemplate strict mode throws on missing slots", () => {
    const raw = `Hello {{name}}`;
    const parsed = parsePromptFile("t.prompt", raw);
    expect(() =>
      renderTemplate(parsed.template, {}, { strict: true }),
    ).toThrow(/missing slot/);
  });

  it("renderFewShot composes Input/Output blocks", () => {
    const raw = `---
examples:
  - input: {"x": "1"}
    output: "one"
  - input: {"x": "2"}
    output: "two"
---
Body`;
    const parsed = parsePromptFile("f.prompt", raw);
    const fs = renderFewShot(parsed.template);
    expect(fs).toContain('Input: {"x":"1"}');
    expect(fs).toContain("Output: one");
    expect(fs).toContain("Output: two");
  });
});

describe("augengine / ai phase 4 — PersonaRunner", () => {
  function buildEntity(): import("../scene/types.js").Entity {
    const e = {
      id: "npc-1" as EntityId,
      name: "Maya",
      components: {},
    };
    ensurePersona(e, {
      traits: ["curious", "guarded"],
      mood: 0.2,
    });
    return e;
  }

  it("dialogue dispatches through scheduler + provider", async () => {
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    provider.reply = "Quiet — listen.";
    const runner = new PersonaRunner({ scheduler, provider });
    const entity = buildEntity();
    const r = await runner.dialogue({
      entity,
      user: { message: "Where are we?", fromName: "player" },
      timeoutMs: 1000,
      fallback: "...",
    });
    expect(r).toBe("Quiet — listen.");
    expect(provider.lastReq?.system).toContain("Maya");
  });

  it("dialogue uses .prompt template when provided", async () => {
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    const runner = new PersonaRunner({ scheduler, provider });
    const entity = buildEntity();
    const tplRaw = `---
slots: ["persona_name", "scene_mood"]
---
You are {{persona_name}}. Scene mood: {{scene_mood}}. Reply briefly.`;
    await runner.dialogue({
      entity,
      user: { message: "Hi" },
      template: {
        path: "test.prompt",
        raw: tplRaw,
        slots: { scene_mood: "calm" },
      },
      timeoutMs: 1000,
      fallback: "...",
    });
    expect(provider.lastReq?.system).toContain("You are Maya");
    expect(provider.lastReq?.system).toContain("Scene mood: calm");
  });

  it("dialogue returns fallback on provider error", async () => {
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    provider.shouldThrow = true;
    const runner = new PersonaRunner({ scheduler, provider });
    const entity = buildEntity();
    const r = await runner.dialogue({
      entity,
      user: { message: "Hello" },
      timeoutMs: 500,
      fallback: "deterministic-fallback",
    });
    expect(r).toBe("deterministic-fallback");
  });

  it("dialogue caches by (memoryId, prompt-hash, world-hash)", async () => {
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    provider.reply = "first";
    const runner = new PersonaRunner({ scheduler, provider });
    const entity = buildEntity();
    entity.components[Tag.Persona]!.memoryId = "mem-1";
    const r1 = await runner.dialogue({
      entity,
      user: { message: "Hello" },
      timeoutMs: 1000,
      fallback: "...",
      worldStateHash: "ws-A",
    });
    provider.reply = "second";
    const r2 = await runner.dialogue({
      entity,
      user: { message: "Hello" },
      timeoutMs: 1000,
      fallback: "...",
      worldStateHash: "ws-A",
    });
    expect(r1).toBe("first");
    // Cache hit — same key, returns "first" again, not the new "second" reply.
    expect(r2).toBe("first");
  });

  it("dialogue cache changes when recalled memory changes", async () => {
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    provider.reply = "first";
    let memoryHash = "memory-v1";
    const runner = new PersonaRunner({
      scheduler,
      provider,
      memory: {
        async recall() {
          return [];
        },
        async remember() {
          /* no-op */
        },
        async recallContext() {
          return {
            lines: ["the player helped Maya yesterday"],
            summary: "Maya trusts the player a little.",
            artifactHash: memoryHash,
          };
        },
      },
    });
    const entity = buildEntity();
    entity.components[Tag.Persona]!.memoryId = "mem-1";
    const r1 = await runner.dialogue({
      entity,
      user: { message: "Hello" },
      timeoutMs: 1000,
      fallback: "...",
      worldStateHash: "ws-A",
    });
    provider.reply = "second";
    memoryHash = "memory-v2";
    const r2 = await runner.dialogue({
      entity,
      user: { message: "Hello" },
      timeoutMs: 1000,
      fallback: "...",
      worldStateHash: "ws-A",
    });
    expect(r1).toBe("first");
    expect(r2).toBe("second");
    expect(provider.lastReq?.system).toContain("Maya trusts the player a little.");
    expect(provider.lastReq?.system).toContain("the player helped Maya yesterday");
  });
});

describe("augengine / ai phase 4 — MnemoPay persona memory adapter", () => {
  it("writes observations with persona namespace and recalls summarized context", async () => {
    const remembered: unknown[] = [];
    const adapter = createMnemoPayMemoryAdapter({
      summaryBudget: 256,
      client: {
        async remember(input) {
          remembered.push(input);
          return {};
        },
        async recall() {
          return {
            memories: [
              { content: "Maya remembers the player kept a promise.", score: 0.91 },
              { text: "The player prefers direct answers.", score: 0.74 },
              { content: "Maya distrusts rushed strangers.", score: 0.6 },
              { summary: "The atrium scene is unresolved.", score: 0.5 },
            ],
          };
        },
        async summarize(input) {
          return { summary: `summary:${input.docs.length}` };
        },
      },
    });

    await adapter.rememberObservation?.("npc/maya", {
      content: "The player brought the blue key.",
      channel: "dialogue",
      source: "player" as EntityId,
      meta: { scene: "atrium" },
    });
    const context = await adapter.recallContext?.("npc/maya", "what does Maya remember?", 4);

    expect(remembered).toEqual([
      expect.objectContaining({
        namespace: "npc/maya",
        content: "The player brought the blue key.",
        tags: ["augengine", "persona", "dialogue"],
        metadata: expect.objectContaining({
          source: "player",
          channel: "dialogue",
          scene: "atrium",
        }),
      }),
    ]);
    expect(context?.summary).toBe("summary:4");
    expect(context?.lines).toEqual([
      "Maya remembers the player kept a promise.",
      "The player prefers direct answers.",
      "Maya distrusts rushed strangers.",
      "The atrium scene is unresolved.",
    ]);
    expect(context?.artifactHash).toMatch(/^[a-f0-9]+$/);
  });
});

describe("augengine / ai phase 4 — scene + persona round-trip", () => {
  it("a Scene with a Persona entity drives a runner end-to-end", async () => {
    const spec: SceneSpec = {
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: [
        {
          id: "guide" as EntityId,
          name: "Guide",
          components: {
            [Tag.Persona]: {
              traits: ["wise", "concise"],
              voice: null,
              memoryId: null,
              goals: [],
              mood: 0.5,
            },
          },
        },
      ],
    };
    const scene = new Scene(spec);
    const guide = scene.get("guide" as EntityId)!;
    const scheduler = new AIScheduler();
    const provider = new MockProvider();
    provider.reply = "Walk east, then turn.";
    const runner = new PersonaRunner({ scheduler, provider });
    const r = await runner.dialogue({
      entity: guide,
      user: { message: "Where now?" },
      timeoutMs: 500,
      fallback: "Wait.",
    });
    expect(r).toBe("Walk east, then turn.");
  });
});

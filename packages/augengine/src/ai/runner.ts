/**
 * PersonaRunner — the orchestration layer that turns an Entity with a
 * Persona into an actual LLM round-trip.
 *
 * Wires together:
 *   - The AIScheduler (frame-budgeted, fallback-always per decision #1)
 *   - A registered LLMProvider (Anthropic / OpenAI / OpenRouter)
 *   - The PromptTemplate (decision #3 — every authored line is a template)
 *   - PersonaMemoryAdapter (MnemoPay or null) for long-term recall
 *
 * Phase 4 surface. Phase 5 (first playable) wires this into actual gameplay.
 */

import { AIScheduler } from "./scheduler.js";
import type { LLMProvider, LLMRequest } from "./providers/index.js";
import {
  parsePromptFile,
  renderTemplate,
  renderFewShot,
} from "./templates/index.js";
import {
  composeTurnContext,
  type PersonaMemoryAdapter,
  NULL_PERSONA_MEMORY,
  type PersonaPromptInput,
} from "./persona.js";
import type { Entity, PersonaComponent } from "../scene/types.js";
import { Tag } from "../scene/types.js";
import { fnv1a64 } from "../persistence/manifest.js";

export interface PersonaRunnerOptions {
  scheduler: AIScheduler;
  provider: LLMProvider;
  memory?: PersonaMemoryAdapter;
}

export interface DialogueRequest {
  /** The speaking entity. Must carry a Persona component. */
  entity: Entity;
  /** What the user/other entity is saying or doing. */
  user: PersonaPromptInput;
  /** Optional template path — if provided, the runner loads + renders it
   *  for the system prompt. Otherwise composeTurnPrompt is used. */
  template?: { path: string; raw: string; slots?: Record<string, string> };
  /** Hard timeout for the AI call. Default 800ms (decision #1 default). */
  timeoutMs?: number;
  /** Deterministic fallback line. Required — never blocks the frame. */
  fallback: string;
  /** Hash of current world state — used as cache key + AI artifact key. */
  worldStateHash?: string;
}

export class PersonaRunner {
  private scheduler: AIScheduler;
  private provider: LLMProvider;
  private memory: PersonaMemoryAdapter;
  private lastMemoryHash = "no-memory";

  constructor(opts: PersonaRunnerOptions) {
    this.scheduler = opts.scheduler;
    this.provider = opts.provider;
    this.memory = opts.memory ?? NULL_PERSONA_MEMORY;
    // Register a dialogue handler that uses our provider.
    this.scheduler.register("dialogue", async (payload, signal) => {
      const r = await this.provider.complete(payload as LLMRequest, signal);
      return r.text;
    });
  }

  /** One dialogue turn for an entity with a persona. */
  async dialogue(req: DialogueRequest): Promise<string> {
    const persona = req.entity.components[Tag.Persona] as
      | import("../scene/types.js").PersonaComponent
      | undefined;
    if (!persona) {
      throw new Error(`PersonaRunner.dialogue: entity ${req.entity.id} has no Persona`);
    }

    const built = await this.buildLLMRequest(persona, req);
    const llmReq = built.request;
    this.lastMemoryHash = built.memoryHash;
    const cacheKey = this.cacheKey(persona, req);

    return await this.scheduler.dispatch<string>({
      task: "dialogue",
      payload: llmReq,
      timeoutMs: req.timeoutMs ?? 800,
      priority: 2,
      fallback: () => req.fallback,
      cacheKey,
    });
  }

  private async buildLLMRequest(
    persona: PersonaComponent,
    req: DialogueRequest,
  ): Promise<{ request: LLMRequest; memoryHash: string }> {
    let system: string;
    let memoryHash = "template-no-memory";
    if (req.template) {
      // Decision #3 path: render the template + few-shot block as system prompt.
      const parsed = parsePromptFile(req.template.path, req.template.raw);
      const slots = req.template.slots ?? {};
      const slotsWithDefaults: Record<string, string> = {
        persona_name: req.entity.name,
        persona_traits: persona.traits.join(", "),
        persona_mood: String(persona.mood),
        ...slots,
      };
      const fewShot = renderFewShot(parsed.template);
      const body = renderTemplate(parsed.template, slotsWithDefaults);
      system = fewShot ? `${fewShot}\n\n${body}` : body;
    } else {
      // Phase-1 fallback: compose-turn-prompt does the legwork.
      const context = await composeTurnContext({
        entityId: req.entity.id,
        entityName: req.entity.name,
        persona,
        user: req.user,
        memory: this.memory,
      });
      system = context.prompt;
      memoryHash = context.memoryHash;
    }

    return {
      request: {
        system,
        messages: [{ role: "user", content: req.user.message }],
        maxTokens: 256,
        temperature: 0.7,
      },
      memoryHash,
    };
  }

  private cacheKey(persona: PersonaComponent, req: DialogueRequest): string {
    const personaPart = persona.memoryId ?? "anon";
    const userPart = fnv1a64(req.user.message);
    const worldPart = req.worldStateHash ?? "_";
    return `dialogue:${personaPart}:${userPart}:${worldPart}:${this.lastMemoryHash}`;
  }
}

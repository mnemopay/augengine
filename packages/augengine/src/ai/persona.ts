/**
 * Persona helpers — Decision #2 of AUGENGINE-ARCHITECTURE.md.
 *
 * Persona is a component, not a class. Any entity *can* carry one; most
 * don't. These helpers compose persona state into LLM prompts and route
 * memory through MnemoPay's recall when configured.
 */

import type { Entity, EntityId } from "../scene/types.js";
import type { PersonaComponent, PersonaGoal } from "../scene/types.js";
import { Tag } from "../scene/types.js";

export interface PersonaPromptInput {
  /** What the user (or other entity) is saying / doing. */
  message: string;
  /** Speaker identity for context. */
  fromName?: string;
  /** Recent observation history snippet (formatted). */
  context?: string;
}

/**
 * Render a Persona into the system-prompt prefix consumed by an LLM.
 * Phase-1 minimal renderer — phase-3 Replaces with real `.prompt` template
 * loader (decision #3) once we ship the prompt-file pipeline.
 */
export function renderPersonaSystem(p: PersonaComponent): string {
  const lines: string[] = [];
  if (p.role) {
    lines.push(`Role: ${p.role}${p.faction ? ` (${p.faction})` : ""}.`);
  }
  if (p.traits.length > 0) {
    lines.push(`Traits: ${p.traits.join(", ")}.`);
  }
  if (p.biasTags && p.biasTags.length > 0) {
    lines.push(`Biases: ${p.biasTags.join(", ")}.`);
  }
  if (p.axes) {
    // Surface only the axes that meaningfully deviate from neutral (0.5).
    // Keeps the system prompt compact — LLMs over-anchor on every value
    // you put in front of them.
    const notable: string[] = [];
    const push = (label: string, v: number) => {
      const direction = v > 0.66 ? "high" : v < 0.34 ? "low" : null;
      if (direction) notable.push(`${label}=${direction}`);
    };
    push("aggression", p.axes.aggression);
    push("fear", p.axes.fear);
    push("loyalty", p.axes.loyalty);
    push("curiosity", p.axes.curiosity);
    if (notable.length > 0) lines.push(`Disposition: ${notable.join(", ")}.`);
  }
  if (p.mood !== 0) {
    const moodWord =
      p.mood > 0.5 ? "elated" : p.mood > 0 ? "pleased" : p.mood < -0.5 ? "angry" : "displeased";
    lines.push(`Current mood: ${moodWord} (${p.mood.toFixed(2)}).`);
  }
  if (p.goals.length > 0) {
    lines.push("Active goals:");
    for (const g of p.goals.slice().sort((a, b) => b.priority - a.priority)) {
      lines.push(`  - ${g.description} (priority ${g.priority})`);
    }
  }
  return lines.join("\n");
}

/**
 * Default-fill missing axes with the neutral 0.5 baseline. Useful for
 * the scheduler which expects an `axes` to be present — call this once
 * at NPC spawn and store the result.
 */
export function ensurePersonaAxes(p: PersonaComponent): PersonaComponent {
  if (p.axes) return p;
  return {
    ...p,
    axes: { aggression: 0.5, fear: 0.5, loyalty: 0.5, curiosity: 0.5 },
  };
}

export function activeGoals(p: PersonaComponent): PersonaGoal[] {
  return [...p.goals].sort((a, b) => b.priority - a.priority);
}

export function getPersona(e: Entity): PersonaComponent | undefined {
  return e.components[Tag.Persona] as PersonaComponent | undefined;
}

export function ensurePersona(e: Entity, defaults?: Partial<PersonaComponent>): PersonaComponent {
  const existing = e.components[Tag.Persona] as PersonaComponent | undefined;
  if (existing) return existing;
  const persona: PersonaComponent = {
    traits: defaults?.traits ?? [],
    voice: defaults?.voice ?? null,
    memoryId: defaults?.memoryId ?? null,
    goals: defaults?.goals ?? [],
    mood: defaults?.mood ?? 0,
  };
  e.components[Tag.Persona] = persona;
  return persona;
}

/**
 * Adapter callback shape for MnemoPay-backed persona memory. Wired by
 * the consumer at app startup to bridge `Persona.memoryId` → real recall.
 */
export interface PersonaMemoryAdapter {
  recall(memoryId: string, query: string, limit?: number): Promise<string[]>;
  remember(memoryId: string, fact: string, tags?: string[]): Promise<void>;
  recallContext?(memoryId: string, query: string, limit?: number): Promise<PersonaMemoryContext>;
  rememberObservation?(memoryId: string, observation: PersonaObservation): Promise<void>;
}

export interface PersonaMemoryContext {
  lines: string[];
  summary?: string;
  artifactHash: string;
}

export interface PersonaObservation {
  content: string;
  tags?: string[];
  source?: EntityId | null;
  channel?: string;
  ts?: string;
  meta?: Record<string, unknown>;
}

/** Default no-op adapter — used in dev / offline / before MnemoPay is wired. */
export const NULL_PERSONA_MEMORY: PersonaMemoryAdapter = {
  async recall() {
    return [];
  },
  async remember() {
    /* no-op */
  },
};

export interface PersonaTurnInput {
  entityId: EntityId;
  entityName?: string;
  persona: PersonaComponent;
  user: PersonaPromptInput;
  memory?: PersonaMemoryAdapter;
}

export interface PersonaTurnContext {
  prompt: string;
  memoryHash: string;
}

/**
 * Compose the full prompt body for an NPC's next turn. Uses the persona
 * traits + recalled memory + provided context. Returned string is fed to
 * an LLM via the AIScheduler with a "dialogue" task.
 */
export async function composeTurnContext(input: PersonaTurnInput): Promise<PersonaTurnContext> {
  const { entityName, persona, user, memory } = input;
  const sys = renderPersonaSystem(persona);
  const identity = entityName ? `You are ${entityName}.` : `Entity: ${input.entityId}.`;
  let memoryBlock = "";
  let memoryHash = "no-memory";
  if (persona.memoryId && memory) {
    const context = memory.recallContext
      ? await memory.recallContext(persona.memoryId, user.message, 5)
      : {
          lines: await memory.recall(persona.memoryId, user.message, 5),
          artifactHash: "legacy-memory",
        };
    memoryHash = context.artifactHash;
    const recalled = context.summary
      ? [context.summary, ...context.lines]
      : context.lines;
    if (recalled.length > 0) {
      memoryBlock = `\nRecalled memories:\n${recalled.map((r) => `  - ${r}`).join("\n")}`;
    }
  }
  const ctx = user.context ? `\n\nRecent context:\n${user.context}` : "";
  return {
    prompt: `${identity}\n${sys}${memoryBlock}${ctx}\n\nUser (${user.fromName ?? "speaker"}): ${user.message}\nReply:`,
    memoryHash,
  };
}

export async function composeTurnPrompt(input: PersonaTurnInput): Promise<string> {
  return (await composeTurnContext(input)).prompt;
}

/**
 * MnemoPay-backed PersonaMemoryAdapter.
 *
 * Wires `Persona.memoryId` to MnemoPay's `recall` + `remember` so persona
 * memory persists across sessions, multiplayer clients, and even across
 * different blackpig titles (the memoryId is portable).
 *
 * Optional dependency on `@mnemopay/sdk`. If the SDK isn't installed at
 * runtime, fall back to NULL_PERSONA_MEMORY semantics.
 */

import type {
  PersonaMemoryAdapter,
  PersonaMemoryContext,
  PersonaObservation,
} from "../persona.js";
import { fnv1a64 } from "../../persistence/manifest.js";

export interface MnemoPayClientLike {
  recall(input: { memoryId?: string; namespace?: string; query: string; limit?: number; mode?: string }): Promise<
    | Array<string | { content?: string; text?: string; summary?: string; score?: number }>
    | { memories?: Array<{ content?: string; text?: string; summary?: string; score?: number }>; results?: Array<{ content?: string; text?: string; summary?: string; score?: number }> }
  >;
  remember(input: { memoryId?: string; namespace?: string; content: string; tags?: string[]; metadata?: Record<string, unknown> }): Promise<unknown>;
  summarize?(input: { docs: string[]; maxTokens?: number }): Promise<string | { summary?: string; text?: string }>;
  postProcess?(input: { text: string; grounding?: string[]; persona?: string }): Promise<string | { text?: string; output?: string }>;
}

export interface MnemoPayMemoryOptions {
  /** A pre-instantiated MnemoPay client. Caller wires the SDK. */
  client: MnemoPayClientLike;
  mode?: "hyde" | "vector" | "hybrid" | "score";
  summaryBudget?: number;
}

export function createMnemoPayMemoryAdapter(
  opts: MnemoPayMemoryOptions,
): PersonaMemoryAdapter {
  const { client } = opts;
  return {
    async recall(memoryId, query, limit = 5) {
      return recallLines(client, memoryId, query, limit, opts.mode);
    },
    async remember(memoryId, fact, tags) {
      await client.remember({
        namespace: memoryId,
        content: fact,
        ...(tags ? { tags } : {}),
      });
    },
    async rememberObservation(memoryId, observation: PersonaObservation) {
      await client.remember({
        namespace: memoryId,
        content: observation.content,
        tags: observation.tags ?? compact(["augengine", "persona", observation.channel]),
        metadata: {
          source: observation.source ?? null,
          channel: observation.channel ?? null,
          ts: observation.ts ?? new Date().toISOString(),
          ...(observation.meta ?? {}),
        },
      });
    },
    async recallContext(memoryId, query, limit = 8): Promise<PersonaMemoryContext> {
      const lines = await recallLines(client, memoryId, query, limit, opts.mode);
      const summary = await summarize(client, lines, opts.summaryBudget ?? 1500);
      const material = [memoryId, query, summary ?? "", ...lines].join("\n---\n");
      return {
        lines,
        ...(summary ? { summary } : {}),
        artifactHash: fnv1a64(material),
      };
    },
  };
}

async function recallLines(
  client: MnemoPayClientLike,
  memoryId: string,
  query: string,
  limit: number,
  mode?: string,
): Promise<string[]> {
  const response = await client.recall({
    namespace: memoryId,
    memoryId,
    query,
    limit,
    ...(mode ? { mode } : {}),
  });
  const rows = Array.isArray(response)
    ? response
    : response.memories ?? response.results ?? [];
  return rows
    .map((row) => typeof row === "string" ? row : row.content ?? row.text ?? row.summary ?? "")
    .map((line) => line.trim())
    .filter(Boolean);
}

async function summarize(
  client: MnemoPayClientLike,
  lines: string[],
  maxTokens: number,
): Promise<string | undefined> {
  if (!client.summarize || lines.length <= 3) return undefined;
  const result = await client.summarize({ docs: lines, maxTokens });
  const text = typeof result === "string" ? result : result.summary ?? result.text;
  return text?.trim() || undefined;
}

function compact(values: Array<string | null | undefined>): string[] {
  return values.filter((value): value is string => Boolean(value));
}

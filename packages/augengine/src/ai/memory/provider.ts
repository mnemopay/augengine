import type { PersonaMemoryAdapter, PersonaMemoryContext, PersonaObservation } from "../persona.js";
import type { InProcessBrain } from "./brain.js";
import { fnv1a64 } from "../../persistence/manifest.js";

/**
 * Wraps the InProcessBrain to satisfy the PersonaMemoryAdapter interface required by the persona runner.
 *
 * `InProcessBrain.recall()` returns the `MnemoPayClientLike` union — either an
 * array of rows or an object with `memories` / `results`. Normalize both shapes
 * down to a flat `string[]` of memory texts in one place.
 */
type RecallRow = string | { content?: string; text?: string; summary?: string };
type RecallResult =
  | RecallRow[]
  | { memories?: RecallRow[]; results?: RecallRow[] };

function normalize(res: RecallResult): string[] {
  const rows: RecallRow[] = Array.isArray(res)
    ? res
    : (res.memories ?? res.results ?? []);
  return rows
    .map((row) => (typeof row === "string" ? row : row.content ?? row.text ?? row.summary ?? ""))
    .map((line) => line.trim())
    .filter(Boolean);
}

export class BrainMemoryProvider implements PersonaMemoryAdapter {
  constructor(private brain: InProcessBrain) {}

  async recall(memoryId: string, query: string, limit = 5): Promise<string[]> {
    const res = await this.brain.recall({ namespace: memoryId, query, limit });
    return normalize(res as RecallResult);
  }

  async remember(memoryId: string, fact: string, tags?: string[]): Promise<void> {
    await this.brain.remember({
      namespace: memoryId,
      content: fact,
      ...(tags ? { tags } : {}),
    });
  }

  async recallContext(memoryId: string, query: string, limit = 5): Promise<PersonaMemoryContext> {
    const res = await this.brain.recall({ namespace: memoryId, query, limit });
    const lines = normalize(res as RecallResult);

    // Hash the retrieved lines to use as a cache key for the LLM runner
    const combined = lines.join("|");
    const artifactHash = combined.length > 0 ? fnv1a64(combined) : "empty-memory";

    return { lines, artifactHash };
  }

  async rememberObservation(memoryId: string, observation: PersonaObservation): Promise<void> {
    const metadata: Record<string, unknown> = { ...(observation.meta ?? {}) };
    if (observation.source != null) metadata.source = observation.source;
    if (observation.channel != null) metadata.channel = observation.channel;
    if (observation.ts != null) metadata.ts = observation.ts;
    await this.brain.remember({
      namespace: memoryId,
      content: observation.content,
      ...(observation.tags ? { tags: observation.tags } : {}),
      metadata,
    });
  }
}
